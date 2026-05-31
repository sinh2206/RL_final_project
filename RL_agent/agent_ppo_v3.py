"""
agent_ppo_v3.py - Orbit Wars
Hybrid MAPPO inference + self-play inspired heuristic fallback.

Design goals:
- Never crash on Kaggle raw-python loader (__file__ may be missing).
- Keep per-turn compute under ~0.8s with strict deadline checks.
- Always return valid actions; avoid sun-crossing trajectories.
- If model/weights are missing, use strong deterministic fallback.
"""

import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Keep heavy ML imports out of module import path to avoid cold-start timeouts.
ort = None
ORT_AVAILABLE = False
_ORT_IMPORT_TRIED = False

# Torch fallback is disabled by default; enable explicitly via env if needed.
torch = None
nn = None
TORCH_AVAILABLE = False
ENABLE_ONNX_INFERENCE = os.environ.get("MAPPO_USE_ONNX", "1") != "0"
ENABLE_TORCH_INFERENCE = os.environ.get("MAPPO_USE_TORCH", "0") == "1"


# -----------------------------
# Constants
# -----------------------------

BOARD = 100.0
SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 1.5
LAUNCH_CLEARANCE = 0.1
ROTATION_LIMIT = 50.0
MAX_SPEED = 6.0

MAX_PLANETS = 48
MAX_FLEETS = 96
SHIP_FRACTIONS = (0.20, 0.35, 0.50, 0.65, 0.80)
NUM_FRAC = len(SHIP_FRACTIONS)
MAX_MOVES_PER_TURN = 8
SELFPLAY_TOP_K = 6
COMET_MIN_TTL_BUFFER = 2.0

# Deadline policy: keep safety margin below 1s actTimeout.
TURN_BUDGET = 0.78
HARD_STOP_BUFFER = 0.04
NO_MODEL_FALLBACK_BUDGET = 0.74

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2


def _base_dir() -> str:
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


def _resolve_weights_path() -> str:
    # 1) explicit env override
    env_path = os.environ.get("MAPPO_WEIGHTS_PATH", "").strip()
    if env_path:
        return env_path
    # 2) project-local defaults
    base = _base_dir()
    cands = [
        os.path.join(base, "ppo_v3_weights.pt"),
        os.path.join(base, "ppo_v2_weights.pt"),
        os.path.join(base, "mappo_weights.pt"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    # keep first as canonical output path for offline training
    return cands[0]


def _resolve_onnx_path() -> str:
    env_path = os.environ.get("MAPPO_ONNX_PATH", "").strip()
    if env_path:
        return env_path
    base = _base_dir()
    cands = [
        os.path.join(base, "mappo_v3.onnx"),
        os.path.join(base, "ppo_v3.onnx"),
        os.path.join(base, "ppo_model.onnx"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return cands[0]


ONNX_PATH = _resolve_onnx_path()
WEIGHTS_PATH = _resolve_weights_path()
DEVICE = torch.device("cpu") if TORCH_AVAILABLE else None


# -----------------------------
# Safe IO helpers
# -----------------------------

def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _warn_once(msg: str):
    if not hasattr(_warn_once, "_seen"):
        _warn_once._seen = set()  # type: ignore[attr-defined]
    seen = _warn_once._seen  # type: ignore[attr-defined]
    if msg in seen:
        return
    seen.add(msg)
    print(msg, file=sys.stderr, flush=True)


# -----------------------------
# Geometry / physics
# -----------------------------

def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def fleet_speed(ships: float) -> float:
    ships = max(1.0, _safe_float(ships, 1.0))
    ratio = math.log(ships) / math.log(1000.0)
    ratio = _clip(ratio, 0.0, 1.0)
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def line_seg_min_dist(x1, y1, x2, y2, px, py):
    dx = x2 - x1
    dy = y2 - y1
    den = dx * dx + dy * dy
    if den <= 1e-12:
        return dist(x1, y1, px, py)
    t = ((px - x1) * dx + (py - y1) * dy) / den
    t = _clip(t, 0.0, 1.0)
    qx = x1 + t * dx
    qy = y1 + t * dy
    return dist(qx, qy, px, py)


def path_crosses_sun(src_x, src_y, angle, src_radius, flight_dist=130.0):
    sx = src_x + math.cos(angle) * (src_radius + LAUNCH_CLEARANCE)
    sy = src_y + math.sin(angle) * (src_radius + LAUNCH_CLEARANCE)
    ex = sx + math.cos(angle) * flight_dist
    ey = sy + math.sin(angle) * flight_dist
    d = line_seg_min_dist(sx, sy, ex, ey, SUN_X, SUN_Y)
    return d < (SUN_RADIUS + SUN_MARGIN)


def _segment_hits_sun_to_target(src, tx, ty, trg_radius):
    angle = math.atan2(ty - src["y"], tx - src["x"])
    sx = src["x"] + math.cos(angle) * (src["radius"] + LAUNCH_CLEARANCE)
    sy = src["y"] + math.sin(angle) * (src["radius"] + LAUNCH_CLEARANCE)
    hit_dist = max(0.0, dist(src["x"], src["y"], tx, ty) - src["radius"] - trg_radius)
    ex = sx + math.cos(angle) * hit_dist
    ey = sy + math.sin(angle) * hit_dist
    d = line_seg_min_dist(sx, sy, ex, ey, SUN_X, SUN_Y)
    return d < (SUN_RADIUS + SUN_MARGIN)


def _is_orbiting_planet(p):
    r = dist(p["x"], p["y"], SUN_X, SUN_Y)
    return (r + p["radius"]) < ROTATION_LIMIT


def predict_orbit(x, y, omega, turns):
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    r = dist(x, y, SUN_X, SUN_Y)
    th2 = theta + omega * turns
    return SUN_X + r * math.cos(th2), SUN_Y + r * math.sin(th2)


def estimate_arrival_turns(src, tx, ty, trg_radius, ships):
    speed = max(1e-6, fleet_speed(ships))
    d = max(0.0, dist(src["x"], src["y"], tx, ty) - src["radius"] - trg_radius)
    return d / speed


def find_angle_to_moving_planet(src, trg, ships, omega):
    t = estimate_arrival_turns(src, trg["x"], trg["y"], trg["radius"], ships)
    t = _clip(t, 0.0, 140.0)
    tx, ty = trg["x"], trg["y"]

    for _ in range(16):
        tx, ty = predict_orbit(trg["x"], trg["y"], omega, t)
        t2 = estimate_arrival_turns(src, tx, ty, trg["radius"], ships)
        if abs(t2 - t) < 0.05:
            t = t2
            break
        t = t2

    if _segment_hits_sun_to_target(src, tx, ty, trg["radius"]):
        return None, None, None, None
    angle = math.atan2(ty - src["y"], tx - src["x"])
    return angle, t, tx, ty


# -----------------------------
# Parse world
# -----------------------------

def _build_comet_index(obs):
    comet_ids = set()
    raw_ids = _read(obs, "comet_planet_ids", []) or []
    for pid in raw_ids:
        comet_ids.add(_safe_int(pid, -1))

    comet_index = {}
    raw_comets = _read(obs, "comets", []) or []
    for group in raw_comets:
        if not isinstance(group, dict):
            continue
        pids = group.get("planet_ids", [])
        paths = group.get("paths", [])
        path_index = max(0, _safe_int(group.get("path_index", 0), 0))
        if not isinstance(pids, list) or not isinstance(paths, list):
            continue
        for i, pid in enumerate(pids):
            pid_i = _safe_int(pid, -1)
            comet_ids.add(pid_i)
            path = paths[i] if i < len(paths) and isinstance(paths[i], list) else []
            remaining = max(0, len(path) - path_index)
            comet_index[pid_i] = {
                "remaining": float(remaining),
                "path_index": path_index,
                "path": path,
            }

    for pid in comet_ids:
        if pid not in comet_index:
            comet_index[pid] = {"remaining": 999.0, "path_index": 0, "path": []}
    return comet_index


def _parse_world(obs) -> Dict:
    player = _safe_int(_read(obs, "player", 0), 0)
    step = _safe_int(_read(obs, "step", 0), 0)
    omega = _safe_float(_read(obs, "angular_velocity", 0.0), 0.0)

    raw_p = _read(obs, "planets", []) or []
    raw_f = _read(obs, "fleets", []) or []
    comet_index = _build_comet_index(obs)

    planets = []
    raw_planet_map = {}
    for row in raw_p:
        if row is None or len(row) < 7:
            continue
        pid = _safe_int(row[0], -1)
        cinfo = comet_index.get(pid, None)
        p = {
            "id": pid,
            "owner": _safe_int(row[1], -1),
            "x": _safe_float(row[2], 0.0),
            "y": _safe_float(row[3], 0.0),
            "radius": max(0.1, _safe_float(row[4], 1.0)),
            "ships": max(0.0, _safe_float(row[5], 0.0)),
            "production": max(0.0, _safe_float(row[6], 0.0)),
            "is_comet": cinfo is not None,
            "comet_ttl": _safe_float(cinfo["remaining"], 999.0) if cinfo is not None else 999.0,
        }
        p["is_orbiting"] = _is_orbiting_planet(p)
        planets.append(p)
        raw_planet_map[pid] = row

    fleets = []
    for row in raw_f:
        if row is None or len(row) < 7:
            continue
        fleets.append(
            {
                "id": _safe_int(row[0], -1),
                "owner": _safe_int(row[1], -1),
                "x": _safe_float(row[2], 0.0),
                "y": _safe_float(row[3], 0.0),
                "angle": _safe_float(row[4], 0.0),
                "from_planet_id": _safe_int(row[5], -1),
                "ships": max(0.0, _safe_float(row[6], 0.0)),
            }
        )

    my_planets = [p for p in planets if p["owner"] == player]
    targets = [p for p in planets if p["owner"] != player]
    enemy_planets = [p for p in planets if p["owner"] not in (-1, player)]
    return {
        "player": player,
        "step": step,
        "omega": omega,
        "planets": planets,
        "fleets": fleets,
        "my_planets": my_planets,
        "targets": targets,
        "enemy_planets": enemy_planets,
        "comet_index": comet_index,
        "raw_planet_map": raw_planet_map,
    }


# -----------------------------
# MAPPO model (optional runtime)
# -----------------------------

if TORCH_AVAILABLE:
    class SetEncoder(nn.Module):
        def __init__(self, d_in, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
            super().__init__()
            self.proj = nn.Linear(d_in, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=2 * d_model,
                batch_first=True,
                activation="gelu",
                dropout=0.0,
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

        def forward(self, x, mask):
            return self.enc(self.proj(x), src_key_padding_mask=mask)


    class MAPPONet(nn.Module):
        PLANET_FEATS = 12
        FLEET_FEATS = 9

        def __init__(self):
            super().__init__()
            self.planet_enc = SetEncoder(self.PLANET_FEATS)
            self.fleet_enc = SetEncoder(self.FLEET_FEATS)
            self.cross = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)

            self.target_score = nn.Sequential(
                nn.Linear(2 * D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, 1)
            )
            self.noop_head = nn.Sequential(
                nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, 1)
            )
            self.frac_head = nn.Sequential(
                nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, NUM_FRAC)
            )

            self.critic = nn.Sequential(
                nn.Linear(2 * D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, 1)
            )

        def forward(self, planets, p_mask, fleets, f_mask):
            hp = self.planet_enc(planets, p_mask)
            hf = self.fleet_enc(fleets, f_mask)
            hpf, _ = self.cross(hp, hf, hf, key_padding_mask=f_mask, need_weights=False)
            h = hp + hpf

            b, p, d = h.shape
            pair = torch.cat(
                [
                    h.unsqueeze(2).expand(b, p, p, d),
                    h.unsqueeze(1).expand(b, p, p, d),
                ],
                dim=-1,
            )
            tgt_scores = self.target_score(pair).squeeze(-1)
            tgt_scores = tgt_scores.masked_fill(p_mask.unsqueeze(1).expand(b, p, p), -1e9)

            noop = self.noop_head(h).squeeze(-1).unsqueeze(-1)
            target_logits = torch.cat([tgt_scores, noop], dim=-1)
            frac_logits = self.frac_head(h)

            pad = p_mask.unsqueeze(-1)
            denom = (~p_mask).sum(1, keepdim=True).clamp_min(1).float()
            mean_pool = h.masked_fill(pad, 0.0).sum(1) / denom
            max_pool = h.masked_fill(pad, -1e9).max(1).values
            value = self.critic(torch.cat([mean_pool, max_pool], dim=-1)).squeeze(-1)
            return target_logits, frac_logits, value


def _encode_obs(obs, player):
    raw_p = _read(obs, "planets", []) or []
    raw_f = _read(obs, "fleets", []) or []
    comet_ids = set(_safe_int(pid, -1) for pid in (_read(obs, "comet_planet_ids", []) or []))
    ang_v = _safe_float(_read(obs, "angular_velocity", 0.0), 0.0)

    p = min(len(raw_p), MAX_PLANETS)
    f = min(len(raw_f), MAX_FLEETS)

    planet_dim = 12
    fleet_dim = 9

    parr = np.zeros((MAX_PLANETS, planet_dim), dtype=np.float32)
    pmask = np.ones(MAX_PLANETS, dtype=bool)
    pids = [-1] * MAX_PLANETS

    for i in range(p):
        if raw_p[i] is None or len(raw_p[i]) < 7:
            continue
        pid, owner, x, y, r, ships, prod = raw_p[i]
        is_comet = float(_safe_int(pid, -1) in comet_ids)
        parr[i] = (
            x / BOARD,
            y / BOARD,
            (x - 50.0) / BOARD,
            (y - 50.0) / BOARD,
            r / 5.0,
            math.log1p(max(0.0, _safe_float(ships, 0.0))) / 7.0,
            _safe_float(prod, 0.0) / 5.0,
            float(owner == player),
            float(owner != player and owner != -1),
            float(owner == -1),
            ang_v,
            is_comet,
        )
        pmask[i] = False
        pids[i] = _safe_int(pid, -1)

    farr = np.zeros((MAX_FLEETS, fleet_dim), dtype=np.float32)
    fmask = np.ones(MAX_FLEETS, dtype=bool)
    for i in range(f):
        if raw_f[i] is None or len(raw_f[i]) < 7:
            continue
        fid, owner, x, y, ang, frm, ships = raw_f[i]
        farr[i] = (
            x / BOARD,
            y / BOARD,
            math.cos(_safe_float(ang, 0.0)),
            math.sin(_safe_float(ang, 0.0)),
            math.log1p(max(0.0, _safe_float(ships, 0.0))) / 7.0,
            float(owner == player),
            1.0 - float(owner == player),
            (_safe_int(frm, 0) % 100) / 100.0,
            1.0,
        )
        fmask[i] = False
    # Avoid all-True masks that can create NaN attention outputs.
    if p > 0 and np.all(fmask):
        fmask[0] = False
    return parr, pmask, pids, farr, fmask


_MODEL = None
_MODEL_USABLE = False
_MODEL_LOAD_TRIED = False
_MODEL_KIND = None
_ONNX_INPUT_NAMES = ()


def _ensure_onnxruntime():
    global ort, ORT_AVAILABLE, _ORT_IMPORT_TRIED
    if _ORT_IMPORT_TRIED:
        return ORT_AVAILABLE
    _ORT_IMPORT_TRIED = True
    try:
        import onnxruntime as _ort  # type: ignore[import-not-found]
        ort = _ort
        ORT_AVAILABLE = True
    except Exception:
        ort = None
        ORT_AVAILABLE = False
    return ORT_AVAILABLE


def _load_model():
    global _MODEL, _MODEL_USABLE, _MODEL_LOAD_TRIED, _MODEL_KIND, _ONNX_INPUT_NAMES
    if _MODEL_LOAD_TRIED:
        return _MODEL if _MODEL_USABLE else None
    _MODEL_LOAD_TRIED = True

    # Prefer ONNX for lower cold-start cost in Kaggle simulation.
    if ENABLE_ONNX_INFERENCE and os.path.exists(ONNX_PATH) and _ensure_onnxruntime():
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(
                ONNX_PATH,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            _MODEL = sess
            _MODEL_KIND = "onnx"
            _ONNX_INPUT_NAMES = tuple(inp.name for inp in sess.get_inputs())
            _MODEL_USABLE = True
            return _MODEL
        except Exception as exc:
            _warn_once(f"[agent_ppo_v3] onnx load failed: {exc}")
    elif ENABLE_ONNX_INFERENCE and os.path.exists(ONNX_PATH) and not ORT_AVAILABLE:
        _warn_once("[agent_ppo_v3] onnxruntime unavailable -> heuristic mode.")

    if not ENABLE_TORCH_INFERENCE:
        _warn_once(
            f"[agent_ppo_v3] model disabled/unavailable ({ONNX_PATH}, {WEIGHTS_PATH}) -> heuristic mode."
        )
        _MODEL_USABLE = False
        return None

    # Optional torch fallback, only when explicitly enabled.
    if not TORCH_AVAILABLE:
        try:
            import torch as _torch  # type: ignore[import-not-found]
            import torch.nn as _nn  # type: ignore[import-not-found]
            globals()["torch"] = _torch
            globals()["nn"] = _nn
            globals()["TORCH_AVAILABLE"] = True
            globals()["DEVICE"] = _torch.device("cpu")
        except Exception:
            _warn_once("[agent_ppo_v3] torch import failed -> heuristic mode.")
            _MODEL_USABLE = False
            return None

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    if not os.path.exists(WEIGHTS_PATH):
        _warn_once(
            f"[agent_ppo_v3] missing models ({ONNX_PATH}, {WEIGHTS_PATH}) -> heuristic mode."
        )
        _MODEL_USABLE = False
        return None

    if "MAPPONet" not in globals():
        _warn_once("[agent_ppo_v3] torch model classes unavailable -> heuristic mode.")
        _MODEL_USABLE = False
        return None

    try:
        m = MAPPONet().to(DEVICE)  # type: ignore[misc]
        state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        if isinstance(state, dict):
            m.load_state_dict(state, strict=False)
        else:
            _warn_once("[agent_ppo_v3] invalid weight format -> heuristic mode.")
            _MODEL_USABLE = False
            return None
        m.eval()
        _MODEL = m
        _MODEL_KIND = "torch"
        _MODEL_USABLE = True
        return _MODEL
    except Exception as exc:
        _warn_once(f"[agent_ppo_v3] load model failed: {exc} -> heuristic mode.")
        _MODEL_USABLE = False
        return None


def _run_model(model, obs, player):
    parr, pmask, pids, farr, fmask = _encode_obs(obs, player)
    if _MODEL_KIND == "onnx":
        feeds = {}
        names = _ONNX_INPUT_NAMES

        def _choose_name(cands, fallback_idx):
            low = {n.lower(): n for n in names}
            for c in cands:
                for k, n in low.items():
                    if c in k:
                        return n
            if 0 <= fallback_idx < len(names):
                return names[fallback_idx]
            return None

        if len(names) >= 4:
            n_planets = _choose_name(("planet", "planets", "obs_planet", "p_in"), 0)
            n_pmask = _choose_name(("p_mask", "pmask", "planet_mask"), 1)
            n_fleets = _choose_name(("fleet", "fleets", "obs_fleet", "f_in"), 2)
            n_fmask = _choose_name(("f_mask", "fmask", "fleet_mask"), 3)
            if None in (n_planets, n_pmask, n_fleets, n_fmask):
                return None, None, None
            feeds[n_planets] = parr[None, ...].astype(np.float32)
            feeds[n_pmask] = pmask[None, ...].astype(np.bool_)
            feeds[n_fleets] = farr[None, ...].astype(np.float32)
            feeds[n_fmask] = fmask[None, ...].astype(np.bool_)
        elif len(names) == 1:
            flat = np.concatenate(
                [parr.reshape(-1), pmask.astype(np.float32), farr.reshape(-1), fmask.astype(np.float32)]
            ).astype(np.float32)
            feeds[names[0]] = flat[None, ...]
        else:
            return None, None, None

        outs = model.run(None, feeds)
        if len(outs) < 2:
            return None, None, None
        tl = np.asarray(outs[0])
        fl = np.asarray(outs[1])
        if tl.ndim == 3:
            tl = tl[0]
        if fl.ndim == 3:
            fl = fl[0]
        return pids, tl, fl

    if _MODEL_KIND == "torch":
        with torch.no_grad():
            tl, fl, _ = model(
                torch.from_numpy(parr).unsqueeze(0),
                torch.from_numpy(pmask).unsqueeze(0),
                torch.from_numpy(farr).unsqueeze(0),
                torch.from_numpy(fmask).unsqueeze(0),
            )
        return pids, tl.squeeze(0).cpu().numpy(), fl.squeeze(0).cpu().numpy()

    return None, None, None


# -----------------------------
# Decision helpers
# -----------------------------

def _enemy_pressure_on_target(target, world):
    player = world["player"]
    pressure = 0.0
    for ep in world["enemy_planets"]:
        d = dist(ep["x"], ep["y"], target["x"], target["y"])
        if d > 55.0:
            continue
        pressure += ep["ships"] / (d + 9.0)
    for f in world["fleets"]:
        if f["owner"] in (-1, player):
            continue
        d = dist(f["x"], f["y"], target["x"], target["y"])
        if d > 30.0:
            continue
        pressure += f["ships"] / (d + 6.0)
    return pressure


def _friendly_pressure_on_target(target, world):
    player = world["player"]
    pressure = 0.0
    for mp in world["my_planets"]:
        if mp["id"] == target["id"]:
            continue
        d = dist(mp["x"], mp["y"], target["x"], target["y"])
        if d > 55.0:
            continue
        pressure += mp["ships"] / (d + 10.0)
    for f in world["fleets"]:
        if f["owner"] != player:
            continue
        d = dist(f["x"], f["y"], target["x"], target["y"])
        if d > 30.0:
            continue
        pressure += f["ships"] / (d + 6.0)
    return pressure


def _planet_danger(planet, world):
    enemy = _enemy_pressure_on_target(planet, world)
    friendly = _friendly_pressure_on_target(planet, world)
    deficit = enemy * 7.0 - (planet["ships"] + 4.2 * friendly)
    danger = max(0.0, deficit)
    if planet["production"] >= 3.0:
        danger += 0.45 * enemy
    return danger


def _dynamic_reserve(src, world):
    step = world["step"]
    enemy_near = 0.0
    friendly_near = 0.0
    for ep in world["enemy_planets"]:
        d = dist(src["x"], src["y"], ep["x"], ep["y"])
        if d > 60.0:
            continue
        enemy_near += ep["ships"] / (d + 8.0)
    for mp in world["my_planets"]:
        if mp["id"] == src["id"]:
            continue
        d = dist(src["x"], src["y"], mp["x"], mp["y"])
        if d > 60.0:
            continue
        friendly_near += mp["ships"] / (d + 8.0)

    reserve = 2.0 + 1.8 * src["production"]
    if step < 140:
        reserve += 2.0
    elif step > 340:
        reserve -= 1.5
    reserve += 2.8 * max(0.0, enemy_near - 0.75 * friendly_near)
    reserve = _clip(reserve, 2.0, max(4.0, src["ships"] * 0.7))
    return int(reserve)


def _estimated_need(src, trg, world, eta):
    player = world["player"]
    if trg.get("is_comet", False) and eta > max(0.0, trg.get("comet_ttl", 999.0) - COMET_MIN_TTL_BUFFER):
        return float("inf")

    growth = trg["production"] * eta
    if trg["owner"] == player:
        need = max(3.0, _planet_danger(trg, world) + 2.0)
    elif trg["owner"] == -1:
        need = trg["ships"] + 1.0 + 0.22 * growth
    else:
        need = trg["ships"] + 2.0 + 0.60 * growth

    pressure = _enemy_pressure_on_target(trg, world)
    support = _friendly_pressure_on_target(trg, world)
    if trg["owner"] != player:
        need += 0.50 * pressure
        need = max(1.0, need - 0.25 * support)
    return need


def _target_base_score(src, trg, player, step):
    d = dist(src["x"], src["y"], trg["x"], trg["y"])
    if trg["owner"] == player:
        owner_bonus = 2.8
    elif trg["owner"] == -1:
        owner_bonus = 8.4 if step < 230 else 6.2
    else:
        owner_bonus = 11.5 if step < 250 else 14.5

    dist_pen = (0.10 if step < 220 else 0.07) * d
    score = owner_bonus + 5.3 * trg["production"] - 0.11 * trg["ships"] - dist_pen
    if trg["is_orbiting"]:
        score -= 0.55
    if trg.get("is_comet", False):
        ttl = trg.get("comet_ttl", 999.0)
        if ttl < 8.0:
            score -= 8.0
        elif ttl < 16.0:
            score -= 3.0
        else:
            score += min(4.0, 0.12 * ttl)
    return score


def _pick_support_target(src, world, deadline):
    best = None
    best_score = -1e18
    for t in world["my_planets"]:
        if t["id"] == src["id"]:
            continue
        if time.perf_counter() > deadline - HARD_STOP_BUFFER:
            break
        d = dist(src["x"], src["y"], t["x"], t["y"])
        if d > 65.0:
            continue
        danger = _planet_danger(t, world)
        if danger < 2.0:
            continue
        score = 1.35 * danger + 1.15 * t["production"] - 0.10 * d - 0.05 * t["ships"]
        if score > best_score:
            best_score = score
            best = t
    return best, best_score


def _pick_target_selfplay_scored(src, targets, world, ships_to_send, deadline):
    player = world["player"]
    step = world["step"]
    omega = world["omega"]

    ranked = []
    for t in targets:
        if t["id"] == src["id"]:
            continue
        base = _target_base_score(src, t, player, step)
        ranked.append((base, t))
    if not ranked:
        return None, -1e18

    ranked.sort(key=lambda x: x[0], reverse=True)
    top = ranked[:SELFPLAY_TOP_K]

    best = None
    best_score = -1e18
    for base, t in top:
        if time.perf_counter() > deadline - HARD_STOP_BUFFER:
            break

        angle, eta, _, _ = _compute_attack_angle(src, t, ships_to_send, omega)
        if angle is None or eta is None:
            continue
        if t.get("is_comet", False):
            ttl = t.get("comet_ttl", 999.0)
            if eta > max(0.0, ttl - COMET_MIN_TTL_BUFFER):
                continue

        need = _estimated_need(src, t, world, eta)
        if not math.isfinite(need):
            continue

        capture_like = _clip((ships_to_send + 1.0) / max(1.0, need), 0.0, 1.75)
        pressure = _enemy_pressure_on_target(t, world)
        retaliation = max(0.0, pressure - 0.80 * ships_to_send)
        src_after = max(0.0, src["ships"] - ships_to_send)
        exposed = max(0.0, _dynamic_reserve(src, world) - src_after)

        score = (
            0.58 * base
            + 4.5 * capture_like
            - 0.64 * retaliation
            - 0.15 * eta
            - 0.70 * exposed
        )

        if t["owner"] == player:
            score += 0.85 * _planet_danger(t, world)
        if score > best_score:
            best_score = score
            best = t

    if best is None:
        return top[0][1], top[0][0] - 8.0
    return best, best_score


def _pick_target_selfplay(src, targets, world, ships_to_send, deadline):
    trg, _ = _pick_target_selfplay_scored(src, targets, world, ships_to_send, deadline)
    return trg


def _compute_attack_angle(src, trg, ships, omega):
    if trg["is_orbiting"]:
        return find_angle_to_moving_planet(src, trg, ships, omega)
    angle = math.atan2(trg["y"] - src["y"], trg["x"] - src["x"])
    if _segment_hits_sun_to_target(src, trg["x"], trg["y"], trg["radius"]):
        return None, None, None, None
    eta = estimate_arrival_turns(src, trg["x"], trg["y"], trg["radius"], ships)
    return angle, eta, trg["x"], trg["y"]


def _planet_by_id(raw_planet_map, pid):
    return raw_planet_map.get(pid, None)


def _sanitize_moves(moves, world):
    valid = []
    reserves = {}
    mine = {p["id"]: p for p in world["my_planets"]}
    for mv in moves:
        if not isinstance(mv, (list, tuple)) or len(mv) != 3:
            continue
        src_id = _safe_int(mv[0], -1)
        angle = _safe_float(mv[1], float("nan"))
        ships = _safe_int(mv[2], 0)
        src = mine.get(src_id)
        if src is None:
            continue
        if not math.isfinite(angle):
            continue
        if path_crosses_sun(src["x"], src["y"], angle, src["radius"]):
            continue

        already = reserves.get(src_id, 0)
        avail = int(src["ships"]) - already - 1
        if avail <= 0:
            continue
        ships = max(1, min(ships, avail))
        valid.append([int(src_id), float(angle), int(ships)])
        reserves[src_id] = already + ships
        if len(valid) >= MAX_MOVES_PER_TURN:
            break
    return valid


def _decode_model_moves(obs, world, pids, tgt_logits, frac_logits, deadline):
    player = world["player"]
    omega = world["omega"]
    targets = world["targets"]
    raw_map = world.get("raw_planet_map", {})
    moves = []
    reserved = {}
    if tgt_logits is None or frac_logits is None:
        return []
    if tgt_logits.ndim != 2 or frac_logits.ndim != 2:
        return []
    p_count = int(tgt_logits.shape[0])
    if p_count <= 0:
        return []

    own_idxs = []
    for i, pid in enumerate(pids):
        if pid == -1:
            continue
        row = _planet_by_id(raw_map, pid)
        if row is not None and _safe_int(row[1], -1) == player:
            own_idxs.append(i)

    for i in own_idxs:
        if time.perf_counter() > deadline - HARD_STOP_BUFFER:
            break

        src_pid = pids[i]
        src_row = _planet_by_id(raw_map, src_pid)
        if src_row is None:
            continue
        src_now = {
            "id": _safe_int(src_row[0], -1),
            "owner": _safe_int(src_row[1], -1),
            "x": _safe_float(src_row[2], 0.0),
            "y": _safe_float(src_row[3], 0.0),
            "radius": max(0.1, _safe_float(src_row[4], 1.0)),
            "ships": max(0.0, _safe_float(src_row[5], 0.0)),
            "production": max(0.0, _safe_float(src_row[6], 0.0)),
            "is_orbiting": False,
            "is_comet": _safe_int(src_row[0], -1) in world.get("comet_index", {}),
            "comet_ttl": world.get("comet_index", {}).get(_safe_int(src_row[0], -1), {}).get("remaining", 999.0),
        }
        src_now["is_orbiting"] = _is_orbiting_planet(src_now)

        reserve_need = _dynamic_reserve(src_now, world)
        avail = int(src_now["ships"]) - int(reserved.get(src_now["id"], 0)) - reserve_need
        if avail <= 1:
            continue

        if i >= tgt_logits.shape[0] or i >= frac_logits.shape[0]:
            continue
        local_logits = np.array(tgt_logits[i], copy=True)
        if not np.isfinite(local_logits).any():
            continue
        if i < p_count:
            local_logits[i] = -1e9
        choice = int(np.argmax(local_logits))
        if choice == p_count:
            continue
        if choice < 0 or choice >= len(pids) or pids[choice] == -1:
            continue

        frac_row = np.array(frac_logits[i], copy=False)
        if not np.isfinite(frac_row).any():
            frac_idx = NUM_FRAC // 2
        else:
            frac_idx = int(np.argmax(frac_row))
        frac_idx = _clip(frac_idx, 0, NUM_FRAC - 1)
        frac = SHIP_FRACTIONS[int(frac_idx)]

        send = int(avail * frac)
        send = max(1, min(send, avail - 1))
        if send <= 0:
            continue

        # MAPPO chooses coarse destination; heuristic re-ranks top candidates for robustness.
        trg0 = None
        chosen_pid = pids[choice]
        for t in targets:
            if t["id"] == chosen_pid:
                trg0 = t
                break

        if trg0 is None:
            trg, _ = _pick_target_selfplay_scored(src_now, targets, world, send, deadline)
        else:
            cands = [trg0]
            alt, alt_score = _pick_target_selfplay_scored(src_now, targets, world, send, deadline)
            if alt is not None and alt["id"] != trg0["id"]:
                cands.append(alt)
            best = None
            best_score = -1e18
            for t in cands:
                base = _target_base_score(src_now, t, player, world["step"])
                score = base + (0.5 * alt_score if alt is not None and t["id"] == alt["id"] else 0.0)
                if score > best_score:
                    best_score = score
                    best = t
            trg = best

        if trg is None:
            continue

        angle, eta, _, _ = _compute_attack_angle(src_now, trg, send, omega)
        if angle is None or eta is None:
            continue
        if trg.get("is_comet", False) and eta > max(0.0, trg.get("comet_ttl", 999.0) - COMET_MIN_TTL_BUFFER):
            continue

        need = _estimated_need(src_now, trg, world, eta)
        if trg["owner"] != player and math.isfinite(need):
            send = max(send, int(need + 1))
        send = max(1, min(send, avail - 1))

        if path_crosses_sun(src_now["x"], src_now["y"], angle, src_now["radius"]):
            continue
        if not math.isfinite(angle):
            continue
        if send <= 0 or send > avail or src_now["ships"] < send:
            continue

        moves.append([int(src_now["id"]), float(angle), int(send)])
        reserved[src_now["id"]] = reserved.get(src_now["id"], 0) + int(send)

    return _sanitize_moves(moves, world)


# -----------------------------
# Heuristic fallback
# -----------------------------

def _fallback_heuristic(world, deadline):
    moves = []
    omega = world["omega"]
    my_planets = sorted(
        world["my_planets"],
        key=lambda p: (p["ships"] + 2.2 * p["production"]),
        reverse=True,
    )
    targets = world["targets"]
    reserved = {}

    for src in my_planets:
        if time.perf_counter() > deadline - HARD_STOP_BUFFER:
            break
        reserve_need = _dynamic_reserve(src, world)
        avail = int(src["ships"]) - int(reserved.get(src["id"], 0)) - reserve_need
        if avail <= 2:
            continue

        support_target, support_score = _pick_support_target(src, world, deadline)
        attack_target, attack_score = _pick_target_selfplay_scored(
            src, targets, world, max(1, int(avail * 0.55)), deadline
        )

        step = world["step"]
        choose_support = support_target is not None and support_score > (attack_score + 0.8)
        trg = support_target if choose_support else attack_target
        if trg is None:
            continue

        if choose_support:
            danger = _planet_danger(trg, world)
            send = int(min(avail - 1, max(2.0, 0.48 * avail, danger + 1.0)))
        elif trg["owner"] == -1:
            ratio = 0.56 if step < 260 else 0.70
            send = int(avail * ratio)
        elif trg["owner"] == world["player"]:
            send = int(avail * (0.34 if step < 280 else 0.26))
        else:
            ratio = 0.64 if step < 260 else 0.80
            send = int(avail * ratio)

        send = max(1, min(send, avail - 1))
        if send <= 0:
            continue

        angle, eta, _, _ = _compute_attack_angle(src, trg, send, omega)
        if angle is None or eta is None:
            continue
        if trg.get("is_comet", False) and eta > max(0.0, trg.get("comet_ttl", 999.0) - COMET_MIN_TTL_BUFFER):
            continue

        if not choose_support:
            need = _estimated_need(src, trg, world, eta)
            if math.isfinite(need):
                send = max(send, int(need + 1))
                send = max(1, min(send, avail - 1))
                if send <= 0:
                    continue

        if path_crosses_sun(src["x"], src["y"], angle, src["radius"]):
            continue
        if not math.isfinite(angle):
            continue
        if send > avail:
            continue

        moves.append([int(src["id"]), float(angle), int(send)])
        reserved[src["id"]] = reserved.get(src["id"], 0) + int(send)
        if len(moves) >= MAX_MOVES_PER_TURN:
            break

    return _sanitize_moves(moves, world)


# -----------------------------
# Agent entrypoint
# -----------------------------

def agent(obs, config=None):
    start = time.perf_counter()
    act_timeout = 1.0
    if config is not None:
        act_timeout = max(0.2, _safe_float(_read(config, "actTimeout", 1.0), 1.0))
    turn_budget = min(TURN_BUDGET, max(0.16, act_timeout * 0.82))
    deadline = start + turn_budget

    try:
        world = _parse_world(obs)
        if not world["my_planets"] or not world["targets"]:
            return []

        # If close to budget already, use direct fallback quickly.
        if time.perf_counter() > start + 0.06:
            return _fallback_heuristic(world, min(deadline, start + NO_MODEL_FALLBACK_BUDGET))

        model = _load_model()
        if model is None:
            return _fallback_heuristic(world, min(deadline, start + NO_MODEL_FALLBACK_BUDGET))

        if time.perf_counter() > deadline - 0.26:
            return _fallback_heuristic(world, min(deadline, start + NO_MODEL_FALLBACK_BUDGET))

        player = world["player"]
        if time.perf_counter() > deadline - HARD_STOP_BUFFER:
            return _fallback_heuristic(world, min(deadline, start + NO_MODEL_FALLBACK_BUDGET))

        pids, tl, fl = _run_model(model, obs, player)
        if pids is None or tl is None or fl is None:
            return _fallback_heuristic(world, min(deadline, start + NO_MODEL_FALLBACK_BUDGET))
        tl = np.asarray(tl)
        fl = np.asarray(fl)
        if tl.ndim != 2 or fl.ndim != 2:
            return _fallback_heuristic(world, min(deadline, start + NO_MODEL_FALLBACK_BUDGET))
        if not np.isfinite(tl).any() or not np.isfinite(fl).any():
            return _fallback_heuristic(world, min(deadline, start + NO_MODEL_FALLBACK_BUDGET))

        if time.perf_counter() > deadline - HARD_STOP_BUFFER:
            return _fallback_heuristic(world, min(deadline, start + NO_MODEL_FALLBACK_BUDGET))

        moves = _decode_model_moves(
            obs,
            world,
            pids,
            tl,
            fl,
            deadline,
        )
        if not moves:
            return _fallback_heuristic(world, min(deadline, start + NO_MODEL_FALLBACK_BUDGET))
        return _sanitize_moves(moves, world)
    except Exception as exc:
        _warn_once(f"[agent_ppo_v3] runtime exception: {exc}")
        try:
            world = _parse_world(obs)
            return _fallback_heuristic(world, min(time.perf_counter() + 0.08, start + NO_MODEL_FALLBACK_BUDGET))
        except Exception:
            return []


# -----------------------------
# Offline training hook (placeholder)
# -----------------------------

def train_selfplay(*args, **kwargs):
    """
    Placeholder for offline MAPPO + self-play training pipeline.
    Keep training outside Kaggle inference environment.
    """
    raise NotImplementedError(
        "Run MAPPO self-play training offline and export weights to ppo_v3_weights.pt"
    )
