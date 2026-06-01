"""
agent_ducanh_v2.py
Hybrid scaffold: rule-candidate generator + optional RL scorer (ONNX) + safety filter.

Goals:
- Keep per-turn inference under 1s.
- Never crash / never return invalid actions.
- Allow incremental upgrade from rule-based v1 to RL-guided policy.
"""

import math
import os
import sys
import time
import importlib.util
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Runtime constants
# -----------------------------

BOARD = 100.0
SUN_X = 50.0
SUN_Y = 50.0
SUN_R = 10.0
SUN_MARGIN = 1.5
LAUNCH_CLEARANCE = 0.1

TURN_BUDGET_FRAC = 0.82
TURN_BUDGET_MIN = 0.22
TURN_BUDGET_MAX = 0.90
HARD_STOP_BUFFER = 0.04

MAX_MOVES_PER_TURN = 24
MAX_CANDIDATES = 8

MODEL_ENV = "DUCANH_V2_ONNX"
DEFAULT_MODEL_NAME = "ducanh_v2_actor.onnx"


# -----------------------------
# Global caches
# -----------------------------

_BASE_AGENT = None
_BASE_AGENT_TRIED = False

_ORT = None
_ORT_READY = False
_ORT_TRIED = False

_MODEL = None
_MODEL_INPUT_NAME = None
_MODEL_TRIED = False


# -----------------------------
# Small utils
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


def _base_dir():
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


def _warn_once(msg: str):
    if not hasattr(_warn_once, "_seen"):
        _warn_once._seen = set()  # type: ignore[attr-defined]
    seen = _warn_once._seen  # type: ignore[attr-defined]
    if msg in seen:
        return
    seen.add(msg)
    print(msg, file=sys.stderr, flush=True)


def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def _line_seg_min_dist(x1, y1, x2, y2, px, py):
    dx = x2 - x1
    dy = y2 - y1
    den = dx * dx + dy * dy
    if den <= 1e-12:
        return _dist(x1, y1, px, py)
    t = ((px - x1) * dx + (py - y1) * dy) / den
    t = _clip(t, 0.0, 1.0)
    qx = x1 + t * dx
    qy = y1 + t * dy
    return _dist(qx, qy, px, py)


def _segment_hits_sun(sx, sy, ex, ey):
    return _line_seg_min_dist(sx, sy, ex, ey, SUN_X, SUN_Y) < (SUN_R + SUN_MARGIN)


def _path_crosses_sun(src_x, src_y, angle, src_r, flight_dist=130.0):
    lx = src_x + math.cos(angle) * (src_r + LAUNCH_CLEARANCE)
    ly = src_y + math.sin(angle) * (src_r + LAUNCH_CLEARANCE)
    ex = lx + math.cos(angle) * flight_dist
    ey = ly + math.sin(angle) * flight_dist
    return _segment_hits_sun(lx, ly, ex, ey)


# -----------------------------
# Base rule policy loader (v1)
# -----------------------------

def _load_base_agent():
    global _BASE_AGENT, _BASE_AGENT_TRIED
    if _BASE_AGENT_TRIED:
        return _BASE_AGENT
    _BASE_AGENT_TRIED = True

    # 1) package-style import
    try:
        mod = __import__("RL_agent.agent_ducanh_v1", fromlist=["agent"])
        fn = getattr(mod, "agent", None)
        if callable(fn):
            _BASE_AGENT = fn
            return _BASE_AGENT
    except Exception:
        pass

    # 2) local file import
    try:
        path = os.path.join(_base_dir(), "agent_ducanh_v1.py")
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("agent_ducanh_v1_fallback", path)
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                fn = getattr(mod, "agent", None)
                if callable(fn):
                    _BASE_AGENT = fn
                    return _BASE_AGENT
    except Exception as exc:
        _warn_once(f"[agent_ducanh_v2] failed to import v1 fallback: {exc}")

    _warn_once("[agent_ducanh_v2] no v1 fallback found, using internal greedy fallback.")
    return None


# -----------------------------
# Optional ONNX scorer
# -----------------------------

def _ensure_ort():
    global _ORT, _ORT_READY, _ORT_TRIED
    if _ORT_TRIED:
        return _ORT_READY
    _ORT_TRIED = True
    try:
        import onnxruntime as ort
        _ORT = ort
        _ORT_READY = True
    except Exception:
        _ORT = None
        _ORT_READY = False
    return _ORT_READY


def _resolve_model_path():
    env_path = os.environ.get(MODEL_ENV, "").strip()
    if env_path:
        return env_path
    return os.path.join(_base_dir(), DEFAULT_MODEL_NAME)


def _load_model():
    global _MODEL, _MODEL_TRIED, _MODEL_INPUT_NAME
    if _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True

    model_path = _resolve_model_path()
    if not os.path.exists(model_path):
        return None
    if not _ensure_ort():
        _warn_once("[agent_ducanh_v2] onnxruntime unavailable, skip RL scorer.")
        return None

    try:
        opts = _ORT.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        sess = _ORT.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        _MODEL = sess
        _MODEL_INPUT_NAME = sess.get_inputs()[0].name
        return _MODEL
    except Exception as exc:
        _warn_once(f"[agent_ducanh_v2] failed to load ONNX model: {exc}")
        _MODEL = None
        return None


# -----------------------------
# Safety and candidate utils
# -----------------------------

def _sanitize_moves(obs, moves):
    planets = _read(obs, "planets", []) or []
    player = _safe_int(_read(obs, "player", 0), 0)
    own = {}
    for row in planets:
        if row is None or len(row) < 7:
            continue
        pid = _safe_int(row[0], -1)
        owner = _safe_int(row[1], -1)
        if owner == player:
            own[pid] = {
                "ships": max(0, _safe_int(row[5], 0)),
                "x": _safe_float(row[2], 0.0),
                "y": _safe_float(row[3], 0.0),
                "r": max(0.1, _safe_float(row[4], 1.0)),
            }

    if not isinstance(moves, list):
        return []

    spent = {}
    out = []
    for mv in moves:
        if not isinstance(mv, (list, tuple)) or len(mv) != 3:
            continue
        src = _safe_int(mv[0], -1)
        ang = _safe_float(mv[1], float("nan"))
        ships = _safe_int(mv[2], 0)
        if src not in own or not math.isfinite(ang) or ships <= 0:
            continue

        left = own[src]["ships"] - spent.get(src, 0) - 1
        if left <= 0:
            continue
        ships = min(ships, left)
        if ships <= 0:
            continue

        if _path_crosses_sun(own[src]["x"], own[src]["y"], ang, own[src]["r"]):
            continue

        out.append([int(src), float(ang), int(ships)])
        spent[src] = spent.get(src, 0) + ships
        if len(out) >= MAX_MOVES_PER_TURN:
            break
    return out


def _freeze_moves(moves):
    return tuple((int(m[0]), round(float(m[1]), 6), int(m[2])) for m in moves)


def _scale_moves(obs, moves, factor):
    scaled = []
    for mv in moves:
        ships = max(1, int(round(_safe_int(mv[2], 1) * factor)))
        scaled.append([_safe_int(mv[0], -1), _safe_float(mv[1], 0.0), ships])
    return _sanitize_moves(obs, scaled)


def _build_candidate_sets(obs, base_moves, deadline):
    cands = []
    seen = set()

    def add(mv):
        key = _freeze_moves(mv)
        if key in seen:
            return
        seen.add(key)
        cands.append(mv)

    base = _sanitize_moves(obs, base_moves)
    add(base)
    if time.perf_counter() > deadline - HARD_STOP_BUFFER:
        return cands

    if base:
        for f in (0.80, 0.92, 1.08, 1.20):
            if len(cands) >= MAX_CANDIDATES:
                break
            if time.perf_counter() > deadline - HARD_STOP_BUFFER:
                break
            add(_scale_moves(obs, base, f))

        for k in (1, 2, 4):
            if len(cands) >= MAX_CANDIDATES:
                break
            add(_sanitize_moves(obs, base[:k]))
    return cands[:MAX_CANDIDATES]


def _world_features(obs):
    player = _safe_int(_read(obs, "player", 0), 0)
    step = _safe_int(_read(obs, "step", 0), 0)
    planets = _read(obs, "planets", []) or []
    fleets = _read(obs, "fleets", []) or []

    my_planets = enemy_planets = 0
    my_ships = enemy_ships = 0.0
    my_prod = enemy_prod = 0.0

    for p in planets:
        if p is None or len(p) < 7:
            continue
        owner = _safe_int(p[1], -1)
        ships = max(0.0, _safe_float(p[5], 0.0))
        prod = max(0.0, _safe_float(p[6], 0.0))
        if owner == player:
            my_planets += 1
            my_ships += ships
            my_prod += prod
        elif owner != -1:
            enemy_planets += 1
            enemy_ships += ships
            enemy_prod += prod

    for f in fleets:
        if f is None or len(f) < 7:
            continue
        owner = _safe_int(f[1], -1)
        ships = max(0.0, _safe_float(f[6], 0.0))
        if owner == player:
            my_ships += ships
        elif owner != -1:
            enemy_ships += ships

    return np.array(
        [
            step / 500.0,
            my_planets / 32.0,
            enemy_planets / 32.0,
            my_ships / 2000.0,
            enemy_ships / 2000.0,
            my_prod / 80.0,
            enemy_prod / 80.0,
        ],
        dtype=np.float32,
    )


def _candidate_features(moves):
    if not moves:
        return np.zeros(8, dtype=np.float32)
    ships = np.array([max(0, _safe_float(m[2], 0.0)) for m in moves], dtype=np.float32)
    ang = np.array([_safe_float(m[1], 0.0) for m in moves], dtype=np.float32)
    return np.array(
        [
            len(moves) / 24.0,
            float(np.sum(ships)) / 2000.0,
            float(np.mean(ships)) / 300.0,
            float(np.max(ships)) / 1000.0,
            float(np.std(ships)) / 300.0,
            float(np.mean(np.sin(ang))),
            float(np.mean(np.cos(ang))),
            float(np.std(ang)) / math.pi,
        ],
        dtype=np.float32,
    )


def _score_candidate_with_model(model, obs_feat, moves):
    cand_feat = _candidate_features(moves)
    x = np.concatenate([obs_feat, cand_feat], axis=0).astype(np.float32)[None, :]
    try:
        out = model.run(None, {_MODEL_INPUT_NAME: x})
        if not out:
            return -1e18
        y = np.asarray(out[0]).reshape(-1)
        if y.size == 0 or not math.isfinite(float(y[0])):
            return -1e18
        return float(y[0])
    except Exception:
        return -1e18


def _score_candidate_heuristic(obs_feat, moves):
    # Conservative proxy score when RL model is absent.
    move_pen = _candidate_features(moves)
    return float(2.0 * obs_feat[5] - 1.6 * obs_feat[6] - 0.15 * move_pen[1] + 0.05 * move_pen[0])


def _fallback_greedy(obs):
    player = _safe_int(_read(obs, "player", 0), 0)
    planets = _read(obs, "planets", []) or []
    my = []
    targets = []
    for row in planets:
        if row is None or len(row) < 7:
            continue
        p = {
            "id": _safe_int(row[0], -1),
            "owner": _safe_int(row[1], -1),
            "x": _safe_float(row[2], 0.0),
            "y": _safe_float(row[3], 0.0),
            "r": max(0.1, _safe_float(row[4], 1.0)),
            "ships": max(0, _safe_int(row[5], 0)),
            "prod": max(0, _safe_int(row[6], 0)),
        }
        if p["owner"] == player:
            my.append(p)
        elif p["owner"] != player:
            targets.append(p)
    my.sort(key=lambda p: p["ships"], reverse=True)
    if not my or not targets:
        return []

    moves = []
    for src in my[:2]:
        avail = src["ships"] - 1
        if avail < 8:
            continue
        best = None
        best_score = -1e18
        for t in targets:
            d = _dist(src["x"], src["y"], t["x"], t["y"])
            score = 4.5 * t["prod"] - 0.08 * t["ships"] - 0.06 * d
            if score > best_score:
                best_score = score
                best = t
        if best is None:
            continue
        ang = math.atan2(best["y"] - src["y"], best["x"] - src["x"])
        send = max(8, int(avail * 0.55))
        moves.append([src["id"], ang, send])
    return _sanitize_moves(obs, moves)


# -----------------------------
# Entrypoint
# -----------------------------

def agent(obs, config=None):
    start = time.perf_counter()
    act_timeout = 1.0 if config is None else max(0.2, _safe_float(_read(config, "actTimeout", 1.0), 1.0))
    budget = _clip(act_timeout * TURN_BUDGET_FRAC, TURN_BUDGET_MIN, TURN_BUDGET_MAX)
    deadline = start + budget

    try:
        base_agent = _load_base_agent()
        base_moves = []
        if base_agent is not None:
            try:
                base_moves = base_agent(obs, config)
            except Exception:
                base_moves = []
        else:
            base_moves = _fallback_greedy(obs)

        base_moves = _sanitize_moves(obs, base_moves)
        if time.perf_counter() > deadline - HARD_STOP_BUFFER:
            return base_moves

        cands = _build_candidate_sets(obs, base_moves, deadline)
        if not cands:
            return base_moves
        if len(cands) == 1:
            return cands[0]

        model = _load_model()
        obs_feat = _world_features(obs)
        best = cands[0]
        best_score = -1e18

        for cand in cands:
            if time.perf_counter() > deadline - HARD_STOP_BUFFER:
                break
            if model is not None:
                s = _score_candidate_with_model(model, obs_feat, cand)
                if s <= -1e17:
                    s = _score_candidate_heuristic(obs_feat, cand)
            else:
                s = _score_candidate_heuristic(obs_feat, cand)
            if s > best_score:
                best_score = s
                best = cand

        return _sanitize_moves(obs, best)
    except Exception:
        return []


__all__ = ["agent"]

