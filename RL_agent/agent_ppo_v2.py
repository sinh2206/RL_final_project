import math, os, time, sys, importlib.util

import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None

def _load_fallback_agent(module_name, file_name, alias_name):
    try:
        mod = __import__(f"RL_agent.{module_name}", fromlist=["agent"])
        fn = getattr(mod, "agent", None)
        if callable(fn):
            return fn
    except Exception:
        pass

    try:
        mod = __import__(module_name, fromlist=["agent"])
        fn = getattr(mod, "agent", None)
        if callable(fn):
            return fn
    except Exception:
        pass

    try:
        base = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        path = os.path.join(base, file_name)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location(alias_name, path)
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                fn = getattr(mod, "agent", None)
                if callable(fn):
                    return fn
    except Exception:
        pass
    return None


_rule_base_v1_agent = _load_fallback_agent("agent_rule_base_v1", "agent_rule_base_v1.py", "agent_rule_base_v1_fallback_ppo")
_rule_base_v4_agent = _load_fallback_agent("agent_rule_base_v4", "agent_rule_base_v4.py", "agent_rule_base_v4_fallback_ppo")
_rule_base_v3_agent = _load_fallback_agent("agent_rule_base_v3", "agent_rule_base_v3.py", "agent_rule_base_v3_fallback_ppo")
_rule_base_v2_agent = _load_fallback_agent("agent_rule_base_v2", "agent_rule_base_v2.py", "agent_rule_base_v2_fallback_ppo")

# ============================================================
# Constants
# ============================================================

BOARD_SIZE = 100.0
SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
SUN_SAFETY = 1.5
ROTATION_LIMIT = 50.0
MAX_SPEED = 6.0
LAUNCH_CLEARANCE = 0.1

DEFAULT_DEADLINE_SEC = 0.85
MIN_REMAINING_SEC = 0.05

TOTAL_STEPS = 500.0
MAX_PLANETS = 28
MAX_FLEETS = 20
MAX_COMETS = 8
PLANET_FEAT_DIM = 11
FLEET_FEAT_DIM = 7
COMET_FEAT_DIM = 8
GLOBAL_FEAT_DIM = 16
SOURCE_FEAT_DIM = 10
OBS_DIM = 512

COMET_LIFE_BUFFER = 2.0
LOCAL_MODEL_FRACTION = 0.9
FALLBACK_MAX_ORDERS = 5
FALLBACK_MIN_RESERVE = 3
FALLBACK_ATTACK_BUFFER = 1
FALLBACK_COORD_SOURCES = 2
LATE_FALLBACK_START = 250
VERY_LATE_FALLBACK_START = 340
SELFPLAY_TOP_K = 3
SELFPLAY_MIN_TIME_LEFT = 0.06
SELFPLAY_COUNTER_HORIZON = 8.0
SELFPLAY_COUNTER_FRACTION = 0.35
SELFPLAY_RISK_WEIGHT = 0.55
FOUR_PLAYER_OPENING_TURN = 130
FALLBACK_MAX_RESERVE = 16


# ============================================================
# Model cache
# ============================================================

_MODEL = None
_MODEL_INPUT_NAME = None
_MODEL_LOAD_ERROR = None
_MODEL_WARNING_PRINTED = False
_FALLBACK_WARNING_PRINTED = False


def _warn_once(msg):
    global _MODEL_WARNING_PRINTED
    if _MODEL_WARNING_PRINTED:
        return
    _MODEL_WARNING_PRINTED = True
    print(msg, file=sys.stderr, flush=True)


def _warn_fallback_once(msg):
    global _FALLBACK_WARNING_PRINTED
    if _FALLBACK_WARNING_PRINTED:
        return
    _FALLBACK_WARNING_PRINTED = True
    print(msg, file=sys.stderr, flush=True)


def get_model():
    global _MODEL, _MODEL_INPUT_NAME, _MODEL_LOAD_ERROR

    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_ERROR is not None:
        return None

    if ort is None:
        _MODEL_LOAD_ERROR = "onnxruntime import failed"
        _warn_once("[agent_ppo_v2] ONNX runtime not available, fallback to heuristic.")
        return None

    try:
        base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        model_path = os.path.join(base_dir, "ppo_model.onnx")
        if not os.path.exists(model_path):
            _MODEL_LOAD_ERROR = "ppo_model.onnx not found"
            _warn_once("[agent_ppo_v2] Missing ppo_model.onnx, fallback to heuristic.")
            return None

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        _MODEL = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        _MODEL_INPUT_NAME = _MODEL.get_inputs()[0].name
        return _MODEL
    except Exception as exc:
        _MODEL_LOAD_ERROR = str(exc)
        _warn_once(f"[agent_ppo_v2] Failed to load ONNX model: {exc}. Fallback to heuristic.")
        _MODEL = None
        _MODEL_INPUT_NAME = None
        return None


# ============================================================
# Utility helpers
# ============================================================

def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def fleet_speed(ships):
    ships = max(1.0, _safe_float(ships, 1.0))
    ratio = math.log(ships) / math.log(1000.0)
    ratio = _clip(ratio, 0.0, 1.0)
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def line_seg_min_dist(x1, y1, x2, y2, px, py):
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        return dist(x1, y1, px, py)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = _clip(t, 0.0, 1.0)
    qx = x1 + t * dx
    qy = y1 + t * dy
    return dist(qx, qy, px, py)


def path_crosses_sun(px, py, angle, src_radius, max_dist=130.0):
    # Signature explicitly requested: check path from source launch point along angle.
    sx = px + math.cos(angle) * (src_radius + LAUNCH_CLEARANCE)
    sy = py + math.sin(angle) * (src_radius + LAUNCH_CLEARANCE)
    ex = sx + math.cos(angle) * max_dist
    ey = sy + math.sin(angle) * max_dist
    d = line_seg_min_dist(sx, sy, ex, ey, SUN_X, SUN_Y)
    return d < (SUN_RADIUS + SUN_SAFETY)


def _segment_hits_sun_to_target(src, tx, ty, target_radius):
    angle = math.atan2(ty - src["y"], tx - src["x"])
    sx = src["x"] + math.cos(angle) * (src["radius"] + LAUNCH_CLEARANCE)
    sy = src["y"] + math.sin(angle) * (src["radius"] + LAUNCH_CLEARANCE)
    hit_dist = max(0.0, dist(src["x"], src["y"], tx, ty) - (src["radius"] + LAUNCH_CLEARANCE) - target_radius)
    ex = sx + math.cos(angle) * hit_dist
    ey = sy + math.sin(angle) * hit_dist
    d = line_seg_min_dist(sx, sy, ex, ey, SUN_X, SUN_Y)
    return d < (SUN_RADIUS + SUN_SAFETY)


def _wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _is_orbiting_planet(planet):
    r = dist(planet["x"], planet["y"], SUN_X, SUN_Y)
    return (r + planet["radius"]) < ROTATION_LIMIT


def predict_orbit(x, y, omega, turns):
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    r = dist(x, y, SUN_X, SUN_Y)
    th2 = theta + omega * turns
    return SUN_X + r * math.cos(th2), SUN_Y + r * math.sin(th2)


def _build_comet_index(raw_comets):
    comet_index = {}
    for group in raw_comets or []:
        pids = group.get("planet_ids", []) if isinstance(group, dict) else []
        paths = group.get("paths", []) if isinstance(group, dict) else []
        path_index = _safe_int(group.get("path_index", 0), 0) if isinstance(group, dict) else 0
        for i, pid in enumerate(pids):
            if i >= len(paths):
                continue
            path = paths[i] if isinstance(paths[i], list) else []
            comet_index[_safe_int(pid)] = {
                "path": path,
                "path_index": path_index,
                "life": max(0, len(path) - path_index),
            }
    return comet_index


def predict_comet_position(comet_index, planet_id, turns, default_x, default_y):
    info = comet_index.get(planet_id)
    if info is None:
        return default_x, default_y
    idx = info["path_index"] + int(max(0, turns))
    path = info["path"]
    if 0 <= idx < len(path):
        pt = path[idx]
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            return _safe_float(pt[0], default_x), _safe_float(pt[1], default_y)
    return default_x, default_y


def comet_remaining_life(comet_index, planet_id):
    info = comet_index.get(planet_id)
    if info is None:
        return 0
    return _safe_int(info.get("life", 0), 0)


def predict_target_position(target, turns, omega, comet_index):
    if target.get("is_comet", False):
        return predict_comet_position(comet_index, target["id"], turns, target["x"], target["y"])
    if target.get("is_orbiting", False):
        return predict_orbit(target["x"], target["y"], omega, turns)
    return target["x"], target["y"]


def estimate_arrival_turns(src, tx, ty, target_radius, ships):
    speed = max(1e-6, fleet_speed(ships))
    d = max(0.0, dist(src["x"], src["y"], tx, ty) - src["radius"] - target_radius)
    return d / speed


def find_angle_to_moving_planet(src, target, ships, omega, comet_index):
    t = estimate_arrival_turns(src, target["x"], target["y"], target["radius"], ships)
    t = _clip(t, 0.0, 140.0)

    tx = target["x"]
    ty = target["y"]
    for _ in range(16):
        tx, ty = predict_target_position(target, t, omega, comet_index)
        t2 = estimate_arrival_turns(src, tx, ty, target["radius"], ships)
        if abs(t2 - t) < 0.05:
            t = t2
            break
        t = t2

    angle = math.atan2(ty - src["y"], tx - src["x"])
    if _segment_hits_sun_to_target(src, tx, ty, target["radius"]):
        return None, None, None, None
    return angle, t, tx, ty


# ============================================================
# World parsing
# ============================================================

def _parse_planets(raw_planets, comet_ids):
    planets = []
    by_id = {}
    for row in raw_planets or []:
        if row is None or len(row) < 7:
            continue
        pid = _safe_int(row[0], -1)
        owner = _safe_int(row[1], -1)
        x = _safe_float(row[2], 0.0)
        y = _safe_float(row[3], 0.0)
        radius = max(0.1, _safe_float(row[4], 1.0))
        ships = max(0.0, _safe_float(row[5], 0.0))
        production = max(0.0, _safe_float(row[6], 0.0))
        p = {
            "id": pid,
            "owner": owner,
            "x": x,
            "y": y,
            "radius": radius,
            "ships": ships,
            "production": production,
            "is_orbiting": False,  # fill after append
            "is_comet": pid in comet_ids,
        }
        p["is_orbiting"] = _is_orbiting_planet(p)
        planets.append(p)
        by_id[pid] = p
    return planets, by_id


def _parse_fleets(raw_fleets):
    fleets = []
    for row in raw_fleets or []:
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
    return fleets


def _infer_extra_comet_ids(planets, comet_ids):
    # Heuristic fallback if environment does not provide comet ids.
    if comet_ids:
        return comet_ids
    out = set()
    for p in planets:
        if p["radius"] < 1.5 and p["production"] <= 1.0:
            out.add(p["id"])
    return out


def _parse_world(obs):
    player = _safe_int(_read(obs, "player", 0), 0)
    step = _safe_int(_read(obs, "step", 0), 0)
    omega = _safe_float(_read(obs, "angular_velocity", 0.0), 0.0)
    raw_planets = _read(obs, "planets", []) or []
    raw_fleets = _read(obs, "fleets", []) or []
    raw_comets = _read(obs, "comets", []) or []
    comet_ids = set(_read(obs, "comet_planet_ids", []) or [])

    comet_index = _build_comet_index(raw_comets)
    comet_ids.update(comet_index.keys())

    planets, planets_by_id = _parse_planets(raw_planets, comet_ids)
    comet_ids = _infer_extra_comet_ids(planets, comet_ids)
    for p in planets:
        p["is_comet"] = p["id"] in comet_ids

    fleets = _parse_fleets(raw_fleets)
    my_planets = [p for p in planets if p["owner"] == player]
    targets = [p for p in planets if p["owner"] != player]
    owner_ids = {player}
    for p in planets:
        if p["owner"] >= 0:
            owner_ids.add(p["owner"])
    for f in fleets:
        if f["owner"] >= 0:
            owner_ids.add(f["owner"])

    return {
        "player": player,
        "step": step,
        "omega": omega,
        "planets": planets,
        "planets_by_id": planets_by_id,
        "fleets": fleets,
        "raw_comets": raw_comets,
        "comet_ids": comet_ids,
        "comet_index": comet_index,
        "my_planets": my_planets,
        "targets": targets,
        "player_count": len(owner_ids),
    }


# ============================================================
# Observation vector
# ============================================================

def _planet_owner_code(owner, player):
    if owner == player:
        return 1.0
    if owner == -1:
        return 0.0
    return -1.0


def _planet_features(src, planet, player):
    rel_x = (planet["x"] - src["x"]) / 50.0
    rel_y = (planet["y"] - src["y"]) / 50.0
    return [
        _clip(planet["x"] / BOARD_SIZE, -1.0, 1.0),
        _clip(planet["y"] / BOARD_SIZE, -1.0, 1.0),
        _clip(rel_x, -2.0, 2.0),
        _clip(rel_y, -2.0, 2.0),
        _clip(math.log1p(planet["ships"]) / math.log1p(2000.0), 0.0, 1.0),
        _clip(planet["production"] / 10.0, 0.0, 1.0),
        _clip(planet["radius"] / 6.0, 0.0, 1.0),
        _planet_owner_code(planet["owner"], player),
        1.0 if planet["is_orbiting"] else 0.0,
        1.0 if planet["is_comet"] else 0.0,
        _clip(dist(src["x"], src["y"], planet["x"], planet["y"]) / 80.0, 0.0, 2.0),
    ]


def _fleet_features(src, fleet, player):
    return [
        _clip((fleet["x"] - src["x"]) / 50.0, -2.0, 2.0),
        _clip((fleet["y"] - src["y"]) / 50.0, -2.0, 2.0),
        1.0 if fleet["owner"] == player else -1.0,
        _clip(math.log1p(fleet["ships"]) / math.log1p(2000.0), 0.0, 1.0),
        _clip(math.sin(fleet["angle"]), -1.0, 1.0),
        _clip(math.cos(fleet["angle"]), -1.0, 1.0),
        _clip(dist(src["x"], src["y"], fleet["x"], fleet["y"]) / 80.0, 0.0, 2.0),
    ]


def _comet_features(src, planet, comet_index):
    pid = planet["id"]
    cx0, cy0 = predict_comet_position(comet_index, pid, 0, planet["x"], planet["y"])
    cx1, cy1 = predict_comet_position(comet_index, pid, 1, planet["x"], planet["y"])
    cx3, cy3 = predict_comet_position(comet_index, pid, 3, planet["x"], planet["y"])
    cx5, cy5 = predict_comet_position(comet_index, pid, 5, planet["x"], planet["y"])
    life = comet_remaining_life(comet_index, pid)
    return [
        _clip((cx0 - src["x"]) / 50.0, -2.0, 2.0),
        _clip((cy0 - src["y"]) / 50.0, -2.0, 2.0),
        _clip((cx1 - src["x"]) / 50.0, -2.0, 2.0),
        _clip((cy1 - src["y"]) / 50.0, -2.0, 2.0),
        _clip((cx3 - src["x"]) / 50.0, -2.0, 2.0),
        _clip((cy3 - src["y"]) / 50.0, -2.0, 2.0),
        _clip((cx5 - src["x"]) / 50.0, -2.0, 2.0),
        _clip(life / 100.0, 0.0, 1.0),
    ]


def build_obs_vector(world, src):
    player = world["player"]
    step = world["step"]
    omega = world["omega"]
    planets = world["planets"]
    fleets = world["fleets"]
    comet_index = world["comet_index"]
    my_planets = world["my_planets"]
    targets = world["targets"]

    feats = []
    feats.extend(
        [
            _clip(step / TOTAL_STEPS, 0.0, 1.0),
            _clip(omega / 0.08, -2.0, 2.0),
            _clip(len(planets) / 40.0, 0.0, 1.0),
            _clip(len(fleets) / 80.0, 0.0, 1.0),
            _clip(len(my_planets) / 20.0, 0.0, 1.0),
            _clip(len(targets) / 30.0, 0.0, 1.0),
            _clip(sum(p["production"] for p in my_planets) / 80.0, 0.0, 2.0),
            _clip(sum(p["ships"] for p in my_planets) / 3000.0, 0.0, 2.0),
            _clip(sum(p["production"] for p in planets if p["owner"] not in (-1, player)) / 80.0, 0.0, 2.0),
            _clip(sum(p["ships"] for p in planets if p["owner"] not in (-1, player)) / 3000.0, 0.0, 2.0),
            _clip(src["x"] / BOARD_SIZE, 0.0, 1.0),
            _clip(src["y"] / BOARD_SIZE, 0.0, 1.0),
            _clip(src["radius"] / 6.0, 0.0, 1.0),
            _clip(math.log1p(src["ships"]) / math.log1p(2000.0), 0.0, 1.0),
            _clip(src["production"] / 10.0, 0.0, 1.0),
            1.0 if src["is_orbiting"] else 0.0,
        ]
    )

    # Source repeated block for easy architecture compatibility.
    feats.extend(
        [
            _clip((src["x"] - SUN_X) / 50.0, -1.0, 1.0),
            _clip((src["y"] - SUN_Y) / 50.0, -1.0, 1.0),
            _clip(dist(src["x"], src["y"], SUN_X, SUN_Y) / 70.0, 0.0, 2.0),
            _clip(src["ships"] / 1500.0, 0.0, 2.0),
            _clip(src["production"] / 8.0, 0.0, 1.0),
            _planet_owner_code(src["owner"], player),
            1.0 if src["is_orbiting"] else 0.0,
            1.0 if src["is_comet"] else 0.0,
            _clip(len(world["comet_ids"]) / 10.0, 0.0, 1.0),
            _clip(player / 3.0, 0.0, 1.0),
        ]
    )

    nearest_planets = sorted(planets, key=lambda p: dist(src["x"], src["y"], p["x"], p["y"]))[:MAX_PLANETS]
    for p in nearest_planets:
        feats.extend(_planet_features(src, p, player))
    if len(nearest_planets) < MAX_PLANETS:
        feats.extend([0.0] * (PLANET_FEAT_DIM * (MAX_PLANETS - len(nearest_planets))))

    nearest_fleets = sorted(fleets, key=lambda f: dist(src["x"], src["y"], f["x"], f["y"]))[:MAX_FLEETS]
    for f in nearest_fleets:
        feats.extend(_fleet_features(src, f, player))
    if len(nearest_fleets) < MAX_FLEETS:
        feats.extend([0.0] * (FLEET_FEAT_DIM * (MAX_FLEETS - len(nearest_fleets))))

    comet_planets = [p for p in planets if p.get("is_comet", False)]
    comet_planets = sorted(comet_planets, key=lambda p: dist(src["x"], src["y"], p["x"], p["y"]))[:MAX_COMETS]
    for c in comet_planets:
        feats.extend(_comet_features(src, c, comet_index))
    if len(comet_planets) < MAX_COMETS:
        feats.extend([0.0] * (COMET_FEAT_DIM * (MAX_COMETS - len(comet_planets))))

    vec = np.asarray(feats, dtype=np.float32)
    if vec.size < OBS_DIM:
        pad = np.zeros((OBS_DIM - vec.size,), dtype=np.float32)
        vec = np.concatenate([vec, pad], axis=0)
    elif vec.size > OBS_DIM:
        vec = vec[:OBS_DIM]
    return vec


# ============================================================
# Model decision + hybrid targeting
# ============================================================

def _run_model(obs_vec):
    model = get_model()
    if model is None:
        return None

    try:
        x = np.asarray(obs_vec, dtype=np.float32)
        in_shape = model.get_inputs()[0].shape
        rank = len(in_shape) if in_shape is not None else 2
        if rank == 1:
            feed = x
        else:
            feed = x.reshape(1, -1)
        outs = model.run(None, {_MODEL_INPUT_NAME: feed})
        if not outs:
            return None
        arr = np.asarray(outs[0], dtype=np.float32).reshape(-1)
        if arr.size < 2:
            return None
        a0 = float(arr[0])
        a1 = float(arr[1])
        if not (math.isfinite(a0) and math.isfinite(a1)):
            return None
        return a0, a1
    except Exception as exc:
        _warn_once(f"[agent_ppo_v2] ONNX inference failed, fallback to heuristic: {exc}")
        return None


def _decode_action(model_out, available_ships):
    angle_norm, ratio_norm = model_out
    angle = _wrap_pi(angle_norm * math.pi)
    ship_ratio = _clip((ratio_norm + 1.0) * 0.5, 0.0, 1.0)
    if available_ships <= 1:
        return None, 0
    ships_to_send = int(ship_ratio * available_ships * LOCAL_MODEL_FRACTION)
    ships_to_send = max(1, min(ships_to_send, max(1, available_ships - 1)))
    return angle, ships_to_send


def _base_target_score(src, target, player):
    d = dist(src["x"], src["y"], target["x"], target["y"])
    owner_bonus = 12.0 if target["owner"] not in (-1, player) else 8.0
    value = target["production"] * 4.0 + owner_bonus - target["ships"] * 0.12 - d * 0.10
    if target["is_comet"]:
        value += 1.5
    return value


def _enemy_counter_threat(world, target, arrival_turn):
    player = world["player"]
    omega = world["omega"]
    comet_index = world["comet_index"]
    threat = 0.0

    horizon = arrival_turn + SELFPLAY_COUNTER_HORIZON
    for ep in world["planets"]:
        if ep["owner"] in (-1, player):
            continue
        if ep["ships"] <= 1:
            continue
        probe = max(1, int(ep["ships"] * 0.45))
        _, eta, _, _ = _compute_attack_angle(ep, target, probe, omega, comet_index)
        if eta is None:
            continue
        if eta <= horizon:
            threat += max(0.0, (ep["ships"] - 1.0) * SELFPLAY_COUNTER_FRACTION)
    return threat


def _self_play_score(src, target, send, world):
    player = world["player"]
    omega = world["omega"]
    comet_index = world["comet_index"]

    angle, eta, _, _ = _compute_attack_angle(src, target, send, omega, comet_index)
    if angle is None or eta is None:
        return -1e9

    need = _fallback_needed_ships(src, target, world, probe_send=max(1, send))
    capture_margin = float(send - need)
    capture_like = 1.0 if capture_margin >= 0 else _clip((send + 1.0) / max(1.0, need), 0.0, 1.0)

    own_term = 0.0
    if target["owner"] == player:
        own_term += 3.0
    elif target["owner"] == -1:
        own_term += 9.0
    else:
        own_term += 13.0

    growth_term = target["production"] * max(0.0, (TOTAL_STEPS - world["step"] - eta)) * 0.025
    speed_term = -0.22 * eta
    ship_cost = -0.08 * send

    counter_threat = _enemy_counter_threat(world, target, eta)
    risk_term = -SELFPLAY_RISK_WEIGHT * max(0.0, counter_threat - max(0.0, send - 1.0))

    return own_term + growth_term + speed_term + ship_cost + risk_term + 3.8 * capture_like


def _select_target_from_angle(src, targets, player, predicted_angle):
    best = None
    best_score = -1e18
    for t in targets:
        if t["id"] == src["id"]:
            continue
        to_t = math.atan2(t["y"] - src["y"], t["x"] - src["x"])
        direction = math.cos(to_t - predicted_angle)
        score = _base_target_score(src, t, player) + direction * 6.0
        if score > best_score:
            best_score = score
            best = t
    return best


def _select_target_with_selfplay(src, targets, player, predicted_angle, ships_to_send, world, deadline):
    remaining = deadline - time.perf_counter()
    # Keep <1s: skip self-play refinement if close to deadline.
    if remaining <= SELFPLAY_MIN_TIME_LEFT:
        return _select_target_from_angle(src, targets, player, predicted_angle)

    candidates = []
    for t in targets:
        if t["id"] == src["id"]:
            continue
        to_t = math.atan2(t["y"] - src["y"], t["x"] - src["x"])
        direction = math.cos(to_t - predicted_angle)
        score = _base_target_score(src, t, player) + direction * 6.0
        candidates.append((score, t))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = candidates[:SELFPLAY_TOP_K]

    best = None
    best_score = -1e18
    for base_score, target in candidates:
        if time.perf_counter() > deadline - MIN_REMAINING_SEC:
            break
        sp_score = _self_play_score(src, target, ships_to_send, world)
        score = 0.45 * base_score + 0.55 * sp_score
        if score > best_score:
            best_score = score
            best = target

    if best is None:
        return candidates[0][1]
    return best


def _choose_heuristic_target(src, targets, player):
    best = None
    best_score = -1e18
    for t in targets:
        if t["id"] == src["id"]:
            continue
        score = _base_target_score(src, t, player)
        if score > best_score:
            best_score = score
            best = t
    return best


def _compute_attack_angle(src, target, ships_to_send, omega, comet_index):
    # Moving targets use predictive intercept; static targets use direct line.
    if target["is_orbiting"] or target["is_comet"]:
        angle, eta, tx, ty = find_angle_to_moving_planet(src, target, ships_to_send, omega, comet_index)
        return angle, eta, tx, ty

    angle = math.atan2(target["y"] - src["y"], target["x"] - src["x"])
    if _segment_hits_sun_to_target(src, target["x"], target["y"], target["radius"]):
        return None, None, None, None
    eta = estimate_arrival_turns(src, target["x"], target["y"], target["radius"], ships_to_send)
    return angle, eta, target["x"], target["y"]


def _is_action_valid(src, angle, ships_to_send, available_ships):
    if angle is None or not math.isfinite(angle):
        return False
    if ships_to_send <= 0:
        return False
    if ships_to_send > available_ships:
        return False
    if src["ships"] < ships_to_send:
        return False
    return True


# ============================================================
# Fallback heuristic policy
# ============================================================

def _fallback_target_value(src, target, player):
    d = dist(src["x"], src["y"], target["x"], target["y"])
    owner = target["owner"]
    if owner == player:
        owner_bonus = 2.5
    elif owner == -1:
        owner_bonus = 8.0
    else:
        owner_bonus = 13.0

    value = target["production"] * 4.8 + owner_bonus - target["ships"] * 0.10 - d * 0.10
    if target["is_orbiting"]:
        value += 0.8
    if target["is_comet"]:
        value += 0.6
    return value


def _fallback_needed_ships(src, target, world, probe_send):
    probe_send = max(1, int(probe_send))
    eta = estimate_arrival_turns(src, target["x"], target["y"], target["radius"], probe_send)
    growth = target["production"] * eta
    if target["owner"] == -1:
        need = target["ships"] + 1 + growth * 0.20
    elif target["owner"] == world["player"]:
        need = max(5.0, target["production"] * 2.0)
    else:
        need = target["ships"] + 2 + growth * 0.58
    return int(max(1.0, need))


def _estimate_local_enemy_pressure(src, world):
    player = world["player"]
    pressure = 0.0
    for ep in world["planets"]:
        if ep["owner"] in (-1, player):
            continue
        d = dist(src["x"], src["y"], ep["x"], ep["y"])
        if d > 48.0:
            continue
        pressure += ep["ships"] / (d + 10.0)
    return pressure


def _fallback_heuristic(world, deadline):
    moves = []
    player = world["player"]
    omega = world["omega"]
    comet_index = world["comet_index"]
    targets = world["targets"]
    my_planets = sorted(world["my_planets"], key=lambda p: p["ships"], reverse=True)
    reserved = {}
    opening = world["step"] <= 110
    late = world["step"] >= LATE_FALLBACK_START
    very_late = world["step"] >= VERY_LATE_FALLBACK_START
    four_player = world.get("player_count", 2) >= 4
    max_orders = FALLBACK_MAX_ORDERS + (2 if late else 0)

    for src in my_planets:
        if time.perf_counter() > deadline:
            break

        if len(moves) >= max_orders:
            break

        reserve_floor = 1 if late else FALLBACK_MIN_RESERVE
        local_pressure = _estimate_local_enemy_pressure(src, world)
        reserve_floor = max(reserve_floor, int(local_pressure * (1.6 if four_player else 1.2)))
        reserve_floor = min(FALLBACK_MAX_RESERVE, reserve_floor)
        available = int(src["ships"]) - int(reserved.get(src["id"], 0))
        if available <= reserve_floor + 1:
            continue

        picked = False

        # Opening: prioritize feasible neutral captures to snowball production.
        if opening:
            neutrals = [t for t in targets if t["owner"] == -1 and t["id"] != src["id"]]
            neutrals.sort(key=lambda t: (dist(src["x"], src["y"], t["x"], t["y"]) - 2.5 * t["production"], t["ships"]))
            for target in neutrals[:4]:
                need = _fallback_needed_ships(src, target, world, probe_send=max(1, int(available * 0.62)))
                if available - 1 < need + FALLBACK_ATTACK_BUFFER:
                    continue
                send = max(need + FALLBACK_ATTACK_BUFFER, int(available * (0.64 if late else 0.60)))
                send = max(1, min(send, available - 1))
                angle, eta, _, _ = _compute_attack_angle(src, target, send, omega, comet_index)
                if angle is None:
                    continue
                if target["is_comet"]:
                    life = comet_remaining_life(comet_index, target["id"])
                    if eta is None or eta > max(0.0, life - COMET_LIFE_BUFFER):
                        continue
                if path_crosses_sun(src["x"], src["y"], angle, src["radius"]):
                    continue
                if _is_action_valid(src, angle, send, available):
                    moves.append([src["id"], float(angle), int(send)])
                    reserved[src["id"]] = reserved.get(src["id"], 0) + int(send)
                    picked = True
                    break
            if picked:
                continue

        ranked = []
        for t in [x for x in targets if x["id"] != src["id"]]:
            need_probe = _fallback_needed_ships(src, t, world, probe_send=max(1, int(available * 0.58)))
            feas = _clip((available - 1) / max(1.0, need_probe), 0.0, 1.6)
            base = _fallback_target_value(src, t, player)
            if four_player and world["step"] <= FOUR_PLAYER_OPENING_TURN and t["owner"] not in (-1, player):
                base -= 6.0
            score = base + 4.8 * feas
            # Lightweight self-play shaping for fallback (only when time allows).
            if deadline - time.perf_counter() > SELFPLAY_MIN_TIME_LEFT and not opening:
                probe_send = max(1, min(need_probe + FALLBACK_ATTACK_BUFFER, available - 1))
                sp = _self_play_score(src, t, probe_send, world)
                score = 0.58 * score + 0.42 * sp
            ranked.append((score, t))
        ranked.sort(key=lambda x: x[0], reverse=True)
        ranked = [t for _, t in ranked[:6]]

        for target in ranked:
            if time.perf_counter() > deadline:
                break
            owner = target["owner"]

            need = _fallback_needed_ships(src, target, world, probe_send=max(1, int(available * 0.60)))
            if owner != player and not late and available - 1 < need + FALLBACK_ATTACK_BUFFER:
                # Do not throw tiny fleets into larger neutral/enemy stacks.
                continue

            # In 4-player opening, avoid early wars unless we have clear local advantage.
            if four_player and world["step"] <= FOUR_PLAYER_OPENING_TURN and owner not in (-1, player):
                if available < int(max(need + 1, src["ships"] * 0.58)):
                    continue

            if owner == player:
                send = max(1, min(int(available * (0.34 if not late else 0.22)), available - 1))
            elif owner == -1:
                send = max(need + FALLBACK_ATTACK_BUFFER, int(available * (0.56 if not late else 0.66)))
            else:
                send = max(need + FALLBACK_ATTACK_BUFFER, int(available * (0.64 if not late else 0.78)))
            send = max(1, min(send, available - 1))
            if send <= 0:
                continue

            angle, eta, _, _ = _compute_attack_angle(src, target, send, omega, comet_index)
            if angle is None:
                continue

            # Comets are useful only if reachable before they vanish.
            if target["is_comet"]:
                life = comet_remaining_life(comet_index, target["id"])
                if eta is None or eta > max(0.0, life - COMET_LIFE_BUFFER):
                    continue

            # Additional defensive sun check by signature requested.
            if path_crosses_sun(src["x"], src["y"], angle, src["radius"]):
                continue

            if _is_action_valid(src, angle, send, available):
                moves.append([src["id"], float(angle), int(send)])
                reserved[src["id"]] = reserved.get(src["id"], 0) + int(send)
                picked = True
                break

        if picked:
            continue

    # Two-source coordinated capture for stubborn targets.
    if len(moves) < max_orders and time.perf_counter() < deadline:
        if four_player and world["step"] <= FOUR_PLAYER_OPENING_TURN:
            # Delay large coordinated all-ins in crowded early game.
            return moves
        non_owned = [t for t in targets if t["owner"] != player]
        non_owned.sort(key=lambda t: (t["production"] - 0.08 * t["ships"]), reverse=True)

        for target in non_owned[: (6 if late else 4)]:
            if len(moves) >= max_orders:
                break
            if time.perf_counter() > deadline:
                break

            source_cands = []
            total_can_send = 0
            need = 0
            for src in my_planets:
                reserve_floor = 1 if late else FALLBACK_MIN_RESERVE
                available = int(src["ships"]) - int(reserved.get(src["id"], 0))
                if available <= reserve_floor + 1:
                    continue
                probe = max(1, int(available * 0.55))
                local_need = _fallback_needed_ships(src, target, world, probe_send=probe)
                need = max(need, local_need)
                send_cap = max(1, available - 1)
                angle, eta, _, _ = _compute_attack_angle(src, target, min(probe, send_cap), omega, comet_index)
                if angle is None:
                    continue
                if target["is_comet"]:
                    life = comet_remaining_life(comet_index, target["id"])
                    if eta is None or eta > max(0.0, life - COMET_LIFE_BUFFER):
                        continue
                source_cands.append((src, send_cap, angle))
                total_can_send += send_cap

            if len(source_cands) < FALLBACK_COORD_SOURCES:
                continue
            if total_can_send < need + FALLBACK_ATTACK_BUFFER:
                continue

            remain = need + FALLBACK_ATTACK_BUFFER
            max_sources = FALLBACK_COORD_SOURCES + (1 if very_late else 0)
            for src, send_cap, angle in sorted(source_cands, key=lambda x: x[1], reverse=True)[:max_sources]:
                if remain <= 0 or len(moves) >= max_orders:
                    break
                available = int(src["ships"]) - int(reserved.get(src["id"], 0))
                send = min(send_cap, max(1, remain))
                send = min(send, max(1, available - 1))
                if send <= 0:
                    continue
                if _is_action_valid(src, angle, send, available):
                    moves.append([src["id"], float(angle), int(send)])
                    reserved[src["id"]] = reserved.get(src["id"], 0) + int(send)
                    remain -= send

            if remain <= 0:
                break

    return moves


def _strong_fallback(obs, config, world, deadline):
    # Prefer stronger packaged rule-based policies if available.
    for name, fn in (
        ("rule_base_v1", _rule_base_v1_agent),
        ("rule_base_v4", _rule_base_v4_agent),
        ("rule_base_v3", _rule_base_v3_agent),
        ("rule_base_v2", _rule_base_v2_agent),
    ):
        if fn is None:
            continue
        try:
            _warn_fallback_once(f"[agent_ppo_v2] Using {name} fallback policy.")
            out = fn(obs, config)
            if isinstance(out, list):
                return out
        except TypeError:
            try:
                out = fn(obs)
                if isinstance(out, list):
                    return out
            except Exception:
                continue
        except Exception as exc:
            _warn_fallback_once(f"[agent_ppo_v2] {name} fallback failed: {exc}")
    return _fallback_heuristic(world, deadline)


# ============================================================
# Agent entrypoint
# ============================================================

def agent(obs, config=None):
    start_time = time.perf_counter()
    deadline = start_time + DEFAULT_DEADLINE_SEC

    try:
        if time.perf_counter() > deadline:
            return []

        world = _parse_world(obs)
        my_planets = world["my_planets"]
        if not my_planets:
            return []

        # Model unavailable -> full heuristic fallback.
        session = get_model()
        if session is None:
            return _strong_fallback(obs, config, world, deadline)

        moves = []
        player = world["player"]
        omega = world["omega"]
        targets = world["targets"]
        comet_index = world["comet_index"]

        for src in my_planets:
            now = time.perf_counter()
            if now > deadline:
                break
            if deadline - now < MIN_REMAINING_SEC:
                break

            available = int(src["ships"])
            if available <= 1:
                continue

            obs_vec = build_obs_vector(world, src)
            if time.perf_counter() > deadline:
                break

            model_out = _run_model(obs_vec)
            if model_out is None:
                # Degrade locally to heuristic for this planet.
                target = _choose_heuristic_target(src, targets, player)
                if target is None:
                    continue
                ships_to_send = max(1, min(int(available * 0.35), available - 1))
                if deadline - time.perf_counter() > SELFPLAY_MIN_TIME_LEFT:
                    target = _select_target_with_selfplay(
                        src,
                        targets,
                        player,
                        math.atan2(target["y"] - src["y"], target["x"] - src["x"]),
                        ships_to_send,
                        world,
                        deadline,
                    ) or target
                angle, eta, _, _ = _compute_attack_angle(src, target, ships_to_send, omega, comet_index)
                if angle is None:
                    continue
                if target["is_comet"]:
                    life = comet_remaining_life(comet_index, target["id"])
                    if eta is None or eta > max(0.0, life - COMET_LIFE_BUFFER):
                        continue
                if path_crosses_sun(src["x"], src["y"], angle, src["radius"]):
                    continue
                if _is_action_valid(src, angle, ships_to_send, available):
                    moves.append([src["id"], float(angle), int(ships_to_send)])
                continue

            pred_angle, ships_to_send = _decode_action(model_out, available)
            if pred_angle is None or ships_to_send <= 0:
                continue

            target = _select_target_with_selfplay(
                src,
                targets,
                player,
                pred_angle,
                ships_to_send,
                world,
                deadline,
            )
            if target is None:
                continue

            # PPO proposes ratio; robust code computes physically accurate firing angle.
            angle, eta, _, _ = _compute_attack_angle(src, target, ships_to_send, omega, comet_index)
            if angle is None:
                continue

            # Comets should be exploited only if they survive until arrival.
            if target["is_comet"]:
                life = comet_remaining_life(comet_index, target["id"])
                if eta is None or eta > max(0.0, life - COMET_LIFE_BUFFER):
                    continue

            # Final absolute safety gate against sun collision.
            if path_crosses_sun(src["x"], src["y"], angle, src["radius"]):
                continue

            if _is_action_valid(src, angle, ships_to_send, available):
                moves.append([src["id"], float(angle), int(ships_to_send)])

        # If model produced no action at all, fallback to stronger deterministic policy.
        if not moves:
            return _strong_fallback(obs, config, world, deadline)

        return moves

    except Exception as exc:
        print(f"[agent_ppo_v2] runtime exception: {exc}", file=sys.stderr, flush=True)
        try:
            world = _parse_world(obs)
            return _strong_fallback(obs, config, world, start_time + DEFAULT_DEADLINE_SEC)
        except Exception:
            return []
