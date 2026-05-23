import math, os, time, sys, importlib.util

import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


# ============================================================
# Constants
# ============================================================

BOARD_SIZE = 100.0
SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 1.5
ROTATION_LIMIT = 50.0
MAX_SPEED = 6.0
LAUNCH_CLEARANCE = 0.1

SOFT_DEADLINE = 0.88
EARLY_EXIT_BUFFER = 0.05

MAX_PLANETS = 28
MAX_FLEETS = 20
GLOBAL_PLANET_FEAT = 8
GLOBAL_FLEET_FEAT = 6
GLOBAL_STRATEGY_FEAT = 12
SOURCE_FEAT = 12
OBS_DIM = 512

TOP_K_TARGETS = 8
MAX_ORDERS = 4
MIN_SEND_RATIO = 0.08
MAX_SEND_RATIO = 0.95
COMET_LIFE_BUFFER = 2.0


# ============================================================
# Global model cache
# ============================================================

_MODEL = None
_MODEL_INPUT_NAME = None
_MODEL_INPUT_DIM = OBS_DIM
_MODEL_LOAD_ERROR = None
_MODEL_WARNED = False
_FALLBACK_WARNED = False


# ============================================================
# Strong fallback policy
# ============================================================

try:
    from RL_agent.agent_rule_base_v2 import agent as _rule_base_v2_agent
except Exception:
    try:
        from agent_rule_base_v2 import agent as _rule_base_v2_agent
    except Exception:
        _rule_base_v2_agent = None
        try:
            _base = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
            _fallback_path = os.path.join(_base, "agent_rule_base_v2.py")
            if os.path.exists(_fallback_path):
                _spec = importlib.util.spec_from_file_location("agent_rule_base_v2_fallback_marl", _fallback_path)
                if _spec is not None and _spec.loader is not None:
                    _mod = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    if hasattr(_mod, "agent") and callable(_mod.agent):
                        _rule_base_v2_agent = _mod.agent
        except Exception:
            _rule_base_v2_agent = None


# ============================================================
# Utility
# ============================================================


def _warn_model_once(msg):
    global _MODEL_WARNED
    if _MODEL_WARNED:
        return
    _MODEL_WARNED = True
    print(msg, file=sys.stderr, flush=True)


def _warn_fallback_once(msg):
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED = True
    print(msg, file=sys.stderr, flush=True)


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


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def fleet_speed(ships):
    ships = max(1.0, _safe_float(ships, 1.0))
    ratio = math.log(ships) / math.log(1000.0)
    ratio = _clip(ratio, 0.0, 1.0)
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = _clip(t, 0.0, 1.0)
    qx = x1 + t * dx
    qy = y1 + t * dy
    return dist(px, py, qx, qy)


def _segment_crosses_sun(x1, y1, x2, y2, margin=SUN_MARGIN):
    d = point_to_segment_distance(SUN_X, SUN_Y, x1, y1, x2, y2)
    return d < (SUN_RADIUS + margin)


def _is_orbiting(planet):
    r = dist(planet["x"], planet["y"], SUN_X, SUN_Y)
    return (r + planet["radius"]) < ROTATION_LIMIT


def predict_orbit(x, y, omega, turns):
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    r = dist(x, y, SUN_X, SUN_Y)
    theta2 = theta + omega * turns
    return SUN_X + r * math.cos(theta2), SUN_Y + r * math.sin(theta2)


def _build_comet_index(raw_comets):
    comet_index = {}
    for g in raw_comets or []:
        if not isinstance(g, dict):
            continue
        pids = g.get("planet_ids", [])
        paths = g.get("paths", [])
        pidx = _safe_int(g.get("path_index", 0), 0)
        for i, pid in enumerate(pids):
            if i >= len(paths):
                continue
            path = paths[i] if isinstance(paths[i], list) else []
            comet_index[_safe_int(pid)] = {
                "path": path,
                "path_index": pidx,
                "life": max(0, len(path) - pidx),
            }
    return comet_index


def _predict_comet_position(comet_index, planet_id, turns, default_x, default_y):
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


def _comet_life(comet_index, planet_id):
    info = comet_index.get(planet_id)
    if info is None:
        return 0
    return _safe_int(info.get("life", 0), 0)


def _predict_target_position(target, turns, omega, comet_index):
    if target.get("is_comet", False):
        return _predict_comet_position(comet_index, target["id"], turns, target["x"], target["y"])
    if target.get("is_orbiting", False):
        return predict_orbit(target["x"], target["y"], omega, turns)
    return target["x"], target["y"]


def _estimate_eta(src, tx, ty, target_radius, ships):
    speed = fleet_speed(ships)
    d = dist(src["x"], src["y"], tx, ty)
    hit_d = max(0.0, d - (src["radius"] + LAUNCH_CLEARANCE) - target_radius)
    return hit_d / max(1e-6, speed)


def _route_crosses_sun(src, tx, ty, target_radius):
    angle = math.atan2(ty - src["y"], tx - src["x"])
    sx = src["x"] + math.cos(angle) * (src["radius"] + LAUNCH_CLEARANCE)
    sy = src["y"] + math.sin(angle) * (src["radius"] + LAUNCH_CLEARANCE)
    d = dist(src["x"], src["y"], tx, ty)
    hit_d = max(0.0, d - (src["radius"] + LAUNCH_CLEARANCE) - target_radius)
    ex = sx + math.cos(angle) * hit_d
    ey = sy + math.sin(angle) * hit_d
    return _segment_crosses_sun(sx, sy, ex, ey)


def find_angle_to_moving_planet(src, target, ships, omega, comet_index):
    tx = target["x"]
    ty = target["y"]
    eta = _estimate_eta(src, tx, ty, target["radius"], ships)

    for _ in range(12):
        tx, ty = _predict_target_position(target, eta, omega, comet_index)
        eta2 = _estimate_eta(src, tx, ty, target["radius"], ships)
        if abs(eta2 - eta) < 0.35:
            eta = eta2
            break
        eta = eta2

    if _route_crosses_sun(src, tx, ty, target["radius"]):
        return None, None, None, None

    angle = math.atan2(ty - src["y"], tx - src["x"])
    return angle, eta, tx, ty


# ============================================================
# World + Features (CTDE style)
# ============================================================


def _parse_world(obs):
    player = _safe_int(_read(obs, "player", 0), 0)
    step = _safe_int(_read(obs, "step", 0), 0)
    omega = _safe_float(_read(obs, "angular_velocity", 0.0), 0.0)

    planets = []
    for raw in _read(obs, "planets", []) or []:
        if not isinstance(raw, (list, tuple)) or len(raw) < 7:
            continue
        planets.append({
            "id": _safe_int(raw[0], -1),
            "owner": _safe_int(raw[1], -1),
            "x": _safe_float(raw[2], 0.0),
            "y": _safe_float(raw[3], 0.0),
            "radius": _safe_float(raw[4], 0.0),
            "ships": _safe_int(raw[5], 0),
            "production": _safe_float(raw[6], 0.0),
            "is_orbiting": False,
            "is_comet": False,
        })

    fleets = []
    for raw in _read(obs, "fleets", []) or []:
        if not isinstance(raw, (list, tuple)) or len(raw) < 7:
            continue
        fleets.append({
            "id": _safe_int(raw[0], -1),
            "owner": _safe_int(raw[1], -1),
            "x": _safe_float(raw[2], 0.0),
            "y": _safe_float(raw[3], 0.0),
            "angle": _safe_float(raw[4], 0.0),
            "from_id": _safe_int(raw[5], -1),
            "ships": _safe_int(raw[6], 0),
        })

    raw_comets = _read(obs, "comets", []) or []
    comet_index = _build_comet_index(raw_comets)

    for p in planets:
        p["is_orbiting"] = _is_orbiting(p)
        p["is_comet"] = p["id"] in comet_index

    return {
        "player": player,
        "step": step,
        "omega": omega,
        "planets": planets,
        "fleets": fleets,
        "comets": comet_index,
    }


def _owner_sign(owner, me):
    if owner == me:
        return 1.0
    if owner == -1:
        return 0.0
    return -1.0


def build_global_state(world, player):
    planets = sorted(world["planets"], key=lambda p: p["id"])
    fleets = world["fleets"]

    max_ships = 300.0
    max_prod = 8.0

    p_feat = np.zeros(MAX_PLANETS * GLOBAL_PLANET_FEAT, dtype=np.float32)
    for i, p in enumerate(planets[:MAX_PLANETS]):
        b = i * GLOBAL_PLANET_FEAT
        p_feat[b + 0] = p["x"] / BOARD_SIZE
        p_feat[b + 1] = p["y"] / BOARD_SIZE
        p_feat[b + 2] = _clip(p["ships"] / max_ships, 0.0, 2.0)
        p_feat[b + 3] = _clip(p["production"] / max_prod, 0.0, 2.0)
        p_feat[b + 4] = _owner_sign(p["owner"], player)
        p_feat[b + 5] = 1.0 if p.get("is_orbiting", False) else 0.0
        p_feat[b + 6] = 1.0 if p.get("is_comet", False) else 0.0
        p_feat[b + 7] = _clip(p["radius"] / 8.0, 0.0, 2.0)

    f_feat = np.zeros(MAX_FLEETS * GLOBAL_FLEET_FEAT, dtype=np.float32)
    for i, f in enumerate(fleets[:MAX_FLEETS]):
        b = i * GLOBAL_FLEET_FEAT
        f_feat[b + 0] = f["x"] / BOARD_SIZE
        f_feat[b + 1] = f["y"] / BOARD_SIZE
        f_feat[b + 2] = math.sin(f["angle"])
        f_feat[b + 3] = math.cos(f["angle"])
        f_feat[b + 4] = _clip(f["ships"] / max_ships, 0.0, 2.0)
        f_feat[b + 5] = _owner_sign(f["owner"], player)

    my_planets = [p for p in planets if p["owner"] == player]
    enemy_planets = [p for p in planets if p["owner"] not in (-1, player)]
    neutral_planets = [p for p in planets if p["owner"] == -1]

    my_planet_ships = float(sum(p["ships"] for p in my_planets))
    enemy_planet_ships = float(sum(p["ships"] for p in enemy_planets))
    my_fleet_ships = float(sum(f["ships"] for f in fleets if f["owner"] == player))
    enemy_fleet_ships = float(sum(f["ships"] for f in fleets if f["owner"] not in (-1, player)))

    my_prod = float(sum(p["production"] for p in my_planets))
    enemy_prod = float(sum(p["production"] for p in enemy_planets))

    g = np.zeros(GLOBAL_STRATEGY_FEAT, dtype=np.float32)
    g[0] = _clip(world["step"] / 500.0, 0.0, 1.0)
    g[1] = _clip(world["omega"] / 0.25, -1.0, 1.0)
    g[2] = _clip(my_planet_ships / max_ships, 0.0, 4.0)
    g[3] = _clip(enemy_planet_ships / max_ships, 0.0, 4.0)
    g[4] = _clip(my_fleet_ships / max_ships, 0.0, 4.0)
    g[5] = _clip(enemy_fleet_ships / max_ships, 0.0, 4.0)
    g[6] = _clip(my_prod / max_prod, 0.0, 5.0)
    g[7] = _clip(enemy_prod / max_prod, 0.0, 5.0)
    g[8] = _clip(len(my_planets) / max(1, len(planets)), 0.0, 1.0)
    g[9] = _clip(len(enemy_planets) / max(1, len(planets)), 0.0, 1.0)
    g[10] = _clip(len(neutral_planets) / max(1, len(planets)), 0.0, 1.0)
    g[11] = 1.0

    return np.concatenate([p_feat, f_feat, g], axis=0).astype(np.float32)


def _build_source_features(src, world):
    player = world["player"]
    planets = world["planets"]

    enemy_planets = [p for p in planets if p["owner"] not in (-1, player)]
    neutral_planets = [p for p in planets if p["owner"] == -1]

    nearest_enemy = min((dist(src["x"], src["y"], p["x"], p["y"]) for p in enemy_planets), default=BOARD_SIZE)
    nearest_neutral = min((dist(src["x"], src["y"], p["x"], p["y"]) for p in neutral_planets), default=BOARD_SIZE)

    feats = np.zeros(SOURCE_FEAT, dtype=np.float32)
    feats[0] = src["x"] / BOARD_SIZE
    feats[1] = src["y"] / BOARD_SIZE
    feats[2] = _clip(src["ships"] / 300.0, 0.0, 2.0)
    feats[3] = _clip(src["production"] / 8.0, 0.0, 2.0)
    feats[4] = 1.0 if src.get("is_orbiting", False) else 0.0
    feats[5] = src["radius"] / 8.0
    feats[6] = _clip(nearest_enemy / BOARD_SIZE, 0.0, 2.0)
    feats[7] = _clip(nearest_neutral / BOARD_SIZE, 0.0, 2.0)
    feats[8] = _clip((500 - world["step"]) / 500.0, 0.0, 1.0)
    feats[9] = _clip(world["omega"] / 0.25, -1.0, 1.0)
    feats[10] = 1.0
    feats[11] = 0.0
    return feats


def _actor_obs(global_state, source_feat):
    return np.concatenate([global_state, source_feat], axis=0).astype(np.float32)


# ============================================================
# Model inference
# ============================================================


def get_model():
    global _MODEL, _MODEL_INPUT_NAME, _MODEL_INPUT_DIM, _MODEL_LOAD_ERROR

    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_ERROR is not None:
        return None

    if ort is None:
        _MODEL_LOAD_ERROR = "onnxruntime import failed"
        _warn_model_once("[agent_marl_v1] ONNX runtime unavailable, fallback heuristic.")
        return None

    try:
        base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        model_path = os.path.join(base_dir, "marl_actor.onnx")
        if not os.path.exists(model_path):
            _MODEL_LOAD_ERROR = "marl_actor.onnx not found"
            _warn_model_once("[agent_marl_v1] Missing marl_actor.onnx, fallback heuristic.")
            return None

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        _MODEL = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])

        inp = _MODEL.get_inputs()[0]
        _MODEL_INPUT_NAME = inp.name
        shape = inp.shape if hasattr(inp, "shape") else None
        if isinstance(shape, (list, tuple)) and len(shape) >= 2 and isinstance(shape[-1], int):
            _MODEL_INPUT_DIM = int(shape[-1])
        else:
            _MODEL_INPUT_DIM = OBS_DIM

        return _MODEL
    except Exception as exc:
        _MODEL_LOAD_ERROR = str(exc)
        _warn_model_once(f"[agent_marl_v1] Failed to load ONNX model: {exc}")
        _MODEL = None
        return None


def _fit_dim(vec, dim):
    if dim <= 0:
        return vec
    if vec.size == dim:
        return vec
    if vec.size > dim:
        return vec[:dim]
    out = np.zeros(dim, dtype=np.float32)
    out[:vec.size] = vec
    return out


def get_marl_action(actor_input):
    session = get_model()
    if session is None:
        return None, None

    x = _fit_dim(actor_input.astype(np.float32), _MODEL_INPUT_DIM).reshape(1, -1)

    try:
        outputs = session.run(None, {_MODEL_INPUT_NAME: x})
    except Exception as exc:
        _warn_model_once(f"[agent_marl_v1] ONNX inference failed: {exc}")
        return None, None

    if not outputs:
        return None, None

    if len(outputs) >= 2:
        target_logits = np.ravel(np.asarray(outputs[0], dtype=np.float32))
        ratio_raw = np.ravel(np.asarray(outputs[1], dtype=np.float32))
        ship_ratio_raw = float(ratio_raw[0]) if ratio_raw.size > 0 else 0.0
    else:
        arr = np.ravel(np.asarray(outputs[0], dtype=np.float32))
        if arr.size < 2:
            return None, None
        target_logits = arr[:-1]
        ship_ratio_raw = float(arr[-1])

    if -1.0 <= ship_ratio_raw <= 1.0:
        ship_ratio = (ship_ratio_raw + 1.0) * 0.5
    else:
        ship_ratio = 1.0 / (1.0 + math.exp(-_clip(ship_ratio_raw, -20.0, 20.0)))

    ship_ratio = _clip(ship_ratio, MIN_SEND_RATIO, MAX_SEND_RATIO)
    return target_logits, ship_ratio


# ============================================================
# Action selection + safety filter
# ============================================================


def _planet_danger_score(p, planets, player):
    enemy = [e for e in planets if e["owner"] not in (-1, player)]
    if not enemy:
        return 0.0
    near = min(dist(p["x"], p["y"], e["x"], e["y"]) for e in enemy)
    return 1.0 - _clip(near / BOARD_SIZE, 0.0, 1.0)


def _target_candidates(src, planets, player, omega):
    cands = []
    for idx, t in enumerate(planets):
        if t["id"] == src["id"]:
            continue

        d = dist(src["x"], src["y"], t["x"], t["y"])
        eta = _estimate_eta(src, t["x"], t["y"], t["radius"], max(1, src["ships"] // 2))

        if t["owner"] == player:
            # Reinforce only high-risk own planets
            danger = _planet_danger_score(t, planets, player)
            if danger < 0.55:
                continue
            need = int(max(4, t["production"] * 2))
            score = 0.8 * danger + 0.35 * t["production"] - 0.02 * d
        elif t["owner"] == -1:
            need = t["ships"] + 1
            score = 1.6 * t["production"] - 0.35 * need - 0.02 * d
        else:
            need = int(t["ships"] + t["production"] * max(0.0, eta) * 0.45 + 2)
            score = 2.0 * t["production"] - 0.25 * need - 0.015 * d

        if t.get("is_comet", False):
            score += 0.35

        cands.append((score, idx, need))

    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[:TOP_K_TARGETS]


def _decode_and_validate_action(src, target, ship_ratio, world, committed):
    available = src["ships"] - committed.get(src["id"], 0)
    if available <= 1:
        return None

    send = int(max(1, ship_ratio * available * 0.9))
    send = max(1, min(send, available - 1))
    if send <= 0 or send >= available:
        return None

    angle, eta, tx, ty = find_angle_to_moving_planet(src, target, send, world["omega"], world["comets"])
    if angle is None or eta is None:
        return None

    if target.get("is_comet", False):
        life = _comet_life(world["comets"], target["id"])
        if life > 0 and eta > max(0.0, life - COMET_LIFE_BUFFER):
            return None

    if not math.isfinite(angle):
        return None

    if _route_crosses_sun(src, tx, ty, target["radius"]):
        return None

    if send > available - 1:
        return None

    move = [int(src["id"]), float(angle), int(send)]
    committed[src["id"]] = committed.get(src["id"], 0) + send
    return move


def _select_target_with_mask(candidates, logits, planets):
    if not candidates:
        return None

    # Invalid action masking by assigning -inf to non-candidate slots.
    masked = []
    logit_len = 0 if logits is None else int(logits.size)
    for score, planet_idx, need in candidates:
        logit = 0.0
        if logit_len > 0 and planet_idx < logit_len:
            logit = float(logits[planet_idx])
        masked_score = logit + 0.08 * float(score)
        masked.append((masked_score, score, planet_idx, need))

    masked.sort(key=lambda x: x[0], reverse=True)
    _, _, best_idx, _ = masked[0]
    return planets[best_idx]


def _fallback(obs, config):
    if _rule_base_v2_agent is None:
        return []
    _warn_fallback_once("[agent_marl_v1] fallback to rule-based policy.")
    try:
        return _rule_base_v2_agent(obs, config)
    except TypeError:
        return _rule_base_v2_agent(obs)


# ============================================================
# Main agent
# ============================================================


def agent(obs, config=None):
    start = time.perf_counter()
    deadline = start + SOFT_DEADLINE

    try:
        world = _parse_world(obs)
        player = world["player"]

        planets = world["planets"]
        if not planets:
            return []

        my_planets = [p for p in planets if p["owner"] == player and p["ships"] > 1]
        if not my_planets:
            return []

        # In decentralized execution, each source planet uses the same global state
        # plus local source features to query actor policy.
        global_state = build_global_state(world, player)
        if time.perf_counter() > deadline - EARLY_EXIT_BUFFER:
            return []

        session = get_model()
        if session is None:
            return _fallback(obs, config)

        moves = []
        committed = {}

        # Strong planets first to reduce risk of invalid tiny launches.
        my_planets.sort(key=lambda p: p["ships"], reverse=True)

        for src in my_planets:
            if len(moves) >= MAX_ORDERS:
                break
            if time.perf_counter() > deadline - EARLY_EXIT_BUFFER:
                break

            actor_input = _actor_obs(global_state, _build_source_features(src, world))
            target_logits, ship_ratio = get_marl_action(actor_input)

            if target_logits is None or ship_ratio is None:
                continue

            candidates = _target_candidates(src, planets, player, world["omega"])
            target = _select_target_with_mask(candidates, target_logits, planets)
            if target is None:
                continue

            mv = _decode_and_validate_action(src, target, ship_ratio, world, committed)
            if mv is None:
                continue

            moves.append(mv)

        if not moves:
            return _fallback(obs, config)

        return moves

    except Exception as exc:
        print(f"[agent_marl_v1] exception: {exc}", file=sys.stderr, flush=True)
        return _fallback(obs, config)
