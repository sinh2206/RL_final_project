import math, os, time, sys

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


# ============================================================
# Model cache
# ============================================================

_MODEL = None
_MODEL_INPUT_NAME = None
_MODEL_LOAD_ERROR = None
_MODEL_WARNING_PRINTED = False


def _warn_once(msg):
    global _MODEL_WARNING_PRINTED
    if _MODEL_WARNING_PRINTED:
        return
    _MODEL_WARNING_PRINTED = True
    print(msg, file=sys.stderr, flush=True)


def get_model():
    global _MODEL, _MODEL_INPUT_NAME, _MODEL_LOAD_ERROR

    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_ERROR is not None:
        return None

    if ort is None:
        _MODEL_LOAD_ERROR = "onnxruntime import failed"
        _warn_once("[agent_PPO_v1] ONNX runtime not available, fallback to heuristic.")
        return None

    try:
        base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        model_path = os.path.join(base_dir, "ppo_model.onnx")
        if not os.path.exists(model_path):
            _MODEL_LOAD_ERROR = "ppo_model.onnx not found"
            _warn_once("[agent_PPO_v1] Missing ppo_model.onnx, fallback to heuristic.")
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
        _warn_once(f"[agent_PPO_v1] Failed to load ONNX model: {exc}. Fallback to heuristic.")
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
        _warn_once(f"[agent_PPO_v1] ONNX inference failed, fallback to heuristic: {exc}")
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

def _fallback_heuristic(world, deadline):
    moves = []
    player = world["player"]
    omega = world["omega"]
    comet_index = world["comet_index"]
    targets = world["targets"]

    for src in world["my_planets"]:
        if time.perf_counter() > deadline:
            break

        available = int(src["ships"])
        if available < 8:
            continue

        target = _choose_heuristic_target(src, targets, player)
        if target is None:
            continue

        need = int(target["ships"] + 1)
        if target["owner"] != -1:
            eta_rough = estimate_arrival_turns(src, target["x"], target["y"], target["radius"], max(need, 1))
            need += int(target["production"] * eta_rough * 0.6) + 1
        need = max(1, min(need, max(1, available - 1)))
        if need <= 0:
            continue

        angle, eta, _, _ = _compute_attack_angle(src, target, need, omega, comet_index)
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

        if _is_action_valid(src, angle, need, available):
            moves.append([src["id"], float(angle), int(need)])

    return moves


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
            return _fallback_heuristic(world, deadline)

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

            target = _select_target_from_angle(src, targets, player, pred_angle)
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

        return moves

    except Exception as exc:
        print(f"[agent_PPO_v1] runtime exception: {exc}", file=sys.stderr, flush=True)
        return []
