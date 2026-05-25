import math, os, time, sys, traceback, importlib.util

import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


# ============================================================
# Constants
# ============================================================

SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 1.5
MAX_SPEED = 6.0
ROTATION_LIMIT = 50.0
LAUNCH_CLEARANCE = 0.1

MCTS_MAX_SIMULATIONS = 42
MCTS_MAX_DEPTH = 6
MCTS_C_PUCT = 1.25

TOP_K_TARGETS = 3
MAX_ACTIONS_PER_NODE = 28
FINAL_MAX_ORDERS = 3
MIN_SOURCE_SHIPS = 10

OBS_DIM = 512
MAX_PLANETS_FEAT = 28
MAX_FLEETS_FEAT = 18
PLANET_FEAT_DIM = 10
FLEET_FEAT_DIM = 6
GLOBAL_FEAT_DIM = 16

SOFT_DEADLINE = 0.78
EARLY_EXIT_BUFFER = 0.07

COMET_MARGIN = 0.5
COMET_MIN_DANGER_SHIPS = 6
COMET_TARGET_MIN_REMAINING = 2.0
FALLBACK_MAX_ORDERS = 4
FALLBACK_MIN_SOURCE_SHIPS = 12
SELFPLAY_TOP_K = 3
SELFPLAY_MIN_TIME_LEFT = 0.06
SELFPLAY_COUNTER_HORIZON = 8.0
SELFPLAY_COUNTER_FRACTION = 0.32
SELFPLAY_RISK_WEIGHT = 0.52


# ============================================================
# Model cache
# ============================================================

_MODEL = None
_MODEL_INPUT = None
_MODEL_LOAD_FAILED = False
_MODEL_WARNED = False
_FALLBACK_WARNED = False


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


def get_model():
    global _MODEL, _MODEL_INPUT, _MODEL_LOAD_FAILED

    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_FAILED:
        return None
    if ort is None:
        _MODEL_LOAD_FAILED = True
        _warn_model_once("[agent_alphazero_v1] onnxruntime unavailable, fallback heuristic.")
        return None

    try:
        base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        model_path = os.path.join(base_dir, "alphazero_model.onnx")
        if not os.path.exists(model_path):
            _MODEL_LOAD_FAILED = True
            _warn_model_once("[agent_alphazero_v1] Missing alphazero_model.onnx, fallback heuristic.")
            return None

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        _MODEL = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        _MODEL_INPUT = _MODEL.get_inputs()[0].name
        return _MODEL
    except Exception as exc:
        _MODEL_LOAD_FAILED = True
        _warn_model_once(f"[agent_alphazero_v1] Model load failed: {exc}")
        _MODEL = None
        _MODEL_INPUT = None
        return None


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


_rule_base_v4_agent = _load_fallback_agent("agent_rule_base_v4", "agent_rule_base_v4.py", "agent_rule_base_v4_fallback_alpha")
_rule_base_v3_agent = _load_fallback_agent("agent_rule_base_v3", "agent_rule_base_v3.py", "agent_rule_base_v3_fallback_alpha")
_rule_base_v2_agent = _load_fallback_agent("agent_rule_base_v2", "agent_rule_base_v2.py", "agent_rule_base_v2_fallback_alpha")
_rule_base_v1_agent = _load_fallback_agent("agent_rule_base_v1", "agent_rule_base_v1.py", "agent_rule_base_v1_fallback_alpha")


# ============================================================
# Data structures
# ============================================================

class Action:
    __slots__ = ("src_idx", "tgt_idx", "ships", "heuristic", "pass_action")

    def __init__(self, src_idx, tgt_idx, ships, heuristic, pass_action=False):
        self.src_idx = _safe_int(src_idx, -1)
        self.tgt_idx = _safe_int(tgt_idx, -1)
        self.ships = _safe_int(ships, 0)
        self.heuristic = _safe_float(heuristic, 0.0)
        self.pass_action = bool(pass_action)


class State:
    __slots__ = ("owners", "ships", "current_player", "step")

    def __init__(self, owners, ships, current_player, step):
        self.owners = owners
        self.ships = ships
        self.current_player = _safe_int(current_player, 0)
        self.step = _safe_int(step, 0)


class Node:
    __slots__ = (
        "state",
        "parent",
        "action_from_parent",
        "prior_from_parent",
        "visit_count",
        "value_sum",
        "expanded",
        "actions",
        "priors",
        "children",
    )

    def __init__(self, state, parent=None, action_from_parent=None, prior_from_parent=1.0):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior_from_parent = _safe_float(prior_from_parent, 1.0)
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False
        self.actions = []
        self.priors = []
        self.children = {}  # action_index -> Node

    def q(self):
        if self.visit_count <= 0:
            return 0.0
        return self.value_sum / self.visit_count


# ============================================================
# Utility + physics
# ============================================================

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


def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def fleet_speed(ships):
    ships = max(1.0, _safe_float(ships, 1.0))
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    seg = dx * dx + dy * dy
    if seg <= 1e-12:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg
    t = max(0.0, min(1.0, t))
    qx = x1 + t * dx
    qy = y1 + t * dy
    return dist(px, py, qx, qy)


def collides(x1, y1, x2, y2, cx, cy, radius):
    return point_to_segment_distance(cx, cy, x1, y1, x2, y2) <= radius


def path_crosses_sun(x1, y1, x2, y2, margin=SUN_MARGIN):
    return collides(x1, y1, x2, y2, SUN_X, SUN_Y, SUN_RADIUS + margin)


def comet_collision(x1, y1, x2, y2, comet):
    return collides(x1, y1, x2, y2, comet["x"], comet["y"], comet["radius"] + COMET_MARGIN)


def predict_orbit(x, y, omega, dt):
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    r = dist(x, y, SUN_X, SUN_Y)
    th2 = theta + omega * dt
    return SUN_X + r * math.cos(th2), SUN_Y + r * math.sin(th2)


def _is_orbiting(p):
    r = dist(p["x"], p["y"], SUN_X, SUN_Y)
    return (r + p["radius"]) < ROTATION_LIMIT


def simulate_comet_path(comet_id, comets, current_step):
    _ = current_step
    for group in comets:
        if not isinstance(group, dict):
            continue
        pids = group.get("planet_ids", [])
        if comet_id not in pids:
            continue
        idx = pids.index(comet_id)
        paths = group.get("paths", [])
        pidx = _safe_int(group.get("path_index", 0), 0)
        if idx >= len(paths):
            return [], 0
        path = paths[idx] if isinstance(paths[idx], list) else []
        future = path[pidx:] if pidx < len(path) else []
        return future, max(0, len(path) - pidx)
    return [], 0


def _build_comet_map(planets, raw_comets, step):
    cmap = {}
    for p in planets:
        if not p["is_comet"]:
            continue
        path, life = simulate_comet_path(p["id"], raw_comets, step)
        cmap[p["id"]] = {"path": path, "life": life}
    return cmap


def _predict_comet_position(comet_map, pid, turns, default_x, default_y):
    info = comet_map.get(pid)
    if info is None:
        return default_x, default_y
    path = info["path"]
    i = int(max(0, turns))
    if i < len(path):
        pt = path[i]
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            return _safe_float(pt[0], default_x), _safe_float(pt[1], default_y)
    return default_x, default_y


def _target_position(target, turns, omega, comet_map):
    if target["is_comet"]:
        return _predict_comet_position(comet_map, target["id"], turns, target["x"], target["y"])
    if target["is_orbiting"]:
        return predict_orbit(target["x"], target["y"], omega, turns)
    return target["x"], target["y"]


def _estimate_eta(src, tx, ty, tr, ships):
    speed = max(1e-6, fleet_speed(ships))
    d = max(0.0, dist(src["x"], src["y"], tx, ty) - src["radius"] - tr)
    return d / speed


def find_angle_to_moving_planet(src, target, ships, omega, comet_map):
    t = _estimate_eta(src, target["x"], target["y"], target["radius"], ships)
    t = max(0.0, min(140.0, t))
    tx, ty = target["x"], target["y"]
    for _ in range(16):
        tx, ty = _target_position(target, t, omega, comet_map)
        t2 = _estimate_eta(src, tx, ty, target["radius"], ships)
        if abs(t2 - t) < 0.05:
            t = t2
            break
        t = t2

    angle = math.atan2(ty - src["y"], tx - src["x"])
    sx = src["x"] + math.cos(angle) * (src["radius"] + LAUNCH_CLEARANCE)
    sy = src["y"] + math.sin(angle) * (src["radius"] + LAUNCH_CLEARANCE)
    ex = tx - math.cos(angle) * target["radius"]
    ey = ty - math.sin(angle) * target["radius"]
    if path_crosses_sun(sx, sy, ex, ey):
        return None, None, None, None
    return angle, t, tx, ty


# ============================================================
# World parsing
# ============================================================

def _parse_world(obs):
    player = _safe_int(_read(obs, "player", 0), 0)
    step = _safe_int(_read(obs, "step", 0), 0)
    omega = _safe_float(_read(obs, "angular_velocity", 0.0), 0.0)
    raw_planets = _read(obs, "planets", []) or []
    raw_fleets = _read(obs, "fleets", []) or []
    raw_comets = _read(obs, "comets", []) or []
    comet_ids = set(_read(obs, "comet_planet_ids", []) or [])

    planets = []
    id_to_idx = {}
    owners = []
    ships = []
    production = []
    player_ids = {player}

    for row in raw_planets:
        if row is None or len(row) < 7:
            continue
        pid = _safe_int(row[0], -1)
        owner = _safe_int(row[1], -1)
        x = _safe_float(row[2], 0.0)
        y = _safe_float(row[3], 0.0)
        radius = max(0.1, _safe_float(row[4], 1.0))
        sh = max(0.0, _safe_float(row[5], 0.0))
        prod = max(0.0, _safe_float(row[6], 0.0))
        p = {
            "id": pid,
            "x": x,
            "y": y,
            "radius": radius,
            "production": prod,
            "is_comet": pid in comet_ids,
            "is_orbiting": False,
        }
        p["is_orbiting"] = _is_orbiting(p)
        id_to_idx[pid] = len(planets)
        planets.append(p)
        owners.append(owner)
        ships.append(sh)
        production.append(prod)
        if owner >= 0:
            player_ids.add(owner)

    # Comet fallback detect
    if not comet_ids:
        for p in planets:
            if p["radius"] < 1.5 and p["production"] <= 1.0:
                p["is_comet"] = True
                comet_ids.add(p["id"])
    else:
        for p in planets:
            p["is_comet"] = p["id"] in comet_ids

    for row in raw_fleets:
        if row is None or len(row) < 2:
            continue
        fo = _safe_int(row[1], -1)
        if fo >= 0:
            player_ids.add(fo)

    player_order = sorted(player_ids)
    if len(player_order) <= 1:
        player_order = [player, 1 - player]

    comet_map = _build_comet_map(planets, raw_comets, step)
    state = State(
        owners=np.asarray(owners, dtype=np.int16),
        ships=np.asarray(ships, dtype=np.float32),
        current_player=player,
        step=step,
    )
    return {
        "player": player,
        "step": step,
        "omega": omega,
        "planets": planets,
        "id_to_idx": id_to_idx,
        "production": np.asarray(production, dtype=np.float32),
        "player_order": player_order,
        "comet_ids": comet_ids,
        "comet_map": comet_map,
        "raw_comets": raw_comets,
        "state": state,
    }


# ============================================================
# AlphaZero-style MCTS helpers
# ============================================================

def _next_player(player_order, current):
    if current not in player_order:
        return player_order[0]
    i = player_order.index(current)
    return player_order[(i + 1) % len(player_order)]


def _action_heuristic(state, world, src_idx, tgt_idx):
    psrc = world["planets"][src_idx]
    ptgt = world["planets"][tgt_idx]
    d = dist(psrc["x"], psrc["y"], ptgt["x"], ptgt["y"])
    t_owner = int(state.owners[tgt_idx])
    t_ship = float(state.ships[tgt_idx])
    if t_owner == state.current_player:
        owner_bonus = 2.5 if t_ship < psrc["ships"] * 0.55 else -6.0
        ship_penalty = 0.05
    elif t_owner == -1:
        owner_bonus = 8.0
        ship_penalty = 0.10
    else:
        owner_bonus = 14.0
        ship_penalty = 0.14

    orbit_bonus = 0.8 if ptgt["is_orbiting"] else 0.0
    comet_bonus = 0.4 if ptgt["is_comet"] else 0.0
    return ptgt["production"] * 5.0 + owner_bonus + orbit_bonus + comet_bonus - ship_penalty * t_ship - 0.10 * d


def _generate_actions(state, world, deadline):
    player = state.current_player
    owners = state.owners
    ships = state.ships

    source_idxs = [i for i in range(len(owners)) if owners[i] == player and ships[i] >= MIN_SOURCE_SHIPS]
    source_idxs.sort(key=lambda i: (ships[i], world["production"][i]), reverse=True)
    source_idxs = source_idxs[: min(6, len(source_idxs))]

    target_idxs = [i for i in range(len(owners)) if owners[i] != player]
    weak_owned = [i for i in range(len(owners)) if owners[i] == player]
    weak_owned.sort(key=lambda i: (ships[i] - world["production"][i] * 1.5))
    target_idxs.extend(weak_owned[:2])
    actions = []

    for sidx in source_idxs:
        if time.perf_counter() > deadline - EARLY_EXIT_BUFFER:
            break
        avail = int(ships[sidx])
        if avail <= 1:
            continue

        ranked_targets = sorted(
            target_idxs,
            key=lambda tidx: _action_heuristic(state, world, sidx, tidx),
            reverse=True,
        )[:TOP_K_TARGETS]

        for tidx in ranked_targets:
            if tidx == sidx:
                continue
            t_owner = int(owners[tidx])
            t_ships = float(ships[tidx])
            eta_rough = _estimate_eta(world["planets"][sidx], world["planets"][tidx]["x"], world["planets"][tidx]["y"], world["planets"][tidx]["radius"], max(1, avail))
            takeover_need = int(t_ships + 1 + (world["planets"][tidx]["production"] * eta_rough * (0.6 if t_owner != -1 else 0.2)))

            send_candidates = {
                max(1, int(avail * 0.30)),
                max(1, int(avail * 0.50)),
                max(1, takeover_need),
            }
            if t_owner == player:
                send_candidates.add(max(1, int(avail * 0.36)))
            for send in send_candidates:
                send = min(send, max(1, avail - 1))
                if send <= 0:
                    continue
                h = _action_heuristic(state, world, sidx, tidx) + min(send, takeover_need) * 0.018
                actions.append(Action(src_idx=sidx, tgt_idx=tidx, ships=int(send), heuristic=h, pass_action=False))
                if len(actions) >= MAX_ACTIONS_PER_NODE:
                    break
            if len(actions) >= MAX_ACTIONS_PER_NODE:
                break
        if len(actions) >= MAX_ACTIONS_PER_NODE:
            break

    # Always include pass action for safety.
    actions.append(Action(src_idx=-1, tgt_idx=-1, ships=0, heuristic=-0.2, pass_action=True))
    return actions


def _heuristic_priors(actions):
    if not actions:
        return []
    scores = np.asarray([a.heuristic for a in actions], dtype=np.float32)
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    s = float(np.sum(probs))
    if s <= 1e-12:
        return [1.0 / len(actions)] * len(actions)
    return list((probs / s).astype(np.float64))


def _state_value_heuristic(state, world):
    p = state.current_player
    owners = state.owners
    ships = state.ships
    prod = world["production"]
    my_ship = float(np.sum(ships[owners == p]))
    my_prod = float(np.sum(prod[owners == p]))
    best_other = 1.0
    for op in world["player_order"]:
        if op == p:
            continue
        op_score = float(np.sum(ships[owners == op]) + 2.5 * np.sum(prod[owners == op]))
        if op_score > best_other:
            best_other = op_score
    me = my_ship + 2.5 * my_prod
    return float(math.tanh((me - best_other) / 120.0))


def _build_obs_vector(state, world):
    planets = world["planets"]
    owners = state.owners
    ships = state.ships
    prod = world["production"]
    p = state.current_player
    step_norm = min(1.0, state.step / 500.0)
    omega = world["omega"]

    feats = [
        step_norm,
        max(-2.0, min(2.0, omega / 0.08)),
        float(p) / 3.0,
        min(1.0, len(planets) / 40.0),
    ]

    my_mask = owners == p
    en_mask = (owners >= 0) & (owners != p)
    feats += [
        min(2.0, float(np.sum(ships[my_mask])) / 3000.0),
        min(2.0, float(np.sum(ships[en_mask])) / 3000.0),
        min(2.0, float(np.sum(prod[my_mask])) / 80.0),
        min(2.0, float(np.sum(prod[en_mask])) / 80.0),
        min(1.0, float(np.sum(owners == -1)) / 25.0),
        min(1.0, float(np.sum(my_mask)) / 20.0),
        min(1.0, float(np.sum(en_mask)) / 20.0),
        min(1.0, len(world["comet_ids"]) / 10.0),
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    # planet block
    planet_order = sorted(range(len(planets)), key=lambda i: planets[i]["id"])[:MAX_PLANETS_FEAT]
    for i in planet_order:
        pi = planets[i]
        owner_code = 1.0 if owners[i] == p else (0.0 if owners[i] == -1 else -1.0)
        feats.extend(
            [
                pi["x"] / 100.0,
                pi["y"] / 100.0,
                min(1.0, pi["radius"] / 6.0),
                min(1.0, float(prod[i]) / 10.0),
                min(1.0, math.log1p(float(ships[i])) / math.log1p(2000.0)),
                owner_code,
                1.0 if pi["is_orbiting"] else 0.0,
                1.0 if pi["is_comet"] else 0.0,
                min(2.0, dist(pi["x"], pi["y"], SUN_X, SUN_Y) / 70.0),
                0.0,
            ]
        )
    if len(planet_order) < MAX_PLANETS_FEAT:
        feats.extend([0.0] * ((MAX_PLANETS_FEAT - len(planet_order)) * PLANET_FEAT_DIM))

    # lightweight fleet-like proxy features from ownership concentration
    my_top = np.sort(ships[my_mask])[-MAX_FLEETS_FEAT:] if np.any(my_mask) else np.array([], dtype=np.float32)
    en_top = np.sort(ships[en_mask])[-MAX_FLEETS_FEAT:] if np.any(en_mask) else np.array([], dtype=np.float32)
    for i in range(MAX_FLEETS_FEAT):
        m = float(my_top[-1 - i]) if i < len(my_top) else 0.0
        e = float(en_top[-1 - i]) if i < len(en_top) else 0.0
        feats.extend([min(1.0, m / 600.0), min(1.0, e / 600.0), 0.0, 0.0, 0.0, 0.0])

    vec = np.asarray(feats, dtype=np.float32)
    if vec.size < OBS_DIM:
        vec = np.concatenate([vec, np.zeros((OBS_DIM - vec.size,), dtype=np.float32)], axis=0)
    elif vec.size > OBS_DIM:
        vec = vec[:OBS_DIM]
    return vec


def _model_policy_value(state, world, actions):
    model = get_model()
    if model is None:
        return None, None

    try:
        vec = _build_obs_vector(state, world)
        feed = vec.reshape(1, -1)
        outs = model.run(None, {_MODEL_INPUT: feed})
        if not outs:
            return None, None

        if len(outs) >= 2:
            logits = np.asarray(outs[0], dtype=np.float32).reshape(-1)
            value = float(np.asarray(outs[1], dtype=np.float32).reshape(-1)[0])
        else:
            arr = np.asarray(outs[0], dtype=np.float32).reshape(-1)
            if arr.size >= len(actions) + 1:
                logits = arr[:len(actions)]
                value = float(arr[len(actions)])
            elif arr.size >= 2:
                logits = arr[:-1]
                value = float(arr[-1])
            else:
                return None, None

        value = max(-1.0, min(1.0, value))

        if len(actions) <= 0:
            return [], value

        if logits.size < len(actions):
            # Pad by repeating last logit to fit dynamic action size.
            if logits.size <= 0:
                logits = np.zeros((len(actions),), dtype=np.float32)
            else:
                pad = np.full((len(actions) - logits.size,), float(logits[-1]), dtype=np.float32)
                logits = np.concatenate([logits, pad], axis=0)
        logits = logits[:len(actions)]
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        s = float(np.sum(probs))
        if s <= 1e-12:
            return None, value
        priors = list((probs / s).astype(np.float64))
        return priors, value
    except Exception:
        return None, None


def _evaluate_node(state, world, actions):
    priors_model, value_model = _model_policy_value(state, world, actions)
    if priors_model is None:
        priors = _heuristic_priors(actions)
    else:
        priors = priors_model
    if value_model is None:
        value = _state_value_heuristic(state, world)
    else:
        value = value_model
    return priors, value


def _apply_action(state, action, world):
    owners = state.owners.copy()
    ships = state.ships.copy()
    p = state.current_player

    if not action.pass_action:
        sidx, tidx, send = action.src_idx, action.tgt_idx, int(action.ships)
        if 0 <= sidx < len(owners) and 0 <= tidx < len(owners) and owners[sidx] == p and ships[sidx] > 1:
            send = max(1, min(send, int(ships[sidx]) - 1))
            ships[sidx] -= send

            if owners[tidx] == p:
                ships[tidx] += send
            else:
                defense = ships[tidx]
                if send > defense:
                    owners[tidx] = p
                    ships[tidx] = send - defense
                else:
                    ships[tidx] = defense - send

    # Light growth approximation for one ply.
    prod = world["production"]
    for i in range(len(owners)):
        if owners[i] >= 0:
            ships[i] += prod[i] * 0.12

    nxt = _next_player(world["player_order"], p)
    return State(owners=owners, ships=ships, current_player=nxt, step=state.step + 1)


def _expand_node(node, world, deadline):
    if node.expanded:
        return _state_value_heuristic(node.state, world)
    actions = _generate_actions(node.state, world, deadline)
    priors, value = _evaluate_node(node.state, world, actions)
    node.actions = actions
    node.priors = priors
    node.expanded = True
    return value


def _select_child(node):
    best_i = None
    best_score = -1e18
    sqrt_n = math.sqrt(max(1, node.visit_count))
    for i, _ in enumerate(node.actions):
        ch = node.children.get(i)
        q = 0.0 if ch is None or ch.visit_count <= 0 else ch.value_sum / ch.visit_count
        p = node.priors[i] if i < len(node.priors) else 0.0
        u = MCTS_C_PUCT * p * sqrt_n / (1 + (0 if ch is None else ch.visit_count))
        score = q + u
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def _simulation_budget(world, state, deadline):
    remaining = max(0.0, deadline - time.perf_counter())
    if remaining <= EARLY_EXIT_BUFFER + 0.10:
        return 8

    my_count = int(np.sum(state.owners == state.current_player))
    total_planets = max(1, len(world["planets"]))
    density = total_planets / 24.0

    base = MCTS_MAX_SIMULATIONS
    if my_count >= 8:
        base -= 12
    elif my_count <= 3:
        base += 4
    base = int(base / max(1.0, density))
    base = max(14, min(MCTS_MAX_SIMULATIONS, base))

    # Hard cap by remaining time.
    time_cap = int(remaining / 0.010)
    return max(8, min(base, time_cap))


def mcts_search(world, root_state, deadline):
    root = Node(state=root_state)
    _expand_node(root, world, deadline)
    simulations = 0
    budget = _simulation_budget(world, root_state, deadline)
    max_depth = 5 if len(world["planets"]) >= 22 else MCTS_MAX_DEPTH

    while simulations < budget and time.perf_counter() < deadline - EARLY_EXIT_BUFFER:
        node = root
        path = [node]

        # Selection
        depth = 0
        while node.expanded and node.actions and depth < max_depth:
            if time.perf_counter() >= deadline - EARLY_EXIT_BUFFER:
                break
            aidx = _select_child(node)
            if aidx is None:
                break
            child = node.children.get(aidx)
            if child is None:
                child_state = _apply_action(node.state, node.actions[aidx], world)
                child = Node(
                    state=child_state,
                    parent=node,
                    action_from_parent=node.actions[aidx],
                    prior_from_parent=node.priors[aidx] if aidx < len(node.priors) else 0.0,
                )
                node.children[aidx] = child
            node = child
            path.append(node)
            depth += 1
            if not node.expanded:
                break

        # Expansion + evaluation
        value = _expand_node(node, world, deadline)

        # Backpropagation with sign flip (value from current player's perspective)
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

        simulations += 1

    return root


# ============================================================
# Final action conversion + safety checks
# ============================================================

def _dangerous_comet_on_path(world, state, src, target, tx, ty, eta, send):
    angle = math.atan2(ty - src["y"], tx - src["x"])
    sx = src["x"] + math.cos(angle) * (src["radius"] + LAUNCH_CLEARANCE)
    sy = src["y"] + math.sin(angle) * (src["radius"] + LAUNCH_CLEARANCE)
    ex = tx - math.cos(angle) * target["radius"]
    ey = ty - math.sin(angle) * target["radius"]

    current_player = state.current_player
    for cid in world["comet_ids"]:
        if cid == target["id"]:
            continue
        cidx = world["id_to_idx"].get(cid)
        if cidx is None:
            continue
        cp = world["planets"][cidx]
        owner = int(state.owners[cidx])
        cships = float(state.ships[cidx])

        # Only avoid clearly risky comets.
        if owner == current_player and cships < send * 0.6:
            continue
        if owner != current_player and cships < max(COMET_MIN_DANGER_SHIPS, send * 0.45):
            continue

        # Current position collision
        c_now = {"x": cp["x"], "y": cp["y"], "radius": cp["radius"]}
        if comet_collision(sx, sy, ex, ey, c_now):
            return True

        # Near-future position collision
        mx, my = _predict_comet_position(world["comet_map"], cid, eta * 0.5, cp["x"], cp["y"])
        c_mid = {"x": mx, "y": my, "radius": cp["radius"]}
        if comet_collision(sx, sy, ex, ey, c_mid):
            return True

    return False


def _estimate_counter_threat(world, state, target_idx, arrival_turn):
    player = state.current_player
    owners = state.owners
    ships = state.ships
    planets = world["planets"]
    target = planets[target_idx]

    horizon = arrival_turn + SELFPLAY_COUNTER_HORIZON
    threat = 0.0
    for i, src in enumerate(planets):
        if owners[i] in (-1, player):
            continue
        eships = float(ships[i])
        if eships <= 1.0:
            continue
        eta = _estimate_eta(src, target["x"], target["y"], target["radius"], max(1, int(eships * 0.45)))
        if eta <= horizon:
            threat += max(0.0, eships - 1.0) * SELFPLAY_COUNTER_FRACTION
    return threat


def _selfplay_action_score(world, state, sidx, tidx, send):
    if sidx < 0 or tidx < 0:
        return -1e9
    planets = world["planets"]
    src = planets[sidx]
    tgt = planets[tidx]
    owners = state.owners
    ships = state.ships
    p = state.current_player

    send = max(1, int(send))
    eta = _estimate_eta(src, tgt["x"], tgt["y"], tgt["radius"], send)
    t_owner = int(owners[tidx])
    t_ships = float(ships[tidx])
    prod = float(world["production"][tidx])
    need = t_ships + 1.0 + (prod * eta * (0.62 if t_owner != -1 else 0.20))
    capture_like = 1.0 if send >= need else max(0.0, min(1.0, (send + 1.0) / max(1.0, need)))

    own_term = 3.0 if t_owner == p else (8.5 if t_owner == -1 else 13.0)
    growth_term = prod * max(0.0, 500.0 - state.step - eta) * 0.025
    speed_term = -0.22 * eta
    ship_cost = -0.08 * send
    counter_threat = _estimate_counter_threat(world, state, tidx, eta)
    risk_term = -SELFPLAY_RISK_WEIGHT * max(0.0, counter_threat - max(0.0, send - 1.0))
    return own_term + growth_term + speed_term + ship_cost + risk_term + 3.6 * capture_like


def _selfplay_reorder_actions(ranked, world, state, deadline):
    # Time-safe light re-ranking: only top-k candidates and only when we have slack.
    if deadline - time.perf_counter() <= SELFPLAY_MIN_TIME_LEFT:
        return ranked

    top = ranked[: max(SELFPLAY_TOP_K, 1)]
    rest = ranked[max(SELFPLAY_TOP_K, 1) :]
    rescored = []
    for visits, idx, act in top:
        if act.pass_action:
            score = -1e9
        else:
            sp = _selfplay_action_score(world, state, act.src_idx, act.tgt_idx, act.ships)
            score = 0.48 * float(visits) + 0.52 * sp
        rescored.append((score, visits, idx, act))

    rescored.sort(reverse=True, key=lambda x: (x[0], x[1], x[3].heuristic))
    top_new = [(v, i, a) for _, v, i, a in rescored]
    return top_new + rest


def _fallback_target_score(state, world, sidx, tidx):
    src = world["planets"][sidx]
    tgt = world["planets"][tidx]
    d = dist(src["x"], src["y"], tgt["x"], tgt["y"])
    owner = int(state.owners[tidx])
    tships = float(state.ships[tidx])
    prod = float(world["production"][tidx])

    if owner == state.current_player:
        owner_term = 2.0
    elif owner == -1:
        owner_term = 7.5
    else:
        owner_term = 12.0

    orbit_term = 0.9 if tgt["is_orbiting"] else 0.0
    comet_term = 0.4 if tgt["is_comet"] else 0.0
    return owner_term + prod * 4.6 + orbit_term + comet_term - 0.11 * tships - 0.09 * d


def _fallback_send_amount(state, world, sidx, tidx, avail):
    owner = int(state.owners[tidx])
    tships = float(state.ships[tidx])
    prod = float(world["production"][tidx])
    src = world["planets"][sidx]
    tgt = world["planets"][tidx]
    eta = _estimate_eta(src, tgt["x"], tgt["y"], tgt["radius"], max(1, int(avail * 0.6)))
    growth = prod * eta

    if owner == state.current_player:
        need = max(5, int(prod * 2.0))
    elif owner == -1:
        need = int(tships + 1 + 0.20 * growth)
    else:
        need = int(tships + 2 + 0.62 * growth)

    need = max(1, need)
    return max(1, min(need, max(1, avail - 1)))


def _internal_fallback_moves(world, deadline):
    state = world["state"]
    p = state.current_player
    owners = state.owners
    ships = state.ships
    planets = world["planets"]

    my_sources = [i for i in range(len(planets)) if owners[i] == p and ships[i] >= FALLBACK_MIN_SOURCE_SHIPS]
    if not my_sources:
        my_sources = [i for i in range(len(planets)) if owners[i] == p and ships[i] > 1]
    if not my_sources:
        return []

    my_sources.sort(key=lambda i: (ships[i], world["production"][i]), reverse=True)
    my_sources = my_sources[: min(7, len(my_sources))]

    target_idxs = [i for i in range(len(planets)) if i not in my_sources or owners[i] != p]
    moves = []
    used_sources = set()
    reserved = {}

    for sidx in my_sources:
        if len(moves) >= FALLBACK_MAX_ORDERS:
            break
        if time.perf_counter() > deadline - EARLY_EXIT_BUFFER:
            break
        if sidx in used_sources:
            continue

        avail = int(ships[sidx]) - int(reserved.get(sidx, 0))
        if avail <= 1:
            continue

        ranked_scores = []
        for tidx in target_idxs:
            base = _fallback_target_score(state, world, sidx, tidx)
            if deadline - time.perf_counter() > SELFPLAY_MIN_TIME_LEFT:
                probe_send = _fallback_send_amount(state, world, sidx, tidx, avail)
                sp = _selfplay_action_score(world, state, sidx, tidx, probe_send)
                base = 0.56 * base + 0.44 * sp
            ranked_scores.append((base, tidx))
        ranked_scores.sort(reverse=True, key=lambda x: x[0])
        ranked = [tidx for _, tidx in ranked_scores[:TOP_K_TARGETS]]

        picked = False
        for tidx in ranked:
            if tidx == sidx:
                continue
            target = planets[tidx]
            src = planets[sidx]
            send = _fallback_send_amount(state, world, sidx, tidx, avail)
            if send <= 0:
                continue

            angle, eta, tx, ty = find_angle_to_moving_planet(src, target, send, world["omega"], world["comet_map"])
            if angle is None or eta is None:
                continue
            if target["is_comet"]:
                life = world["comet_map"].get(target["id"], {}).get("life", 0)
                if eta > max(0.0, life - COMET_TARGET_MIN_REMAINING):
                    continue

            if _dangerous_comet_on_path(world, state, src, target, tx, ty, eta, send):
                continue

            if not math.isfinite(angle):
                continue
            if send >= avail:
                continue

            moves.append([int(src["id"]), float(angle), int(send)])
            used_sources.add(sidx)
            reserved[sidx] = reserved.get(sidx, 0) + send
            picked = True
            break

        # no valid target for this source, try next source
        if not picked:
            continue

    return moves


def _strong_fallback(obs, config, world=None, deadline=None):
    if deadline is None:
        deadline = time.perf_counter() + 0.6

    if world is None:
        try:
            world = _parse_world(obs)
        except Exception:
            world = None

    # First, try external strong policies if packaged.
    for name, fn in (
        ("rule_base_v1", _rule_base_v1_agent),
        ("rule_base_v4", _rule_base_v4_agent),
        ("rule_base_v3", _rule_base_v3_agent),
        ("rule_base_v2", _rule_base_v2_agent),
    ):
        if fn is None:
            continue
        try:
            _warn_fallback_once(f"[agent_alphazero_v1] Using {name} fallback.")
            return fn(obs, config)
        except TypeError:
            try:
                return fn(obs)
            except Exception:
                continue
        except Exception:
            continue

    # Always have an internal fallback so the agent never stalls on Kaggle.
    try:
        _warn_fallback_once("[agent_alphazero_v1] Using internal heuristic fallback.")
        if world is None:
            return []
        return _internal_fallback_moves(world, deadline)
    except Exception:
        return []


def _build_moves_from_root(root, world, deadline):
    if not root.actions:
        return []

    ranked = []
    for i, act in enumerate(root.actions):
        ch = root.children.get(i)
        visits = 0 if ch is None else ch.visit_count
        ranked.append((visits, i, act))
    ranked.sort(reverse=True, key=lambda x: (x[0], x[2].heuristic))
    ranked = _selfplay_reorder_actions(ranked, world, root.state, deadline)

    moves = []
    used_sources = set()
    reserved = {}
    state0 = root.state
    omega = world["omega"]
    comet_map = world["comet_map"]

    for _, _, act in ranked:
        if time.perf_counter() > deadline - EARLY_EXIT_BUFFER:
            break
        if len(moves) >= FINAL_MAX_ORDERS:
            break
        if act.pass_action:
            continue

        sidx, tidx = act.src_idx, act.tgt_idx
        if sidx in used_sources:
            continue
        if sidx < 0 or tidx < 0 or sidx >= len(world["planets"]) or tidx >= len(world["planets"]):
            continue
        if int(state0.owners[sidx]) != state0.current_player:
            continue

        src = world["planets"][sidx]
        target = world["planets"][tidx]
        available = int(state0.ships[sidx]) - int(reserved.get(sidx, 0))
        if available <= 1:
            continue

        send = max(1, min(int(act.ships), available - 1))
        if send <= 0:
            continue

        angle, eta, tx, ty = find_angle_to_moving_planet(src, target, send, omega, comet_map)
        if angle is None or eta is None:
            continue

        # Comet lifetime check for comet targets.
        if target["is_comet"]:
            life = world["comet_map"].get(target["id"], {}).get("life", 0)
            if eta > max(0.0, life - COMET_TARGET_MIN_REMAINING):
                continue

        # Sun avoidance.
        launch_x = src["x"] + math.cos(angle) * (src["radius"] + LAUNCH_CLEARANCE)
        launch_y = src["y"] + math.sin(angle) * (src["radius"] + LAUNCH_CLEARANCE)
        hit_x = tx - math.cos(angle) * target["radius"]
        hit_y = ty - math.sin(angle) * target["radius"]
        if path_crosses_sun(launch_x, launch_y, hit_x, hit_y, margin=SUN_MARGIN):
            continue

        # Comet safety.
        if _dangerous_comet_on_path(world, state0, src, target, tx, ty, eta, send):
            continue

        if not math.isfinite(angle):
            continue
        if send <= 0 or send >= available + 1:
            continue

        moves.append([int(src["id"]), float(angle), int(send)])
        used_sources.add(sidx)
        reserved[sidx] = reserved.get(sidx, 0) + send

    return moves


# ============================================================
# Agent
# ============================================================

def agent(obs, config=None):
    start = time.perf_counter()
    deadline = start + SOFT_DEADLINE

    try:
        if time.perf_counter() >= deadline:
            return []

        world = _parse_world(obs)
        root_state = world["state"]
        my_planets = np.sum(root_state.owners == root_state.current_player)
        if my_planets <= 0:
            return []

        # No model available -> robust rule-based fallback.
        if get_model() is None:
            fallback_moves = _strong_fallback(obs, config, world=world, deadline=deadline)
            if isinstance(fallback_moves, list):
                return fallback_moves
            return []

        # Hybrid control: use robust rule-based policy for opening/mid game,
        # then switch to MCTS for late-game conversion.
        if root_state.step <= 300 and time.perf_counter() < deadline - 0.12:
            base_moves = _strong_fallback(obs, config, world=world, deadline=deadline)
            if isinstance(base_moves, list) and base_moves:
                return base_moves

        root = mcts_search(world, root_state, deadline)
        moves = _build_moves_from_root(root, world, deadline)

        if moves:
            return moves

        # If MCTS gives no valid move under strict safety filters.
        fallback_moves = _strong_fallback(obs, config, world=world, deadline=deadline)
        if isinstance(fallback_moves, list):
            return fallback_moves
        return []

    except Exception:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return []
