import math
from collections import defaultdict, deque

import orbit_base as base
import feature46_weights_2p as scorer_2p
import feature46_weights_4p as scorer_4p


_base_nearest_targets = base._nearest_targets

K_SHIPS = 40.0
EPS = 1.0e-6
LAMBDA_ECEP = 0.3
LAMBDA_POTENTIAL = 0.15
BUFFER_SIZE = 5
RHYTHM_SIZE = 10
MAX_ETA = 25.0
TOTAL_STEPS = 500.0
BOARD = 100.0

ATTN_ENABLED = True
ATTN_EXPAND_K_EXTRA_2P = 10
ATTN_EXPAND_K_EXTRA_4P = 10
ATTN_BONUS_2P = 1.45
ATTN_BONUS_4P = 1.25
ATTN_MIN_STEP = 0
ATTN_4P_TIE_JITTER = 1e-06


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def _ship_sigmoid(x):
    x = max(0.0, float(x))
    return x / (x + K_SHIPS)


def _sigmoid(x):
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _dist_obj(a, b):
    return base.dist(a.x, a.y, b.x, b.y)


def _direct_eta_obj(a, b, ships):
    d = _dist_obj(a, b)
    return max(1, int(math.ceil(d / base.fleet_speed(max(1, int(ships)))))), d


def _direct_eta_xy(ax, ay, bx, by, ships):
    d = base.dist(ax, ay, bx, by)
    return max(1, int(math.ceil(d / base.fleet_speed(max(1, int(ships)))))), d


def _center_of_mass(world, owner):
    sx = sy = sw = 0.0
    for p in world.planets:
        if int(p.owner) != int(owner):
            continue
        w = max(1.0, float(p.ships))
        sx += float(p.x) * w
        sy += float(p.y) * w
        sw += w
    if sw <= 0.0:
        for f in world.fleets:
            if int(f.owner) != int(owner):
                continue
            w = max(1.0, float(f.ships))
            sx += float(f.x) * w
            sy += float(f.y) * w
            sw += w
    if sw <= 0.0:
        return base.CENTER_X, base.CENTER_Y
    return sx / sw, sy / sw


def _angular_spread(my_com, enemy_coms):
    if len(enemy_coms) <= 1:
        return 0.0
    angles = sorted(math.atan2(y - my_com[1], x - my_com[0]) for x, y in enemy_coms)
    gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
    gaps.append(angles[0] + 2.0 * math.pi - angles[-1])
    return _clamp(1.0 - max(gaps) / (2.0 * math.pi))


class _FeatureHistory:
    def __init__(self):
        self.ship_share = {i: deque(maxlen=BUFFER_SIZE) for i in range(4)}
        self.targeted = deque(maxlen=BUFFER_SIZE)
        self.incoming = deque(maxlen=RHYTHM_SIZE)
        self.prev_enemy_dists = None
        self.convergence_prev = 0.0
        self.last_step = None
        self.last_outputs = None

    def update(self, step, player, shares, targeted_ratio, hostile_incoming, enemy_dists, convergence_raw):
        step = int(step)
        if self.last_step == step and self.last_outputs is not None:
            return self.last_outputs
        if self.last_step is not None and step <= self.last_step:
            self.__init__()
        momentum = 0.0
        old_my = self.ship_share.get(player, deque())
        if len(old_my) >= BUFFER_SIZE:
            momentum = float(shares.get(player, 0.0)) - float(old_my[0])
        enemy_momentums = []
        for owner in range(4):
            old = self.ship_share.get(owner, deque())
            if owner == player or len(old) < BUFFER_SIZE:
                continue
            enemy_momentums.append(float(shares.get(owner, 0.0)) - float(old[0]))
        fastest_gap = (max(enemy_momentums) - momentum) if enemy_momentums else 0.0
        aggression_trend = float(targeted_ratio) - float(self.targeted[0]) if len(self.targeted) >= BUFFER_SIZE else 0.0
        enemy_rhythm = 0.0
        if len(self.incoming) >= RHYTHM_SIZE:
            vals = list(float(v) for v in self.incoming)
            mean = sum(vals) / max(1, len(vals))
            var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
            enemy_rhythm = _clamp(var / (mean * mean + EPS))
        approach_rate = 0.0
        if self.prev_enemy_dists:
            total = 0.0
            n = 0
            for owner, cur in enemy_dists.items():
                prev = self.prev_enemy_dists.get(owner)
                if prev is None:
                    continue
                total += max(0.0, float(prev) - float(cur))
                n += 1
            approach_rate = _clamp(total / (max(1, n) * 12.0 + EPS))
        convergence = _clamp(0.7 * self.convergence_prev + 0.3 * float(convergence_raw))
        self.convergence_prev = convergence
        for owner in range(4):
            self.ship_share.setdefault(owner, deque(maxlen=BUFFER_SIZE)).append(float(shares.get(owner, 0.0)))
        self.targeted.append(float(targeted_ratio))
        self.incoming.append(float(hostile_incoming))
        self.prev_enemy_dists = dict(enemy_dists)
        self.last_step = step
        self.last_outputs = {
            "momentum": _clamp(momentum, -1.0, 1.0),
            "fastest_grower_gap": _clamp(fastest_gap, -1.0, 1.0),
            "aggression_trend": _clamp(aggression_trend, -1.0, 1.0),
            "enemy_rhythm": enemy_rhythm,
            "approach_rate": approach_rate,
            "convergence_threat": convergence,
        }
        return self.last_outputs


_HISTORY_BY_PLAYER = {}
_CACHE_BY_PLAYER = {}


def _history_for(world):
    key = int(world.player)
    hist = _HISTORY_BY_PLAYER.get(key)
    if hist is None:
        hist = _FeatureHistory()
        _HISTORY_BY_PLAYER[key] = hist
    return hist


def _nearest_enemy_distance(world, planet):
    if not world.enemy_planets:
        return 1.0e9
    return min(base.dist(planet.x, planet.y, enemy.x, enemy.y) for enemy in world.enemy_planets)


def _feature_cache(world):
    player = int(world.player)
    cached = _CACHE_BY_PLAYER.get(player)
    if cached is not None and cached.get("step") == int(world.step):
        return cached

    totals = defaultdict(lambda: {"planets": 0, "planet_ships": 0.0, "fleet_ships": 0.0, "prod": 0.0, "transit": 0.0, "total_ships": 0.0})
    alive = set()
    for p in world.planets:
        if int(p.owner) == -1:
            continue
        rec = totals[int(p.owner)]
        rec["planets"] += 1
        rec["planet_ships"] += float(p.ships)
        rec["prod"] += float(p.production)
        alive.add(int(p.owner))
    for f in world.fleets:
        rec = totals[int(f.owner)]
        rec["fleet_ships"] += float(f.ships)
        rec["transit"] += float(f.ships)
        alive.add(int(f.owner))
    for rec in totals.values():
        rec["total_ships"] = rec["planet_ships"] + rec["fleet_ships"]

    total_ships = sum(rec.get("total_ships", 0.0) for rec in totals.values())
    total_prod = sum(rec.get("prod", 0.0) for rec in totals.values())
    my = totals.get(player, {})
    my_total = float(my.get("total_ships", 0.0))
    my_prod = float(my.get("prod", 0.0))
    enemy_owners = [o for o in totals if o not in (-1, player) and totals[o].get("total_ships", 0.0) > 0.0]
    max_enemy_ships = max((totals[o].get("total_ships", 0.0) for o in enemy_owners), default=0.0)
    hostile_transit = sum(float(f.ships) for f in world.fleets if int(f.owner) not in (-1, player))
    hostile_to_me = 0.0
    my_planet_ids = {int(p.id) for p in world.my_planets}
    for pid, rows in world.arrivals_by_planet.items():
        if int(pid) not in my_planet_ids:
            continue
        hostile_to_me += sum(float(ships) for _eta, owner, ships in rows if int(owner) not in (-1, player))
    targeted_ratio = hostile_to_me / (hostile_transit + EPS)

    my_com = _center_of_mass(world, player)
    enemy_com = {o: _center_of_mass(world, o) for o in enemy_owners}
    enemy_dists = {o: base.dist(my_com[0], my_com[1], c[0], c[1]) for o, c in enemy_com.items()}
    avg_enemy_dist_raw = sum(enemy_dists.values()) / max(1, len(enemy_dists))
    min_enemy_dist = min(enemy_dists.values(), default=avg_enemy_dist_raw)
    convergence_raw = _clamp(min_enemy_dist / (avg_enemy_dist_raw + EPS)) if enemy_dists else 0.0
    shares = {owner: rec.get("total_ships", 0.0) / (total_ships + EPS) for owner, rec in totals.items()}
    hist = _history_for(world).update(int(world.step), player, shares, targeted_ratio, hostile_to_me, enemy_dists, convergence_raw)

    weakest_enemy = None
    if enemy_owners:
        weakest_enemy = min(enemy_owners, key=lambda o: totals[o].get("total_ships", 0.0))
    weakest_colony = 1.0
    if world.my_planets:
        for p in world.my_planets:
            hostile = sum(
                float(ships)
                for eta, owner, ships in world.arrivals_by_planet.get(p.id, [])
                if int(owner) not in (-1, player) and int(eta) <= MAX_ETA
            )
            weakest_colony = min(weakest_colony, _clamp((float(p.ships) - hostile) / (float(p.ships) + EPS)))
    else:
        weakest_colony = 0.0

    macro = {
        "game_progress": _clamp(float(world.step) / TOTAL_STEPS),
        "players_alive": _clamp(len(alive) / 4.0),
        "my_planet_share": _clamp(float(len(world.my_planets)) / max(1.0, float(len(world.planets)))),
        "my_gdp_share": _clamp(my_prod / (total_prod + EPS)),
        "my_ship_share": _clamp(my_total / (total_ships + EPS)),
        "momentum": hist["momentum"],
        "my_fleet_ratio": _clamp(float(my.get("transit", 0.0)) / (my_total + EPS)),
        "leader_gap": _clamp((max_enemy_ships - my_total) / (my_total + K_SHIPS), -1.0, 2.0),
        "am_i_targeted": _clamp(targeted_ratio),
        "fastest_grower_gap": hist["fastest_grower_gap"],
        "aggression_trend": hist["aggression_trend"],
        "enemy_rhythm": hist["enemy_rhythm"],
        "avg_enemy_distance": _clamp(avg_enemy_dist_raw / (math.sqrt(2.0) * BOARD)),
        "angular_spread": _angular_spread(my_com, list(enemy_com.values())),
        "convergence_threat": hist["convergence_threat"],
        "approach_rate": hist["approach_rate"],
        "weakest_colony": weakest_colony,
    }
    cached = {
        "step": int(world.step),
        "totals": totals,
        "total_ships": total_ships,
        "total_prod": total_prod,
        "my_total": my_total,
        "my_prod": my_prod,
        "weakest_enemy": weakest_enemy,
        "macro": macro,
    }
    _CACHE_BY_PLAYER[player] = cached
    return cached


def _candidate_features(world, src, target, raw_distance):
    cache = _feature_cache(world)
    player = int(world.player)
    src_ships = max(1, int(src.ships))
    send = src_ships
    aim = base.aim_at_target(src, target, send, world.initial_by_id, world.ang_vel, world=world)
    if aim is None:
        eta = max(1, int(math.ceil(float(raw_distance) / base.fleet_speed(send))))
    else:
        _angle, eta = aim

    proj_owner, proj_ships = base.predict_defender_at_arrival(world, target, eta)
    proj_ships = max(0.0, float(proj_ships))
    need = 0.0 if int(proj_owner) == player else float(math.ceil(proj_ships)) + 1.0
    margin = float(send) - need
    target_owner = int(target.owner)
    target_total = cache["totals"].get(target_owner, {}).get("total_ships", 0.0) if target_owner != -1 else 0.0
    owner_power = _clamp(target_total / (cache["total_ships"] + EPS))
    weakest_enemy = cache["weakest_enemy"]

    src_threat = sum(
        _ship_sigmoid(ships) * math.exp(-LAMBDA_ECEP * float(eta0))
        for eta0, owner, ships in world.arrivals_by_planet.get(src.id, [])
        if int(owner) not in (-1, player)
    )
    src_safety = _clamp(1.0 - src_threat / (src_threat + 1.0))
    arrivals = world.arrivals_by_planet.get(target.id, [])
    threat_after = sum(
        _ship_sigmoid(ships) * math.exp(-LAMBDA_ECEP * max(0.0, float(eta0) - float(eta)))
        for eta0, owner, ships in arrivals
        if int(eta0) > int(eta) and int(owner) not in (-1, player)
    )
    support_after = sum(
        _ship_sigmoid(ships) * math.exp(-LAMBDA_ECEP * max(0.0, float(eta0) - float(eta)))
        for eta0, owner, ships in arrivals
        if int(eta0) > int(eta) and int(owner) == player
    )
    pre_volatility = sum(
        _ship_sigmoid(ships) * math.exp(-LAMBDA_ECEP * max(0.0, float(eta) - float(eta0)))
        for eta0, owner, ships in arrivals
        if int(eta0) < int(eta)
    )
    threat_potential_raw = 0.0
    for p in world.enemy_planets:
        e_eta, _ = _direct_eta_obj(p, target, max(1, int(p.ships)))
        threat_potential_raw += _ship_sigmoid(p.ships) * math.exp(-LAMBDA_POTENTIAL * e_eta)
    support_potential_raw = 0.0
    for p in world.my_planets:
        if int(p.id) == int(src.id):
            continue
        s_eta, _ = _direct_eta_obj(p, target, max(1, int(p.ships)))
        support_potential_raw += _ship_sigmoid(p.ships) * math.exp(-LAMBDA_POTENTIAL * s_eta)
    threat_potential = _clamp(min(threat_potential_raw, 2.0) / 2.0)
    support_potential = _clamp(min(support_potential_raw, 2.0) / 2.0)

    if target_owner == player:
        diplomacy = 1.0
        survivors = float(send)
    elif target_owner == -1:
        diplomacy = 0.0
        survivors = max(0.0, float(send) - proj_ships)
    else:
        diplomacy = -1.0
        survivors = max(0.0, float(send) - proj_ships)

    garrison_strength = _ship_sigmoid(survivors)
    defense_sustainability = _clamp(garrison_strength / (float(threat_after) + EPS))
    src_front = _nearest_enemy_distance(world, src)
    tgt_front = _nearest_enemy_distance(world, target)
    src_x2, src_y2 = base.predict_planet_position(src, world.initial_by_id, world.ang_vel, 1)
    tgt_x2, tgt_y2 = base.predict_target_position(target, world, 1) if hasattr(base, "predict_target_position") else base.predict_planet_position(target, world.initial_by_id, world.ang_vel, 1)
    eta_next, _ = _direct_eta_xy(src_x2, src_y2, tgt_x2, tgt_y2, send)
    orbital_trend = _clamp((float(eta_next) - float(eta)) / (float(eta) + EPS), -1.0, 1.0)
    enemy_commitment = 0.0
    if target_owner not in (-1, player):
        rec = cache["totals"].get(target_owner, {})
        enemy_commitment = _clamp(float(rec.get("transit", 0.0)) / (float(rec.get("total_ships", 0.0)) + EPS))
    local_superiority = _clamp(min(support_potential / (threat_potential + 0.1), 3.0) / 3.0)
    economic_impact = _clamp(float(target.production) / (cache["my_prod"] + 1.0))
    enemy_exposed = _clamp(enemy_commitment * (1.0 - owner_power))
    macro = cache["macro"]
    row = [
        macro["game_progress"],
        macro["players_alive"],
        macro["my_planet_share"],
        macro["my_gdp_share"],
        macro["my_ship_share"],
        macro["momentum"],
        macro["my_fleet_ratio"],
        macro["leader_gap"],
        macro["am_i_targeted"],
        macro["fastest_grower_gap"],
        macro["aggression_trend"],
        macro["enemy_rhythm"],
        _ship_sigmoid(src_ships),
        _clamp(float(src.production) / 5.0),
        src_safety,
        diplomacy,
        _clamp(float(target.production) / 5.0),
        1.0 if int(target.id) in world.comet_ids or not base.is_static_planet(target) else 0.0,
        owner_power,
        1.0 if target_owner == weakest_enemy and target_owner not in (-1, player) else 0.0,
        1.0 if int(proj_owner) == player else (0.0 if int(proj_owner) == -1 else -1.0),
        _ship_sigmoid(proj_ships),
        _clamp(threat_after),
        _clamp(support_after),
        _clamp(pre_volatility),
        threat_potential,
        support_potential,
        _clamp(float(eta) / MAX_ETA),
        _clamp(float(send) / (cache["my_total"] + EPS)),
        _clamp(need / (float(src_ships) + EPS), 0.0, 2.0),
        _clamp(margin / (float(send) + K_SHIPS), -1.0, 1.0),
        1.0 if float(send) > need else 0.0,
        garrison_strength,
        defense_sustainability,
        _clamp(float(target.production) / (need + 1.0)),
        _clamp((src_front - tgt_front) / 60.0, -1.0, 1.0),
        macro["avg_enemy_distance"],
        macro["angular_spread"],
        orbital_trend,
        macro["convergence_threat"],
        macro["approach_rate"],
        enemy_commitment,
        macro["weakest_colony"],
        local_superiority,
        economic_impact,
        enemy_exposed,
    ]
    for idx, value in enumerate(row):
        if not math.isfinite(float(value)):
            row[idx] = 0.0
    return row


def _selected_scorer(world):
    if world.is_2p:
        return scorer_2p, ATTN_BONUS_2P, ATTN_EXPAND_K_EXTRA_2P
    return scorer_4p, ATTN_BONUS_4P, ATTN_EXPAND_K_EXTRA_4P


def _tiny_4p_jitter(world, src_id, target_id, idx):
    if world.is_2p:
        return 0.0
    value = (
        int(world.step) * 1103515245
        + int(world.player) * 12345
        + int(src_id) * 2654435761
        + int(target_id) * 97531
        + int(idx) * 31337
    ) & 0xFFFFFFFF
    value ^= value >> 16
    return (value & 0xFFFF) / 65535.0 * ATTN_4P_TIE_JITTER


def _attn_nearest_targets(src, world, K, max_travel, target_locked):
    if not ATTN_ENABLED or world.step < ATTN_MIN_STEP:
        return _base_nearest_targets(src, world, K, max_travel, target_locked)
    scorer, bonus, extra = _selected_scorer(world)
    expanded_k = max(K, min(K + extra, 16))
    candidates = _base_nearest_targets(src, world, expanded_k, max_travel, target_locked)
    if len(candidates) <= 1:
        return candidates
    rows = []
    valid = []
    for idx, (target, raw_distance) in enumerate(candidates):
        try:
            rows.append(_candidate_features(world, src, target, raw_distance))
            valid.append((idx, target, raw_distance))
        except Exception:
            rows.append(None)
            valid.append((idx, target, raw_distance))
    try:
        compact_rows = [row for row in rows if row is not None]
        raw_scores = scorer.score_many(compact_rows)
        score_iter = iter(raw_scores)
        scores = []
        for row in rows:
            scores.append(0.0 if row is None else _sigmoid(float(next(score_iter))))
    except Exception:
        scores = [0.0 for _ in rows]
    scored = []
    for (idx, target, raw_distance), score in zip(valid, scores):
        score = _clamp(float(score) + _tiny_4p_jitter(world, src.id, target.id, idx))
        adjusted = idx - bonus * score
        scored.append((adjusted, -score, idx, target, raw_distance))
    scored.sort()
    return [(target, raw_distance) for _adj, _neg_score, _idx, target, raw_distance in scored[:K]]


base._nearest_targets = _attn_nearest_targets


def agent(obs, config=None):
    return base.agent(obs, config)


__all__ = ["agent"]
