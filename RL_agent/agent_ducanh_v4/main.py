from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass

def _find_here():
    try:
        p = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(p, "model.py")):
            return p
    except Exception:
        pass
    try:
        import inspect
        frame = inspect.currentframe()
        if frame is not None:
            p = os.path.dirname(os.path.abspath(inspect.getfile(frame)))
            if os.path.exists(os.path.join(p, "model.py")):
                return p
    except Exception:
        pass
    for p in ["/kaggle_simulations/agent", "./agent", "."]:
        abs_p = os.path.abspath(p)
        if os.path.exists(os.path.join(abs_p, "model.py")):
            return abs_p
    return os.getcwd()

_HERE = _find_here()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
from torch import Tensor



from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves

# Constants
SUN_RADIUS = 10.0
CENTER = 50.0
ALPHA = 0.5

# Global state
_MODEL_2P = None
_MODEL_4P = None
_MODEL_LOAD_ATTEMPTED = False
_CURRENT_WORLD = None


@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs — extended with Focus Fire and Sun-LOS Risk."""
    horizon: int = 18
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12
    max_defensive_targets: int = 4
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5
    min_ships_to_launch: float = 4.0
    # --- regroup ---
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    # --- Sun-LOS potential-attack risk (regroup gradient) ---
    enable_potential_risk: bool = False
    risk_blend_weight: float = 1.0
    risk_enemy_prod_weight: float = 2.0
    risk_self_prod_weight: float = 2.0
    risk_support_weight: float = 0.5
    # --- Focus Fire (coordinated multi-source attack) ---
    enable_focus_fire: bool = True
    max_strike_sources: int = 4
    # --- Reactive Reinforcement Margin ---
    enable_reinforcement_margin: bool = True
    reinforcement_weight: float = 0.8
    reinforcement_scale: float = 4.0



def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


# ---------------------------------------------------------------------------
# Sun-LOS-gated potential attack risk
# ---------------------------------------------------------------------------

def potential_attack_risk(obs, cache, *, horizon: float, player_id: int, config) -> Tensor:
    """Precautionary potential-attack risk per planet [P].
    
    Builds threat from enemy planets, gated by Sun line-of-sight (zeroes threat
    when the direct segment src->tgt grazes the sun), discounted by friendly
    support from nearby owned planets.
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    
    H = max(float(horizon), 1e-6)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    prod = obs.prod.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))
    reach = (speeds.view(P, 1) * H).clamp(min=1e-6)
    decay = (1.0 - d0 / reach).clamp(min=0.0)
    eye = torch.eye(P, device=device, dtype=torch.bool)
    
    # --- Sun line-of-sight gate ---
    x = obs.x.to(dtype)
    y = obs.y.to(dtype)
    ax = x.view(P, 1); ay = y.view(P, 1)
    bx = x.view(1, P); by = y.view(1, P)
    abx = bx - ax; aby = by - ay
    denom = (abx * abx + aby * aby).clamp(min=1e-12)
    u = (((CENTER - ax) * abx + (CENTER - ay) * aby) / denom).clamp(0.0, 1.0)
    cx = ax + u * abx; cy = ay + u * aby
    sun_dist = torch.sqrt(((cx - CENTER) ** 2 + (cy - CENTER) ** 2).clamp(min=0.0))
    los_clear = (sun_dist >= SUN_RADIUS).to(dtype)
    
    # --- Enemy threat (sun-gated) ---
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    strength = ships + float(config.risk_enemy_prod_weight) * prod
    valid_e = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    threat = torch.where(valid_e, strength.view(P, 1) * decay * los_clear, torch.zeros_like(decay))
    enemy_threat = threat.sum(dim=0)
    
    # --- Friendly support discount (ungated — reinforcement can route around sun) ---
    own = obs.owned & obs.alive
    valid_o = own.view(P, 1) & obs.alive.view(1, P) & ~eye
    support = torch.where(
        valid_o, (1.0 + ships).view(P, 1) * decay, torch.zeros_like(decay)
    ).sum(dim=0)
    
    value = 1.0 + float(config.risk_self_prod_weight) * prod
    return value * enemy_threat / (1.0 + float(config.risk_support_weight) * support)


# ---------------------------------------------------------------------------
# Cheap enemy pressure (same as 1k2_elo_agent)
# ---------------------------------------------------------------------------

def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)


# ---------------------------------------------------------------------------
# Model loading (fixed: only attempts once)
# ---------------------------------------------------------------------------

def load_models():
    global _MODEL_2P, _MODEL_4P, _MODEL_LOAD_ATTEMPTED
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    if _MODEL_LOAD_ATTEMPTED:
        return
    _MODEL_LOAD_ATTEMPTED = True
    
    from model import AttentionRanker
    
    for tag, attr_name in [("2p", "_MODEL_2P"), ("4p", "_MODEL_4P")]:
        try:
            model_path = os.path.join(_HERE, f"best_model_{tag}.pt")
            if not os.path.exists(model_path):
                continue
            try:
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(model_path, map_location="cpu")
            is_ppo = any("critic" in k for k in checkpoint["state_dict"].keys())
            if is_ppo:
                # PPO model — extract actor weights only for inference scoring
                from ppo_agent import PPOActorCritic
                ppo_model = PPOActorCritic(max_candidates=25)
                ppo_model.load_state_dict(checkpoint["state_dict"])
                ppo_model.mean.copy_(torch.tensor(checkpoint["mean"]))
                ppo_model.std.copy_(torch.tensor(checkpoint["std"]))
                ppo_model.eval()
                # Wrap to expose same interface as AttentionRanker
                class _PPOWrapper:
                    def __init__(self, m):
                        self.m = m
                        self.mean = m.mean
                        self.std = m.std
                    def __call__(self, x, mask=None):
                        with torch.no_grad():
                            return self.m(x, mask)[0]
                model = _PPOWrapper(ppo_model)
            else:
                model = AttentionRanker(max_candidates=24)
                model.load_state_dict(checkpoint["state_dict"])
                model.mean.copy_(torch.tensor(checkpoint["mean"]))
                model.std.copy_(torch.tensor(checkpoint["std"]))
                model.eval()
            
            if tag == "2p":
                _MODEL_2P = model
            else:
                _MODEL_4P = model
        except Exception:
            pass  # Silently continue — pure heuristic fallback


# ---------------------------------------------------------------------------
# PPO model blended scoring
# ---------------------------------------------------------------------------

def _score_with_model(phys_scores, obs, source_idx, target_idx, valid, S, T, player_count):
    """Blend physical heuristic scores with PPO model logits.
    
    Formula: blended = phys_score * (1.0 + ALPHA * sigmoid(model_logit))
    Only applied to candidates with phys_score > 0.
    Falls back to pure phys_scores if no model loaded.
    """
    model = _MODEL_4P if player_count >= 4 else _MODEL_2P
    if model is None or _CURRENT_WORLD is None:
        return phys_scores
    
    import orbit_base as base
    import main_agent
    
    device = phys_scores.device
    scores = phys_scores.clone()
    
    for s in range(S):
        src_slot = int(source_idx[s].item())
        if src_slot >= len(_CURRENT_WORLD.planets):
            continue
        src_planet = _CURRENT_WORLD.planets[src_slot]
        
        # Collect valid targets with positive physical score
        targets_info = []
        for t in range(T):
            flat_idx = s * T + t
            if not valid[s, t].item():
                continue
            if phys_scores[flat_idx].item() <= 0.0:
                continue
            tgt_slot = int(target_idx[t].item())
            if tgt_slot >= len(_CURRENT_WORLD.planets):
                continue
            tgt_planet = _CURRENT_WORLD.planets[tgt_slot]
            raw_dist = base.dist(src_planet.x, src_planet.y, tgt_planet.x, tgt_planet.y)
            targets_info.append((t, flat_idx, tgt_planet, raw_dist))
        
        if not targets_info:
            continue
        
        targets_info.sort(key=lambda x: x[3])
        targets_info = targets_info[:24]
        
        features_list = []
        for _, _, tgt_planet, raw_dist in targets_info:
            try:
                feat = main_agent._candidate_features(_CURRENT_WORLD, src_planet, tgt_planet, raw_dist)
            except Exception:
                feat = [0.0] * 46
            features_list.append(feat)
        
        feats_tensor = torch.tensor(features_list, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            model_scores = model(feats_tensor).squeeze(0)
        
        for i, (_, flat_idx, _, _) in enumerate(targets_info):
            probs = torch.sigmoid(model_scores[i])
            scores[flat_idx] = phys_scores[flat_idx] * (1.0 + ALPHA * probs.item())
    
    return scores


# ---------------------------------------------------------------------------
# Plan lite waves — with Focus Fire + model blending
# ---------------------------------------------------------------------------

def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
):
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    
    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))
    
    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)
    
    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]
    
    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )
    
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)
    
    # --- ETA-aware reactive-reinforcement margin ---
    reinforcement = None
    if config.enable_reinforcement_margin and K_eta > 0 and P > 0:
        d0 = cache.cross_dist[0].to(dtype) # [P, P]
        radii = movement.radii.to(dtype)
        gap_matrix = radii.view(P, 1) + radii.view(1, P) + 0.1
        ships = obs.ships.to(dtype)
        speeds = fleet_speed(ships.clamp(min=1e-6)) # [P]
        eta_matrix = torch.ceil((d0 - gap_matrix) / speeds.view(P, 1)).clamp(min=1.0) # [P, P]
        
        tgt_owner = obs.owner_abs[target_idx].view(T, 1) # [T, 1]
        all_owner = obs.owner_abs.view(1, P) # [1, P]
        is_same_owner = (all_owner == tgt_owner) & (all_owner >= 0) # [T, P]
        
        eye_target = (target_idx.view(T, 1) == torch.arange(P, device=device).view(1, P)) # [T, P]
        valid_source = is_same_owner & ~eye_target & obs.alive.view(1, P) # [T, P]
        
        eta_tgt = eta_matrix[:, target_idx].transpose(0, 1) # [T, P]
        
        k_grid = torch.arange(1, K_eta + 1, dtype=dtype, device=device).view(1, 1, K_eta) # [1, 1, K_eta]
        eta_tgt_expanded = eta_tgt.unsqueeze(-1) # [T, P, 1]
        dt = k_grid - eta_tgt_expanded # [T, P, K_eta]
        
        w = (dt / float(config.reinforcement_scale)).clamp(0.0, 1.0) # [T, P, K_eta]
        contrib = w * ships.view(1, P, 1) * float(config.reinforcement_weight) # [T, P, K_eta]
        
        valid_source_expanded = valid_source.unsqueeze(-1) # [T, P, 1]
        reinforcement = torch.where(valid_source_expanded, contrib, torch.zeros_like(contrib)).sum(dim=1) # [T, K_eta]

    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid, reinforcement=reinforcement,
    )
    K = int(floor.shape[-1])
    
    sizes = drain.view(S, 1).expand(S, T).floor()
    
    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),
        target_idx.unsqueeze(0),
        sizes,
        active=active,
    )
    angle = aim["angle"]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))
    
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr
    
    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )
    
    # === Focus Fire branching ===
    if not bool(config.enable_focus_fire):
        # Original single-source path
        L = 1
        C = S * T
        cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
        cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
        cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
        cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
        cand_angle = angle.reshape(C, L)
        cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
        cand_active = valid.reshape(C, L)
        cand_valid = valid.reshape(C)
    else:
        # Focus Fire: single-source candidates + pooled multi-source strikes
        L = max(1, int(config.max_strike_sources))
        ST = S * T
        
        ss_src = torch.zeros(ST, L, dtype=torch.long, device=device)
        ss_src[:, 0] = source_idx.view(S, 1).expand(S, T).reshape(-1)
        ss_send = torch.zeros(ST, L, dtype=dtype, device=device)
        ss_send[:, 0] = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(-1)
        ss_angle = torch.zeros(ST, L, dtype=dtype, device=device)
        ss_angle[:, 0] = angle.reshape(-1)
        ss_eta = torch.ones(ST, L, dtype=dtype, device=device)
        ss_eta[:, 0] = torch.where(valid, eta, torch.ones_like(eta)).reshape(-1)
        ss_active = torch.zeros(ST, L, dtype=torch.bool, device=device)
        ss_active[:, 0] = valid.reshape(-1)
        ss_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(-1)
        ss_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(-1)
        ss_valid = valid.reshape(-1)
        
        # Pooled strikes on offensive (non-owned) targets
        eligible = (
            viable & (sizes >= 1.0) & src_neq_tgt
            & source_exists.view(S, 1) & target_exists.view(1, T)
        )
        step_arr = eta.clamp(min=1.0, max=float(K_eta)).ceil().long()
        pooled = []
        
        if L >= 2 and K > 0:
            for t in range(T):
                # if bool(target_is_mine[t]):
                #     continue
                rows = torch.nonzero(eligible[:, t], as_tuple=False).flatten()
                if int(rows.numel()) < 2:
                    continue
                steps_t = step_arr[rows, t]
                for k in torch.unique(steps_t).tolist():
                    k = int(k)
                    if k < 1 or (k - 1) >= K:
                        continue
                    grp = rows[steps_t == k]
                    if int(grp.numel()) < 2:
                        continue
                    gd = sizes[grp, t]
                    order = torch.argsort(gd, descending=True, stable=True)
                    grp = grp[order]
                    csum = torch.cumsum(gd[order], dim=0)
                    need = floor[t, k - 1]
                    hit = torch.nonzero(csum >= need, as_tuple=False)
                    if int(hit.numel()) == 0:
                        continue
                    j = int(hit[0].item()) + 1
                    if j < 2 or j > L:
                        continue
                    pooled.append((t, grp[:j]))
        
        if pooled:
            C2 = len(pooled)
            p_src = torch.zeros(C2, L, dtype=torch.long, device=device)
            p_send = torch.zeros(C2, L, dtype=dtype, device=device)
            p_angle = torch.zeros(C2, L, dtype=dtype, device=device)
            p_eta = torch.ones(C2, L, dtype=dtype, device=device)
            p_active = torch.zeros(C2, L, dtype=torch.bool, device=device)
            p_tgt_slot = torch.zeros(C2, dtype=torch.long, device=device)
            p_tgt_short = torch.zeros(C2, dtype=torch.long, device=device)
            
            for i, (t, grp) in enumerate(pooled):
                j = int(grp.numel())
                p_src[i, :j] = source_idx[grp]
                p_send[i, :j] = sizes[grp, t]
                p_angle[i, :j] = angle[grp, t]
                p_eta[i, :j] = eta[grp, t]
                p_active[i, :j] = True
                p_tgt_slot[i] = target_idx[t]
                p_tgt_short[i] = t
            
            cand_src = torch.cat([ss_src, p_src], dim=0)
            cand_send = torch.cat([ss_send, p_send], dim=0)
            cand_angle = torch.cat([ss_angle, p_angle], dim=0)
            cand_eta = torch.cat([ss_eta, p_eta], dim=0)
            cand_active = torch.cat([ss_active, p_active], dim=0)
            cand_tgt_slot = torch.cat([ss_tgt_slot, p_tgt_slot], dim=0)
            cand_tgt_short = torch.cat([ss_tgt_short, p_tgt_short], dim=0)
            cand_valid = torch.cat(
                [ss_valid, torch.ones(C2, dtype=torch.bool, device=device)], dim=0
            )
        else:
            cand_src, cand_send, cand_angle = ss_src, ss_send, ss_angle
            cand_eta, cand_active = ss_eta, ss_active
            cand_tgt_slot, cand_tgt_short, cand_valid = ss_tgt_slot, ss_tgt_short, ss_valid
        
        C = int(cand_src.shape[0])
    
    cand_is_def = target_is_mine[cand_tgt_short]
    
    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    
    # Physical heuristic scores
    phys_scores = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )
    
    # Blend with model if available
    score = _score_with_model(phys_scores, obs, source_idx, target_idx, valid, S, T, player_count)
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))
    
    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )
    
    if not bool(config.enable_regroup):
        return wave_entries
    
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    if bool(config.enable_potential_risk):
        enemy_mass = enemy_mass + float(config.risk_blend_weight) * potential_attack_risk(
            obs, cache, horizon=float(K_eta), player_id=pid, config=config,
        )
    
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


# ---------------------------------------------------------------------------
# Run turn — standard orbit_lite pipeline
# ---------------------------------------------------------------------------

def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)
    
    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]
    
    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
    )
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


# ---------------------------------------------------------------------------
# Config presets
# ---------------------------------------------------------------------------

CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
    risk_blend_weight=0.5,
    max_strike_sources=3,
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

class ProducerLiteMemory:
    def __init__(self):
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
    
    def reset(self):
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None


class ProducerLiteRuntime:
    def __init__(self, memory=None):
        self.memory = memory if memory is not None else ProducerLiteMemory()
    
    def reset(self):
        self.memory.reset()
    
    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def agent(obs):
    """Kaggle entry point."""
    global _CURRENT_WORLD
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    step = obs.get("step", 0) if isinstance(obs, dict) else obs.step
    step = int(step)
    
    if step == 0:
        _RUNTIME.reset()
        load_models()
        # Reset orbit_base globals if we have model
        if _MODEL_2P is not None or _MODEL_4P is not None:
            try:
                import orbit_base as base
                import main_agent
                base._agent_step = 0
                base._hammer_plan = None
                base._planet_idle_counts = {}
                base._promoted_stockpiles = set()
                base._pending_commitments = []
                base._game_num_players = None
                base._2p_patient_streak = 0
                base._2p_prod_share_history = []
                base._neutral_prev_ships = {}
                base._neutral_wounded = set()
                base._enemy_prev_ships = {}
                base._enemy_recently_launched = set()
                base._planet_prev_owner = {}
                base._freshly_lost_planets = set()
                base._opp_profile = {}
                base._fleet_target_cache = {}
                base._fleet_target_cache_step = -1
                main_agent._HISTORY_BY_PLAYER.clear()
                main_agent._CACHE_BY_PLAYER.clear()
            except Exception:
                pass
    
    # Build World state only when model is available (for feature extraction)
    if _MODEL_2P is not None or _MODEL_4P is not None:
        try:
            import orbit_base as base
            base._agent_step = step
            _CURRENT_WORLD = base.World(obs, inferred_step=step)
            if not _CURRENT_WORLD.is_2p:
                base._update_opp_profile_4p(_CURRENT_WORLD)
        except Exception:
            _CURRENT_WORLD = None
    else:
        _CURRENT_WORLD = None
    
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)


