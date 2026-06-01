"""
Offline trainer for RL_agent/agent_ducanh_v2.py ONNX scorer.

Pipeline:
1) Generate imitation-ranking data from self-play using agent_ducanh_v1.
2) Build candidate move-sets from agent_ducanh_v2 helper functions.
3) Train a small MLP scorer on 15-D features:
   [world_features(7) + candidate_features(8)].
4) Export ONNX model that outputs scalar score.

Usage:
  python train_ducanh_v2.py --episodes 24 --epochs 16
"""

import argparse
import importlib.util
import json
import math
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _candidate_distance(a: List[List[float]], b: List[List[float]]) -> float:
    if a == b:
        return 0.0
    ax = sorted(a, key=lambda x: int(x[0]))
    bx = sorted(b, key=lambda x: int(x[0]))
    d = abs(len(ax) - len(bx)) * 0.6
    k = min(len(ax), len(bx))
    for i in range(k):
        sa = int(ax[i][0])
        sb = int(bx[i][0])
        aa = float(ax[i][1])
        ab = float(bx[i][1])
        na = float(ax[i][2])
        nb = float(bx[i][2])
        d += 0.7 if sa != sb else 0.0
        d += abs(_wrap_pi(aa - ab)) / math.pi
        d += abs(na - nb) / max(8.0, abs(nb))
    return float(d)


def _extract_samples_from_steps(steps, rewards, v2_mod):
    xs = []
    ys = []
    for step in steps[:-1]:
        for i, seat in enumerate(step):
            obs = _get(seat, "observation", None)
            if obs is None:
                continue
            teacher_raw = _get(seat, "action", [])
            teacher = v2_mod._sanitize_moves(obs, teacher_raw if isinstance(teacher_raw, list) else [])
            if not teacher:
                continue

            deadline = time.perf_counter() + 1.0
            cands = v2_mod._build_candidate_sets(obs, teacher, deadline)
            if not cands:
                continue

            obs_feat = v2_mod._world_features(obs)
            teacher_key = v2_mod._freeze_moves(teacher)

            # Outcome term is intentionally light; main signal is imitation ranking.
            r = rewards[i] if i < len(rewards) else 0.0
            r_term = float(np.tanh(r / 120.0))
            for cand in cands:
                cand_feat = v2_mod._candidate_features(cand)
                x = np.concatenate([obs_feat, cand_feat], axis=0).astype(np.float32)
                cand_key = v2_mod._freeze_moves(cand)
                if cand_key == teacher_key:
                    y = 1.0 + 0.20 * r_term
                else:
                    d = _candidate_distance(cand, teacher)
                    y = (0.25 * math.exp(-d)) + (0.10 * r_term) - 0.20
                xs.append(x)
                ys.append(np.float32(y))

    return xs, ys


def _build_dataset(
    v1_agent,
    v2_mod,
    episodes: int,
    start_seed: int,
    num_agents: int,
):
    try:
        from kaggle_environments import make
    except Exception as exc:
        raise RuntimeError(
            "kaggle_environments is not installed. Use --replays to train from replay files, "
            "or install kaggle-environments for self-play data generation."
        ) from exc

    xs = []
    ys = []
    ep_used = 0

    for ep in range(episodes):
        seed = start_seed + ep
        env = make(
            "orbit_wars",
            configuration={"seed": seed, "episodeSteps": 500},
            debug=False,
        )
        env.run([v1_agent] * num_agents)
        steps = env.steps
        if not steps:
            continue
        last = steps[-1]
        rewards = [float(_get(s, "reward", 0.0) or 0.0) for s in last]
        statuses = [_get(s, "status", "") for s in last]
        if any(st not in ("DONE", "ACTIVE") for st in statuses):
            continue

        sx, sy = _extract_samples_from_steps(steps, rewards, v2_mod)
        if sx:
            xs.extend(sx)
            ys.extend(sy)
        ep_used += 1

    if not xs:
        raise RuntimeError("No dataset samples were generated.")
    x_arr = np.stack(xs).astype(np.float32)
    y_arr = np.array(ys, dtype=np.float32).reshape(-1, 1)
    return x_arr, y_arr, ep_used


def _build_dataset_from_replays(v2_mod, replay_paths: List[Path]):
    xs = []
    ys = []
    used = 0
    for rp in replay_paths:
        try:
            obj = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            continue
        steps = obj.get("steps", [])
        if not steps:
            continue
        rewards = obj.get("rewards", []) or []
        if not rewards and steps:
            rewards = [float(_get(s, "reward", 0.0) or 0.0) for s in steps[-1]]
        rewards = [float(r or 0.0) for r in rewards]
        sx, sy = _extract_samples_from_steps(steps, rewards, v2_mod)
        if sx:
            xs.extend(sx)
            ys.extend(sy)
            used += 1
    if not xs:
        raise RuntimeError("No samples extracted from replay files.")
    x_arr = np.stack(xs).astype(np.float32)
    y_arr = np.array(ys, dtype=np.float32).reshape(-1, 1)
    return x_arr, y_arr, used


def _train_model(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
):
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for offline training. Install torch before running this script."
        ) from exc

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n = x.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = max(1, int(n * 0.9))
    tr_idx = idx[:split]
    va_idx = idx[split:] if split < n else idx[:1]

    x_tr = torch.from_numpy(x[tr_idx])
    y_tr = torch.from_numpy(y[tr_idx])
    x_va = torch.from_numpy(x[va_idx])
    y_va = torch.from_numpy(y[va_idx])

    model = nn.Sequential(
        nn.Linear(15, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    best_state = None
    best_val = float("inf")
    hist = []

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(x_tr.shape[0])
        total = 0.0
        count = 0
        for s in range(0, x_tr.shape[0], batch_size):
            b = perm[s : s + batch_size]
            xb = x_tr[b]
            yb = y_tr[b]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            count += xb.shape[0]

        model.eval()
        with torch.no_grad():
            val = loss_fn(model(x_va), y_va).item()
        tr = total / max(1, count)
        hist.append({"epoch": ep, "train_loss": tr, "val_loss": float(val)})
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 2 == 0 or ep == 1 or ep == epochs:
            print(f"[train] epoch {ep:02d} train={tr:.6f} val={val:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def _export_onnx(model, out_path: str):
    import torch

    try:
        import onnx  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'onnx' for export. Install with: pip install onnx"
        ) from exc

    model.eval()
    dummy = torch.randn(1, 15, dtype=torch.float32)
    kwargs = dict(
        input_names=["x"],
        output_names=["score"],
        dynamic_axes={"x": {0: "batch"}, "score": {0: "batch"}},
        opset_version=13,
    )
    # Force legacy exporter to avoid hard dependency on onnxscript.
    try:
        torch.onnx.export(model, dummy, out_path, dynamo=False, **kwargs)
        return
    except TypeError:
        # Older torch versions don't expose dynamo kwarg.
        torch.onnx.export(model, dummy, out_path, **kwargs)
        return
    except Exception as exc:
        raise RuntimeError(f"ONNX export failed: {exc}") from exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--start-seed", type=int, default=1000)
    ap.add_argument("--num-agents", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path("RL_agent") / "ducanh_v2_actor.onnx"),
    )
    ap.add_argument(
        "--replays",
        type=str,
        default="",
        help="Comma-separated replay json files/globs (e.g. '78305775.json,78311384.json').",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent
    v1_path = repo / "RL_agent" / "agent_ducanh_v1.py"
    v2_path = repo / "RL_agent" / "agent_ducanh_v2.py"
    if not v1_path.exists() or not v2_path.exists():
        raise RuntimeError("Missing RL_agent/agent_ducanh_v1.py or RL_agent/agent_ducanh_v2.py")

    v1_mod = _load_module(str(v1_path), "agent_ducanh_v1_train")
    v2_mod = _load_module(str(v2_path), "agent_ducanh_v2_train")
    v1_agent = getattr(v1_mod, "agent", None)
    if not callable(v1_agent):
        raise RuntimeError("agent_ducanh_v1.py does not expose callable agent()")

    print("[data] generating dataset...")
    if args.replays.strip():
        replay_paths = []
        for token in args.replays.split(","):
            token = token.strip()
            if not token:
                continue
            replay_paths.extend(sorted(repo.glob(token)))
        if not replay_paths:
            raise RuntimeError("No replay files matched --replays patterns.")
        x, y, used = _build_dataset_from_replays(v2_mod=v2_mod, replay_paths=replay_paths)
    else:
        x, y, used = _build_dataset(
            v1_agent=v1_agent,
            v2_mod=v2_mod,
            episodes=args.episodes,
            start_seed=args.start_seed,
            num_agents=args.num_agents,
        )
    print(f"[data] episodes_used={used} samples={x.shape[0]} dim={x.shape[1]}")

    print("[train] training scorer model...")
    model, hist = _train_model(
        x=x,
        y=y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] writing ONNX to {out_path}")
    _export_onnx(model, str(out_path))

    meta = {
        "episodes": args.episodes,
        "episodes_used": used,
        "samples": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "history_tail": hist[-5:],
        "model_path": str(out_path),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
