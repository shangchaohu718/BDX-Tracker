#!/usr/bin/env python
# [BFM-DIAG-NAN] -- instrumentation added by Claude
#
# Offline first-update debugger for BFM-Zero. Loads the BFM-DIAG-SNAPSHOT captured at
# the end of the random-action seed phase (num_seed_steps = 10*N_env pure random rollouts)
# and re-runs the first agent.update() with STAGED finiteness checks, to locate exactly
# where the first non-finite value appears:
#   Stage 0 : raw random-rollout buffer (the prime suspect) + expert batch
#   Stage 1 : BatchRank observation normalization (stats update + eval normalize)
#   Stage 2 : network forward outputs (expert_z, backward_map, forward_map, discriminator)
#   Stage 3 : full agent.update() (faithful reproduction of the first update)
#
# Usage (remote):
#   cd /root/epfs/tcl/bdx_BFMzero
#   BFM_ZERO_SNAPSHOT=1 nohup .venv/bin/python -m humanoidverse.train > snapshot.log 2>&1 &
#   # ...wait for "[BFM-DIAG-SNAPSHOT] exiting after snapshot capture."...
#   .venv/bin/python -m humanoidverse.debug_first_update --snap_dir results/bfmzero-isaac-diag/debug_snapshot
import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.nn_models import eval_mode


class _FixedExpertBuffer:
    """Sham replay-buffer entry whose .sample() always returns the saved expert batch
    (expert data is clean demo trajectories -- not a NaN suspect -- so one fixed batch is enough)."""

    def __init__(self, batch):
        self._batch = batch

    def sample(self, batch_size, seq_length=None):
        return self._batch


def _leaf_stats(name, t):
    if not torch.is_tensor(t):
        return None
    t = t.float()
    has_fin = torch.isfinite(t).any().item()
    return {
        "name": name,
        "shape": str(tuple(t.shape)),
        "nan": int(torch.isnan(t).sum().item()),
        "inf": int(torch.isinf(t).sum().item()),
        "min": float(t[torch.isfinite(t)].min().item()) if has_fin else float("nan"),
        "max": float(t[torch.isfinite(t)].max().item()) if has_fin else float("nan"),
    }


def _scan_tree(storage, prefix=""):
    rows = []

    def _walk(d, pfx):
        for k, v in d.items():
            key = f"{pfx}/{k}" if pfx else k
            if isinstance(v, dict):
                _walk(v, key)
            elif torch.is_tensor(v):
                rows.append(_leaf_stats(key, v))

    _walk(storage, prefix)
    return rows


def _print_scan(title, rows):
    print(f"\n--- {title} ---")
    tot_nan = sum(r["nan"] for r in rows)
    tot_inf = sum(r["inf"] for r in rows)
    print(f"    total NaN={tot_nan}, inf={tot_inf} across {len(rows)} tensors")
    for r in rows:
        flag = "OK " if (r["nan"] == 0 and r["inf"] == 0) else "BAD"
        print(f"    [{flag}] {r['name']:44s} shape={r['shape']:20s} "
              f"NaN={r['nan']:>9d} inf={r['inf']:>9d} min={r['min']:+.4e} max={r['max']:+.4e}")
    return tot_nan == 0 and tot_inf == 0


def _single(name, t):
    return _print_scan(name, [r for r in [_leaf_stats(name, t)] if r is not None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap_dir", required=True, help="path to debug_snapshot/ produced by BFM_ZERO_SNAPSHOT=1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_rng", action="store_true", help="do not restore RNG states (faster, less faithful sampling)")
    ap.add_argument("--num_updates", type=int, default=1,
                    help="if >1, run this many updates in a loop on a fresh agent and report the first NaN update "
                         "(tests whether the networks blow up over time on finite data)")
    args = ap.parse_args()

    snap = Path(args.snap_dir)
    dev = args.device
    print(f"[BFM-DIAG-NAN] loading snapshot from {snap}")

    # ---------------------------------------------------------------- load state
    train_buf = TrajectoryDictBufferMultiDim.load(snap / "train_buffer", device=dev)
    print(f"[BFM-DIAG-NAN] train buffer: len={len(train_buf)} capacity={train_buf.capacity} size(transitions)={train_buf.size()}")

    agent = FBcprAuxAgent.load(str(snap / "agent"), device=dev)
    bs = agent.cfg.train.batch_size
    print(f"[BFM-DIAG-NAN] agent: batch_size={bs} seq_length={agent.cfg.model.seq_length} amp={agent.cfg.model.amp} device={dev}")

    expert_batch = torch.load(snap / "expert_batch.pt", weights_only=False, map_location=dev)
    expert_batch = tree_map(lambda x: x.to(dev) if torch.is_tensor(x) else x, expert_batch)

    with open(snap / "rng_states.pkl", "rb") as f:
        rng = pickle.load(f)
    if not args.skip_rng:
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        print(f"[BFM-DIAG-NAN] restored RNG (snapshot t={rng['t']})")
    else:
        print(f"[BFM-DIAG-NAN] RNG not restored (snapshot t={rng['t']})")

    first_bad_stage = None

    # ------------------------------------------------------ Stage 0: raw data
    print("\n" + "=" * 78)
    print("STAGE 0 : raw random-rollout buffer finiteness  (THE PRIME SUSPECT)")
    print("=" * 78)
    raw_ok = _print_scan("train buffer storage (ALL random-rollout transitions)", _scan_tree(train_buf.storage))
    _print_scan("fixed expert batch", _scan_tree(expert_batch))
    train_batch = train_buf.sample(bs)
    _print_scan("one sampled train batch (as update() sees it)", _scan_tree(train_batch))
    if not raw_ok:
        first_bad_stage = 0
        print("\n>>> STAGE 0 BAD: the random-rollout buffer ALREADY contains NaN/inf.")
        print(">>> The simulator produces non-finite values under random actions, BEFORE any PyTorch update.")
    else:
        print("\n>>> STAGE 0 OK: raw rollout data is finite. NaN arises downstream (stages 1-3).")

    # ------------------------------------------------------ Stage 1: normalize
    print("\n" + "=" * 78)
    print("STAGE 1 : observation normalization (BatchRank running-stats update + eval normalize)")
    print("=" * 78)
    try:
        train_obs = tree_map(lambda x: x.to(dev), train_batch["observation"])
        train_next_obs = tree_map(lambda x: x.to(dev), train_batch["next"]["observation"])
        expert_obs = tree_map(lambda x: x.to(dev), expert_batch["observation"])
        expert_next_obs = tree_map(lambda x: x.to(dev), expert_batch["next"]["observation"])
        # mimic update(): update running stats (train mode), then normalize (eval mode)
        agent._model._obs_normalizer(train_obs)
        agent._model._obs_normalizer(train_next_obs)
        with torch.no_grad(), eval_mode(agent._model._obs_normalizer):
            n_train_obs = agent._model._obs_normalizer(train_obs)
            n_train_next = agent._model._obs_normalizer(train_next_obs)
            n_expert_obs = agent._model._obs_normalizer(expert_obs)
            n_expert_next = agent._model._obs_normalizer(expert_next_obs)
        s1_ok = (
            _print_scan("normalized train_obs", _scan_tree(n_train_obs))
            and _print_scan("normalized train_next_obs", _scan_tree(n_train_next))
            and _print_scan("normalized expert_obs", _scan_tree(n_expert_obs))
        )
        if not s1_ok and first_bad_stage is None:
            first_bad_stage = 1
            print("\n>>> STAGE 1 BAD: normalization produces NaN/inf (BatchRank stats likely poisoned by raw data).")
        else:
            print("\n>>> STAGE 1 OK: normalized observations are finite.")
    except Exception as e:  # noqa
        import traceback
        traceback.print_exc()
        if first_bad_stage is None:
            first_bad_stage = 1
        print(f"\n>>> STAGE 1 EXCEPTION: {e}")

    # ------------------------------------------------------ Stage 2: forwards
    print("\n" + "=" * 78)
    print("STAGE 2 : network forward outputs (no grad)")
    print("=" * 78)
    try:
        train_z = train_batch["z"].to(dev)
        train_action = train_batch["action"].to(dev)
        with torch.no_grad():
            expert_z = agent.encode_expert(next_obs=n_expert_next)
            B = agent._model._backward_map(n_train_next)
            F = agent._model._forward_map(n_train_obs, train_z, train_action)
            disc_logits = agent._model._discriminator.compute_logits(obs=n_train_obs, z=train_z)
        s2_ok = (
            _single("expert_z (encode_expert)", expert_z)
            and _single("backward_map B", B)
            and _single("forward_map F", F)
            and _single("discriminator logits", disc_logits)
        )
        if not s2_ok and first_bad_stage is None:
            first_bad_stage = 2
            print("\n>>> STAGE 2 BAD: a network forward produces NaN/inf.")
        else:
            print("\n>>> STAGE 2 OK: all network forwards are finite.")
    except Exception as e:  # noqa
        import traceback
        traceback.print_exc()
        if first_bad_stage is None:
            first_bad_stage = 2
        print(f"\n>>> STAGE 2 EXCEPTION: {e}")

    # ------------------------------------------------------ Stage 3: full update (fresh agent)
    print("\n" + "=" * 78)
    if args.num_updates > 1:
        print(f"STAGE 3 : {args.num_updates} agent.update() on a FRESH agent (instability / blow-up test on finite data)")
    else:
        print("STAGE 3 : full agent.update() on a FRESH agent (faithful first update)")
    print("=" * 78)
    try:
        fresh_agent = FBcprAuxAgent.load(str(snap / "agent"), device=dev)
        if not args.skip_rng:
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        rb = {"train": train_buf, "expert_slicer": _FixedExpertBuffer(expert_batch)}

        def _mean(v):
            return float((v if torch.is_tensor(v) else torch.as_tensor(v)).float().mean().item())

        def _bad_keys(metrics):
            out = []
            for k, v in metrics.items():
                if k.endswith("_ok"):
                    continue
                t = (v if torch.is_tensor(v) else torch.as_tensor(v)).float().mean()
                if not torch.isfinite(t).item():
                    out.append(k)
            return out

        if args.num_updates <= 1:
            metrics = fresh_agent.update(rb, step=rng["t"])
            nf = {}
            print("\n    per-metric mean + finite flag:")
            for k in sorted(metrics):
                m = _mean(metrics[k])
                fin = bool(torch.isfinite(torch.as_tensor(m).float()).item())
                if not fin:
                    nf[k] = m
                print(f"    {k:32s} = {m:+.6e}  finite={fin}")
            ok_flags = {k: _mean(metrics[k]) for k in metrics if k.endswith("_ok")}
            print(f"\n    non-finite metrics ({len(nf)}): {nf}")
            print(f"    sub-update _ok flags: {ok_flags}")
            if nf and first_bad_stage is None:
                first_bad_stage = 3
                print("\n>>> STAGE 3 BAD: first update() reproduces NaN. The _ok flags above show WHICH sub-update.")
            elif not nf:
                print("\n>>> STAGE 3 OK: first update() is FINITE here.")
        else:
            watch = ["fb_loss", "fb_bwd_grad_norm", "fb_fwd_grad_norm", "orth_loss", "Q_fb", "Q_fb_abs",
                     "B_norm", "F_norm", "disc_loss", "critic_loss", "aux_critic_loss", "actor_loss",
                     "z_norm", "target_M", "M1", "mean_disc_reward"]
            print("\n    running updates (logging watched metrics); stops at first NaN:\n")
            first_nan = None
            for u in range(args.num_updates):
                metrics = fresh_agent.update(rb, step=rng["t"])
                bad = _bad_keys(metrics)
                vals = {k: _mean(metrics[k]) for k in watch if k in metrics}
                if (u + 1) % 5 == 0 or u == 0 or bad:
                    print(f"    upd {u + 1:4d}: " + " ".join(f"{k}={vals[k]:+.3e}" for k in
                          ["fb_loss", "fb_bwd_grad_norm", "Q_fb_abs", "B_norm", "orth_loss", "disc_loss"] if k in vals))
                if bad:
                    first_nan = (u + 1, bad)
                    print(f"\n    >>> FIRST non-finite at update {u + 1}/{args.num_updates}: {bad}")
                    print(f"    >>> watched metrics at blow-up: " + ", ".join(f"{k}={vals[k]:+.3e}" for k in watch if k in vals))
                    break
            if first_nan is None:
                print(f"\n>>> STAGE 3 OK: all {args.num_updates} updates FINITE -> networks are stable on this (finite) data.")
                print(">>>           The live-training NaN therefore came from a NaN-containing BATCH (intermittent sim NaN).")
            else:
                if first_bad_stage is None:
                    first_bad_stage = 3
                print(">>> STAGE 3 BAD: networks blow up over updates on finite data -> instability (needs grad clip / loss scaling).")
    except Exception as e:  # noqa
        import traceback
        traceback.print_exc()
        if first_bad_stage is None:
            first_bad_stage = 3
        print(f"\n>>> STAGE 3 EXCEPTION: {e}")

    # ------------------------------------------------------ verdict
    print("\n" + "=" * 78)
    if first_bad_stage is None:
        print("VERDICT: no non-finite value found in any stage (data + update all finite).")
        print("         Live-training NaN may depend on later sampling / a different batch / RNG.")
    else:
        print(f"VERDICT: FIRST non-finite value appears at STAGE {first_bad_stage}.")
        hints = {
            0: "Simulator produces NaN/inf under random actions. Fix in env/reward terms (clamp, terminate-on-NaN) or the mujoco_warp backend.",
            1: "BatchRank normalizer goes non-finite -> check its running stats vs the raw obs ranges from Stage 0.",
            2: "A network forward diverges -> check input magnitudes (Stage 1) and weight init; inspect which net (expert_z/B/F/disc).",
            3: "Inputs are finite but a sub-update loss/grad explodes -> inspect the _ok flags (disc/fb/critic/aux/actor) and WGAN-GP double-backward.",
        }
        print(f"         {hints.get(first_bad_stage, '')}")
    print("=" * 78)


if __name__ == "__main__":
    main()
