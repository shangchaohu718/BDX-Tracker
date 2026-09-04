"""Gate-1 verification for the expert-sim pool (no RL training).

Arms (fresh discriminator, same expert-z for both classes):

  self_consistency : pool rows vs simulator obs regenerated at the SAME frames
                     (both through the warp sim pipeline) -> expect AUC ~ 0.5.
                     AUC > 0.5 means the pool generator diverges from the
                     reference injection path (a second domain).
  domain_B         : kinematic expert rows vs sim-regenerated rows at the same
                     frames (the original domain test, warp path) -> expect
                     AUC ~ 1.0; together with self_consistency this proves the
                     pool genuinely lives in the sim domain.
  purity           : local_body_pos (noise-free channel) raw diff between pool
                     rows and regenerated rows -> catches frame misalignment /
                     contaminated injections (should be ~1e-3, not O(0.1+)).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import copy
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("BFM_ZERO_HARD_LIMIT_RESET", "0")
os.environ.setdefault("BFM_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import (
    load_expert_trajectories_from_motion_lib,
)
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.scripts.generate_expert_sim_pool import (
    build_warp_env, build_cpu_env, inject_round, inject_read_exact_lean, OBS_KEYS,
)

PRIV_VEL = (151, 202)
PRIV_POS = (1, 49)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_expert(model, next_obs, seq_length):
    b = model._backward_map(next_obs).detach()
    b = b.view(b.shape[0] // seq_length, seq_length, b.shape[-1]).mean(1)
    return torch.repeat_interleave(model.project_z(b), seq_length, dim=0)


def reset_module(module):
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()


def metrics(disc, obs_pos, z, obs_neg):
    with torch.no_grad():
        e = disc.compute_logits(obs_pos, z).float().flatten()
        p = disc.compute_logits(obs_neg, z).float().flatten()
    labels = np.concatenate([np.ones(e.numel()), np.zeros(p.numel())])
    scores = np.concatenate([e.cpu().numpy(), p.cpu().numpy()])
    return {
        "pos_logit": e.mean().item(),
        "neg_logit": p.mean().item(),
        "margin": (e.mean() - p.mean()).item(),
        "accuracy_at_zero": 0.5 * ((e > 0).float().mean() + (p < 0).float().mean()).item(),
        "auc": float(roc_auc_score(labels, scores)),
    }


def train_fresh_d(template, obs_pos, z, obs_neg, steps, batch, lr, seed, checkpoints):
    seed_all(seed)
    disc = copy.deepcopy(template)
    reset_module(disc)
    disc.requires_grad_(True)
    opt = torch.optim.Adam(disc.parameters(), lr=lr)
    n = z.shape[0]
    history = []
    checkpoints = sorted(set(checkpoints) | {0, steps})
    for step in range(steps + 1):
        if step in checkpoints:
            history.append({"step": step, **metrics(disc, obs_pos, z, obs_neg)})
        if step == steps:
            break
        idx = torch.randint(n, (batch,), device=z.device)
        pos = tree_map(lambda x: x[idx], obs_pos)
        neg = tree_map(lambda x: x[idx], obs_neg)
        el = disc.compute_logits(pos, z[idx])
        pl = disc.compute_logits(neg, z[idx])
        loss = (-F.logsigmoid(el) + F.softplus(pl)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return disc, history


def regenerate(wrapped, frames, num_envs, settle_reps=2, backend="warp"):
    """frames: list of (motion_id, time). Returns obs dict rows [N, ...] on cpu."""
    inner = wrapped._env
    out = []
    if backend == "cpu":
        for f in frames:
            obs = inject_read_exact_lean(wrapped, torch.tensor([f[0]], device=inner.device),
                                         torch.tensor([f[1]], dtype=torch.float32, device=inner.device))
            out.append({k: v[:1] for k, v in obs.items()})
        return tree_map(lambda *xs: torch.cat([x for x in xs], 0)[: len(frames)], *out)
    for i0 in range(0, len(frames), num_envs):
        batch = frames[i0:i0 + num_envs]
        while len(batch) < num_envs:
            batch.append(batch[0])
        ids = torch.tensor([f[0] for f in batch], dtype=torch.long, device=inner.device)
        times = torch.tensor([f[1] for f in batch], dtype=torch.float32, device=inner.device)
        obs, _ = inject_round(wrapped, ids, times, settle_reps=settle_reps)
        out.append({k: v[: len(batch)] for k, v in obs.items()})
    return tree_map(lambda *xs: torch.cat([x for x in xs], 0)[: len(frames)], *out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="humanoidverse/data/bdx_expert_sim_pool.pt")
    ap.add_argument("--model-dir", default="results/bfmzero-bdx-full")
    ap.add_argument("--train-config", default="results/bfm-dynamics-baseline/config.json")
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--backend", choices=["warp", "cpu"], default="warp")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--output", default="results/bfmzero-bdx-full/expert_sim_pool_gate1.json")
    args = ap.parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    seq = 8

    payload = torch.load(args.pool, map_location="cpu")
    episodes, meta = payload["episodes"], payload["meta"]
    dt = float(payload["stats"]["dt"])
    print(f"pool: {len(episodes)} episodes, {sum(m['length'] for m in meta)} frames, dt={dt}", flush=True)

    cfg = json.loads(Path(args.train_config).read_text())
    wrapped = (
        build_cpu_env(cfg, device) if args.backend == "cpu"
        else build_warp_env(cfg, device, args.num_envs)
    )
    inner = wrapped._env
    agent_cfg = FBcprAuxAgentConfig(**cfg["agent"])
    kbuf = load_expert_trajectories_from_motion_lib(inner, agent_cfg, device="cpu", base_ang_vel_frame="body")

    model = load_model_from_checkpoint_dir(Path(args.model_dir) / "checkpoint", device="cuda")
    model.eval()
    model._obs_normalizer.eval()

    n = args.samples - args.samples % seq
    rng = np.random.default_rng(args.seed + 1)

    # ---- sample windows from pool episodes ----
    windows = []  # (episode_idx, w)
    while len(windows) < n // seq:
        e = int(rng.integers(0, len(episodes)))
        T = episodes[e]["observation"]["state"].shape[0]
        w = int(rng.integers(0, T - seq))  # need rows w..w+8 (obs + next)
        windows.append((e, w))
    pool_obs = {k: torch.cat([episodes[e]["observation"][k][w:w + seq] for e, w in windows]) for k in OBS_KEYS}
    pool_next = {k: torch.cat([episodes[e]["observation"][k][w + 1:w + 1 + seq] for e, w in windows]) for k in OBS_KEYS}
    reg_frames = [(meta[e]["motion_id"], meta[e]["t0"] + (w + k) * dt) for e, w in windows for k in range(seq)]

    wrapped.reset(to_numpy=False)
    regen1 = regenerate(wrapped, reg_frames, args.num_envs, backend=args.backend)
    regen2 = regenerate(wrapped, reg_frames, args.num_envs, backend=args.backend)  # fresh pass: holdout negatives
    # purity: noise-free local_body_pos raw diff, same frames
    lo, hi = PRIV_POS
    purity = (pool_obs["privileged_state"][:, lo:hi] - regen1["privileged_state"][:, lo:hi]).abs().flatten()
    purity_stats = {
        "mean": purity.mean().item(), "p50": purity.median().item(),
        "p95": torch.quantile(purity, 0.95).item(), "frac_gt_0.1": (purity > 0.1).float().mean().item(),
    }

    # ---- domain_B arm: frame-aligned kinematic rows vs regenerated rows (warp) ----
    traj = rng.integers(0, len(kbuf.lengths), size=n // seq)
    offsets = []
    for t in traj:
        length = int(kbuf.lengths[t])
        offsets.append(int(rng.integers(0, length - seq - 1)))
    kin_frames, regB_frames = [], []
    for t, off in zip(traj, offsets):
        s = int(kbuf.start_idx[t, 0])
        kin_frames.extend(range(s + off, s + off + seq))
        m_id = int(kbuf.storage["motion_id"][s])
        regB_frames.extend([(m_id, (off + k) * dt) for k in range(seq)])
    idx = torch.tensor(kin_frames, dtype=torch.long)
    kin_obs = {k: kbuf.storage["observation"][k][idx].float() for k in OBS_KEYS}
    regB = regenerate(wrapped, regB_frames, args.num_envs, backend=args.backend)

    # ---- normalize + z (checkpoint normalizer; z from pool next obs) ----
    with torch.no_grad():
        norm = lambda d: model._obs_normalizer(tree_map(lambda x: x.to(device), d))
        pool_n, regen1_n, regen2_n = norm(pool_obs), norm(regen1), norm(regen2)
        kin_n, regB_n = norm(kin_obs), norm(regB)
        z = encode_expert(model, tree_map(lambda x: x.to(device), pool_next), seq)

    keys = ("state", "privileged_state")
    to = lambda d, dev: {k: d[k].to(dev) for k in keys}
    checks = [0, 1, 10, 50, 100, 250, 500, 1000, args.steps]
    n_train = n - n // 4  # first 3/4 of windows train, disjoint last 1/4 holdout
    sl = slice(0, n_train)
    hs = slice(n_train, n)
    disc, hist_self = train_fresh_d(
        model._discriminator,
        to(tree_map(lambda x: x[sl], pool_n), device), z[sl],
        to(tree_map(lambda x: x[sl], regen1_n), device),
        args.steps, args.batch, 1e-5, args.seed + 3, checks)
    # holdout on DISJOINT frames: positives = pool rows never trained on,
    # negatives = fresh regeneration of those same unseen frames. Memorization
    # of trained rows cannot help here; a generalizable pass-level difference
    # (a second domain) keeps this AUC high.
    holdout = metrics(
        disc, to(tree_map(lambda x: x[hs], pool_n), device), z[hs],
        to(tree_map(lambda x: x[hs], regen2_n), device))
    # channel attribution on the disjoint holdout: which obs key carries the
    # generalizable pass signature?
    pos_h = to(tree_map(lambda x: x[hs], pool_n), device)
    neg_h = to(tree_map(lambda x: x[hs], regen2_n), device)
    holdout_state_swapped = metrics(
        disc, {**pos_h, "state": neg_h["state"]}, z[hs],
        {**neg_h, "state": pos_h["state"]})
    holdout_priv_swapped = metrics(
        disc, {**pos_h, "privileged_state": neg_h["privileged_state"]}, z[hs],
        {**neg_h, "privileged_state": pos_h["privileged_state"]})
    # control: two independent regen passes against each other (both from the
    # verify-time env). If these are also separable, ANY two generation passes
    # differ; if ~0.5, the signature is specific to the pool-generation pass.
    disc_r, hist_rr = train_fresh_d(
        model._discriminator,
        to(tree_map(lambda x: x[sl], regen1_n), device), z[sl],
        to(tree_map(lambda x: x[sl], regen2_n), device),
        args.steps, args.batch, 1e-5, args.seed + 3, checks)
    control_rr_holdout = metrics(
        disc_r, to(tree_map(lambda x: x[hs], regen1_n), device), z[hs],
        to(tree_map(lambda x: x[hs], regen2_n), device))
    result = {
        "pool": str(args.pool), "samples": n, "train_rows": n_train, "steps": args.steps,
        "purity_local_body_pos_raw": purity_stats,
        "self_consistency_train": hist_self,
        "self_consistency_holdout_disjoint_fresh": holdout,
        "holdout_state_swapped": holdout_state_swapped,
        "holdout_priv_swapped": holdout_priv_swapped,
        "control_regen1_vs_regen2_train_final": hist_rr[-1],
        "control_regen1_vs_regen2_holdout": control_rr_holdout,
        "domain_B_warp": train_fresh_d(
            model._discriminator, to(regB_n, device), z, to(kin_n, device),
            args.steps, args.batch, 1e-5, args.seed + 3, checks)[1],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
