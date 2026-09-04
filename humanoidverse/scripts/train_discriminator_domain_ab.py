"""Domain-separation ablation for the fresh discriminator.

Arms (all use the SAME expert-z for both classes, so z carries no label):

  B "domain"   : dataset-expert obs (kinematic pipeline)  vs
                 same expert motion states injected into MuJoCo (sim pipeline)
  C "behavior" : expert-sim obs (sim pipeline, positive)  vs
                 old-policy rollout obs (sim pipeline, negative)

If B reaches AUC~1 while C stays near 0.5, the training discriminator's task
was dominated by an obs-pipeline domain watermark (e.g. the sigma=2-smoothed
expert local_body_vel vs instantaneous sim velocities), not by behavior.
Channel-swap attribution on arm B replaces privileged_state blocks between
classes and reports the AUC change:

  privileged_state layout: [0:1] root_height | [1:49] local_body_pos |
    [49:151] local_body_rot | [151:202] local_body_vel | [202:253] local_body_ang_vel
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import (
    HumanoidVerseIsaacConfig,
    load_expert_trajectories_from_motion_lib,
)
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir

PRIV_SLICES = {
    "root_height": (0, 1),
    "local_body_pos": (1, 49),
    "local_body_rot": (49, 151),
    "local_body_vel": (151, 202),
    "local_body_ang_vel": (202, 253),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(tree, device):
    return tree_map(lambda x: x.to(device), tree)


def encode_expert(model, next_obs, seq_length):
    b = model._backward_map(next_obs).detach()
    b = b.view(b.shape[0] // seq_length, seq_length, b.shape[-1]).mean(1)
    return torch.repeat_interleave(model.project_z(b), seq_length, dim=0)


def reset_module(module):
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()


def build_env(cfg, device, disable_obs_noise):
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["device"] = str(device)
    env_cfg["lafan_tail_path"] = str(Path("humanoidverse/data/bdx_14dof_train.pkl").resolve())
    env_cfg["disable_obs_noise"] = bool(disable_obs_noise)
    overrides = [o for o in env_cfg["hydra_overrides"] if not o.startswith("simulator.config.sim.")]
    overrides += [
        "env.config.max_episode_length_s=10000",
        "env.config.headless=True",
        "simulator=mujoco",
        "robot.motion.smooth_angular_velocity=false",
    ]
    env_cfg["hydra_overrides"] = overrides
    # NOTE: num_envs > 1 hits a buffer-shape init bug in this repo's mujoco path
    # (gravity_vec [N,3] vs base_quat [1,4]); the prior fresh-D script also uses 1.
    wrapped, _ = HumanoidVerseIsaacConfig(**env_cfg).build(1)
    return wrapped


def sample_chunks(buffer, n_chunks, seq, seed):
    rng = np.random.default_rng(seed)
    chunks = []
    for _ in range(n_chunks):
        e = int(rng.integers(0, len(buffer.lengths)))
        length = int(buffer.lengths[e])
        offset = int(rng.integers(0, length - seq - 1))
        chunks.append((e, offset))
    return chunks


def inject_and_read(wrapped, ids, times):
    """Inject per-env (motion, time) states, step once, return obs at those states."""
    inner = wrapped._env
    res = inner._motion_lib.get_motion_state(ids, times)
    n_env = ids.shape[0]
    dof_states = torch.zeros(n_env, inner.num_dof, 2, device=inner.device)
    root_states = torch.zeros(n_env, 13, device=inner.device)
    dof_states[..., 0] = res["dof_pos"]
    dof_states[..., 1] = res["dof_vel"]
    root_states[:, :3] = res["root_pos"]
    root_states[:, 3:7] = res["root_rot"]
    root_states[:, 7:10] = res["root_vel"]
    root_states[:, 10:13] = res["root_ang_vel"]
    env_ids = torch.arange(n_env, device=inner.device)
    inner.reset_envs_idx(env_ids, target_states={
        "dof_states": dof_states, "root_states": root_states,
    })
    action = torch.zeros(n_env, inner.num_dof, device=inner.device)
    obs, _, _, _, _ = wrapped.step(action, to_numpy=False)
    return obs


def collect_expert_sim(wrapped, chunks, seq, device):
    """Return chunk-major sim obs rows: chunk c, frame k -> row c*seq + k.

    Single-env env: each chunk is injected frame by frame (seq resets + steps).
    """
    inner = wrapped._env
    dt = inner.dt
    rows = []
    for e, off in chunks:
        for k in range(seq):
            ids = torch.tensor([e], dtype=torch.long, device=inner.device)
            times = torch.tensor([(off + k) * dt], dtype=torch.float32, device=inner.device)
            obs = inject_and_read(wrapped, ids, times)
            rows.append(to_device(obs, device))
    return tree_map(lambda *xs: torch.cat([x for x in xs], 0), *rows)


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


def swap_priv_block(obs, other, name):
    lo, hi = PRIV_SLICES[name]
    out = dict(obs)
    out["privileged_state"] = torch.cat(
        [obs["privileged_state"][:, :lo], other["privileged_state"][:, lo:hi],
         obs["privileged_state"][:, hi:]], dim=1
    )
    return out


def collect_policy(model, wrapped, command_z, n, device):
    obs, _ = wrapped.reset(to_numpy=False)
    obs = to_device(obs, device)
    rows = []
    while len(rows) < n:
        with torch.no_grad():
            action = model.act(obs, command_z, mean=False)
        obs, _, _, _, _ = wrapped.step(action, to_numpy=False)
        obs = to_device(obs, device)
        rows.append(tree_map(lambda x: x.detach().clone(), obs))
    return tree_map(lambda *xs: torch.cat(xs, 0)[:n], *rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/bfmzero-bdx-full")
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--disable-obs-noise", action="store_true")
    ap.add_argument("--output", default="results/bfmzero-bdx-full/discriminator_domain_ab.json")
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("BFM_DISABLE_TORCH_COMPILE", "1")
    device = torch.device(args.device)
    run = Path(args.run_dir)
    cfg = json.loads((run / "config.json").read_text())
    seq = cfg["agent"]["model"]["seq_length"]
    n = args.samples - args.samples % seq
    model = load_model_from_checkpoint_dir(run / "checkpoint", device=str(device)); model.eval()
    agent_cfg = FBcprAuxAgentConfig(**cfg["agent"])

    wrapped = build_env(cfg, device, args.disable_obs_noise)
    inner = wrapped._env
    inner._motion_lib.mesh_parsers.cfg["smooth_angular_velocity"] = False
    buffer = load_expert_trajectories_from_motion_lib(inner, agent_cfg, device="cpu", base_ang_vel_frame="body")

    wrapped.reset(to_numpy=False)
    chunks = sample_chunks(buffer, n // seq, seq, args.seed + 1)
    sim_obs = collect_expert_sim(wrapped, chunks, seq, device)

    # expert buffer rows aligned with the injected frames (chunk-major)
    idxs = []
    for e, off in chunks:
        s = int(buffer.start_idx[e, 0])
        idxs.extend(range(s + off, s + off + seq))
    idx = torch.tensor(idxs, dtype=torch.long)
    expert_obs = tree_map(lambda x: x[idx].to(device), buffer.storage["observation"])
    expert_next = tree_map(lambda x: x[idx + 1].to(device), buffer.storage["observation"])

    assert "state" in sim_obs and "privileged_state" in sim_obs, list(sim_obs.keys())
    model._obs_normalizer.eval()
    with torch.no_grad():
        norm = lambda d: model._obs_normalizer(d)
        expert_norm = norm(expert_obs)
        expert_next_norm = norm(expert_next)
        sim_norm = norm(sim_obs)
        z = encode_expert(model, expert_next_norm, seq)

    # pairing sanity: local_body_pos is unsmoothed on both pipelines -> must match
    lo, hi = PRIV_SLICES["local_body_pos"]
    diff_norm = (expert_norm["privileged_state"][:, lo:hi]
                 - sim_norm["privileged_state"][:, lo:hi]).abs().flatten()
    diff_raw = (expert_obs["privileged_state"][:, lo:hi]
                - sim_obs["privileged_state"][:, lo:hi]).abs().flatten()
    pairing = {
        "norm_mean": diff_norm.mean().item(),
        "norm_p50": diff_norm.median().item(),
        "norm_p95": torch.quantile(diff_norm, 0.95).item(),
        "norm_frac_gt_0.5": (diff_norm > 0.5).float().mean().item(),
        "raw_mean": diff_raw.mean().item(),
        "raw_p50": diff_raw.median().item(),
        "raw_p95": torch.quantile(diff_raw, 0.95).item(),
    }

    seed_all(args.seed + 2)
    command_z = z[:1]
    policy_raw = collect_policy(model, wrapped, command_z, n, device)
    policy_obs = model._obs_normalizer(policy_raw)

    keys = ("state", "privileged_state")
    B_pos = {k: expert_norm[k] for k in keys}
    B_neg = {k: sim_norm[k] for k in keys}
    C_pos = {k: sim_norm[k] for k in keys}
    C_neg = {k: policy_obs[k] for k in keys}

    checks = [0, 1, 10, 50, 100, 250, 500, 1000, args.steps]
    disc_B, hist_B = train_fresh_d(model._discriminator, B_pos, z, B_neg,
                                   args.steps, args.batch, 1e-5, args.seed + 3, checks)
    _, hist_C = train_fresh_d(model._discriminator, C_pos, z, C_neg,
                              args.steps, args.batch, 1e-5, args.seed + 3, checks)

    attribution = {}
    # swap-all control: identical privileged_state on both sides must collapse to AUC 0.5,
    # validating that any remaining separation comes only from the state channel.
    attribution["sim_gets_expert_ALL"] = metrics(
        disc_B, B_pos, z, {**B_neg, "privileged_state": B_pos["privileged_state"]})
    for name in PRIV_SLICES:
        attribution[f"sim_gets_expert_{name}"] = metrics(
            disc_B, B_pos, z, swap_priv_block(B_neg, B_pos, name))
    for name in PRIV_SLICES:
        attribution[f"expert_gets_sim_{name}"] = metrics(
            disc_B, swap_priv_block(B_pos, B_neg, name), z, B_neg)

    result = {
        "samples": n,
        "steps": args.steps,
        "disable_obs_noise": bool(args.disable_obs_noise),
        "pairing_local_body_pos": pairing,
        "arm_B_domain": hist_B,
        "arm_C_behavior": hist_C,
        "arm_B_channel_attribution": attribution,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
