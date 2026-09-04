"""Fresh-discriminator 2x2 ablation for expert angular-velocity semantics.

Factors are motion-library angular smoothing (smooth/instant) and root angular
velocity frame (world/body). Samples, policy rollout, frozen networks,
discriminator initialization, minibatch order and optimizer are held fixed.
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
from torch import nn
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import (
    HumanoidVerseIsaacConfig,
    load_expert_trajectories_from_motion_lib,
)
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir


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


def build_env(cfg, smooth, device):
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["device"] = str(device)
    env_cfg["lafan_tail_path"] = str(Path("humanoidverse/data/bdx_14dof_train.pkl").resolve())
    overrides = [o for o in env_cfg["hydra_overrides"] if not o.startswith("simulator.config.sim.")]
    overrides += [
        "env.config.max_episode_length_s=10000",
        "env.config.headless=True",
        "simulator=mujoco",
        f"robot.motion.smooth_angular_velocity={'true' if smooth else 'false'}",
    ]
    env_cfg["hydra_overrides"] = overrides
    wrapped, _ = HumanoidVerseIsaacConfig(**env_cfg).build(1)
    actual = bool(wrapped._env.config.robot.motion.get("smooth_angular_velocity", True))
    if actual != smooth:
        raise RuntimeError(f"runtime smooth_angular_velocity={actual}, expected {smooth}")
    return wrapped


def paired_expert_indices(buffer, n, seq_length, seed):
    rng = np.random.default_rng(seed)
    num_slices = n // seq_length
    traj = rng.integers(0, len(buffer.lengths), size=num_slices)
    idxs = []
    for t in traj:
        start = int(buffer.start_idx[t, 0])
        length = int(buffer.lengths[t])
        offset = int(rng.integers(0, length - seq_length - 1))
        idxs.extend(range(start + offset, start + offset + seq_length))
    return torch.tensor(idxs, dtype=torch.long, device=buffer.storage["observation"]["state"].device)


def expert_at_indices(buffer, idx):
    return {
        "observation": tree_map(lambda x: x[idx], buffer.storage["observation"]),
        "next": {"observation": tree_map(lambda x: x[idx + 1], buffer.storage["observation"])},
    }


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


def metrics(disc, expert_obs, expert_z, policy_obs, policy_z):
    with torch.no_grad():
        e = disc.compute_logits(expert_obs, expert_z).float().flatten()
        p = disc.compute_logits(policy_obs, policy_z).float().flatten()
    labels = np.concatenate([np.ones(e.numel()), np.zeros(p.numel())])
    scores = np.concatenate([e.cpu().numpy(), p.cpu().numpy()])
    return {
        "expert_logit": e.mean().item(),
        "policy_logit": p.mean().item(),
        "margin": (e.mean() - p.mean()).item(),
        "accuracy_at_zero": 0.5 * ((e > 0).float().mean() + (p < 0).float().mean()).item(),
        "auc": float(roc_auc_score(labels, scores)),
    }


def train_full_d(template, expert_obs, expert_z, policy_obs, policy_z, steps, batch, lr, seed, checkpoints):
    seed_all(seed)
    disc = copy.deepcopy(template)
    reset_module(disc)
    disc.requires_grad_(True)
    opt = torch.optim.Adam(disc.parameters(), lr=lr)
    n = expert_z.shape[0]
    history = []
    checkpoints = set(checkpoints) | {0, steps}
    for step in range(steps + 1):
        if step in checkpoints:
            row = {"step": step, **metrics(disc, expert_obs, expert_z, policy_obs, policy_z)}
            history.append(row)
        if step == steps:
            break
        # Seed and call order are identical in both A/B runs.
        ei = torch.randint(n, (batch,), device=expert_z.device)
        pi = torch.randint(n, (batch,), device=expert_z.device)
        eo = tree_map(lambda x: x[ei], expert_obs)
        po = tree_map(lambda x: x[pi], policy_obs)
        el = disc.compute_logits(eo, expert_z[ei])
        pl = disc.compute_logits(po, policy_z[pi])
        loss = (-F.logsigmoid(el) + F.softplus(pl)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step + 1 in checkpoints:
            history[-1 if history and history[-1]["step"] == step + 1 else len(history):]
    return history


class AngularClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(51, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).flatten()


def train_angular(expert, policy, steps, batch, lr, seed, checkpoints):
    seed_all(seed)
    net = AngularClassifier().to(expert.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = expert.shape[0]
    out = []
    checkpoints = set(checkpoints) | {0, steps}
    for step in range(steps + 1):
        if step in checkpoints:
            with torch.no_grad():
                e = net(expert).float(); p = net(policy).float()
                labels = np.concatenate([np.ones(n), np.zeros(n)])
                scores = np.concatenate([e.cpu().numpy(), p.cpu().numpy()])
                out.append({
                    "step": step,
                    "expert_logit": e.mean().item(),
                    "policy_logit": p.mean().item(),
                    "margin": (e.mean() - p.mean()).item(),
                    "accuracy_at_zero": 0.5 * ((e > 0).float().mean() + (p < 0).float().mean()).item(),
                    "auc": float(roc_auc_score(labels, scores)),
                })
        if step == steps:
            break
        ei = torch.randint(n, (batch,), device=expert.device)
        pi = torch.randint(n, (batch,), device=expert.device)
        el = net(expert[ei]); pl = net(policy[pi])
        loss = (-F.logsigmoid(el) + F.softplus(pl)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/bfmzero-bdx-full")
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--output", default="results/bfmzero-bdx-full/discriminator_ab.json")
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

    # Build all four expert pipelines. The loader's world-frame option is a
    # diagnostic-only reproduction of the legacy bug; production defaults body.
    env_new = build_env(cfg, smooth=False, device=device)
    motion_lib = env_new._env._motion_lib
    motion_lib.mesh_parsers.cfg["smooth_angular_velocity"] = False
    buf_instant_body = load_expert_trajectories_from_motion_lib(
        env_new._env, agent_cfg, device="cpu", base_ang_vel_frame="body"
    )
    if bool(motion_lib.mesh_parsers.cfg.get("smooth_angular_velocity", True)):
        raise RuntimeError("fixed expert runtime config did not resolve to false")
    # HumanoidVerse is process-global and forbids constructing a second config.
    # Recompute the same motion library with only the FK smoothing flag changed.
    motion_lib.mesh_parsers.cfg["smooth_angular_velocity"] = True
    motion_lib.all_motions_loaded = False
    buf_smooth_body = load_expert_trajectories_from_motion_lib(
        env_new._env, agent_cfg, device="cpu", base_ang_vel_frame="body"
    )
    buf_smooth_world = load_expert_trajectories_from_motion_lib(
        env_new._env, agent_cfg, device="cpu", base_ang_vel_frame="world"
    )
    if not bool(motion_lib.mesh_parsers.cfg.get("smooth_angular_velocity", False)):
        raise RuntimeError("legacy expert runtime config did not resolve to true")
    # Restore the fixed setting before collecting policy observations.
    motion_lib.mesh_parsers.cfg["smooth_angular_velocity"] = False
    motion_lib.all_motions_loaded = False
    buf_instant_world = load_expert_trajectories_from_motion_lib(
        env_new._env, agent_cfg, device="cpu", base_ang_vel_frame="world"
    )
    paired_idx = paired_expert_indices(buf_instant_body, n, seq, args.seed + 1)
    buffers = {
        "v0_smooth_world": buf_smooth_world,
        "v1_instant_world": buf_instant_world,
        "v2_smooth_body": buf_smooth_body,
        "v3_instant_body": buf_instant_body,
    }
    batches = {k: expert_at_indices(v, paired_idx) for k, v in buffers.items()}
    raw = {k: to_device(v["observation"], device) for k, v in batches.items()}
    next_raw = {k: to_device(v["next"]["observation"], device) for k, v in batches.items()}
    new_raw = raw["v3_instant_body"]
    old_raw = raw["v0_smooth_world"]
    # State and all privileged blocks before angular velocity must be bit-identical.
    # state[-3:] is base_ang_vel and intentionally comes from the same filtered
    # body_ang_vel_t root channel. All other state features must match exactly.
    if not torch.equal(new_raw["state"][:, :31], old_raw["state"][:, :31]):
        raise RuntimeError("A/B non-angular expert state samples are not identical")
    if not torch.equal(new_raw["privileged_state"][:, :202], old_raw["privileged_state"][:, :202]):
        raise RuntimeError("A/B non-angular privileged samples are not identical")
    angular_max_delta = (
        raw["v3_instant_body"]["privileged_state"][:, 202:253]
        - raw["v2_smooth_body"]["privileged_state"][:, 202:253]
    ).abs().max().item()
    if angular_max_delta <= 1e-6:
        raise RuntimeError("A/B angular samples are identical; legacy motions were not recomputed")

    with torch.no_grad():
        observations = {k: model._obs_normalizer(v) for k, v in raw.items()}
        next_observations = {k: model._obs_normalizer(v) for k, v in next_raw.items()}
        zs = {k: encode_expert(model, next_observations[k], seq) for k in observations}
        command = zs["v3_instant_body"][:1]
        seed_all(args.seed + 2)
        policy_raw = collect_policy(model, env_new, command, n, device)
        policy_obs = model._obs_normalizer(policy_raw)
        policy_z = model.project_z(model._backward_map(policy_obs))

    checks = [0, 1, 10, 50, 100, 250, 500, 1000, args.steps]
    result = {
        "samples": n,
        "steps": args.steps,
        "batch": args.batch,
        "factors": {
            "v0_smooth_world": {"smooth": True, "base_frame": "world"},
            "v1_instant_world": {"smooth": False, "base_frame": "world"},
            "v2_smooth_body": {"smooth": True, "base_frame": "body"},
            "v3_instant_body": {"smooth": False, "base_frame": "body"},
        },
        "expert_angular_max_ab_delta": angular_max_delta,
        "full_discriminator": {
            k: train_full_d(model._discriminator, observations[k], zs[k], policy_obs, policy_z, args.steps, args.batch, 1e-5, args.seed + 3, checks)
            for k in observations
        },
        "angular_only": {
            "smooth": train_angular(raw["v2_smooth_body"]["privileged_state"][:, 202:253], policy_raw["privileged_state"][:, 202:253], args.steps, args.batch, 3e-4, args.seed + 4, checks),
            "instant": train_angular(raw["v3_instant_body"]["privileged_state"][:, 202:253], policy_raw["privileged_state"][:, 202:253], args.steps, args.batch, 3e-4, args.seed + 4, checks),
        },
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
