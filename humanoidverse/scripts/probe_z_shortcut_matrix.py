"""Offline discriminator obs/z crossover probe.

The original 100M replay buffer was not checkpointed.  Consequently ``policy_z``
below is an explicitly labelled reconstruction of the rollout-z generator, not a
sample from the historical replay buffer.  The script never mutates the model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import (
    HumanoidVerseIsaacConfig,
    load_expert_trajectories_from_motion_lib,
)
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir


def to_device(tree, device: torch.device):
    return tree_map(lambda x: x.to(device), tree)


def take(tree, n: int):
    return tree_map(lambda x: x[:n], tree)


def repeat_rows(tree, repeats: int):
    return tree_map(lambda x: x.repeat((repeats,) + (1,) * (x.ndim - 1)), tree)


def encode_expert(model, next_obs, seq_length: int):
    """Exact non-compiled equivalent of FBcprAgent.encode_expert."""
    b = model._backward_map(next_obs).detach()
    b = b.view(b.shape[0] // seq_length, seq_length, b.shape[-1]).mean(dim=1)
    z = model.project_z(b)
    return torch.repeat_interleave(z, seq_length, dim=0)


def stats(model, obs, z):
    logits = model._discriminator.compute_logits(obs=obs, z=z).float().flatten()
    rewards = model._discriminator.compute_reward(obs=obs, z=z).float().flatten()
    q = torch.quantile(rewards, torch.tensor([0.1, 0.5, 0.9], device=rewards.device))
    return {
        "n": logits.numel(),
        "mean_logit": logits.mean().item(),
        "std_logit": logits.std(unbiased=False).item(),
        "mean_reward": rewards.mean().item(),
        "std_reward": rewards.std(unbiased=False).item(),
        "p10_reward": q[0].item(),
        "p50_reward": q[1].item(),
        "p90_reward": q[2].item(),
    }


def distribution_stats(x):
    flat = x.float().flatten()
    q = torch.quantile(flat, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=flat.device))
    return {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "p01": q[0].item(),
        "p05": q[1].item(),
        "p50": q[2].item(),
        "p95": q[3].item(),
        "p99": q[4].item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "near_zero_fraction": (flat.abs() < 1e-6).float().mean().item(),
    }


def obs_blocks(obs):
    """Named D-input blocks for this BDX checkpoint."""
    state_dim = obs["state"].shape[-1]
    privileged_dim = obs["privileged_state"].shape[-1]
    if state_dim != 34 or privileged_dim != 253:
        raise ValueError(f"unexpected D input dimensions: state={state_dim}, privileged_state={privileged_dim}")
    return {
        "dof_pos": ("state", slice(0, 14)),
        "dof_vel": ("state", slice(14, 28)),
        "projected_gravity": ("state", slice(28, 31)),
        "base_ang_vel": ("state", slice(31, 34)),
        # 17 bodies, root position removed; root height enabled, contact disabled.
        "root_height": ("privileged_state", slice(0, 1)),
        "local_body_pos": ("privileged_state", slice(1, 49)),
        "local_body_rot": ("privileged_state", slice(49, 151)),
        "local_body_vel": ("privileged_state", slice(151, 202)),
        "local_body_ang_vel": ("privileged_state", slice(202, 253)),
    }


def replace_block(base, donor, key, slc):
    out = {k: v.clone() for k, v in base.items()}
    out[key][:, slc] = donor[key][:, slc]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="results/bfmzero-bdx-full")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="results/bfmzero-bdx-full/z_shortcut_matrix_proxy.json")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("BFM_DISABLE_TORCH_COMPILE", "1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir)
    device = torch.device(args.device)
    cfg = json.loads((run_dir / "config.json").read_text())
    seq_length = cfg["agent"]["model"]["seq_length"]
    n = args.samples - args.samples % seq_length
    if n <= 0:
        raise ValueError("samples must be at least seq_length")

    model = load_model_from_checkpoint_dir(run_dir / "checkpoint", device=str(device))
    model.eval()

    env_dict = dict(cfg["env"])
    env_dict["device"] = str(device)
    env_dict["lafan_tail_path"] = str(Path("humanoidverse/data/bdx_14dof_train.pkl").resolve())
    overrides = [o for o in env_dict["hydra_overrides"] if not o.startswith("simulator.config.sim.")]
    overrides += ["env.config.max_episode_length_s=10000", "env.config.headless=True", "simulator=mujoco"]
    env_dict["hydra_overrides"] = overrides
    wrapped, _ = HumanoidVerseIsaacConfig(**env_dict).build(1)
    expert_buffer = load_expert_trajectories_from_motion_lib(
        wrapped._env, FBcprAuxAgentConfig(**cfg["agent"]), device="cpu"
    )

    # Sampling with the buffer's seq_length preserves the exact 8-frame grouping
    # required by encode_expert.
    expert_batch = expert_buffer.sample(n, seq_length=seq_length)
    expert_obs_raw = to_device(expert_batch["observation"], device)
    expert_next_raw = to_device(expert_batch["next"]["observation"], device)

    with torch.no_grad():
        # Build a policy rollout.  A fixed expert encoding is merely the rollout
        # command used to obtain policy observations; it is not used as policy_z.
        command_z = encode_expert(model, take(expert_next_raw, seq_length), seq_length)[:1]
        obs, _ = wrapped.reset(to_numpy=False)
        obs = to_device(obs, device)
        policy_rows = []
        while sum(x["state"].shape[0] for x in policy_rows) < n:
            action = model.act(obs, command_z.repeat(obs["state"].shape[0], 1), mean=False)
            obs, _, _, _, _ = wrapped.step(action, to_numpy=False)
            obs = to_device(obs, device)
            policy_rows.append(tree_map(lambda x: x.detach().clone(), obs))
        policy_obs_raw = tree_map(lambda *xs: torch.cat(xs, dim=0)[:n], *policy_rows)

        # D is trained on normalized observations.  Do not update BN statistics.
        expert_obs = model._obs_normalizer(expert_obs_raw)
        expert_next = model._obs_normalizer(expert_next_raw)
        policy_obs = model._obs_normalizer(policy_obs_raw)
        expert_z = encode_expert(model, expert_next, seq_length)

        # Historical replay was not saved. Reconstruct the late-training z-buffer
        # mixture documented in sample_mixed_z: 20% policy goals, 60% expert
        # encodings, 20% random projected z. This is the closest reproducible proxy.
        policy_goal_z = model.project_z(model._backward_map(policy_obs))
        random_z = model.sample_z(n, device=device)
        source = torch.multinomial(torch.tensor([0.2, 0.6, 0.2], device=device), n, replacement=True)
        perm = torch.randperm(n, device=device)
        policy_z = random_z.clone()
        policy_z = torch.where((source == 0)[:, None], policy_goal_z, policy_z)
        policy_z = torch.where((source == 1)[:, None], expert_z[perm], policy_z)
        zero_z = torch.zeros_like(expert_z)

        # Causal observation audit: replace one raw block, then apply the exact
        # same checkpoint normalizer. Keep z fixed to the original paired
        # expert encoding so each score isolates the changed observation block.
        blocks = obs_blocks(expert_obs_raw)
        feature_swap = {
            "expert_baseline": stats(model, expert_obs, expert_z),
            "policy_baseline_with_expert_z": stats(model, policy_obs, expert_z),
        }
        observation_audit = {}
        for block_name, (key, slc) in blocks.items():
            expert_block = expert_obs_raw[key][:, slc]
            policy_block = policy_obs_raw[key][:, slc]
            observation_audit[block_name] = {
                "expert_raw": distribution_stats(expert_block),
                "policy_raw": distribution_stats(policy_block),
                "expert_normalized": distribution_stats(expert_obs[key][:, slc]),
                "policy_normalized": distribution_stats(policy_obs[key][:, slc]),
                "standardized_mean_gap_raw": (
                    (expert_block.float().mean() - policy_block.float().mean()).abs()
                    / (0.5 * (expert_block.float().std(unbiased=False) + policy_block.float().std(unbiased=False)) + 1e-8)
                ).item(),
            }
            expert_with_policy = replace_block(expert_obs_raw, policy_obs_raw, key, slc)
            policy_with_expert = replace_block(policy_obs_raw, expert_obs_raw, key, slc)
            feature_swap[f"expert_plus_policy_{block_name}"] = stats(
                model, model._obs_normalizer(expert_with_policy), expert_z
            )
            feature_swap[f"policy_plus_expert_{block_name}"] = stats(
                model, model._obs_normalizer(policy_with_expert), expert_z
            )

        body_names = list(wrapped._env.body_names)
        expert_av = expert_obs_raw["privileged_state"][:, 202:253].view(n, 17, 3)
        policy_av = policy_obs_raw["privileged_state"][:, 202:253].view(n, 17, 3)
        angular_velocity_by_body_axis = []
        for body_idx, body_name in enumerate(body_names):
            for axis_idx, axis_name in enumerate(("x", "y", "z")):
                ev = expert_av[:, body_idx, axis_idx]
                pv = policy_av[:, body_idx, axis_idx]
                angular_velocity_by_body_axis.append({
                    "body_index": body_idx,
                    "body_name": body_name,
                    "axis": axis_name,
                    "expert_zero_fraction": (ev.abs() < 1e-6).float().mean().item(),
                    "policy_zero_fraction": (pv.abs() < 1e-6).float().mean().item(),
                    "expert_std": ev.float().std(unbiased=False).item(),
                    "policy_std": pv.float().std(unbiased=False).item(),
                    "expert_mean_abs": ev.float().abs().mean().item(),
                    "policy_mean_abs": pv.float().abs().mean().item(),
                })

        matrix = {}
        for obs_name, matrix_obs in (("expert_obs", expert_obs), ("policy_obs", policy_obs)):
            for z_name, matrix_z in (
                ("encode_expert_z", expert_z),
                ("shuffled_expert_z", expert_z[perm]),
                ("policy_goal_z", policy_goal_z),
                ("random_z", random_z),
                ("reconstructed_policy_z", policy_z),
                ("zero_z", zero_z),
            ):
                matrix[f"{obs_name}__{z_name}"] = stats(model, matrix_obs, matrix_z)

    result = {
        "checkpoint": str(run_dir / "checkpoint"),
        "seed": args.seed,
        "samples": n,
        "warning": "reconstructed_policy_z is a generator-matched proxy; the historical replay buffer was not checkpointed",
        "policy_z_source_counts": {str(i): int((source == i).sum()) for i in range(3)},
        "matrix": matrix,
        "feature_swap": feature_swap,
        "observation_audit": observation_audit,
        "body_names": body_names,
        "angular_velocity_by_body_axis": angular_velocity_by_body_axis,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
