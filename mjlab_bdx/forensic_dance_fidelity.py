"""Forensic: does the policy actually PERFORM a stationary-family motion?

Property test (not proxy): per-joint movement amplitude ratio
(robot_excursion / reference_excursion) + trajectory correlation, plus
mean joint error — for a pinned clip on the stock play path.

Usage (from bdx_rl_mjlab root):
  .venv/bin/python forensic_dance_fidelity.py <task> <checkpoint> \
      <motion_idx> <n_steps> <out_json>
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict

import numpy as np
import torch


def main():
    task, ckpt, pin, n_steps, out = (
        sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
        sys.argv[5],
    )
    device = "cpu"
    import tracking_bdx.config.bdx_v4  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    agent_cfg = load_rl_cfg(task)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task)
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    mc = env.unwrapped.command_manager.get_term("motion")
    names = [str(f).split("/")[-1] for f in mc.cfg.motion_files]
    print(f"pinning {pin} = {names[pin]}")
    mc.set_motion_index(pin)

    def _pin_assign(env_ids, _mc=mc, _pin=pin):
        _mc.motion_indices[env_ids] = _pin

    mc._assign_motions = _pin_assign
    obs, _ = env.reset()

    qs, cmds = [], []
    with torch.inference_mode():
        for _ in range(n_steps):
            qs.append(mc.robot.data.joint_pos[0].cpu().numpy().copy())
            cmds.append(mc.joint_pos[0].cpu().numpy().copy())
            obs, _, _, _ = env.step(policy(obs))

    q = np.array(qs)
    c = np.array(cmds)
    skip = 25  # drop birth transient
    q, c = q[skip:], c[skip:]
    ref_amp = c.max(0) - c.min(0)
    rob_amp = q.max(0) - q.min(0)
    ratio = rob_amp / np.maximum(ref_amp, 1e-6)
    cors = []
    for j in range(q.shape[1]):
        a, b = q[:, j] - q[:, j].mean(), c[:, j] - c[:, j].mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        cors.append(float((a * b).sum() / d) if d > 1e-9 else float("nan"))
    jerr = float(np.abs(q[-50:] - c[-50:]).mean())
    moving = ref_amp > 0.15
    result = {
        "task": task, "checkpoint": ckpt, "motion": names[pin],
        "steps": n_steps, "transient_skip": skip,
        "ref_amp_per_joint": np.round(ref_amp, 3).tolist(),
        "robot_amp_per_joint": np.round(rob_amp, 3).tolist(),
        "amp_ratio_per_joint": np.round(ratio, 3).tolist(),
        "corr_per_joint": [round(x, 3) for x in cors],
        "jerr_last50": round(jerr, 4),
        "mean_amp_ratio_on_moving_joints": round(float(ratio[moving].mean()), 3)
        if moving.any() else None,
        "mean_corr_on_moving_joints": round(
            float(np.nanmean(np.array(cors)[moving])), 3)
        if moving.any() else None,
        "n_moving_joints": int(moving.sum()),
    }
    print(json.dumps(result, indent=2))
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
