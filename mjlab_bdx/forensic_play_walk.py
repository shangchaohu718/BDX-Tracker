"""Forensic: does the mixed134 policy actually WALK on the stock play path?

Replicates pool_play.py's exact loading (play cfg, runner.load, inference
policy, motion pin via _assign_motions monkeypatch) but WITHOUT viser —
pure step loop, measuring:
  - policy root xy travel + leg joint excursion (does the robot walk?)
  - reference root xy travel (what it was supposed to do)
Adjudicates: viewer-display bug vs policy/eval-chain P0.

Usage (from bdx_rl_mjlab root):
  .venv/bin/python forensic_play_walk.py <task> <checkpoint> <motion_idx> \
      <n_steps> <out_json>
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

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
    names = [Path(str(f)).name for f in mc.cfg.motion_files]
    print(f"pinning {pin} = {names[pin]}")
    mc.set_motion_index(pin)

    def _pin_assign(env_ids, _mc=mc, _pin=pin):
        _mc.motion_indices[env_ids] = _pin

    mc._assign_motions = _pin_assign

    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]

    def snap() -> dict:
        root = robot.data.root_link_pos_w[0].cpu().numpy()
        q = robot.data.joint_pos[0].cpu().numpy()
        cmd_j = mc.joint_pos[0, :3].cpu().numpy()
        return {
            "root_xy": [round(float(root[0]), 3), round(float(root[1]), 3)],
            "root_z": round(float(root[2]), 3),
            "q_legs": [round(float(x), 3) for x in q[:10]],
            "cmd_hipL": round(float(cmd_j[0]), 3),
            "ref_t": int(mc.time_steps[0]),
        }

    trace = [snap()]
    ref0 = None
    with torch.inference_mode():
        for t in range(n_steps):
            obs, _, _, _ = env.step(policy(obs))
            if ref0 is None:
                ref0 = mc.motion.body_pos_w[0, 0, :2].cpu().numpy() \
                    if hasattr(mc, "motion") else None
            if t % 25 == 0 or t == n_steps - 1:
                s = snap()
                s["t"] = t
                trace.append(s)

    ref_end = mc.motion.body_pos_w[0, 0, :2].cpu().numpy() \
        if hasattr(mc, "motion") else None
    starts = np.array([t["root_xy"] for t in trace])
    travel = float(np.linalg.norm(starts[-1] - starts[0]))
    q = np.array([t["q_legs"] for t in trace])
    excursion = float((q.max(axis=0) - q.min(axis=0)).max())
    ref_travel = float(np.linalg.norm(ref_end - ref0)) if ref0 is not None else None
    verdict = "WALKS" if travel > 0.5 else "STANDS"
    result = {
        "task": task, "checkpoint": ckpt, "motion": names[pin],
        "steps": n_steps, "policy_root_travel_m": round(travel, 3),
        "policy_leg_excursion_rad": round(excursion, 3),
        "ref_travel_m_at_final_frame": round(ref_travel, 3)
        if ref_travel is not None else None,
        "verdict": verdict, "trace": trace,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "trace"},
                     indent=2))
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
