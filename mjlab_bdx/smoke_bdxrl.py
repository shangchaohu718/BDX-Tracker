"""Smoke for the auto-registered ss4 gesture task inside bdx_rl_mjlab.

Run from repo root of bdx_rl_mjlab with ITS venv:
  .venv/bin/python /home/tcl/Desktop/start/BFM-zero/mjlab_bdx/smoke_bdxrl.py \
      [--num-envs 64] [--device cuda:0]

Stages:
  1. Registry: task exists; motion command frames == 293; env fps == 50.
  2. Reset: teleport states finite; anchor pos error sane (env origins offset).
  3. Zero-action probe (10 steps): obs finite, errors/terminations reported.
  4. Memory: torch peak for the env batch.
"""

from __future__ import annotations

import argparse
import json

import torch


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--task", default="Mjlab-Tracking-Flat-BDX-V4-Gesture-Side-Step-4")
  ap.add_argument("--frames", type=int, default=293)
  args = ap.parse_args()

  import tracking_bdx.config.bdx_v4  # noqa: F401  (registers tasks)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  TASK = args.task
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  obs, _ = env.reset()
  motion = env.command_manager.get_term("motion")
  frames = int(motion.motion.time_step_total)
  fps = 1.0 / env.step_dt
  dims = {
    "task": TASK,
    "num_envs": args.num_envs,
    "actor_obs_dim": int(obs["actor"].shape[-1]),
    "action_dim": int(env.action_manager.total_action_dim),
    "motion_frames": frames,
    "env_fps": fps,
    "episode_length_s": cfg.episode_length_s,
  }
  print("[1]", json.dumps(dims))
  assert frames == args.frames, frames
  assert abs(fps - 50.0) < 1e-6, fps
  assert dims["action_dim"] == 14

  a0 = float(motion.metrics["error_anchor_pos"].mean())
  print(f"[2] reset anchor_pos_err mean={a0:.4f} (post-teleport, should be ~0)")
  assert torch.isfinite(obs["actor"]).all()

  n_term = 0
  for _ in range(10):
    obs, _, terminated, truncated, _ = env.step(torch.zeros(
      args.num_envs, dims["action_dim"], device=args.device))
    n_term += int(terminated.sum())
  report = {
    "anchor_pos_err": round(float(motion.metrics["error_anchor_pos"].mean()), 4),
    "anchor_rot_err": round(float(motion.metrics["error_anchor_rot"].mean()), 4),
    "body_pos_err": round(float(motion.metrics["error_body_pos"].mean()), 4),
    "body_rot_err": round(float(motion.metrics["error_body_rot"].mean()), 4),
    "joint_pos_err": round(float(motion.metrics["error_joint_pos"].mean()), 4),
    "terminations_in_10_steps": n_term,
    "finite_obs": bool(torch.isfinite(obs["actor"]).all()),
  }
  print("[3]", json.dumps(report))
  assert report["finite_obs"]
  # Orientation-convention gate: run mjlab_bdx/smoke_quat_reset.py on the same
  # npz for the per-body orientation residual check (catches xyzw/wxyz swaps
  # that root-level identity checks cannot).

  if args.device.startswith("cuda"):
    print(f"[4] torch peak {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
  print("SMOKE PASS")


if __name__ == "__main__":
  main()
