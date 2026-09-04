"""Post-reset orientation smoke: proves quat convention by measurement.

For a motion npz, builds the gesture env DIRECTLY (cfg factory, no registry
side effects), teleports via reset, and checks:
  - error_anchor_rot  (motion metric, computed by mjlab in wxyz)
  - error_body_rot
  - max per-body orientation residual (angle between motion body_quat_w and
    the robot's actual FK body quats right after the teleport)

A convention-correct file yields ~0 anchor rot and small body residuals
(float32 + settle-step noise). An xyzw-in-wxyz file yields residuals on the
order of radians — demonstrated by the --demo-bad mode on the quarantined
pre-fix ss4 artifact.

Usage (bdx_rl_mjlab venv, repo root):
  .venv/bin/python smoke_quat_reset.py --motion <npz> [--duration 5.02]
  .venv/bin/python smoke_quat_reset.py --demo-bad
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def quat_angle_between(q0, q1):
  """Angle between two wxyz quats (rad)."""
  d = np.abs(np.sum(q0 * q1, axis=-1))
  return 2 * np.arccos(np.clip(d, -1, 1))


def check(motion_file: str, duration: float, device: str = "cpu") -> dict:
  import tracking_bdx.config.bdx_v4  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from tracking_bdx.config.bdx_v4.env_cfgs import bdx_v4_gesture_tracking_env_cfg

  cfg = bdx_v4_gesture_tracking_env_cfg(motion_file, duration, play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  motion = env.command_manager.get_term("motion")
  robot = env.scene["robot"]
  env.reset()

  # Root-level buffers are fresh right after the teleport (error_anchor_pos/rot
  # read 0.0), but BODY-level buffers are one step stale — refresh with one
  # hold step: reference-PD action keeps the pose at the reference while the
  # kinematics buffers sync.
  q_ref = motion.joint_pos[0]
  hold = ((q_ref - robot.data.default_joint_pos[0]) / 0.25)[None]
  env.step(hold)

  e_anchor = float(motion.metrics["error_anchor_rot"][0])
  e_body = float(motion.metrics["error_body_rot"][0])

  ref = motion.body_quat_w[0].cpu().numpy()  # (17,4) wxyz at current frame
  act = robot.data.body_link_quat_w[0][motion.body_indexes].cpu().numpy()
  per_body = quat_angle_between(ref, act)
  worst_i = int(np.argmax(per_body))
  names = motion.cfg.body_names
  out = {
    "motion_file": motion_file,
    "error_anchor_rot": round(e_anchor, 5),
    "error_body_rot": round(e_body, 5),
    "max_body_ori_residual_rad": round(float(per_body.max()), 5),
    "worst_body": names[worst_i],
    "per_body_p50_rad": round(float(np.median(per_body)), 5),
  }
  env.close()
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--motion")
  ap.add_argument("--duration", type=float)
  ap.add_argument("--demo-bad", action="store_true",
                  help="Run on the quarantined pre-fix xyzw artifact.")
  args = ap.parse_args()

  if args.demo_bad:
    r = check("/tmp/ss4_contaminated_quat_demo.npz", 5.86)
    print("[BAD  pre-fix xyzw artifact]", r)
    assert r["error_anchor_rot"] > 0.2 or r["max_body_ori_residual_rad"] > 0.2, (
      "contaminated artifact NOT detected — check thresholds")
    print("DETECTED as expected (residual is rad-scale)")
    return

  r = check(args.motion, args.duration)
  print("[CHECK]", r)
  assert r["error_anchor_rot"] < 0.05, "anchor rot residual too large"
  assert r["max_body_ori_residual_rad"] < 0.2, "body ori residual too large"
  print("QUAT RESET SMOKE PASS")


if __name__ == "__main__":
  main()
