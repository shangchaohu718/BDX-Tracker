"""Headless eval of checkpoints in the exact play env (frame-0 spawn, no DR).

Run from bdx_rl_mjlab repo root with ITS venv:
  cd /home/tcl/Desktop/start/bdx_rl_mjlab && .venv/bin/python \
    /home/tcl/Desktop/start/BFM-zero/mjlab_bdx/eval_play_headless.py

Tests (1 env, deterministic):
  A. model_500 policy      — trained policy, dense first-50-step trace.
  B. reference-PD control  — action=(q_ref-q_default)/scale (perfect joint PD).
  C. model_0 policy        — untrained baseline.
Plus post-reset kinematic consistency: motion error metrics immediately after
the frame-0 teleport (before any physics step).
"""

from __future__ import annotations

import json
import sys

import torch

TASK = "Mjlab-Tracking-Flat-BDX-V4-Gesture-Excited-Wiggle"
RUN_DIR = (
  "/home/tcl/Desktop/start/bdx_rl_mjlab/logs/rsl_rl/bdx_v4_tracking/"
  "2026-08-25_17-21-09_excited_wiggle_v1"
)
ACTION_SCALE = 0.25
OUT = "/home/tcl/Desktop/start/BFM-zero/results/mjlab_bdx/eval_play_headless.json"


def load_actor(path: str, device: str):
  ck = torch.load(path, map_location="cpu", weights_only=False)
  sd = ck["actor_state_dict"]
  mean = sd["obs_normalizer._mean"].float().to(device)
  std = sd["obs_normalizer._std"].float().to(device)
  # Zero-variance obs channels (constant dims) make _std 0 -> NaN actions.
  # For those channels obs always equals mean, so dividing by 1 is exact.
  std = torch.where(std > 1e-8, std, torch.ones_like(std))
  layers = [(sd[f"mlp.{i}.weight"].float().to(device),
             sd[f"mlp.{i}.bias"].float().to(device)) for i in (0, 2, 4, 6)]

  @torch.no_grad()
  def act(obs: torch.Tensor) -> torch.Tensor:
    x = (obs - mean) / std
    for w, b in layers[:-1]:
      x = torch.nn.functional.elu(x @ w.T + b)
    w, b = layers[-1]
    return x @ w.T + b

  return act, ck["iter"]


def z_of(robot):
  return float(robot.data.root_link_pose_w[0, 2])


def term_names(env):
  return list(env.termination_manager.active_terms)


def fired_terms(env):
  tm = env.termination_manager
  return [n for n in tm.active_terms if bool(tm.get_term(n)[0])]


def reset_snapshot(env, motion, robot):
  """Right after reset: motion error metrics (kinematic consistency probe)."""
  snap = {
    k: round(float(v[0]), 4)
    for k, v in motion.metrics.items()
    if k.startswith("error_")
  }
  # Per-body z mismatch at spawn: ref (world, origin-added) vs actual FK.
  ref_w = motion.body_pos_w[0]  # (17,3), env origins already added
  act_w = robot.data.body_link_pos_w[0][motion.body_indexes]
  dz = (act_w - ref_w)[:, 2]
  order = torch.argsort(dz.abs(), descending=True)
  names = motion.cfg.body_names
  snap["spawn_body_dz_mm"] = [
    (names[i], round(float(dz[i]) * 1000, 1)) for i in order[:6].tolist()
  ]
  # Joint soft-limit violations in clip frame 0.
  q0 = motion.motion.joint_pos[0]
  lo = robot.data.soft_joint_pos_limits[0, :, 0]
  hi = robot.data.soft_joint_pos_limits[0, :, 1]
  viol = [
    (robot.joint_names[i], round(float(q0[i]), 3),
     (round(float(lo[i]), 3), round(float(hi[i]), 3)))
    for i in range(len(q0)) if q0[i] < lo[i] - 1e-4 or q0[i] > hi[i] + 1e-4
  ]
  snap["clip_frame0_softlimit_violations"] = viol
  return snap


def run(env, motion, robot, mode: str, n_steps: int, act=None, dense_until=50):
  obs, _ = env.reset()
  snap0 = reset_snapshot(env, motion, robot)
  falls = []
  dense = []
  trace = []
  for t in range(n_steps):
    if mode == "policy":
      a = act(obs["actor"])
    elif mode == "ref_pd":
      # motion.joint_pos is a per-env property: (num_envs, 14) at current frame.
      a = (motion.joint_pos[0] - robot.data.default_joint_pos[0]) / ACTION_SCALE
    elif mode == "zero":
      a = torch.zeros(14)
    else:
      raise ValueError(mode)
    a = a if a.dim() == 2 else a[None]
    z_pre = z_of(robot)
    obs, _, terminated, truncated, _ = env.step(a)
    fired = fired_terms(env)
    fell = bool(terminated[0]) or bool(truncated[0])
    rec = (t, int(motion.time_steps[0].item()), round(z_pre, 3),
           round(z_of(robot), 3), fired if fell else [])
    if t < dense_until:
      dense.append(rec)
    if fell:
      falls.append(rec)
    if t % 25 == 0:
      trace.append(rec)
  return {
    "mode": mode, "steps": n_steps, "falls": len(falls),
    "first_fall": falls[0] if falls else None,
    "post_reset_errors": snap0,
    "final_root_z": round(z_of(robot), 3),
    "dense_first50": dense,
    "trace": trace,
  }


def main():
  device = "cpu"
  task = sys.argv[1] if len(sys.argv) > 1 else TASK
  out = sys.argv[2] if len(sys.argv) > 2 else OUT
  import tracking_bdx.config.bdx_v4  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  motion = env.command_manager.get_term("motion")
  robot = env.scene["robot"]
  print(f"env ok: frames={motion.motion.time_step_total} "
        f"terms={term_names(env)}")

  results = {"task": task, "device": device,
             "termination_terms": term_names(env)}

  act500, it500 = load_actor(f"{RUN_DIR}/model_500.pt", device)
  print(f"== A: model_500 (iter {it500})")
  results["A_model500"] = run(env, motion, robot, "policy", 600, act500)
  print(json.dumps({k: v for k, v in results["A_model500"].items()
                    if k not in ("dense_first50", "trace")}))

  print("== B: reference-PD")
  results["B_ref_pd"] = run(env, motion, robot, "ref_pd", 300)
  print(json.dumps({k: v for k, v in results["B_ref_pd"].items()
                    if k not in ("dense_first50", "trace")}))

  act0, _ = load_actor(f"{RUN_DIR}/model_0.pt", device)
  print("== C: model_0")
  results["C_model0"] = run(env, motion, robot, "policy", 100, act0)
  print(json.dumps({k: v for k, v in results["C_model0"].items()
                    if k not in ("dense_first50", "trace")}))

  with open(out, "w") as f:
    json.dump(results, f, indent=2)
  print("saved", OUT)


if __name__ == "__main__":
  sys.exit(main())
