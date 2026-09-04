"""mjlab+BDX smoke: entity loads, actuator mapping, standing, motion command.

Stages (each prints numbers; any FAIL exits non-zero):
  1. Entity build: 14 joints, 17 bodies, 14 actuators, keyframe values.
  2. STAND env (no motion command, no pushes): zero action == PD at default
     joint targets. 200 steps @ 50 Hz — BDX must stand (base_z floor, no
     NaN). This is the pure "mjlab loads BDX + action mapping" check.
  3. TRACK env: MotionCommand loads 293 frames; reset teleports to
     reference; 10 zero-action steps — obs finite, errors reported (no
     no-fall assertion: zero action from mid-dance poses SHOULD often
     terminate; that is the task working).

Usage: .venv-mjlab/bin/python mjlab_bdx/smoke_bdx.py [--num-envs 64]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bdx_constants import BDX_ACTION_SCALE, get_bdx_robot_cfg  # noqa: E402


def zero_action(env, obs):
  return torch.zeros(env.scene.num_envs,
                     env.action_manager.total_action_dim,
                     device=obs["actor"].device)


def stage_entity() -> None:
  from mjlab.entity import Entity
  entity = Entity(get_bdx_robot_cfg())
  names_j, names_b = list(entity.joint_names), list(entity.body_names)
  assert len(names_j) == 14, f"expected 14 joints, got {len(names_j)}"
  assert len(names_b) == 17, f"expected 17 bodies, got {len(names_b)}"
  n_act = entity.num_actuators
  assert n_act == 14, f"expected 14 actuators, got {n_act}"
  print("[1] entity OK: joints", len(names_j), "bodies", len(names_b),
        "actuators", n_act)
  print("    joints:", names_j)


def build_stand_env(num_envs: int, device: str):
  """Minimal standing env: same robot/action, no command/push/motion terms."""
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
  from mjlab.envs.mdp.actions import JointPositionActionCfg
  from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
  )
  from mjlab.scene import SceneCfg
  from mjlab.sim import MujocoCfg, SimulationCfg
  from mjlab.terrains import TerrainEntityCfg
  from mjlab.envs.mdp import observations as mdp_obs

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp_obs.base_ang_vel, params={}),
    "joint_pos": ObservationTermCfg(func=mdp_obs.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp_obs.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp_obs.last_action),
  }
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"),
                   num_envs=num_envs,
                   entities={"robot": get_bdx_robot_cfg()},
                   sensors=()),
    observations={"actor": ObservationGroupCfg(terms=actor_terms,
                                               concatenate_terms=True)},
    actions={"joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",),
      scale=BDX_ACTION_SCALE, use_default_offset=True)},
    commands={},
    events={},
    rewards={},
    terminations={},
    sim=SimulationCfg(nconmax=35, njmax=250,
                      mujoco=MujocoCfg(timestep=0.005, iterations=10,
                                       ls_iterations=20)),
    decimation=4,
    episode_length_s=10.0,
  )
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def stage_stand(num_envs: int, device: str, steps: int = 200) -> None:
  env = build_stand_env(num_envs, device)
  obs = env.reset()
  root_z = lambda: env.scene["robot"].data.root_link_pose_w[:, 2]
  z0 = float(root_z().mean())
  zs = []
  for _ in range(steps):
    obs, _, terminated, truncated, _ = env.step(zero_action(env, obs))
    zs.append(float(root_z().mean()))
  zmin_all = float(root_z().min())
  report = {
    "base_z_start": round(z0, 4), "base_z_end": round(zs[-1], 4),
    "base_z_min_over_envs": round(zmin_all, 4),
    "base_z_traj_min": round(min(zs), 4), "base_z_traj_max": round(max(zs), 4),
    "finite_obs": bool(torch.isfinite(obs["actor"]).all()),
  }
  print("[2] stand:", json.dumps(report))
  assert report["finite_obs"], "NaN/Inf in standing obs"
  assert min(zs) > 0.15, f"base sagged below 0.15 m: {report}"
  assert zmin_all > 0.15, f"some env fell below 0.15 m: {report}"
  print("    stand OK (zero action == default PD targets)")


def stage_track(num_envs: int, device: str) -> None:
  import bdx_tracking
  bdx_tracking.register()
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  cfg = load_env_cfg(bdx_tracking.TASK_ID)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  obs = env.reset()
  motion = env.command_manager.get_term("motion")
  frames = int(motion.motion.time_step_total)
  assert frames == 293, f"motion frames {frames} != 293"
  assert abs(1.0 / env.step_dt - 50.0) < 1e-6, f"env fps {1.0/env.step_dt}"
  dims = {
    "actor_obs_dim": int(obs["actor"].shape[-1]),
    "action_dim": int(env.action_manager.total_action_dim),
    "motion_frames": frames,
  }
  print("[3] track env:", json.dumps(dims))
  assert dims["action_dim"] == 14

  n_term = 0
  for _ in range(10):
    obs, _, terminated, truncated, _ = env.step(zero_action(env, obs))
    n_term += int(terminated.sum())
  report = {
    "anchor_pos_err": round(float(motion.metrics["error_anchor_pos"].mean()), 4),
    "joint_pos_err": round(float(motion.metrics["error_joint_pos"].mean()), 4),
    "terminations_in_10_steps": n_term,
    "finite_obs": bool(torch.isfinite(obs["actor"]).all()),
  }
  print("    zero-action probe:", json.dumps(report))
  assert report["finite_obs"], "NaN in tracking probe"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--device", default="cuda:0")
  args = ap.parse_args()

  stage_entity()
  stage_stand(args.num_envs, args.device)
  stage_track(args.num_envs, args.device)

  if args.device.startswith("cuda"):
    print(f"[mem] torch peak {torch.cuda.max_memory_allocated()/1024**2:.0f} MB "
          f"for {args.num_envs} envs")
  print("SMOKE PASS")


if __name__ == "__main__":
  main()
