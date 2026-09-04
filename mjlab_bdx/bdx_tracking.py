"""BDX single-motion tracking task: side_step_4 dance via mjlab + RSL-RL PPO.

Standard mjlab tracking recipe (BeyondMimic re-implementation) with BDX
robot-specific wiring only. Deliberately NOT ported from the old trainer:
custom PPO, BFM actor, objective-v2, old termination code, old action
normalization chain, F/P对照 (quarantined line stays archived).

Motion: results/mjlab_bdx/ss4_mjlab_motion.npz — the validated open-loop PD
re-roll (293 frames @ 50 Hz). Episode length = 293/50 = 5.86 s so one
episode == one full pass (non-cyclic, motion end == episode end).

Deviations from the G1 config, all BDX-motivated:
  - anchor/base_link, all 17 bodies tracked, episode 5.86 s
  - foot friction regex for left/right_foot_collision geoms
  - self_collision sensor + reward dropped for v1 (ContactMatch subtree
    semantics unverified for BDX; MJCF self-collisions stay physically on)
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from bdx_constants import BDX_ACTION_SCALE, BDX_BODY_NAMES, get_bdx_robot_cfg

ROOT = Path(__file__).resolve().parents[1]
MOTION_FILE = ROOT / "results" / "mjlab_bdx" / "ss4_mjlab_motion.npz"

TASK_ID = "Mjlab-Tracking-Flat-BDX-SS4"
EPISODE_LENGTH_S = 5.86  # 293 frames @ 50 Hz


def bdx_ss4_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_bdx_robot_cfg()}
  cfg.scene.sensors = ()

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = BDX_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert MOTION_FILE.exists(), (
    f"motion file missing: {MOTION_FILE} — run convert_motion.py first")
  motion_cmd.motion_file = str(MOTION_FILE)
  motion_cmd.anchor_body_name = "base_link"
  motion_cmd.body_names = BDX_BODY_NAMES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    r"^(left|right)_foot_collision$"
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_link", "right_ankle_link",
  )

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.6

  cfg.episode_length_s = EPISODE_LENGTH_S

  if play:
    motion_cmd.sampling_mode = "start"
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    cfg.events.pop("push_robot", None)
    cfg.observations["actor"].enable_corruption = False

  return cfg


def bdx_ss4_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="bdx_ss4_tracking",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5000,
  )


def register() -> None:
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=bdx_ss4_tracking_env_cfg(play=False),
    play_env_cfg=bdx_ss4_tracking_env_cfg(play=True),
    rl_cfg=bdx_ss4_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
  )


if __name__ == "__main__":
  raise SystemExit(
    "Use mjlab_bdx/train_bdx.py (registers the task, then calls launch_training).")
