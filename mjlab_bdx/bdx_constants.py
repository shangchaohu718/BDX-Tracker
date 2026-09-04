"""BDX (14-DoF quadruped-style biped) robot constants for mjlab.

All physical values come from the VALIDATED BFM-zero training stack:
  - MJCF: humanoidverse/data/robots/bdx/bdx_14dof.xml (meshes, capsules,
    joint classes act_x8 / act_xh540, joint ranges).
  - PD gains / efforts: humanoidverse/config/robot/bdx/bdx_14dof.yaml
    (stiffness/damping per joint group; motor forceranges from the MJCF).
The open-loop PD re-roll of `re_dancegen_side_step_4` (293/293 frames,
results/mjlab_bdx/ss4_openloop_rollout.npz) ran under exactly these
parameters — BuiltinPositionActuator OVERRIDES armature/frictionloss/
viscous_damping on its target joints, so we pass the xml-identical values
in (override-with-same-value == physics preserved bit-for-bit).
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

BDX_XML: Path = (
    Path(__file__).resolve().parents[1]
    / "humanoidverse" / "data" / "robots" / "bdx" / "bdx_14dof.xml"
)
assert BDX_XML.exists(), f"missing BDX MJCF: {BDX_XML}"

# xml joint-class values (act_x8 / act_xh540) — must stay identical to the
# MJCF defaults so the actuator override is a no-op on these channels.
_X8 = dict(armature=0.0116, frictionloss=0.8, viscous_damping=0.012)
_XH540 = dict(armature=0.0058, frictionloss=0.05, viscous_damping=0.009)


def get_spec() -> mujoco.MjSpec:
  """Load the BDX MJCF and strip the 14 torque motors.

  mjlab attaches its own position actuators (see BDX_ARTICULATION); the
  stock torque motors would otherwise coexist and break actuator regex
  matching. Everything else (joints, geoms, defaults) is untouched.
  """
  spec = mujoco.MjSpec.from_file(str(BDX_XML))
  for act in list(spec.actuators):
    spec.delete(act)
  assert len(spec.actuators) == 0, "failed to strip stock torque motors"
  return spec


# PD gains from bdx_14dof.yaml control.stiffness/damping; effort limits from
# the MJCF motor forceranges (full physical authority).
BDX_ACTUATORS: tuple[BuiltinPositionActuatorCfg, ...] = (
  BuiltinPositionActuatorCfg(
    target_names_expr=("left_hip_yaw_joint", "right_hip_yaw_joint"),
    stiffness=10.0, damping=0.3, effort_limit=23.7, **_X8),
  BuiltinPositionActuatorCfg(
    target_names_expr=("left_hip_roll_joint", "right_hip_roll_joint"),
    stiffness=15.0, damping=0.6, effort_limit=34.0, **_X8),
  BuiltinPositionActuatorCfg(
    target_names_expr=("left_hip_pitch_joint", "right_hip_pitch_joint"),
    stiffness=15.0, damping=0.6, effort_limit=34.0, **_X8),
  BuiltinPositionActuatorCfg(
    target_names_expr=("left_knee_joint", "right_knee_joint"),
    stiffness=15.0, damping=0.6, effort_limit=34.0, **_X8),
  BuiltinPositionActuatorCfg(
    target_names_expr=("left_ankle_joint", "right_ankle_joint"),
    stiffness=10.0, damping=0.3, effort_limit=23.7, **_X8),
  BuiltinPositionActuatorCfg(
    target_names_expr=("neck_forward_joint",),
    stiffness=10.0, damping=0.3, effort_limit=23.7, **_X8),
  BuiltinPositionActuatorCfg(
    target_names_expr=("neck_pitch_joint", "neck_yaw_joint", "neck_roll_joint"),
    stiffness=5.0, damping=0.2, effort_limit=4.8, **_XH540),
)

BDX_ARTICULATION = EntityArticulationInfoCfg(
  actuators=BDX_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)

# Home keyframe: bdx_14dof.yaml init_state.default_joint_angles; base height
# from the settled frame-0 state of the validated rollout (xml's 0.35 is the
# drop height, not the settled standing height).
BDX_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.23499999940395355),
  joint_pos={
    "left_hip_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_pitch_joint": 0.904,
    "left_knee_joint": -1.143,
    "left_ankle_joint": 0.24,
    "right_hip_yaw_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.904,
    "right_knee_joint": -1.143,
    "right_ankle_joint": 0.24,
    "neck_forward_joint": 0.5,
    "neck_pitch_joint": -0.5,
    "neck_yaw_joint": 0.0,
    "neck_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

BDX_BODY_NAMES: tuple[str, ...] = (
  "base_link",
  "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link",
  "left_knee_link", "left_ankle_link",
  "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link",
  "right_knee_link", "right_ankle_link",
  "neck_forward_link", "neck_pitch_link", "neck_yaw_link", "neck_roll_link",
  "left_ear_link", "right_ear_link",
)

BDX_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
  "left_knee_joint", "left_ankle_joint",
  "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
  "right_knee_joint", "right_ankle_joint",
  "neck_forward_joint", "neck_pitch_joint", "neck_yaw_joint", "neck_roll_joint",
)


def get_bdx_robot_cfg() -> EntityCfg:
  """Get a fresh BDX robot configuration instance."""
  return EntityCfg(
    init_state=BDX_HOME_KEYFRAME,
    collisions=(),  # MJCF collision geoms already carry contype/conaffinity.
    spec_fn=get_spec,
    articulation=BDX_ARTICULATION,
  )


# mjlab convention (mirrors G1_ACTION_SCALE): action ±1 reaches the joint
# angle whose PD restoring torque equals 25% of the effort limit.
BDX_ACTION_SCALE: dict[str, float] = {}
for _a in BDX_ACTUATORS:
  for _n in _a.target_names_expr:
    BDX_ACTION_SCALE[_n] = 0.25 * _a.effort_limit / _a.stiffness
