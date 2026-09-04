"""Convert the validated open-loop BDX rollout into mjlab MotionCommand format.

Body-ordering is the known silent-corruption trap (mjlab docs + our own
right-ankle bare-index incident): MotionLoader indexes the motion file's
body axis with the ENTITY's body indexes (robot.find_bodies(cfg.body_names)),
and joint tracks are compared directly against entity joint order. This
converter therefore reorders BOTH axes BY NAME against the live Entity
(built from get_bdx_robot_cfg()), with bijection + per-pair assertions and
a recorded name manifest inside the output npz.

QUATERNION CONTRACT (P0 fix, 2026-08-26):
  mjlab consumes wxyz — proven from source: mjlab/utils/lab_api/math.py
  matrix_from_quat docstring "(w, x, y, z)"; also empirically: v3_planner
  clips (wxyz) produce error_anchor_rot == 0.0 right after RSI teleport.
  The recorder (record_motion.py -> sim._rigid_body_rot) produces xyzw.
  Therefore --in-quat is REQUIRED and the output is ALWAYS wxyz, tagged
  quat_order="wxyz". Convention is never guessed from values; it must be
  declared. (Historical note: ss4_mjlab_motion.npz produced before this fix
  carried xyzw in a wxyz pipeline — marked INVALID.)

Input  (record_motion.py): results/mjlab_bdx/ss4_openloop_rollout.npz
Output (this script):      results/mjlab_bdx/ss4_mjlab_motion.npz
  joint_pos (T,14)  joint_vel (T,14)
  body_pos_w (T,17,3)  body_quat_w (T,17,4)  body_lin_vel_w (T,17,3)
  body_ang_vel_w (T,17,3)
All world frame; quats wxyz.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# recorder layout
REC_BODY_NAMES = (
  "base_link",
  "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link",
  "left_knee_link", "left_ankle_link",
  "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link",
  "right_knee_link", "right_ankle_link",
  "neck_forward_link", "neck_pitch_link", "neck_yaw_link", "neck_roll_link",
  "left_ear_link", "right_ear_link",
)
REC_JOINT_NAMES = (
  "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
  "left_knee_joint", "left_ankle_joint",
  "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
  "right_knee_joint", "right_ankle_joint",
  "neck_forward_joint", "neck_pitch_joint", "neck_yaw_joint", "neck_roll_joint",
)


def perm_by_name(src_names: tuple[str, ...], dst_names: list[str] | tuple[str, ...],
                 axis_desc: str) -> np.ndarray:
  src = list(src_names)
  dst = list(dst_names)
  assert set(src) == set(dst), (
    f"{axis_desc} name-set mismatch: only-in-src={sorted(set(src)-set(dst))} "
    f"only-in-dst={sorted(set(dst)-set(src))}")
  assert len(set(src)) == len(src), f"{axis_desc} duplicate names in src"
  perm = np.array([src.index(n) for n in dst], dtype=np.int64)
  return perm


def quat_to_wxyz(q: np.ndarray, in_order: str) -> np.ndarray:
  """Explicit convention conversion. No value-based guessing."""
  if in_order == "wxyz":
    return q.astype(np.float32)
  if in_order == "xyzw":
    return q[..., [3, 0, 1, 2]].astype(np.float32)
  raise ValueError(f"unknown --in-quat {in_order}")


def rotmat_from(q: np.ndarray, order: str) -> np.ndarray:
  """Rotation matrix from unit quats in the DECLARED order (numpy, (...,4)->(...,3,3))."""
  # xyzw layout is (x,y,z,w): the wxyz components are q[[3,0,1,2]].
  w, x, y, z = (q[..., i] if order == "wxyz" else q[..., [3, 0, 1, 2][i]]
                for i in range(4))
  return np.stack([
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def quat_selftest(n: int = 256, seed: int = 0) -> dict:
  """Random non-trivial unit quats: conversion must preserve the ROTATION.

  Compares rotation matrices built from the original (declared order) and
  the converted (wxyz) quats. Identity-first-frame checks can NOT catch
  permutations where w is small; random full-tilt quats can.
  """
  rng = np.random.default_rng(seed)
  q = rng.normal(size=(n, 4))
  q /= np.linalg.norm(q, axis=-1, keepdims=True)
  out = {}
  for order in ("xyzw", "wxyz"):
    conv = quat_to_wxyz(q, order)
    R0, R1 = rotmat_from(q, order), rotmat_from(conv, "wxyz")
    # geodesic angle between R0 and R1: |angle| = arccos((trace(R0^T R1)-1)/2)
    # trace(R0^T R1) = sum of elementwise products (NO extra transpose)
    tr = np.einsum("...ij,...ij->...", R0, R1)
    ang = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    out[order] = float(np.abs(ang).max())
    # 1e-3 rad: float32 output precision bound; permutation errors are
    # >= ~0.1 rad, so 100x separation from any convention mistake.
    assert np.abs(ang).max() < 1e-3, f"quat selftest FAILED for {order}"
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--rollout", default=str(ROOT / "results/mjlab_bdx/ss4_openloop_rollout.npz"))
  ap.add_argument("--out", default=str(ROOT / "results/mjlab_bdx/ss4_mjlab_motion.npz"))
  ap.add_argument(
    "--in-quat", choices=("xyzw", "wxyz"), required=True,
    help="Quaternion order of the ROLLOUT's body_quat channel. record_motion.py "
         "(sim._rigid_body_rot, repo convention) => xyzw. Never inferred from "
         "values; must be declared by whoever produced the rollout.")
  ap.add_argument(
    "--robot-cfg", choices=("bfmzero", "bdxrl"), default="bfmzero",
    help="Which robot cfg builds the authoritative-ordering Entity. Use "
         "'bdxrl' when running under bdx_rl_mjlab's venv (its Entity is "
         "what the training task will actually index with).")
  args = ap.parse_args()
  rollout_path, out_path = Path(args.rollout).resolve(), Path(args.out).resolve()
  assert not out_path.exists(), f"refusing to overwrite {out_path}"

  # Authoritative ordering: build the actual Entity the task will use.
  from mjlab.entity import Entity
  if args.robot_cfg == "bdxrl":
    from robots.bdx_v4 import get_bdx_v4_robot_cfg
    entity = Entity(get_bdx_v4_robot_cfg())
  else:
    from bdx_constants import get_bdx_robot_cfg
    entity = Entity(get_bdx_robot_cfg())
  ent_body_names = list(entity.body_names)
  ent_joint_names = list(entity.joint_names)
  print(f"entity bodies ({len(ent_body_names)}):", ent_body_names)
  print(f"entity joints ({len(ent_joint_names)}):", ent_joint_names)

  rec = np.load(rollout_path, allow_pickle=False)
  rec_body = [str(x) for x in rec["body_names"]]
  rec_joint = [str(x) for x in rec["joint_names"]]
  # recorder-side sanity: stored names must equal our frozen REC_* tuples.
  assert tuple(rec_body) == REC_BODY_NAMES, "recorder body manifest drifted"
  assert tuple(rec_joint) == REC_JOINT_NAMES, "recorder joint manifest drifted"

  bperm = perm_by_name(tuple(rec_body), ent_body_names, "body")
  jperm = perm_by_name(tuple(rec_joint), ent_joint_names, "joint")

  joint_pos = rec["dof_pos"][:, jperm].astype(np.float32)
  joint_vel = rec["dof_vel"][:, jperm].astype(np.float32)
  body_pos_w = rec["body_pos"][:, bperm].astype(np.float32)
  body_quat_w = quat_to_wxyz(rec["body_quat"][:, bperm], args.in_quat)
  body_lin_vel_w = rec["body_lin_vel"][:, bperm].astype(np.float32)
  body_ang_vel_w = rec["body_ang_vel"][:, bperm].astype(np.float32)

  T = joint_pos.shape[0]
  # 1) conversion selftest on random non-trivial quats (catches any
  #    permutation error identity-frame data cannot expose).
  selftest = quat_selftest()
  # 2) real-data equivalence: rotation matrices of the SOURCE rollout quats
  #    (declared order) vs the CONVERTED output (wxyz), every body, every
  #    frame. This is the per-body rotation-matrix comparison.
  R_src = rotmat_from(rec["body_quat"][:, bperm].astype(np.float64), args.in_quat)
  R_out = rotmat_from(body_quat_w.astype(np.float64), "wxyz")
  tr = np.einsum("...bij,...bij->...", R_src, R_out)
  frame_ang = np.arccos(np.clip((tr - 1) / 2, -1, 1))
  max_body_ang = float(frame_ang.max())
  assert max_body_ang < 1e-3, f"quat conversion changed rotations: {max_body_ang}"
  # 3) unit norms + finiteness (kept).
  norms = np.linalg.norm(body_quat_w, axis=-1)
  assert np.allclose(norms, 1.0, atol=1e-5), f"non-unit quats: min={norms.min()}, max={norms.max()}"
  for arr in (joint_pos, joint_vel, body_pos_w, body_lin_vel_w, body_ang_vel_w):
    assert np.isfinite(arr).all(), "non-finite values in motion"

  np.savez(
    out_path,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
    # manifest (MotionLoader ignores unknown keys; kept for audit)
    body_names=np.array(ent_body_names),
    joint_names=np.array(ent_joint_names),
    # explicit convention tag — loaders may reject unknown/missing values
    quat_order=np.array("wxyz"),
    in_quat=np.array(args.in_quat),
    # fps as (1,) — bdx_rl_mjlab's auto-registration reads _d["fps"][0]
    fps=np.array([float(rec["fps"])]),
  )

  root_z = body_pos_w[:, 0, 2]
  report = {
    "out": str(out_path),
    "frames": int(T),
    "duration_s": float(T / float(rec["fps"])),
    "in_quat": args.in_quat,
    "out_quat": "wxyz",
    "quat_selftest_max_angle": selftest,
    "quat_data_equivalence_max_body_angle_rad": max_body_ang,
    "body_perm": bperm.tolist(),
    "joint_perm": jperm.tolist(),
    "entity_body_names": ent_body_names,
    "entity_joint_names": ent_joint_names,
    "root_z_range": [float(root_z.min()), float(root_z.max())],
    "joint_pos_absmax": float(np.abs(joint_pos).max()),
    "body_lin_vel_absmax": float(np.abs(body_lin_vel_w).max()),
    "converted_at": datetime.now(timezone.utc).isoformat(),
  }
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
