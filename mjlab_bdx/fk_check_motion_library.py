"""FK round-trip gate: motion_library.npz frames replayed on the bdx_rl_mjlab
BDX v4 asset must reproduce the library's body_pos/body_rot exactly.

This is the definitive asset+order+convention check (same gate BFM-zero used:
"FK round-trip 0.00mm"). If max body position error is sub-mm, the library
was recorded on the same asset with the same body/joint ordering and the
wxyz/world-frame declaration holds; the walk pool can then be sliced safely.

Usage (under bdx_rl_mjlab venv, cwd=bdx_rl_mjlab):
  .venv/bin/python /home/tcl/Desktop/start/BFM-zero/mjlab_bdx/fk_check_motion_library.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

LIB = "/home/tcl/Desktop/start/bdx-data/bdx_motion_library/motion_library.npz"
OUT = "/home/tcl/Desktop/start/BFM-zero/results/mjlab_bdx/fk_check_motion_library.json"
N_FRAMES_PER_SEG = 6

BODY_NAMES = (
  "base_link",
  "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link",
  "left_knee_link", "left_ankle_link",
  "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link",
  "right_knee_link", "right_ankle_link",
  "neck_forward_link", "neck_pitch_link", "neck_yaw_link", "neck_roll_link",
  "left_ear_link", "right_ear_link",
)
JOINT_NAMES = (
  "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
  "left_knee_joint", "left_ankle_joint",
  "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
  "right_knee_joint", "right_ankle_joint",
  "neck_forward_joint", "neck_pitch_joint", "neck_yaw_joint", "neck_roll_joint",
)
SEGMENTS = {  # segment id -> name (from README table)
  0: "walk_seed0", 1: "walk_seed1", 2: "walk_seed2", 3: "walk_seed3",
  4: "walk_seed4", 5: "stand_seed0", 6: "stand_seed1", 9: "stand_seed3",
  14: "laugh_giggle", 16: "relax_v2",
}


def quat_angle(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
  """Geodesic angle between unit quats (wxyz), elementwise."""
  d = np.abs(np.einsum("...i,...i->...", q0, q1))
  return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def main() -> None:
  from robots.bdx_v4 import get_bdx_v4_robot_cfg
  from mjlab.entity import Entity

  entity = Entity(get_bdx_v4_robot_cfg())
  assert tuple(entity.body_names) == BODY_NAMES, "entity body order drifted"
  assert tuple(entity.joint_names) == JOINT_NAMES, "entity joint order drifted"
  compiled = entity.compile()
  model = compiled.model if hasattr(compiled, "model") else compiled
  data = mujoco.MjData(model)

  lib = np.load(LIB, allow_pickle=True)
  starts = lib["length_starts"]
  lengths = lib["_motion_num_frames"]
  root_pos, root_rot = lib["root_pos"], lib["root_rot"]
  joint_pos, body_pos, body_rot = lib["joint_pos"], lib["body_pos"], lib["body_rot"]

  # qpos layout: free joint (pos3+quat4) then joints in model order == JOINT_NAMES.
  qadr = model.jnt_dofadr.copy()
  free_jid = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
  assert len(free_jid) == 1, f"expected one free joint, got {len(free_jid)}"
  free_qadr = int(qadr[free_jid[0]])
  joint_qadr = {}
  for name in JOINT_NAMES:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert jid >= 0, f"joint {name} missing in compiled model"
    joint_qadr[name] = int(qadr[jid])
  bid = {
    name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    for name in BODY_NAMES
  }

  rows = []
  worst = {"seg": None, "pos_mm": 0.0, "ang_deg": 0.0}
  for seg, name in SEGMENTS.items():
    s, T = int(starts[seg]), int(lengths[seg])
    for f in np.linspace(0, T - 1, N_FRAMES_PER_SEG).astype(int):
      t = s + f
      q = np.zeros(model.nq)
      q[free_qadr:free_qadr + 3] = root_pos[t]
      q[free_qadr + 3:free_qadr + 7] = root_rot[t]  # wxyz both sides
      for jname in JOINT_NAMES:
        q[joint_qadr[jname]] = joint_pos[t, JOINT_NAMES.index(jname)]
      data.qpos[:] = q
      data.qvel[:] = 0
      mujoco.mj_forward(model, data)
      pos_err_mm, ang_err_deg = 0.0, 0.0
      for bname in BODY_NAMES:
        i = BODY_NAMES.index(bname)
        p_fk = data.xpos[bid[bname]]
        p_lib = body_pos[t, i]
        pos_err_mm = max(pos_err_mm, float(np.linalg.norm(p_fk - p_lib) * 1000.0))
        # MuJoCo xquat is wxyz — same declared order as the library.
        ang_err_deg = max(
          ang_err_deg, float(np.degrees(quat_angle(data.xquat[bid[bname]], body_rot[t, i]))))
      rows.append({
        "segment": name, "frame": int(f),
        "pos_err_mm": round(pos_err_mm, 4), "ang_err_deg": round(ang_err_deg, 4),
      })
      if pos_err_mm > worst["pos_mm"]:
        worst = {"seg": f"{name}@{f}", "pos_mm": pos_err_mm, "ang_deg": ang_err_deg}

  max_pos = max(r["pos_err_mm"] for r in rows)
  max_ang = max(r["ang_err_deg"] for r in rows)
  verdict = "PASS" if max_pos < 1.0 and max_ang < 0.1 else "FAIL"
  report = {
    "lib": LIB, "frames_checked": len(rows), "verdict": verdict,
    "max_pos_err_mm": round(max_pos, 4), "max_ang_err_deg": round(max_ang, 4),
    "worst": worst, "rows": rows,
    "gate": "max_pos_err < 1.0 mm AND max_ang_err < 0.1 deg on all sampled frames",
    "checked_at": datetime.now(timezone.utc).isoformat(),
  }
  assert not Path(OUT).exists(), f"refusing to overwrite {OUT}"
  Path(OUT).write_text(json.dumps(report, indent=2))
  print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
  sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
  main()
