"""Deep data-quality audit of the v3_planner excited_wiggle cohort.

Adjudicates, with numbers, three separate suspicions:
  1. Head whip transient (root lin vel / joint vel / body ang vel at frame 0
     vs the clip's own steady level) — is it systemic across all 50 clips?
  2. Velocity-channel self-consistency everywhere (stored vel vs central
     differences of positions), not just at frame 0.
  3. Kinematic consistency: clip body_pos_w vs TRUE FK of clip joint_pos on
     the bdx_v4 MJCF (settles the 1-2 cm spawn mismatch with no env staleness
     caveat). Controls: their native laugh_big.npz / relax.npz.

Run from bdx_rl_mjlab with its venv (needs mujoco only, no env).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import mujoco
import numpy as np

BDX_V4_XML = (
  "/home/tcl/Desktop/start/bdx_rl_mjlab/src/robots/../../robot_description/"
  "robots/bdx_v4/mjcf/bdx_v4.xml"
)
CLIPS = sorted(
  glob.glob(
    "/home/tcl/Desktop/start/dataset/v3_planner/clips/"
    "re_dancegen_excited_wiggle_*.npz"
  )
)
NATIVE = [
  "/home/tcl/Desktop/start/bdx_rl_mjlab/policy_assets/bdx_v4/npz/episodic/"
  "laugh_big.npz",
  "/home/tcl/Desktop/start/bdx_rl_mjlab/policy_assets/bdx_v4/npz/episodic/"
  "relax.npz",
]
OUT = "/home/tcl/Desktop/start/BFM-zero/results/mjlab_bdx/audit_wiggle_cohort.json"
FPS = 50.0


def head_and_vel_stats(path: str) -> dict:
  d = np.load(path, allow_pickle=True)
  jv, jp = d["joint_vel"], d["joint_pos"]
  bp, bv = d["body_pos_w"], d["body_lin_vel_w"]
  ba = d["body_ang_vel_w"]
  T = jp.shape[0]
  steady = slice(10, T)
  root_v = np.linalg.norm(bv[:, 0, :], axis=-1)
  jv_n = np.linalg.norm(jv.reshape(len(jv), -1), axis=-1)
  ba_n = np.linalg.norm(ba.reshape(len(ba), -1), axis=-1)
  # central-diff consistency for joint vel and root lin vel (frames 1..T-2)
  jd = (jp[2:] - jp[:-2]) * 0.5 * FPS
  jv_resid = np.abs(jv[1:-1] - jd)
  rd = (bp[2:, 0, :] - bp[:-2, 0, :]) * 0.5 * FPS
  r_resid = np.abs(bv[1:-1, 0, :] - rd)
  return {
    "path": Path(path).name,
    "frames": T,
    "head": {
      "f0_root_lin_vel": round(float(root_v[0]), 3),
      "f0_joint_vel_norm": round(float(jv_n[0]), 3),
      "f0_body_ang_vel_norm": round(float(ba_n[0]), 3),
      "f1_root_lin_vel": round(float(root_v[1]), 3),
      "steady_root_lin_vel_med": round(float(np.median(root_v[steady])), 3),
      "steady_joint_vel_med": round(float(np.median(jv_n[steady])), 3),
      "steady_body_ang_vel_med": round(float(np.median(ba_n[steady])), 3),
    },
    "vel_consistency": {
      "joint_central_diff_max_abs": round(float(jv_resid.max()), 3),
      "joint_central_diff_med_abs": round(float(np.median(jv_resid)), 4),
      "root_central_diff_max_abs": round(float(r_resid.max()), 4),
    },
    "quat_norm_min": round(float(np.abs(
      np.linalg.norm(d["body_quat_w"], axis=-1) - 1).max()), 6),
    "root_z": [round(float(bp[:, 0, 2].min()), 3), round(float(bp[:, 0, 2].max()), 3)],
  }


def fk_mismatch(path: str, model, jname2adr, bname2id, body_names, joint_names):
  """Per-frame FK(joint_pos, root) vs clip body_pos_w, in mm."""
  d = np.load(path, allow_pickle=True)
  jp, bp, bq = d["joint_pos"], d["body_pos_w"], d["body_quat_w"]
  data = mujoco.MjData(model)
  qpos = np.zeros(model.nq)
  per_body = {n: [] for n in body_names}
  for t in range(jp.shape[0]):
    qpos[:3] = bp[t, 0]
    qpos[3:7] = bq[t, 0]  # mujoco qpos quat = wxyz; clip slot0 ~ 1 => wxyz
    for jn in joint_names:
      qpos[jname2adr[jn]] = jp[t, joint_names.index(jn)]
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    for bi, bn in enumerate(body_names):
      per_body[bn].append(
        np.linalg.norm(data.body(bname2id[bn]).xpos - bp[t, bi]) * 1000
      )
  return {
    n: {
      "mean_mm": round(float(np.mean(v)), 2),
      "max_mm": round(float(np.max(v)), 2),
    }
    for n, v in per_body.items()
  }


def main():
  clips = [head_and_vel_stats(p) for p in CLIPS]
  head = [c["head"] for c in clips]
  n_whip = sum(
    1 for h in head
    if h["f0_root_lin_vel"] > 0.5
    or h["f0_joint_vel_norm"] > 3 * max(h["steady_joint_vel_med"], 1e-3)
  )
  report = {
    "cohort_size": len(clips),
    "whip_head_count": n_whip,
    "f0_root_lin_vel": {
      "min": min(h["f0_root_lin_vel"] for h in head),
      "median": round(float(np.median([h["f0_root_lin_vel"] for h in head])), 3),
      "max": max(h["f0_root_lin_vel"] for h in head),
    },
    "f0_joint_vel_norm": {
      "median": round(float(np.median([h["f0_joint_vel_norm"] for h in head])), 3),
      "max": max(h["f0_joint_vel_norm"] for h in head),
      "steady_med_median": round(float(np.median(
        [h["steady_joint_vel_med"] for h in head])), 3),
    },
    "vel_consistency_worst_clip": max(
      clips, key=lambda c: c["vel_consistency"]["joint_central_diff_max_abs"]
    )["vel_consistency"],
    "vel_consistency_all_max_joint": max(
      c["vel_consistency"]["joint_central_diff_max_abs"] for c in clips),
    "clips": clips,
  }

  # FK check: source clip, ingested clip, native controls.
  model = mujoco.MjModel.from_xml_path(
    "/home/tcl/Desktop/start/bdx_rl_mjlab/robot_description/robots/bdx_v4/"
    "mjcf/bdx_v4.xml"
  )
  d0 = np.load(CLIPS[0], allow_pickle=True)
  joint_names = [str(x) for x in d0["joint_names"]]
  body_names = [str(x) for x in d0["body_names"]]
  jname2adr = {n: model.joint(n).qposadr[0] for n in joint_names}
  bname2id = {n: model.body(n).id for n in body_names}
  fk = {
    "src_wiggle0": fk_mismatch(CLIPS[0], model, jname2adr, bname2id,
                               body_names, joint_names),
    "src_wiggle1": fk_mismatch(CLIPS[1], model, jname2adr, bname2id,
                               body_names, joint_names),
    "native_laugh_big": fk_mismatch(NATIVE[0], model, jname2adr, bname2id,
                                    body_names, joint_names),
  }
  report["fk_mismatch_mm"] = fk
  Path(OUT).write_text(json.dumps(report, indent=2))
  print(json.dumps({k: v for k, v in report.items() if k != "clips"}, indent=2))
  print("saved", OUT)


if __name__ == "__main__":
  main()
