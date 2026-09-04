"""Continuous difficulty scoring + static audit for the trimmed pool (50 clips).

Philosophy per 2026-08-26 ruling: PD rollout failure is NOT a binary gate on
learnability (the RL policy may track what PD cannot) — every metric here is
a CONTINUOUS feature for curriculum ordering / sampling weights / bucketing.
Hard rejection is reserved for numeric corruption, penetration, or severe
limit violations, which the static section reports.

Static (numpy + mujoco FK on bdx_v4.xml):
  quat norm dev, joint-limit margin (mm of rad to nearest limit), foot-geom
  penetration (min z of *_foot_collision geoms over frames), joint
  vel/acc/jerk max, root XY range, root yaw drift, FK consistency vs
  body_pos_w, head-transient strength (from trim provenance).

Dynamic (reference-PD rollout, gesture env built from cfg factory, CPU):
  3 RSI episodes x 300 steps; survival steps (termination OR height fall
  z<0.12), min root z, anchor-orientation error P95/max, joint-error mean,
  |action|>1 fraction (effort-saturation proxy).

Output: pool_v1/difficulty_index.json (append-only).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import mujoco
import numpy as np

POOL = sorted(glob.glob(
  "/home/tcl/Desktop/start/BFM-zero/results/mjlab_bdx/pool_v1/*.trim*.npz"))
XML = ("/home/tcl/Desktop/start/bdx_rl_mjlab/robot_description/robots/"
       "bdx_v4/mjcf/bdx_v4.xml")
OUT = Path("/home/tcl/Desktop/start/BFM-zero/results/mjlab_bdx/pool_v1/"
           "difficulty_index.json")
FALL_Z = 0.12
EPISODES = 3
STEPS = 300


def yaw_of(q_wxyz):
  w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def static_audit(path: str, model, jadr, foot_geoms) -> dict:
  d = np.load(path, allow_pickle=True)
  jp, jv = d["joint_pos"], d["joint_vel"]
  bp, bq = d["body_pos_w"], d["body_quat_w"]
  T = jp.shape[0]
  qn = float(np.abs(np.linalg.norm(bq, axis=-1) - 1).max())

  lo = np.array([model.joint(n).range[0] for n in
                 (str(x) for x in d["joint_names"])])
  hi = np.array([model.joint(n).range[1] for n in
                 (str(x) for x in d["joint_names"])])
  margin = float(min((jp - lo).min(), (hi - jp).min()))

  data = mujoco.MjData(model)
  qpos = np.zeros(model.nq)
  jn = [str(x) for x in d["joint_names"]]
  min_foot_z, fk_max = 1e9, 0.0
  for t in range(T):
    qpos[:3] = bp[t, 0]
    qpos[3:7] = bq[t, 0]
    for i, n in enumerate(jn):
      qpos[jadr[n]] = jp[t, i]
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    for g in foot_geoms:
      min_foot_z = min(min_foot_z, float(data.geom(g).xpos[2]))
    bn = [str(x) for x in d["body_names"]]
    for bi, n in enumerate(bn):
      fk_max = max(fk_max, float(np.linalg.norm(data.body(n).xpos - bp[t, bi])))

  acc = np.diff(jv, axis=0) * 50.0
  jerk = np.diff(acc, axis=0) * 50.0
  root_yaw = yaw_of(bq[:, 0, :])
  return {
    "frames": T,
    "quat_norm_max_dev": qn,
    "joint_limit_margin_rad": round(margin, 4),
    "min_foot_geom_z_mm": round(min_foot_z * 1000, 2),
    "fk_max_mm": round(fk_max * 1000, 3),
    "joint_vel_norm_max": round(float(np.linalg.norm(jv, axis=-1).max()), 2),
    "joint_acc_norm_max": round(float(np.linalg.norm(acc, axis=-1).max()), 2),
    "joint_jerk_norm_max": round(float(np.linalg.norm(jerk, axis=-1).max()), 1),
    "root_xy_range_m": [round(float(np.ptp(bp[:, 0, 0])), 3),
                        round(float(np.ptp(bp[:, 0, 1])), 3)],
    "root_yaw_drift_rad": round(float(np.abs(np.unwrap(root_yaw)
                                             - root_yaw[0]).max()), 3),
    "mean_root_z": round(float(bp[:, 0, 2].mean()), 3),
  }


def pd_rollout(path: str, duration: float) -> dict:
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from tracking_bdx.config.bdx_v4.env_cfgs import bdx_v4_gesture_tracking_env_cfg
  cfg = bdx_v4_gesture_tracking_env_cfg(path, duration, play=True)
  cfg.scene.num_envs = 1
  cfg.commands["motion"].sampling_mode = "uniform"  # RSI random starts
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  motion = env.command_manager.get_term("motion")
  robot = env.scene["robot"]
  eps = []
  for _ in range(EPISODES):
    env.reset()
    surv, min_z, fell = 0, 1e9, False
    a_ori, j_err, big = [], [], 0
    for t in range(STEPS):
      a = ((motion.joint_pos[0] - robot.data.default_joint_pos[0]) / 0.25)[None]
      z = float(robot.data.root_link_pose_w[0, 2])
      min_z = min(min_z, z)
      big += int((np.abs(a[0].cpu().numpy()) > 1).any())
      env.step(a)
      surv += 1
      a_ori.append(float(motion.metrics["error_anchor_rot"][0]))
      j_err.append(float(motion.metrics["error_joint_pos"][0]))
      if z < FALL_Z:
        fell = True
        break
    eps.append({"survived": surv, "min_root_z": round(min_z, 3),
                "fall": fell,
                "anchor_ori_p95": round(float(np.percentile(a_ori, 95)), 3),
                "anchor_ori_max": round(float(max(a_ori)), 3),
                "joint_err_mean": round(float(np.mean(j_err)), 3),
                "action_gt1_frac": round(big / STEPS, 3)})
  env.close()
  return {
    "survival_ratio": round(sum(e["survived"] for e in eps)
                            / (EPISODES * STEPS), 3),
    "falls": sum(e["fall"] for e in eps),
    "min_root_z_worst": round(min(e["min_root_z"] for e in eps), 3),
    "anchor_ori_p95_max": round(max(e["anchor_ori_p95"] for e in eps), 3),
    "anchor_ori_max": round(max(e["anchor_ori_max"] for e in eps), 3),
    "joint_err_mean_avg": round(float(np.mean(
      [e["joint_err_mean"] for e in eps])), 3),
    "action_gt1_frac_max": round(max(e["action_gt1_frac"] for e in eps), 3),
    "episodes": eps,
  }


def main():
  model = mujoco.MjModel.from_xml_path(XML)
  d0 = np.load(POOL[0], allow_pickle=True)
  jadr = {str(n): model.joint(str(n)).qposadr[0] for n in d0["joint_names"]}
  foot_geoms = [model.geom(n).id for n in
                ("left_foot_collision", "right_foot_collision")]
  trim_idx = {Path(e["src"]).stem: e for e in json.load(open(
    "/home/tcl/Desktop/start/BFM-zero/results/mjlab_bdx/pool_v1/"
    "trim_index.json"))}
  out = []
  for p in POOL:
    stem = Path(p).name.split(".trim")[0]
    fps = float(np.atleast_1d(np.load(p)["fps"])[0])
    T = np.load(p)["joint_pos"].shape[0]
    s = static_audit(p, model, jadr, foot_geoms)
    dyn = pd_rollout(p, T / fps)
    tr = trim_idx.get(stem, {})
    entry = {"clip": Path(p).name, "src": stem,
             "trim_frames": tr.get("trim_frames"),
             "head_transient": tr.get("f0_values"),
             **s, **dyn}
    out.append(entry)
    print(f"[done] {Path(p).name}: surv={dyn['survival_ratio']} "
          f"falls={dyn['falls']} footz={s['min_foot_geom_z_mm']}mm "
          f"margin={s['joint_limit_margin_rad']}", flush=True)
  assert not OUT.exists(), f"refusing overwrite {OUT}"
  OUT.write_text(json.dumps(out, indent=2))
  print("saved", OUT)


if __name__ == "__main__":
  main()
