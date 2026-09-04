"""Closing demo: P2.9a planner in AUTOREGRESSIVE mode driven by a scripted
command timeline (no GT after the 1s seed):
  0-2s  vx=0.30 forward walk
  2-4s  + neck look-left 0.7 rad
  4-6s  vyaw=0.5 turn-in-place
  6-8s  vx=0 stop
Renders the rollout and prints realized per-window velocities (obedience log).

Run: BDX_PLANNER_DATA=.../bdx_planner_v2 .venv/bin/python \
     humanoidverse/scripts/demo_planner_v29.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import sys
from pathlib import Path

import mediapy as media
import mujoco
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, MotionData, WINDOW,
                                           canonical_sparse, load_stats,
                                           quat_wxyz_to_mat, sixd_to_mat)
from humanoidverse.planner.fk import BDXFK, rotmat_to_rotvec
from humanoidverse.planner.model import CommandedEncoder, MotionAE

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))
XML = Path(__file__).parents[1] / "data" / "robots" / "bdx" / "bdx_v4.xml"

# (name, seconds, vx, vy, vyaw, neck_yaw_cmd)
TIMELINE = [
    ("walk fwd",        2.0, 0.30, 0.0, 0.0, 0.0),
    ("fwd + look left", 2.0, 0.30, 0.0, 0.0, 0.7),
    ("turn in place",   2.0, 0.00, 0.0, 0.5, 0.0),
    ("stop",            2.0, 0.00, 0.0, 0.0, 0.0),
]


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from humanoidverse.planner.registry import resolve_canonical
    _reg = resolve_canonical(OUT_DIR)
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(_reg["teacher"], map_location=device,
                                       weights_only=False)["model"])
    model = CommandedEncoder(latent_dim=64, cmd_dim=7).to(device).eval()
    model.load_state_dict(torch.load(_reg["planner"], map_location=device,
                                     weights_only=False)["model"])
    print(f"canonical: {_reg['planner'].name}")
    stats = load_stats(OUT_DIR / "p1_stats.json")
    sp = {k: torch.tensor(v) for k, v in json.loads(
        (OUT_DIR / "p2_stats32.json").read_text()).items()}
    fk = BDXFK()
    md = MotionData()
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    spm, sps = sp["mean"].to(device), sp["std"].to(device)

    # seed: 1 s of GT walking (initial condition only)
    name = next(n for n in md.split_names["train"]
                if n.startswith("v1_walk/") and md.clip_cat[md.name2clip[n]] == "locomotion")
    ci = md.name2clip[name]
    seed_f = md.frames(ci, WINDOW, WINDOW)

    def frames_from_gen(pw, qw):
        bp = torch.from_numpy(pw[:, 0:3].astype(np.float32))
        R = quat_wxyz_to_mat(torch.from_numpy(pw[:, 3:7].astype(np.float32)))
        q = torch.from_numpy(qw.astype(np.float32))
        pos, Rall = fk.forward(bp, R.float(), q, return_rot=True)
        lv = torch.zeros_like(bp); lv[1:] = (bp[1:] - bp[:-1]) * 50
        rv = rotmat_to_rotvec(Rall[1:, 0] @ Rall[:-1, 0].transpose(-1, -2)) * 50
        av = torch.zeros_like(bp); av[1:] = rv
        FEET = [5, 10]
        yaw_f = np.arctan2(Rall[:, FEET].numpy()[..., 1, 0],
                           Rall[:, FEET].numpy()[..., 0, 0])
        fp = pos[:, FEET].numpy()
        return {"base_pos": bp.numpy(), "base_quat": pw[:, 3:7],
                "base_linvel": lv.numpy(), "base_angvel": av.numpy(),
                "head_rotvec": rotmat_to_rotvec(Rall[:, 12]).numpy(),
                "head_angvel": av_h.numpy(),
                "foot_pos": fp.astype(np.float32), "foot_yaw": yaw_f.astype(np.float32),
                "contact": np.zeros((len(pw), 2), np.float32),
                "head_pos": pos[:, 12].numpy(), "q": qw,
                "qdot": np.concatenate([np.zeros((1, 14), np.float32),
                                        np.diff(qw, axis=0) * 50]).astype(np.float32)}

    def plan(f_hist, cmd7, psi):
        sh = canonical_sparse(f_hist, psi)
        c = model(((sh.to(device) - spm) / sps).unsqueeze(0),
                  torch.zeros(1, WINDOW, sh.shape[-1], device=device),
                  cmd7.unsqueeze(0).to(device))
        return (teacher.decode(c)[0][0] * std + mean).cpu()

    seed = np.concatenate([seed_f["base_pos"], seed_f["base_quat"],
                           seed_f["q"]], -1)
    traj = seed.copy()                      # (t, 21)
    reports = []
    for seg_name, dur, vx, vy, vyaw, neck in TIMELINE:
        for _ in range(int(dur * 50) // WINDOW):
            h = traj[-WINDOW:]
            f_hist = frames_from_gen(h[:, :7], h[:, 7:])
            bq_a = torch.from_numpy(h[-1:, 3:7].astype(np.float32))
            psi = float(torch.atan2(
                2 * (bq_a[0, 0] * bq_a[0, 3] + bq_a[0, 1] * bq_a[0, 2]),
                1 - 2 * (bq_a[0, 2] ** 2 + bq_a[0, 3] ** 2)))
            cmd = torch.zeros(WINDOW, 7)
            cmd[:, 0] = torch.from_numpy(
                h[:, 7:][:, 10].astype(np.float32)) / 1.7    # neck_forward hold
            cmd[:, 1] = torch.from_numpy(
                h[:, 7:][:, 11].astype(np.float32)) / 1.7
            cmd[:, 2] = neck                                  # yaw command
            cmd[:, 3] = torch.from_numpy(
                h[:, 7:][:, 13].astype(np.float32)) / 1.7
            cmd[:, 4], cmd[:, 5], cmd[:, 6] = vx / 1.0, vy / 0.5, vyaw / 1.5
            p = plan(f_hist, cmd, psi)
            c_, s_ = np.cos(psi), np.sin(psi)
            Rz = np.array([[c_, s_, 0.], [-s_, c_, 0.], [0., 0., 1.]], np.float32)
            pos_w = p[:, 28:31].numpy() @ Rz + h[-1, 0:3]
            R_w = Rz.T[None] @ sixd_to_mat(p[:, 31:37]).numpy()
            quat_w = np.roll(sRot.from_matrix(R_w).as_quat(), 1, axis=-1)
            for i in range(1, len(quat_w)):
                if np.dot(quat_w[i], quat_w[i - 1]) < 0:
                    quat_w[i] = -quat_w[i]
            traj = np.concatenate(
                [traj, np.concatenate([pos_w, quat_w, p[:, 0:14].numpy()], -1)], 0)
            lv = p[:, 37:40].numpy().mean(0)
            R = sixd_to_mat(p[:, 31:37]).numpy()
            dyaw = np.degrees(np.arctan2(R[-1][1, 0], R[-1][0, 0])
                              - np.arctan2(R[0][1, 0], R[0][0, 0]))
            reports.append((seg_name, vx, vy, vyaw, neck,
                            float(lv[0]), dyaw,
                            float(p[:, 10:14].numpy()[:, 2].mean())))

    print(f"{'segment':<18} {'cmd vx/vyaw/neck':>22} {'realized vx/dyaw(deg)/neckyaw':>30}")
    for r in reports:
        print(f"{r[0]:<18} {r[1]:>7.2f} {r[3]:>6.2f} {r[4]:>6.2f}"
              f"{r[5]:>12.2f} {r[6]:>10.1f} {r[7]:>8.2f}")

    model_mj = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model_mj)
    renderer = mujoco.Renderer(model_mj, height=360, width=480)
    cam = mujoco.MjvCamera()
    frames = []
    for t in range(0, len(traj), 2):          # 25 fps playback
        cam.lookat[:] = [traj[t, 0], traj[t, 1], 0.22]
        cam.distance, cam.elevation, cam.azimuth = 1.7, -12, -90
        data.qpos[:] = traj[t]
        mujoco.mj_forward(model_mj, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    out = OUT_DIR / "demo_planner_v29.mp4"
    media.write_video(str(out), frames, fps=25)
    print(f"demo -> {out}")


if __name__ == "__main__":
    main()
