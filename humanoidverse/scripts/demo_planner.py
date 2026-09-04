"""Planner demo: sparse command -> P2 student -> full 14-DoF motion -> video.

Renders three panels side by side for a 1 s window:
  [ GT motion | sparse-completed motion | completed with edited head command ]

The head edit rewrites the FUTURE sparse head fields (rotvec -> constant
look-left target, angvel -> 0) while keeping the true history — the completed
neck motion should follow the command. Rendering uses MuJoCo offscreen
directly on bdx_v4.xml (no humanoidverse env stack involved).

Run: .venv/bin/python humanoidverse/scripts/demo_planner.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import sys
from pathlib import Path

import mediapy as media
import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, MotionData, WINDOW,
                                           canonical_sparse, load_stats,
                                           sixd_to_mat)
from humanoidverse.planner.model import MotionAE

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))
XML = Path(__file__).parents[1] / "data" / "robots" / "bdx" / "bdx_v4.xml"
HEAD_LOOK_LEFT_YAW = 0.7  # rad, canonical frame


def quat_wxyz_from_R(R):
    from scipy.spatial.transform import Rotation as sRot
    return np.roll(sRot.from_matrix(R).as_quat(), 1, axis=-1)


@torch.no_grad()
def complete(student, teacher, stats, sp_stats, sp_fut, sp_hist, device, cmd):
    def norm(x):
        return ((x.to(device) - sp_stats["mean"].to(device))
                / sp_stats["std"].to(device))

    c = student(norm(sp_hist).unsqueeze(0).to(device),
                norm(sp_fut).unsqueeze(0).to(device), cmd.unsqueeze(0).to(device))
    p = teacher.decode(c)[0][0] * stats["std"].to(device) + stats["mean"].to(device)
    return p.cpu(), c.cpu()[0]


def render_rollout(model, data, renderer, cam_pos, qpos_traj):
    frames = []
    for qpos in qpos_traj:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        frames.append(renderer.render())
    return frames


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from humanoidverse.planner.registry import resolve_canonical
    _reg = resolve_canonical(OUT_DIR)
    print(f"canonical: {_reg['planner'].name} + {_reg['teacher'].name}")
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(_reg["teacher"], map_location=device,
                                       weights_only=False)["model"])
    from humanoidverse.planner.model import CommandedEncoder
    sck = torch.load(_reg["planner"], map_location=device, weights_only=False)
    cmd_dim = 7 if sck["config"].get("loco_cmd") else 4
    student = CommandedEncoder(latent_dim=sck["config"]["latent"],
                               cmd_dim=cmd_dim).to(device).eval()
    student.load_state_dict(sck["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    sp_stats = {k: torch.tensor(v) for k, v in json.loads(
        (OUT_DIR / "p2_stats32.json").read_text()).items()}
    md = MotionData()

    # pick a val locomotion window with clear forward motion
    ds_windows = [(md.name2clip[n], s) for n in md.split_names["val"]
                  for s in [int(md.d["lengths"][md.name2clip[n]]) // 2]
                  if md.clip_cat[md.name2clip[n]] == "locomotion"
                  and int(md.d["lengths"][md.name2clip[n]]) > 2 * WINDOW]
    clip, s = ds_windows[0]
    name = md.clip_names[clip]
    print(f"demo window: {name} start {s} ({md.clip_cat[clip]})")

    f_fut = md.frames(clip, s, WINDOW)
    f_hist = md.frames(clip, s - WINDOW, WINDOW)
    bq0 = f_fut["base_quat"][0].astype(np.float32)
    psi = float(np.arctan2(2 * (bq0[0] * bq0[3] + bq0[1] * bq0[2]),
                           1 - 2 * (bq0[2] ** 2 + bq0[3] ** 2)))
    sp_fut = canonical_sparse(f_fut, psi)
    sp_hist = canonical_sparse(f_hist, psi)

    neck_cmd = torch.from_numpy(f_fut["q"][:, 10:14].astype(np.float32)) / 1.7
    if cmd_dim == 7:
        # v11/v12 lineage: 7D cmd = neck(4)/1.7 + loco(3) from the future
        # window's canonical-frame mean velocities, scales (1.0, 0.5, 1.5)
        # (dataset contract, P2Dataset.__getitem__)
        c2, sn2 = float(np.cos(psi)), float(np.sin(psi))
        Rzf = np.array([[c2, sn2, 0.], [-sn2, c2, 0.], [0., 0., 1.]], np.float32)
        lv_c = f_fut["base_linvel"].astype(np.float32) @ Rzf.T
        av_c = f_fut["base_angvel"].astype(np.float32) @ Rzf.T
        loco = torch.tensor([lv_c[:, 0].mean() / 1.0, lv_c[:, 1].mean() / 0.5,
                             av_c[:, 2].mean() / 1.5], dtype=torch.float32)
        cmd0 = torch.cat([neck_cmd, loco[None, :].expand(WINDOW, 3)], -1)
    else:
        cmd0 = neck_cmd
    p_base, _ = complete(student, teacher, stats, sp_stats, sp_fut, sp_hist, device, cmd0)
    cmd_edit = cmd0.clone()
    cmd_edit[:, 2] = HEAD_LOOK_LEFT_YAW   # explicit neck_yaw command (P2.8)
    p_edit, _ = complete(student, teacher, stats, sp_stats, sp_fut, sp_hist, device, cmd_edit)

    def to_world_qpos(p):
        c, sn = np.cos(psi), np.sin(psi)
        Rz = np.array([[c, sn, 0.], [-sn, c, 0.], [0., 0., 1.]])
        pos_w = p[:, 28:31].numpy() @ Rz + f_fut["base_pos"][0]
        R_w = Rz.T[None] @ sixd_to_mat(p[:, 31:37]).numpy()
        quat_w = quat_wxyz_from_R(R_w)
        for i in range(1, len(quat_w)):
            if np.dot(quat_w[i], quat_w[i - 1]) < 0:
                quat_w[i] = -quat_w[i]
        qpos = np.zeros((len(p), 21), np.float64)
        qpos[:, 0:3], qpos[:, 3:7] = pos_w, quat_w
        qpos[:, 7:] = p[:, 0:14].numpy()
        return qpos

    qp_gt = np.zeros((WINDOW, 21))
    qp_gt[:, 0:3] = f_fut["base_pos"]
    qp_gt[:, 3:7] = f_fut["base_quat"]
    qp_gt[:, 7:] = f_fut["q"]
    qp_base, qp_edit = to_world_qpos(p_base), to_world_qpos(p_edit)

    neck = dict(base=p_base[:, 10:14].numpy(), edit=p_edit[:, 10:14].numpy())
    print("neck response to look-left cmd (rad, mean over window):")
    print(f"  baseline yaw {neck['base'][:, 2].mean():+.3f}  "
          f"edited yaw {neck['edit'][:, 2].mean():+.3f}  "
          f"forward {neck['base'][:, 0].mean():+.3f} -> {neck['edit'][:, 0].mean():+.3f}")

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=320, width=320)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [qp_gt[0, 0] + 0.15, qp_gt[0, 1], 0.22]
    cam.distance, cam.elevation, cam.azimuth = 1.6, -12, -90
    renderer.update_scene(data, camera=cam)  # prime scene
    frames = []
    for i in range(WINDOW):
        row = []
        for qp in (qp_gt, qp_base, qp_edit):
            data.qpos[:] = qp[i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            row.append(renderer.render())
        frames.append(np.concatenate(row, axis=1))
    out = OUT_DIR / "demo_planner.mp4"
    media.write_video(str(out), frames, fps=25)
    print(f"demo video -> {out}  ([ GT | sparse completion | completion+head-cmd ])")
    np.savez(OUT_DIR / "demo_planner.npz", qp_gt=qp_gt, qp_base=qp_base,
             qp_edit=qp_edit, clip=name, start=s)


if __name__ == "__main__":
    main()
