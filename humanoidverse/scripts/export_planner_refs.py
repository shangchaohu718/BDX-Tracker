"""P3: export PLANNER-generated reference pkls for the tracker end-to-end test.

Two modes (--planner):
  tf : teacher-forced — history from GT frames, command from GT window
  ar : autoregressive — history from the planner's OWN previous output
       (real deployment condition; tests compounding drift), hop=WINDOW

Motion = P2.9a CommandedEncoder(cmd_dim 7, future sparse masked) -> frozen
teacher decoder. tf stitching uses trapezoid crossfade; ar appends windows.

Run: BDX_PLANNER_DATA=.../bdx_planner_v2 .venv/bin/python \
     humanoidverse/scripts/export_planner_refs.py --planner tf
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, MotionData, WINDOW,
                                           canonical_sparse, load_stats,
                                           quat_wxyz_to_mat, sixd_to_mat)
from humanoidverse.planner.fk import BDXFK, HEAD, LFOOT, RFOOT, rotmat_to_rotvec
from humanoidverse.planner.model import CommandedEncoder, MotionAE

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))
FEET = [LFOOT, RFOOT]


def yaw_of_R(R):
    """(...,3,3) rotation matrices -> yaw (about z)."""
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["tf", "ar"], default="tf")
    ap.add_argument("--checkpoint", default=None, help="default: canonical registry")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from humanoidverse.planner.registry import resolve_canonical
    _reg = resolve_canonical(OUT_DIR)
    print(f"canonical: {_reg['planner'].name} + {_reg['teacher'].name} (preprocess {_reg['registry'].get('preprocess_version')})")
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(_reg["teacher"], map_location=device,
                                       weights_only=False)["model"])
    model = CommandedEncoder(latent_dim=64, cmd_dim=7).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=False)["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    sp = {k: torch.tensor(v) for k, v in json.loads(
        (OUT_DIR / "p2_stats32.json").read_text()).items()}
    fk = BDXFK()
    md = MotionData()

    mean, std = stats["mean"].to(device), stats["std"].to(device)
    spm, sps = sp["mean"].to(device), sp["std"].to(device)

    meta = json.loads((OUT_DIR / "meta.json").read_text())["clips"]
    val = set(md.split_names["val"])
    picks = [("locomotion", "v1_walk"), ("locomotion", "v1_perturb"),
             ("locomotion", "v1_augment"), ("stand", "v1_stand"),
             ("stand", "v0"), ("walk_head", "v1_walkhead"),
             ("walk_head", "v1_perturb")]
    names = []
    for cat, ver in picks:
        c = sorted(x["name"] for x in meta if x["category"] == cat
                   and x["version"] == ver and x["name"] in val
                   and x["frames"] >= 150)
        if c:
            names.append(c[0])

    def plan(f_hist, cmd_neck, loco, psi):
        sh = canonical_sparse(f_hist, psi)
        cmd = torch.cat([cmd_neck, loco[None].expand(WINDOW, -1)], -1).float()
        c = model(((sh.to(device) - spm) / sps).unsqueeze(0),
                  torch.zeros(1, WINDOW, sh.shape[-1], device=device),
                  cmd.unsqueeze(0).to(device))
        return (teacher.decode(c)[0][0] * std + mean).cpu()

    def frames_from_gen(pw, qw):
        """world pos+quat(7) and q -> sparse-able frames dict via FK"""
        bp = torch.from_numpy(pw[:, 0:3].astype(np.float32))
        R = quat_wxyz_to_mat(torch.from_numpy(pw[:, 3:7].astype(np.float32)))
        q = torch.from_numpy(qw.astype(np.float32))
        pos, Rall = fk.forward(bp, R.float(), q, return_rot=True)
        lv = torch.zeros_like(bp)
        lv[1:] = (bp[1:] - bp[:-1]) * 50
        rv = rotmat_to_rotvec(Rall[1:, 0] @ Rall[:-1, 0].transpose(-1, -2)) * 50
        av = torch.zeros_like(bp)
        av[1:] = rv
        # head angular velocity from the HEAD body's rotation sequence (was
        # base av — wrong body, polluted AR history head tokens)
        rv_h = rotmat_to_rotvec(Rall[1:, 12] @ Rall[:-1, 12].transpose(-1, -2)) * 50
        av_h = torch.zeros_like(bp)
        av_h[1:] = rv_h
        fp = pos[:, FEET].numpy()
        fs = np.concatenate([np.zeros((1, 2)), np.linalg.norm(
            (fp[1:] - fp[:-1]) * 50, axis=-1)], 0)   # (W,2)
        contact = (fs < 0.3) & (fp[..., 2] < 0.08)   # (W,2)
        return {"base_pos": bp.numpy(), "base_quat": pw[:, 3:7],
                "base_linvel": lv.numpy(), "base_angvel": av.numpy(),
                "head_rotvec": rotmat_to_rotvec(Rall[:, HEAD]).numpy(),
                "head_angvel": av_h.numpy(),
                "foot_pos": fp.astype(np.float32),
                "foot_yaw": yaw_of_R(
                    Rall[:, FEET].numpy()).astype(np.float32),
                "contact": contact.astype(np.float32),
                "head_pos": pos[:, HEAD].numpy(),
                "q": qw, "qdot": np.concatenate(
                    [np.zeros((1, 14), np.float32),
                     np.diff(qw, axis=0) * 50]).astype(np.float32)}

    def to_world(p, origin, psi):
        c_, s_ = np.cos(psi), np.sin(psi)
        Rz = np.array([[c_, s_, 0.], [-s_, c_, 0.], [0., 0., 1.]], np.float32)
        pos_w = p[:, 28:31].numpy() @ Rz + origin
        R_w = Rz.T[None] @ sixd_to_mat(p[:, 31:37]).numpy()
        quat_w = np.roll(sRot.from_matrix(R_w).as_quat(), 1, axis=-1)
        for i in range(1, len(quat_w)):
            if np.dot(quat_w[i], quat_w[i - 1]) < 0:
                quat_w[i] = -quat_w[i]
        return np.concatenate([pos_w, quat_w], -1), p[:, 0:14].numpy()

    out = {}
    for name in names:
        ci = md.name2clip[name]
        T = int(md.d["lengths"][ci])
        f_all = md.frames(ci, 0, T)
        hop = WINDOW if args.planner == "ar" else WINDOW // 2
        starts = list(range(WINDOW, T - WINDOW + 1, hop))
        if starts[-1] != T - WINDOW:
            starts.append(T - WINDOW)
        q_acc = np.zeros((T, 14))
        p_acc = np.zeros((T, 7))
        w_acc = np.zeros(T)
        gen_traj = None            # world (t,7)+(t,14), ar only
        for s in starts:
            f_fut = md.frames(ci, s, WINDOW)
            bq0 = f_fut["base_quat"][0].astype(np.float32)
            psi0 = float(np.arctan2(2 * (bq0[0] * bq0[3] + bq0[1] * bq0[2]),
                                    1 - 2 * (bq0[2] ** 2 + bq0[3] ** 2)))
            cmd_neck = torch.from_numpy(
                f_fut["q"][:, 10:14].astype(np.float32)) / 1.7
            lv = f_fut["base_linvel"].astype(np.float32)
            av = f_fut["base_angvel"].astype(np.float32)
            c_, s_ = np.cos(psi0), np.sin(psi0)
            Rzf = np.array([[c_, s_, 0.], [-s_, c_, 0.], [0., 0., 1.]], np.float32)
            loco = torch.tensor([(lv @ Rzf.T)[:, 0].mean() / 1.0,
                                 (lv @ Rzf.T)[:, 1].mean() / 0.5,
                                 (av @ Rzf.T)[:, 2].mean() / 1.5])
            if args.planner == "ar" and gen_traj is not None:
                h_pw, h_q = gen_traj[-WINDOW:, :7], gen_traj[-WINDOW:, 7:]
                f_hist = frames_from_gen(h_pw, h_q)
                bq_a = torch.from_numpy(h_pw[-1:, 3:7].astype(np.float32))
                psi_hist = float(torch.atan2(
                    2 * (bq_a[0, 0] * bq_a[0, 3] + bq_a[0, 1] * bq_a[0, 2]),
                    1 - 2 * (bq_a[0, 2] ** 2 + bq_a[0, 3] ** 2)))
                p = plan(f_hist, cmd_neck, loco, psi_hist)
                pw, qw = to_world(p, h_pw[-1, 0:3], psi_hist)
                gen_traj = np.concatenate([gen_traj,
                                           np.concatenate([pw, qw], -1)], 0)
            else:
                f_hist = md.frames(ci, s - WINDOW, WINDOW)
                p = plan(f_hist, cmd_neck, loco, psi0)
                pw, qw = to_world(p, f_fut["base_pos"][0], psi0)
                if args.planner == "ar":
                    gen_traj = np.concatenate(
                        [np.concatenate([f_all["base_pos"][:WINDOW],
                                         f_all["base_quat"][:WINDOW],
                                         f_all["q"][:WINDOW]], -1),
                         np.concatenate([pw, qw], -1)], 0)
            ramp = np.ones(WINDOW)
            if args.planner == "tf":
                if s != starts[0]:
                    ramp[:hop] = np.linspace(0, 1, hop)
                if s != starts[-1]:
                    ramp[-hop:] = np.linspace(1, 0, hop)
            q_acc[s:s + WINDOW] += ramp[:, None] * qw
            p_acc[s:s + WINDOW] += ramp[:, None] * pw
            w_acc[s:s + WINDOW] += ramp
        # frames before the first window have no coverage: seed with the GT
        # initial condition (history warmup — the planner can't retro-plan it)
        q_acc[:WINDOW] = f_all["q"][:WINDOW]
        p_acc[:WINDOW, 0:3] = f_all["base_pos"][:WINDOW]
        p_acc[:WINDOW, 3:7] = f_all["base_quat"][:WINDOW]
        w_acc[:WINDOW] = 1.0
        w = np.maximum(w_acc, 1e-6)[:, None]
        q_w = (q_acc / w).astype(np.float32)
        pos_w = (p_acc[:, 0:3] / w).astype(np.float32)
        quat_w = (p_acc[:, 3:7] / w).astype(np.float32)
        quat_w /= np.linalg.norm(quat_w, axis=-1, keepdims=True)
        pose_aa = np.zeros((T, 17, 3), np.float32)
        pose_aa[:, 0] = sRot.from_quat(np.roll(quat_w, -1, axis=-1)).as_rotvec()
        for b in range(1, 17):
            if fk.qadr[b] >= 0:
                pose_aa[:, b] = fk.axis[b] * (
                    q_w[:, fk.qadr[b]] - fk.ref[b])[:, None]
        dq = np.abs(q_w[2 * WINDOW:] - f_all["q"][2 * WINDOW:]).mean()
        out[name] = {"root_trans_offset": pos_w, "pose_aa": pose_aa,
                     "dof": q_w, "root_rot": quat_w, "fps": 50}
        print(f"✓ {name}: {T} fr, planner({args.planner}) mean |Δq| "
              f"{dq*1000:.1f} mrad (vs GT, after 2s warmup)", flush=True)

    joblib.dump(out, OUT_DIR / f"tracker_planner_{args.planner}_refs.pkl",
                compress=3)
    print(f"\n✓ {len(out)} clips -> tracker_planner_{args.planner}_refs.pkl")


if __name__ == "__main__":
    main()
