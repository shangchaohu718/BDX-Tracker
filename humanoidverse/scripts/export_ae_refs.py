"""P1.5: export GT and AE-reconstructed reference pkls for the tracker A/B test.

Picks val-split clips (category-stratified), reconstructs each clip through the
P1 autoencoder window-by-window (stride 50 + tail window), inverse-transforms
the canonical window frame back to world, and writes TWO motion pkls in the
humanoidverse schema ({name: {root_trans_offset, pose_aa, dof, root_rot, fps}}):

  tracker_gt_refs.pkl   — entries copied verbatim from bdx_14dof_dataset.pkl
  tracker_ae_refs.pkl   — AE-reconstructed dof/root (pose_aa rebuilt from them;
                          construction verified to 2e-7 vs stored pose_aa)

Run: .venv/bin/python humanoidverse/scripts/export_ae_refs.py [--checkpoint ...]
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
                                           load_stats, quat_wxyz_to_mat,
                                           sixd_to_mat)
from humanoidverse.planner.fk import BDXFK
from humanoidverse.planner.model import MotionAE

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))
SRC_PKL = Path(__file__).parents[1] / "data" / "bdx_14dof_dataset.pkl"

# (category, version) preference, one clip each — covers walk / perturb /
# augment / stand / head / gesture
PICKS = [("locomotion", "v1_walk"), ("locomotion", "v1_perturb"),
         ("locomotion", "v1_augment"), ("stand", "v1_stand"),
         ("stand", "v0"), ("walk_head", "v1_walkhead"),
         ("walk_head", "v1_perturb"), ("gesture", "v0")]
MIN_FRAMES = 150


def pick_clips(md):
    meta = json.loads((OUT_DIR / "meta.json").read_text())["clips"]
    val = set(md.split_names["val"])
    picked = []
    for cat, ver in PICKS:
        cands = sorted(c["name"] for c in meta
                       if c["category"] == cat and c["version"] == ver
                       and c["name"] in val and c["frames"] >= MIN_FRAMES)
        if cands:
            picked.append(cands[0])
    return picked


def mat_to_quat_wxyz(R):
    q = sRot.from_matrix(R.cpu().numpy()).as_quat()  # xyzw
    return np.roll(q, 1, axis=-1)                    # -> wxyz


@torch.no_grad()
def reconstruct_clip(model, md, stats, fk, clip_idx, device, crossfade=True):
    T = int(md.d["lengths"][clip_idx])
    hop = WINDOW // 2 if crossfade else WINDOW
    starts = list(range(0, T - WINDOW + 1, hop))
    if starts[-1] != T - WINDOW:
        starts.append(T - WINDOW)
    f_all = md.frames(clip_idx, 0, T)
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    xs, origins, yaws = [], [], []
    for s in starts:
        f = {k: v[s:s + WINDOW] for k, v in f_all.items()}
        bq = torch.from_numpy(f["base_quat"].astype(np.float32))
        psi = float(torch.atan2(2 * (bq[0, 0] * bq[0, 3] + bq[0, 1] * bq[0, 2]),
                                1 - 2 * (bq[0, 2] ** 2 + bq[0, 3] ** 2)))
        c, sn = float(np.cos(psi)), float(np.sin(psi))
        Rz = torch.tensor([[c, sn, 0.], [-sn, c, 0.], [0., 0., 1.]],
                          dtype=torch.float32)

        q = torch.from_numpy(f["q"].astype(np.float32))
        bp = torch.from_numpy(f["base_pos"].astype(np.float32))
        blv = torch.from_numpy(f["base_linvel"].astype(np.float32))
        bav = torch.from_numpy(f["base_angvel"].astype(np.float32))
        rot = lambda v: v @ Rz.T
        R_c = Rz.unsqueeze(0) @ quat_wxyz_to_mat(bq)
        from humanoidverse.planner.dataset import mat_to_6d
        x = torch.cat([q, torch.from_numpy(f["qdot"].astype(np.float32)),
                       rot(bp - bp[0]), mat_to_6d(R_c), rot(blv), rot(bav)], -1)
        xs.append((x - mean.cpu()) / std.cpu())
        origins.append(bp[0].numpy().copy())
        yaws.append(Rz.numpy().copy())

    c = model.encode(torch.stack(xs).to(device))
    pred, _ = model.decode(c)
    pred = (pred * std + mean).cpu()
    # crossfade accumulation: trapezoid weights so consecutive windows blend
    # over their overlap (stride-50 stitching produced 2-4x Δq jumps at
    # boundaries that the tracker punished — see P1.5 notes)
    q_acc = np.zeros((T, 14), np.float64)
    pos_acc = np.zeros((T, 3), np.float64)
    R_acc = np.zeros((T, 3, 3), np.float64)
    w_acc = np.zeros(T, np.float64)
    for i, s in enumerate(starts):
        ramp = np.ones(WINDOW, np.float64)
        if crossfade:
            if i > 0:
                ramp[:hop] = np.linspace(0, 1, hop)
            if i < len(starts) - 1:
                ramp[-hop:] = np.linspace(1, 0, hop)
        n = min(WINDOW, T - s)
        q_p = pred[i, :n, 0:14].numpy()
        pos_c = pred[i, :n, 28:31].numpy()
        R_c = sixd_to_mat(pred[i, :n, 31:37]).numpy()
        Rzp = yaws[i].T                       # Rz(+psi)
        q_acc[s:s + n] += ramp[:n, None] * q_p
        pos_acc[s:s + n] += ramp[:n, None] * (pos_c @ Rzp.T + origins[i])
        R_acc[s:s + n] += ramp[:n, None, None] * (Rzp[None] @ R_c)
        w_acc[s:s + n] += ramp[:n]
    q_w = (q_acc / w_acc[:, None]).astype(np.float32)
    pos_w = (pos_acc / w_acc[:, None]).astype(np.float32)
    R_blend = torch.from_numpy(R_acc / w_acc[:, None, None]).float()
    U, _, Vh = torch.linalg.svd(R_blend)      # project to SO(3)
    R_w = (U @ Vh).numpy().astype(np.float32)

    quat_w = mat_to_quat_wxyz(torch.from_numpy(R_w))
    for i in range(1, len(quat_w)):          # hemisphere continuity
        if np.dot(quat_w[i], quat_w[i - 1]) < 0:
            quat_w[i] = -quat_w[i]

    pose_aa = np.zeros((T, 17, 3), np.float32)
    pose_aa[:, 0] = sRot.from_quat(np.roll(quat_w, -1, axis=-1)).as_rotvec()
    for b in range(1, 17):
        if fk.qadr[b] >= 0:
            pose_aa[:, b] = fk.axis[b] * (q_w[:, fk.qadr[b]] - fk.ref[b])[:, None]
    return {"root_trans_offset": pos_w, "pose_aa": pose_aa, "dof": q_w,
            "root_rot": quat_w.astype(np.float32), "fps": 50}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="default: canonical teacher")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint is None:
        from humanoidverse.planner.registry import resolve_canonical
        args.checkpoint = str(resolve_canonical(OUT_DIR)["teacher"])
    print(f"canonical teacher: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MotionAE(in_dim=MODEL_DIM, latent_dim=ckpt["config"]["latent"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    stats = load_stats(OUT_DIR / "p1_stats.json")
    fk = BDXFK()
    md = MotionData()

    names = pick_clips(md)
    def build_gt_entry(clip_idx):
        """GT entry straight from the npz fields (v2 clips are not in the
        1154-clip pkl); pose_aa via the construction verified to 2e-7."""
        T = int(md.d["lengths"][clip_idx])
        f = md.frames(clip_idx, 0, T)
        quat = f["base_quat"].astype(np.float32).copy()
        for i in range(1, T):          # hemisphere continuity
            if np.dot(quat[i], quat[i - 1]) < 0:
                quat[i] = -quat[i]
        pose_aa = np.zeros((T, 17, 3), np.float32)
        pose_aa[:, 0] = sRot.from_quat(
            np.roll(f["base_quat"].astype(np.float32), -1, axis=-1)).as_rotvec()
        for b in range(1, 17):
            if fk.qadr[b] >= 0:
                pose_aa[:, b] = fk.axis[b] * (
                    f["q"][:, fk.qadr[b]] - fk.ref[b])[:, None]
        return {"root_trans_offset": f["base_pos"].astype(np.float32),
                "pose_aa": pose_aa, "dof": f["q"].astype(np.float32),
                "root_rot": quat, "fps": 50}

    src = None
    gt_out, ae_out = {}, {}
    for name in names:
        ci = md.name2clip[name]
        gt = build_gt_entry(ci)
        gt_out[name] = gt
        ae_out[name] = reconstruct_clip(model, md, stats, fk, ci, device)
        dq = np.abs(ae_out[name]["dof"] - gt["dof"]).mean()
        dp = np.abs(ae_out[name]["root_trans_offset"]
                    - gt["root_trans_offset"]).max()
        print(f"✓ {name}: {gt['dof'].shape[0]} fr, "
              f"mean |Δq| {dq*1000:.1f} mrad, max |Δroot| {dp*1000:.1f} mm")

    joblib.dump(gt_out, OUT_DIR / "tracker_gt_refs.pkl", compress=3)
    joblib.dump(ae_out, OUT_DIR / f"tracker_ae_refs{args.suffix}.pkl", compress=3)
    print(f"\n✓ {len(names)} clips -> tracker_gt_refs.pkl / "
          f"tracker_ae_refs{args.suffix}.pkl")


if __name__ == "__main__":
    main()
