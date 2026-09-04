"""P3.1.1 pre-fix baseline probe for the backward line (user ruling 2026-09-04).

Records, for (a) the 5 v5 backward-commanded branches whose GT realizes
forward, and (b) a sample of clean `_rev` clips from the PoC v2 pool:
  - root displacement (canonical x) per window
  - realized vx (canonical position finite-diff)
  - frozen teacher (p1_ae_l64, fixed_v1) encode->decode q reconstruction error
Teacher walk-class reference: val locomotion q ~ 20.1 mrad (P0.1 eval_p2).

Run: BDX_PLANNER_DATA=<v2combo-baseline> .venv/bin/python \
     humanoidverse/scripts/probe_backward_baseline.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, WINDOW, load_stats,
                                           mat_to_6d, quat_wxyz_to_mat,
                                           quat_yaw)
from humanoidverse.planner.model import MotionAE

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))
V5 = Path("/home/tcl/Desktop/start/dataset/v5_intervene2/clips")
POC_POOL = Path(__file__).parents[1] / "data" / "bdx_planner_v2"
V5_TARGETS = ["iv2_iv2bg140321_w006g07_b04", "iv2_iv2bg140321_w020g04_b00",
              "iv2_iv2bg140321_w025g04_b07", "iv2_iv2bg140321_w045g02_b04",
              "iv2_iv2bg140321_w048g05_b03"]
REV_SAMPLE = 24


def canonical_from_raw(q, qdot, bp, bq, blv, bav):
    """Mirror P2Dataset.canonical() (fixed_v1 de-yaw contract)."""
    psi = quat_yaw(bq[0])
    c, s = torch.cos(psi), torch.sin(psi)
    Rz = torch.tensor([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])

    def rot(v):
        return v @ Rz.T

    bp_c = rot(bp - bp[0])
    R_c = Rz.unsqueeze(0) @ quat_wxyz_to_mat(bq)
    x = torch.cat([q, qdot, bp_c, mat_to_6d(R_c), rot(blv), rot(bav)], dim=-1)
    return x, bp_c


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(
        OUT_DIR / "p1_ae_l64.pt", map_location=device,
        weights_only=False)["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    report = {"v5_branches": [], "rev_sample": {}, "_note": "pre-fix baseline"}

    def recon_windows(windows_x):
        """windows_x: (N, W, 43) -> per-window q mrad."""
        errs, dxs, vxs = [], [], []
        for x in windows_x:
            xn = ((x.to(device) - mean) / std).unsqueeze(0)
            p = teacher.decode(teacher.encode(xn))[0] * std + mean
            q_p = p[..., 0:14].cpu()
            q_g = x[..., 0:14]
            errs.append(float((q_p - q_g).abs().mean()) * 1000)
            bp_c = x[..., 28:31]
            dxs.append(float(bp_c[-1, 0] - bp_c[0, 0]))
            vxs.append(dxs[-1] * 50 / (x.shape[0] - 1))
        return errs, dxs, vxs

    for cid in V5_TARGETS:
        z = np.load(V5 / f"{cid}.npz", allow_pickle=True)
        T = z["joint_pos"].shape[0]
        rec = {"clip": cid, "frames": T, "windows": []}
        xs = []
        for s in range(0, T - WINDOW + 1, 25):
            sl = slice(s, s + WINDOW)
            x, _ = canonical_from_raw(
                *[torch.from_numpy(z[k][sl].astype(np.float32)) for k in
                  ("joint_pos", "joint_vel")] +
                [torch.from_numpy(z[k][sl, 0].astype(np.float32)) for k in
                 ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")])
            xs.append(x)
        errs, dxs, vxs = recon_windows(xs)
        rec["q_mrad_mean"] = float(np.mean(errs))
        rec["q_mrad_max"] = float(np.max(errs))
        rec["root_dx_mean"] = float(np.mean(dxs))
        rec["realized_vx_mean"] = float(np.mean(vxs))
        report["v5_branches"].append(rec)
        print(f"{cid}: q {rec['q_mrad_mean']:.1f} mrad  dx {rec['root_dx_mean']:+.3f}m  vx {rec['realized_vx_mean']:+.3f} m/s  ({len(xs)} win)")

    # _rev sample from PoC pool (clean pairs only: dx sign flip held).
    # Layout is clip-contiguous with 'lengths' — use cumsum ranges, and load
    # each big field exactly once (masked access re-decompresses the whole
    # field every call on this ~800k-frame npz).
    zp = np.load(POC_POOL / "sparse_full_v2.npz", allow_pickle=True)
    meta = json.load(open(POC_POOL / "meta.json"))
    names = [c["name"] for c in meta["clips"]]
    idx = {n: i for i, n in enumerate(names)}
    lengths = zp["lengths"]
    starts = np.concatenate([[0], np.cumsum(lengths)])
    base_pos_all = zp["base_pos"]
    revs = [n for n in names if n.startswith("v2_backward/") and n.endswith("_rev")]
    clean = []
    for n in revs:
        stem = n.split("/")[1][:-4]
        cands = [nm for nm in names if nm.split("/")[-1] == stem
                 and not nm.startswith("v2_backward")]
        if not cands:
            continue
        io_, ir = idx[cands[0]], idx[n]
        so, sr = slice(starts[io_], starts[io_ + 1]), slice(starts[ir], starts[ir + 1])
        if lengths[io_] < 50 or lengths[ir] < 50:
            continue
        bpo, bpr = base_pos_all[so], base_pos_all[sr]
        dxo = bpo[-1, 0] - bpo[0, 0]
        dxr = bpr[-1, 0] - bpr[0, 0]
        if dxo * dxr <= 0:
            clean.append((n, ir))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(clean), min(REV_SAMPLE, len(clean)), replace=False)
    fields = {k: zp[k] for k in ("q", "qdot", "base_pos", "base_quat",
                                 "base_linvel", "base_angvel")}
    errs, dxs, vxs = [], [], []
    worst = []
    for i in sel:
        n, ic = clean[i]
        sr = slice(starts[ic], starts[ic] + lengths[ic])
        clip = {k: v[sr] for k, v in fields.items()}
        T = len(clip["q"])
        for s in range(0, T - WINDOW + 1, 50):   # non-overlapping windows
            sl = slice(s, s + WINDOW)
            x, _ = canonical_from_raw(*[torch.from_numpy(
                clip[k][sl].astype(np.float32)) for k in
                ("q", "qdot", "base_pos", "base_quat",
                 "base_linvel", "base_angvel")])
            e, dx, vx = recon_windows([x])
            errs.append(e[0]); dxs.append(dx[0]); vxs.append(vx[0])
            worst.append((e[0], n, s))
        print(f"  _rev {n.split('/')[-1][:40]} done (cum {len(errs)} win)", flush=True)
    worst.sort(reverse=True)
    report["rev_sample"] = {
        "n_clips": int(len(sel)), "n_windows": len(errs),
        "q_mrad_mean": float(np.mean(errs)), "q_mrad_p90": float(np.percentile(errs, 90)),
        "root_dx_mean": float(np.mean(dxs)), "realized_vx_mean": float(np.mean(vxs)),
        "realized_vx_neg_frac": float(np.mean(np.array(vxs) < 0)),
        "worst5": [{"clip": n, "start": s, "q_mrad": e} for e, n, s in worst[:5]],
    }
    r = report["rev_sample"]
    print(f"_rev sample: q {r['q_mrad_mean']:.1f} mrad (p90 {r['q_mrad_p90']:.1f})  "
          f"vx {r['realized_vx_mean']:+.3f} m/s  neg {r['realized_vx_neg_frac']*100:.0f}%  "
          f"({r['n_windows']} win / {r['n_clips']} clips)")

    out = Path(os.environ.get("P3_BASELINE_OUT", "p3_prefix_baseline.json"))
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
