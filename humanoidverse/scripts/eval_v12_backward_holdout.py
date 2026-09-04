"""P3.2.3 G2 A-line acceptance: student on the lineage-clean backward_heldout
(29 `_rev` clips, 8 families — never in v12 training in any variant form).

Uses the REAL P2Dataset contract via a temporary splits file whose val side is
exactly the heldout clip list (history = separate prior 1s, cmd routing v1.1,
fixed_v1 stats) — no hand-reimplementation.

Metrics per window (completion form, GT history + GT command, future masked):
  commanded vx vs realized vx (canonical position finite diff) -> sign
  inversion (cmd<=-0.05 & realized>0), |dvx|; q error vs GT.
G2 acceptance: no sign inversion, |Δvx| <= 0.05 m/s, q increment <= 15 mrad
vs the same model's forward-val baseline (reported alongside).

Run: BDX_PLANNER_DATA=<dir> CHECKPOINT=<student.pt> \
     HOLDOUT_OUT=<json> .venv/bin/python \
     humanoidverse/scripts/eval_v12_backward_holdout.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, MotionData, P2Dataset,
                                           WINDOW, load_stats, set_preprocess_version,
                                           sixd_to_mat)
from humanoidverse.planner.model import CommandedEncoder, MotionAE

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA"))
POOL = OUT_DIR / "sparse_full_v1m.npz"
HOLDOUT = Path(os.environ.get(
    "HOLDOUT_REG",
    "/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/"
    "bdx_planner_distill/20260904T1900Z_p321_v12data/backward_heldout.json"))
CKPT = os.environ.get("CHECKPOINT")


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_preprocess_version("fixed_v1")
    reg = json.loads(HOLDOUT.read_text())
    held = sorted(set(reg["heldout_clips"]))

    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(
        OUT_DIR / "p1_ae_l64.pt", map_location=device,
        weights_only=False)["model"])
    student = CommandedEncoder(latent_dim=64, cmd_dim=7).to(device).eval()
    student.load_state_dict(torch.load(CKPT, map_location=device,
                                       weights_only=False)["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    sp = {k: torch.tensor(v) for k, v in json.loads(
        (OUT_DIR / "p2_stats32.json").read_text()).items()}

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump({"source": "heldout-only", "train": [], "val": held}, tf)
        splits_tmp = tf.name
    md = MotionData(npz_path=POOL, meta_path=OUT_DIR / "meta.json",
                    splits_path=splits_tmp)
    ds = P2Dataset(md, "val", stats=stats, sparse_stats=sp)
    print(f"{Path(CKPT).name}: heldout {len(held)} clips -> {len(ds)} windows")
    loader = DataLoader(ds, batch_size=64, num_workers=2)

    dvx, sign_inv, qerr, cmd_v, real_v, per_clip = [], [], [], [], [], {}
    for batch in loader:
        cmd7 = torch.cat([batch["cmd"],
                          batch["loco"][:, None, :].expand(-1, WINDOW, -1)], -1)
        z = student(batch["sp_hist"].to(device),
                    torch.zeros_like(batch["sp_fut"]).to(device),
                    cmd7.to(device))
        p = teacher.decode(z)[0] * std + mean
        pos = p[..., 28:31]
        rvx = (pos[:, -1, 0] - pos[:, 0, 0]) * 50 / (WINDOW - 1)
        cvx = batch["loco"][:, 0]                       # scale 1.0
        qe = (p[..., 0:14].cpu() - batch["q"]).abs().mean(dim=(1, 2)) * 1000
        cats = batch.get("clip") if "clip" in batch else None
        for j in range(len(rvx)):
            c_, r_ = float(cvx[j]), float(rvx[j])
            cmd_v.append(c_); real_v.append(r_)
            dvx.append(abs(r_ - c_)); qerr.append(float(qe[j]))
            if c_ <= -0.05 and r_ > 0:
                sign_inv.append(j)
    os.unlink(splits_tmp)
    cmd_v, real_v = np.array(cmd_v), np.array(real_v)
    res = {
        "checkpoint": Path(CKPT).name,
        "heldout_clips": len(held), "families": reg["heldout_families"],
        "windows": len(cmd_v),
        "neg_cmd_windows": int((cmd_v < -0.05).sum()),
        "cmd_vx_mean": float(cmd_v.mean()),
        "realized_vx_mean": float(real_v.mean()),
        "realized_vx_neg_frac_back": float(
            (real_v[cmd_v < -0.05] < 0).mean()) if (cmd_v < -0.05).sum() else None,
        "dvx_mean": float(np.mean(dvx)), "dvx_p90": float(np.percentile(dvx, 90)),
        "sign_inversions": len(sign_inv),
        "q_mrad_mean": float(np.mean(qerr)),
        "q_mrad_p90": float(np.percentile(qerr, 90)),
        "g2_gate": {"no_sign_inversion": len(sign_inv) == 0,
                    "dvx_le_0.05": float(np.mean(dvx)) <= 0.05},
    }
    print(f"cmd {cmd_v.mean():+.3f} -> realized {real_v.mean():+.3f} m/s | "
          f"neg-cmd windows {(cmd_v < -0.05).sum()} "
          f"(realized<0: {res['realized_vx_neg_frac_back'] and round(res['realized_vx_neg_frac_back'] * 100)}%) | "
          f"|dvx| {np.mean(dvx):.4f} p90 {np.percentile(dvx, 90):.4f} | "
          f"sign-inv {len(sign_inv)} | q {np.mean(qerr):.1f} mrad")
    out = Path(os.environ.get("HOLDOUT_OUT", "backward_holdout_result.json"))
    out.write_text(json.dumps(res, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
