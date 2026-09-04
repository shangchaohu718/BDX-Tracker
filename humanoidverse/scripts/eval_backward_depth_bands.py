"""P3.3 evidence: per-depth-band backward validation on the lineage-clean
heldout (real P2Dataset contract, same windows as eval_v12_backward_holdout).

Outputs (ART_DIR):
  cmd_coverage_holdout.json    dataset-side commanded-vx coverage (no model)
  backward_depth_bands.json    per-band sign/|dvx|/gain/q for v12 AND v11

Band rule (mechanical, from the preregistered |dvx|<=0.05 gate): starting at
-0.05 walk outward while each band has sign-correct 100% AND band-mean |dvx|
<= 0.05; the envelope min is the deepest contiguous valid depth, rounded
TOWARD ZERO onto the 0.05 grid (never exposes beyond observed coverage).

Run: BDX_PLANNER_DATA=<v12data dir> ART_DIR=<artifact dir> \
     .venv/bin/python humanoidverse/scripts/eval_backward_depth_bands.py
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
                                           WINDOW, load_stats, set_preprocess_version)
from humanoidverse.planner.model import CommandedEncoder, MotionAE

OUT_DIR = Path(os.environ["BDX_PLANNER_DATA"])
POOL = OUT_DIR / "sparse_full_v1m.npz"
ART = Path(os.environ["ART_DIR"])
CKPTS = {
    "v12": OUT_DIR / "p2_student_v12.pt",
    "v11": Path("/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/"
               "bdx_planner_v2combo/p2_student_v11.pt"),
}
EDGES = [-0.70, -0.60, -0.50, -0.40, -0.30, -0.20, -0.10, -0.05]


def make_ds():
    set_preprocess_version("fixed_v1")
    reg = json.loads((OUT_DIR / "backward_heldout.json").read_text())
    held = sorted(set(reg["heldout_clips"]))
    stats = load_stats(OUT_DIR / "p1_stats.json")
    sp = {k: torch.tensor(v) for k, v in json.loads(
        (OUT_DIR / "p2_stats32.json").read_text()).items()}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump({"source": "heldout-depth", "train": [], "val": held}, tf)
        splits_tmp = tf.name
    md = MotionData(npz_path=POOL, meta_path=OUT_DIR / "meta.json",
                    splits_path=splits_tmp)
    ds = P2Dataset(md, "val", stats=stats, sparse_stats=sp)
    return ds, splits_tmp


def grid_toward_zero(x, step=0.05):
    n = x / step
    if abs(n - round(n)) < 1e-9:      # fp snap (e.g. -0.60/0.05 -> -11.999...)
        n = round(n)
    elif n < 0:
        n = np.ceil(n)
    else:
        n = np.floor(n)
    return float(n * step)


@torch.no_grad()
def run_model(ds, loader, ckpt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(
        OUT_DIR / "p1_ae_l64.pt", map_location=device, weights_only=False)["model"])
    student = CommandedEncoder(latent_dim=64, cmd_dim=7).to(device).eval()
    student.load_state_dict(torch.load(ckpt, map_location=device,
                                       weights_only=False)["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    cmd_v, real_v, qerr = [], [], []
    for batch in loader:
        cmd7 = torch.cat([batch["cmd"],
                          batch["loco"][:, None, :].expand(-1, WINDOW, -1)], -1)
        z = student(batch["sp_hist"].to(device),
                    torch.zeros_like(batch["sp_fut"]).to(device), cmd7.to(device))
        p = teacher.decode(z)[0] * std + mean
        pos = p[..., 28:31]
        rvx = (pos[:, -1, 0] - pos[:, 0, 0]) * 50 / (WINDOW - 1)
        cmd_v.append(batch["loco"][:, 0].numpy())
        real_v.append(rvx.cpu().numpy())
        qerr.append((p[..., 0:14].cpu() - batch["q"]).abs()
                    .mean(dim=(1, 2)).numpy() * 1000)
    return map(np.concatenate, (cmd_v, real_v, qerr))


def main():
    ds, splits_tmp = make_ds()
    loader = DataLoader(ds, batch_size=64, num_workers=2)

    # 1) dataset-side coverage (no model)
    cvx = np.concatenate([b["loco"][:, 0].numpy() for b in loader])
    os.unlink(splits_tmp)
    neg = cvx[cvx < -0.05]
    (ART / "cmd_coverage_holdout.json").write_text(json.dumps({
        "windows": int(len(cvx)), "neg_windows": int(len(neg)),
        "cmd_vx": {"min": float(cvx.min()), "max": float(cvx.max()),
                   "mean": float(cvx.mean())},
        "neg_cmd_vx": {"min": float(neg.min()), "max": float(neg.max()),
                       "mean": float(neg.mean()),
                       "p1": float(np.percentile(neg, 1)),
                       "p5": float(np.percentile(neg, 5)),
                       "p50": float(np.percentile(neg, 50)),
                       "p95": float(np.percentile(neg, 95)),
                       "p99": float(np.percentile(neg, 99))},
    }, indent=1))
    print(f"coverage: neg n={len(neg)} min {neg.min():+.4f} p50 "
          f"{np.percentile(neg, 50):+.4f}")

    # 2) per-band validation, both candidate students
    res = {"band_rule": "deepest contiguous band chain from -0.05 outward "
                        "with sign 100% and band-mean |dvx|<=0.05; envelope "
                        "min rounded toward zero onto 0.05 grid",
           "grid_edges": EDGES}
    for tag, ck in CKPTS.items():
        cmd_v, real_v, qerr = run_model(ds, loader, ck)
        bands = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (cmd_v >= lo) & (cmd_v < hi)
            if not m.sum():
                bands.append({"band": [lo, hi], "n": 0})
                continue
            negm = cmd_v[m] <= -0.05
            bands.append({
                "band": [lo, hi], "n": int(m.sum()),
                "sign_correct_frac": float((real_v[m][negm] < 0).mean()),
                "dvx_mean": float(np.abs(real_v[m] - cmd_v[m]).mean()),
                "dvx_p90": float(np.percentile(np.abs(real_v[m] - cmd_v[m]), 90)),
                "gain": float(real_v[m].mean() / cmd_v[m].mean()),
                "q_mrad": float(qerr[m].mean()),
            })
        valid_min = -0.05
        for b in reversed(bands):          # shallow -> deep
            if b["n"] == 0:
                break
            if b["sign_correct_frac"] == 1.0 and b["dvx_mean"] <= 0.05:
                valid_min = b["band"][0]
            else:
                break
        # exposable depth bounded by BOTH the contiguous-valid chain and the
        # observed coverage (never claim beyond either), then grid toward zero
        deepest = max(valid_min, float(cmd_v.min()))
        env_min = float(grid_toward_zero(deepest))
        res[tag] = {"checkpoint": Path(ck).name, "bands": bands,
                    "deepest_contiguous_valid": valid_min,
                    "observed_cmd_min": float(cmd_v.min()),
                    "envelope_min_grid_toward_zero": env_min}
        print(f"{tag}: deepest_contiguous_valid {valid_min} "
              f"observed_min {cmd_v.min():+.4f} -> envelope_min {env_min:+.2f}")
    (ART / "backward_depth_bands.json").write_text(json.dumps(res, indent=1))
    print(f"-> {ART}")


if __name__ == "__main__":
    main()
