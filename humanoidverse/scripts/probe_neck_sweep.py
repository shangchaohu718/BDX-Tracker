"""P2.2 neck-channel sweep probe: individual gate numbers for neck_pitch
(+ neck_forward / neck_roll as P4 validation evidence).

Mirrors eval_p29's vx sweep: fixed locomotion history, planner form
(future masked), one neck command channel swept over its envelope,
realized neck joint = window mean of decoded q[10+j]
(j: 0=forward, 1=pitch, 2=yaw, 3=roll; cmd normalized /1.7).

Run: BDX_PLANNER_DATA=<dir> .venv/bin/python humanoidverse/scripts/probe_neck_sweep.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, MotionData, P2Dataset,
                                           WINDOW, load_stats)
from humanoidverse.planner.model import CommandedEncoder, MotionAE

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))
NECK_SCALE = 1.7
NECK_Q_IDX = {"neck_forward": 10, "neck_pitch": 11, "neck_yaw": 12, "neck_roll": 13}
NECK_CMD_IDX = {"neck_forward": 0, "neck_pitch": 1, "neck_yaw": 2, "neck_roll": 3}
# envelopes from canonical.json capability_envelope (home/validated ranges)
SWEEPS = {
    "neck_pitch":  [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.35],
    "neck_forward": [0.0, 0.15, 0.30, 0.50, 0.70],
    "neck_roll":   [-0.30, -0.15, 0.0, 0.15, 0.30],
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="json path (default OUT_DIR/probe_neck_sweep.json)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from humanoidverse.planner.registry import resolve_canonical
    _reg = resolve_canonical(OUT_DIR)
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(_reg["teacher"], map_location=device,
                                       weights_only=False)["model"])
    model = CommandedEncoder(latent_dim=64, cmd_dim=7).to(device).eval()
    ckpt = str(_reg["planner"])
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    sp = {k: torch.tensor(v) for k, v in json.loads(
        (OUT_DIR / "p2_stats32.json").read_text()).items()}
    md = MotionData()
    ds = P2Dataset(md, "val", stats=stats, sparse_stats=sp, max_windows=4000)

    loco_idx = np.where(ds.categories == "locomotion")[0]
    item = ds[int(loco_idx[len(loco_idx) // 2])]
    hist = item["sp_hist"]
    base_cmd = torch.cat([item["cmd"], item["loco"][None].expand(WINDOW, -1)], -1)

    report = {"checkpoint": Path(ckpt).name, "history_window": "locomotion(mid)",
              "sweeps": {}, "fits": {}}
    for ch, vals in SWEEPS.items():
        rows = []
        for v in vals:
            cmd = base_cmd.clone()
            cmd[:, NECK_CMD_IDX[ch]] = v / NECK_SCALE
            c = model(hist.unsqueeze(0).to(device),
                      torch.zeros(1, WINDOW, hist.shape[-1], device=device),
                      cmd.unsqueeze(0).to(device))
            p = teacher.decode(c)[0] * stats["std"].to(device) + stats["mean"].to(device)
            realized = {n: float(p[..., i].mean()) for n, i in NECK_Q_IDX.items()}
            rows.append({"cmd": v, "realized": realized})
        xs = np.array([r["cmd"] for r in rows])
        ys = np.array([r["realized"][ch] for r in rows])
        slope, intercept = np.polyfit(xs, ys, 1)
        r2 = 1 - ((ys - (slope * xs + intercept)) ** 2).sum() / \
                 ((ys - ys.mean()) ** 2).sum()
        signs = sum(1 for r in rows
                    if abs(r["realized"][ch]) < 0.02 or
                    np.sign(r["realized"][ch]) == np.sign(r["cmd"]))
        report["sweeps"][ch] = rows
        report["fits"][ch] = {"slope": float(slope), "intercept": float(intercept),
                              "r2": float(r2), "sign_ok": f"{signs}/{len(rows)}"}
        pts = "  ".join(f"{r['cmd']:+.2f}->{r['realized'][ch]:+.3f}" for r in rows)
        print(f"{ch}: {pts}   slope {slope:.3f} R2 {r2:.4f} sign {signs}/{len(rows)}")

    out = Path(args.out) if args.out else (OUT_DIR / "probe_neck_sweep.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
