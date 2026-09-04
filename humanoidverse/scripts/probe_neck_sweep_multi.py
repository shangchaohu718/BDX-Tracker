"""P4.2 multi-history neck sweep — robustness upgrade of probe_neck_sweep
(P2.2 anchor used a SINGLE fixed history window).

Same grids, same realized definition (window mean of decoded q[10+j]),
K=8 deterministic locomotion history windows (evenly spaced over the val
locomotion indices — no random selection). Aggregation rule FROZEN BEFORE
running: a channel counts as sweep-gate-passing only if the WORST history
slope >= 0.50 AND worst R² >= 0.60; anything else -> FROZEN_WITH_DISCLOSURE
in the P4.2 disposition. Per-history slopes are reported in full — no
selection.

Run: BDX_PLANNER_DATA=<dir> ART_DIR=<out> \
     .venv/bin/python humanoidverse/scripts/probe_neck_sweep_multi.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, MotionData, P2Dataset,
                                           WINDOW, load_stats, set_preprocess_version)
from humanoidverse.planner.model import CommandedEncoder, MotionAE
from humanoidverse.planner.registry import resolve_canonical

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA"))
ART = Path(os.environ["ART_DIR"])
NECK_SCALE = 1.7
NECK_Q_IDX = {"neck_forward": 10, "neck_pitch": 11, "neck_yaw": 12, "neck_roll": 13}
NECK_CMD_IDX = {"neck_forward": 0, "neck_pitch": 1, "neck_yaw": 2, "neck_roll": 3}
SWEEPS = {
    "neck_pitch":  [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.35],
    "neck_forward": [0.0, 0.15, 0.30, 0.50, 0.70],
    "neck_roll":   [-0.30, -0.15, 0.0, 0.15, 0.30],
}
K = 8


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg = resolve_canonical(OUT_DIR)
    set_preprocess_version(reg["registry"]["preprocess_version"])
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(reg["teacher"], map_location=device,
                                       weights_only=False)["model"])
    model = CommandedEncoder(latent_dim=64, cmd_dim=7).to(device).eval()
    model.load_state_dict(torch.load(reg["planner"], map_location=device,
                                     weights_only=False)["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    sp = {k: torch.tensor(v) for k, v in json.loads(
        (OUT_DIR / "p2_stats32.json").read_text()).items()}
    md = MotionData()
    ds = P2Dataset(md, "val", stats=stats, sparse_stats=sp, max_windows=4000)

    loco_idx = np.where(ds.categories == "locomotion")[0]
    hist_ids = np.linspace(0, len(loco_idx) - 1, K).round().astype(int)
    print(f"histories: {K} locomotion windows, ds indices "
          f"{[int(loco_idx[i]) for i in hist_ids]}")

    out = {"artifact": "neck_sweep_multi",
           "checkpoint": Path(reg["planner"]).name, "K": K,
           "aggregation_rule": "channel passes sweep gate iff WORST-history "
                               "slope>=0.50 AND worst R²>=0.60 (frozen before "
                               "running); per-history slopes reported in full",
           "grids": SWEEPS, "channels": {}}
    for ch, vals in SWEEPS.items():
        per_hist = []
        for hi in hist_ids:
            item = ds[int(loco_idx[hi])]
            hist = item["sp_hist"]
            base_cmd = torch.cat([item["cmd"],
                                  item["loco"][None].expand(WINDOW, -1)], -1)
            rows = []
            for v in vals:
                cmd = base_cmd.clone()
                cmd[:, NECK_CMD_IDX[ch]] = v / NECK_SCALE
                c = model(hist.unsqueeze(0).to(device),
                          torch.zeros(1, WINDOW, hist.shape[-1], device=device),
                          cmd.unsqueeze(0).to(device))
                p = teacher.decode(c)[0] * stats["std"].to(device) + stats["mean"].to(device)
                rows.append({"cmd": v, "realized": float(p[..., NECK_Q_IDX[ch]].mean())})
            xs = np.array([r["cmd"] for r in rows])
            ys = np.array([r["realized"] for r in rows])
            slope, intercept = np.polyfit(xs, ys, 1)
            r2 = 1 - ((ys - (slope * xs + intercept)) ** 2).sum() / \
                     ((ys - ys.mean()) ** 2).sum()
            signs = sum(1 for r in rows
                        if abs(r["realized"]) < 0.02 or
                        np.sign(r["realized"]) == np.sign(r["cmd"]))
            per_hist.append({"ds_idx": int(loco_idx[hi]),
                             "slope": round(float(slope), 4),
                             "intercept": round(float(intercept), 4),
                             "r2": round(float(r2), 4),
                             "sign_ok": f"{signs}/{len(rows)}",
                             "points": rows})
        slopes = [h["slope"] for h in per_hist]
        r2s = [h["r2"] for h in per_hist]
        out["channels"][ch] = {
            "per_history": per_hist,
            "slope_min": round(float(min(slopes)), 4),
            "slope_mean": round(float(np.mean(slopes)), 4),
            "slope_max": round(float(max(slopes)), 4),
            "r2_min": round(float(min(r2s)), 4),
            "worst_history_slope": min(per_hist, key=lambda h: h["slope"]),
        }
        c = out["channels"][ch]
        print(f"{ch}: slope min/mean/max {c['slope_min']}/{c['slope_mean']}/"
              f"{c['slope_max']}  r2_min {c['r2_min']}")

    (ART / "neck_sweep_multi.json").write_text(json.dumps(out, indent=1))
    print(f"-> {ART / 'neck_sweep_multi.json'}")


if __name__ == "__main__":
    main()
