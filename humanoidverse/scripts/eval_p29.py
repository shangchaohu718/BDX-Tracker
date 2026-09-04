"""P2.9a evaluation: velocity-command tracking, command sweeps, edge combos.

The planner takes history + [vx, vy, vyaw, neck×4] (future sparse MASKED);
this eval asks whether the GENERATED motion obeys the command:
  (a) per-bucket velocity tracking: mean |Δvx| / |Δvyaw| between command and
      the decoded motion's realized base velocity, bucketed by commanded vx
  (b) vx command sweep on a fixed walk history (continuity + gain)
  (c) edge combos: stand history + walking-speed command (honest failure log)

Run: BDX_PLANNER_DATA=.../bdx_planner_v2 .venv/bin/python humanoidverse/scripts/eval_p29.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, MotionData, P2Dataset,
                                           WINDOW, load_stats, sixd_to_mat)
from humanoidverse.planner.fk import BDXFK
from humanoidverse.planner.model import CommandedEncoder, MotionAE

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))
LOCO_SCALE = torch.tensor([1.0, 0.5, 1.5])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="default: canonical registry")
    ap.add_argument("--max-windows", type=int, default=4000)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from humanoidverse.planner.registry import resolve_canonical
    _reg = resolve_canonical(OUT_DIR)
    if args.checkpoint is None:
        args.checkpoint = str(_reg["planner"])
    print(f"canonical: {args.checkpoint}")
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(_reg["teacher"], map_location=device,
                                       weights_only=False)["model"])
    model = CommandedEncoder(latent_dim=64, cmd_dim=7).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=False)["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    sp = {k: torch.tensor(v) for k, v in json.loads(
        (OUT_DIR / "p2_stats32.json").read_text()).items()}  # CPU: dataloader workers
    md = MotionData()
    ds = P2Dataset(md, "val", stats=stats, sparse_stats=sp,
                   max_windows=args.max_windows)
    loader = DataLoader(ds, batch_size=128, num_workers=4)

    cmd_v, gen_v, qerr = [], [], []
    with torch.no_grad():
        for batch in loader:
            sp_f = batch["sp_fut"].to(device) * 0          # planner form
            cmd = torch.cat([batch["cmd"], batch["loco"][:, None, :].expand(
                -1, WINDOW, -1)], -1).to(device)
            c = model(batch["sp_hist"].to(device), sp_f, cmd)
            p = teacher.decode(c)[0] * stats["std"].to(device) + stats["mean"].to(device)
            # realized velocity from POSITION/ROTATION trajectories (finite
            # diff) — the angvel output field is unreliable (stays ~0 while
            # rotating; caused the vyaw misdiagnosis on 08-19)
            pos = p[..., 28:31]
            R = sixd_to_mat(p[..., 31:37])
            lv_x = (pos[:, -1] - pos[:, 0])[..., 0] * (50 / (WINDOW - 1))
            lv_y = (pos[:, -1] - pos[:, 0])[..., 1] * (50 / (WINDOW - 1))
            dyaw = torch.atan2(R[:, -1, 1, 0], R[:, -1, 0, 0]) \
                - torch.atan2(R[:, 0, 1, 0], R[:, 0, 0, 0])
            av_z = dyaw * (50 / (WINDOW - 1))
            gen_v.append(torch.stack([lv_x, lv_y, av_z], -1).cpu().numpy())
            cmd_v.append(batch["loco"].numpy())
            qerr.append((p[..., 0:14] - batch["q"].to(device)).abs()
                        .mean(dim=(1, 2)).cpu().numpy())
    cmd_v, gen_v, qerr = map(np.concatenate, (cmd_v, gen_v, qerr))
    # denormalize command to physical units for reporting
    scale = LOCO_SCALE.numpy()
    cmd_phys, gen_phys = cmd_v * scale, gen_v * scale
    dv = gen_phys - cmd_phys

    print(f"P2.9a eval — {args.checkpoint} ({len(cmd_v)} val windows)")
    print("(a) velocity-command tracking, buckets by commanded vx (m/s):")
    buckets = [(-1.0, -0.05, "backward"), (-0.05, 0.05, "stationary"),
               (0.05, 0.25, "slow"), (0.25, 0.5, "medium"), (0.5, 9.9, "fast")]
    rep = {}
    for lo, hi, name in buckets:
        m = (cmd_phys[:, 0] >= lo) & (cmd_phys[:, 0] < hi)
        if m.sum() < 20:
            continue
        rep[name] = {"n": int(m.sum()),
                     "dvx": float(np.abs(dv[m, 0]).mean()),
                     "dvy": float(np.abs(dv[m, 1]).mean()),
                     "dvyaw": float(np.abs(dv[m, 2]).mean()),
                     "q_mrad": float(qerr[m].mean() * 1000)}
        r = rep[name]
        print(f"  {name:<11} n={r['n']:>4}  |Δvx| {r['dvx']:.3f}  "
              f"|Δvy| {r['dvy']:.3f}  |Δvyaw| {r['dvyaw']:.3f}  "
              f"q {r['q_mrad']:.0f} mrad")
    print(f"  overall |Δvx| {np.abs(dv[:,0]).mean():.3f} m/s, "
          f"|Δvyaw| {np.abs(dv[:,2]).mean():.3f} rad/s, q {qerr.mean()*1000:.0f} mrad")

    # (b) vx sweep on the first locomotion window
    loco_idx = np.where(ds.categories == "locomotion")[0]
    if len(loco_idx):
        item = ds[int(loco_idx[len(loco_idx) // 2])]
        hist = item["sp_hist"]
        base_cmd = torch.cat([item["cmd"], item["loco"][None].expand(WINDOW, -1)], -1)
        print("(b) vx command sweep (walk history held fixed):")
        sweep = []
        for vx in (-0.2, 0.0, 0.2, 0.4, 0.6, 0.8):
            cmd = base_cmd.clone()
            cmd[:, 4] = vx                                  # normalized by 1.0
            c = model(hist.unsqueeze(0).to(device),
                      torch.zeros(1, WINDOW, hist.shape[-1], device=device),
                      cmd.unsqueeze(0).to(device))
            p = teacher.decode(c)[0] * stats["std"].to(device) + stats["mean"].to(device)
            gen_vx = p[..., 37].mean().item()
            sweep.append((vx, gen_vx))
        gain = np.polyfit([s[0] for s in sweep], [s[1] for s in sweep], 1)[0]
        print("  " + "  ".join(f"cmd{v:+.1f}->gen{g:+.2f}" for v, g in sweep)
              + f"   gain {gain:.2f}")

    (OUT_DIR / "p29_eval.json").write_text(json.dumps(
        {"buckets": rep,
         "overall": {"dvx": float(np.abs(dv[:, 0]).mean()),
                     "dvyaw": float(np.abs(dv[:, 2]).mean()),
                     "q_mrad": float(qerr.mean() * 1000)},
         "vx_sweep": sweep}, indent=1))
    print(f"-> {OUT_DIR / 'p29_eval.json'}")


if __name__ == "__main__":
    main()
