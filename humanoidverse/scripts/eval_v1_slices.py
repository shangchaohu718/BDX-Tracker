"""Planner v1 slice evaluation — the performance map.

Slices the val set by (a) behavior bucket from realized command
(forward/backward/strafe/turn/stationary), (b) manifest tags: command_fidelity
(ok vs moderate vs n/a) and derived (native vs derived), and reports per-slice
command-tracking metrics:
  - vx/vy/vyaw: |command - realized| from rotation/position trajectories
  - q error vs GT
Plus command-gain sweeps on a fixed forward-walk history.

Run: BDX_PLANNER_DATA=.../bdx_planner_v1 .venv/bin/python \
     humanoidverse/scripts/eval_v1_slices.py [--checkpoint ...]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, MotionData, P2Dataset,
                                           WINDOW, load_stats, sixd_to_mat)
from humanoidverse.planner.model import CommandedEncoder, MotionAE

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))


def behavior_bucket(loco_phys):
    vx, vy, vyaw = loco_phys
    if abs(vyaw) > 0.3:
        return "turn"
    if vx < -0.08:
        return "backward"
    if abs(vy) > 0.15:
        return "strafe"
    if abs(vx) <= 0.08:
        return "stationary"
    return "forward"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="default: canonical registry")
    ap.add_argument("--max-windows", type=int, default=6000)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(resolve_canonical(OUT_DIR)["teacher"], map_location=device,
                                       weights_only=False)["model"])
    model = CommandedEncoder(latent_dim=64, cmd_dim=7).to(device).eval()
    from humanoidverse.planner.registry import resolve_canonical
    if args.checkpoint is None:
        args.checkpoint = str(resolve_canonical(OUT_DIR)["planner"])
    print(f"canonical: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=False)["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    sp = {k: torch.tensor(v) for k, v in json.loads(
        (OUT_DIR / "p2_stats32.json").read_text()).items()}
    md = MotionData()
    ds = P2Dataset(md, "val", stats=stats, sparse_stats=sp,
                   max_windows=args.max_windows)
    loader = DataLoader(ds, batch_size=128, num_workers=4)

    # slice metadata per window (from meta.json manifest tags)
    meta = {c["name"]: c for c in json.loads(
        (OUT_DIR / "meta.json").read_text())["clips"]}
    names = [md.clip_names[c] for c, _ in ds.windows]
    fid = np.array([meta[n].get("fidelity", "n/a") for n in names])
    def derived_subtype(name, m):
        if str(m.get("derived")) != "True" and not name.startswith("v2_backward"):
            return "native"
        cid = name.split("/")[-1]
        tags = " ".join(str(t) for t in (m.get("tags") or []))
        if "mirror" in cid or "mirror" in tags: return "der:mirror"
        if "speed" in cid or "speed" in tags: return "der:speed"
        if "_rev" in cid or "reverse" in tags or name.startswith("v2_backward"): return "der:reverse"
        return "der:other"
    der = np.array([derived_subtype(n, meta[n]) for n in names])

    agg = defaultdict(lambda: defaultdict(list))
    SCALE = np.array([1.0, 0.5, 1.5])
    std = stats["std"].to(device)
    mean = stats["mean"].to(device)
    n_seen = 0
    for batch in loader:
        B = batch["cmd"].shape[0]
        cmd = torch.cat([batch["cmd"], batch["loco"][:, None].expand(
            -1, WINDOW, -1)], -1).to(device)
        c = model(batch["sp_hist"].to(device),
                  torch.zeros(B, WINDOW, 32, device=device), cmd)
        p = teacher.decode(c)[0] * std + mean
        pos = p[..., 28:31]
        R = sixd_to_mat(p[..., 31:37])
        gen = torch.stack([
            (pos[:, -1, 0] - pos[:, 0, 0]) * (50 / (WINDOW - 1)),
            (pos[:, -1, 1] - pos[:, 0, 1]) * (50 / (WINDOW - 1)),
            (torch.atan2(R[:, -1, 1, 0], R[:, -1, 0, 0])
             - torch.atan2(R[:, 0, 1, 0], R[:, 0, 0, 0])) * (50 / (WINDOW - 1)),
        ], -1).cpu().numpy()
        cmd_phys = batch["loco"].numpy() * SCALE
        qerr = (p[..., 0:14] - batch["q"].to(device)).abs() \
            .mean(dim=(1, 2)).cpu().numpy()
        for i in range(B):
            b = behavior_bucket(cmd_phys[i])
            fid_i, der_i = fid[n_seen + i], der[n_seen + i]
            dv = np.abs(gen[i] - cmd_phys[i])
            row = {"dvx": dv[0], "dvy": dv[1], "dvyaw": dv[2], "q": qerr[i]}
            for key in (b, f"fid:{fid_i}", der_i):
                for k, v in row.items():
                    agg[key][k].append(v)
        n_seen += B

    print(f"Planner v1 slice map — {args.checkpoint} ({n_seen} val windows)")
    print(f"{'slice':<18} {'n':>5} {'|Δvx|':>7} {'|Δvy|':>7} {'|Δvyaw|':>7} {'q mrad':>7}")
    order = ["forward", "turn", "backward", "strafe", "stationary",
             "fid:ok", "fid:moderate", "fid:n/a",
             "native", "der:mirror", "der:speed", "der:reverse"]
    report = {}
    for key in order:
        if key not in agg:
            continue
        a = agg[key]
        r = {k: float(np.mean(v)) for k, v in a.items()}
        report[key] = {"n": len(a["q"]), **r}
        print(f"{key:<18} {len(a['q']):>5} {r['dvx']:7.3f} {r['dvy']:7.3f} "
              f"{r['dvyaw']:7.3f} {r['q']*1000:7.0f}")
    (OUT_DIR / "v1_slice_map.json").write_text(json.dumps(report, indent=1))
    print(f"-> {OUT_DIR / 'v1_slice_map.json'}")


if __name__ == "__main__":
    main()
