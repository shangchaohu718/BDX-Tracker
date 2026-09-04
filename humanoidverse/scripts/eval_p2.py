"""P2 student evaluation: motion-level metrics vs frozen teacher, per category.

Prints overall + per-category joint MAE / foot FK / head FK for student and
teacher (both decoded through the same frozen decoder), plus per-dim latent
R^2 stored in the checkpoint.

Run: .venv/bin/python humanoidverse/scripts/eval_p2.py [--checkpoint ...]
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
                                           load_stats, sixd_to_mat)
from humanoidverse.planner.fk import BDXFK
from humanoidverse.planner.model import (CommandedEncoder, MotionAE,
                                           SparseEncoder)

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))
TEACHER = None  # resolved via registry at runtime
FKB = [5, 10, 12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="default: canonical registry")
    ap.add_argument("--max-windows", type=int, default=4000)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    from humanoidverse.planner.registry import resolve_canonical
    _reg = resolve_canonical(OUT_DIR)
    teacher.load_state_dict(torch.load(_reg["teacher"], map_location=device,
                                       weights_only=False)["model"])
    if args.checkpoint is None:
        args.checkpoint = str(_reg["planner"])
    print(f"canonical: {args.checkpoint}")
    sck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    enc_cls = CommandedEncoder if sck["config"].get("explicit_cmd") else SparseEncoder
    if enc_cls is CommandedEncoder:
        # v11 lineage: loco_cmd appends (B,3) loco to the (B,W,4) neck cmd -> 7
        cmd_dim = 7 if sck["config"].get("loco_cmd") else 4
        student = enc_cls(latent_dim=sck["config"]["latent"],
                          cmd_dim=cmd_dim).to(device).eval()
    else:
        student = enc_cls(latent_dim=sck["config"]["latent"]).to(device).eval()
    student.load_state_dict(sck["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    sp = {k: torch.tensor(v) for k, v in json.loads(
        (OUT_DIR / "p2_stats32.json").read_text()).items()}
    fk = BDXFK().to(device)
    ds = P2Dataset(MotionData(), "val", stats=stats, sparse_stats=sp,
                   max_windows=args.max_windows)
    loader = DataLoader(ds, batch_size=128, num_workers=4)
    cats = ds.categories

    res = {n: {k: [] for k in ("q", "foot", "head")} for n in ("student", "teacher")}
    with torch.no_grad():
        for batch in loader:
            cmd = batch["cmd"] if "cmd" in batch else None
            if cmd is not None and sck["config"].get("loco_cmd"):
                # same contract as train_p2.build_cmd: neck(4) + loco(3) -> 7
                cmd = torch.cat([cmd, batch["loco"][:, None, :].expand(
                    -1, cmd.shape[1], -1)], -1)
            c_s = student(batch["sp_hist"].to(device), batch["sp_fut"].to(device),
                          cmd.to(device) if cmd is not None else None)
            c_r = teacher.encode(batch["x"].to(device))
            bp_g = fk.forward(batch["base_pos"].to(device),
                              batch["base_R"].to(device),
                              batch["q"].to(device))[..., FKB, :]
            for name, c in [("student", c_s), ("teacher", c_r)]:
                p = teacher.decode(c)[0] * stats["std"].to(device) + stats["mean"].to(device)
                q_p, pos_p = p[..., 0:14], p[..., 28:31]
                bp = fk.forward(pos_p, sixd_to_mat(p[..., 31:37]), q_p)[..., FKB, :]
                res[name]["q"].append((q_p - batch["q"].to(device)).abs().mean(dim=(1, 2)).cpu().numpy())
                res[name]["foot"].append((bp[..., :2] - bp_g[..., :2]).norm(dim=-1).mean(dim=1).cpu().numpy())
                res[name]["head"].append((bp[..., 2] - bp_g[..., 2]).norm(dim=-1).cpu().numpy())
    out = {n: {k: np.concatenate(v) for k, v in d.items()} for n, d in res.items()}
    n = min(len(out["student"]["q"]), len(cats))
    cats = cats[:n]
    abl = sck.get("ablation", "none")
    print(f"P2 eval — {args.checkpoint} (ablation={abl}, epoch {sck['epoch']}, {n} windows)")
    qs, qt = out["student"]["q"][:n] * 1000, out["teacher"]["q"][:n] * 1000
    print(f"overall joint MAE student {qs.mean():.1f} / teacher {qt.mean():.1f} mrad "
          f"(ratio {qs.mean()/qt.mean():.2f})")
    print("per-category  joint (stud/tch mrad, ratio) | foot mm | head mm")
    rows = {}
    for cat in sorted(set(cats)):
        m = cats == cat
        rows[cat] = {
            "q_student": float(qs[m].mean()), "q_teacher": float(qt[m].mean()),
            "q_ratio": float(qs[m].mean() / qt[m].mean()),
            "foot_student": float(out["student"]["foot"][:n][m].mean() * 1000),
            "head_student": float(out["student"]["head"][:n][m].mean() * 1000)}
        r = rows[cat]
        print(f"  {cat:<12} {r['q_student']:6.1f} / {r['q_teacher']:5.1f} ({r['q_ratio']:5.2f}) "
              f"| {r['foot_student']:5.1f} | {r['head_student']:5.1f}")
    r2 = np.array(sck["r2"])
    print(f"latent R2 mean {r2.mean():.3f} ({(r2 > 0.5).sum()}/{len(r2)} dims > 0.5)")
    Path(args.checkpoint).with_suffix(".eval.json").write_text(json.dumps(
        {"ablation": abl, "overall_q_ratio": float(qs.mean() / qt.mean()),
         "per_category": rows, "r2_mean": float(r2.mean())}, indent=1))


if __name__ == "__main__":
    main()
