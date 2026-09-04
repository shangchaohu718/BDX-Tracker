"""P1 training: BDX full-motion autoencoder (Full window -> c_r(R^32) -> Full window).

Losses (5 families, physical units):
  L_q       MSE on joint angles (rad)
  L_qdot    0.1 * MSE on joint velocities (rad/s)
  L_base    10*MSE(base pos, m) + geodesic(base rot, rad) + 0.1*MSE(vels)
  L_fk      MSE on FK feet (L/R ankle) + head (neck_pitch) positions (m),
            computed by planner.fk from predicted q/base vs GT q/base
  L_contact 0.2 * BCE on auxiliary contact logits

Run: .venv/bin/python humanoidverse/scripts/train_p1.py [--epochs 10]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, WINDOW, MotionData,
                                           WindowDataset, compute_stats,
                                           load_stats, save_stats,
                                           sixd_to_mat)
from humanoidverse.planner.fk import BDXFK, HEAD, LFOOT, RFOOT
from humanoidverse.planner.model import MotionAE

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))
FK_BODIES = [LFOOT, RFOOT, HEAD]


def loss_fn(batch, motion_pred, contact_logits, stats, fk, slip_weight=0.0,
            neck_weight=1.0):
    mean = stats["mean"].to(motion_pred.device)
    std = stats["std"].to(motion_pred.device)
    pred = motion_pred * std + mean
    q_p, qdot_p = pred[..., 0:14], pred[..., 14:28]
    pos_p = pred[..., 28:31]
    R_p = sixd_to_mat(pred[..., 31:37])
    lv_p, av_p = pred[..., 37:40], pred[..., 40:43]

    q, qdot = batch["q"], batch["qdot"]
    pos, R = batch["base_pos"], batch["base_R"]
    lv, av = batch["base_linvel"], batch["base_angvel"]

    if neck_weight != 1.0:
        # neck occupies only 4/43 dims of the joint losses; plain MSE leaves
        # the student little pressure to READ the sparse head command
        w = torch.ones(14, device=q_p.device)
        w[10:] = neck_weight
        L_q = ((q_p - q) ** 2 * w).mean()
        L_qdot = ((qdot_p - qdot) ** 2 * w).mean()
    else:
        L_q = F.mse_loss(q_p, q)
        L_qdot = F.mse_loss(qdot_p, qdot)
    cos = ((R_p.transpose(-1, -2) @ R).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    L_rot = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)).mean()
    L_base = 10 * F.mse_loss(pos_p, pos) + L_rot \
        + 0.1 * (F.mse_loss(lv_p, lv) + F.mse_loss(av_p, av))

    bp_pred = fk.forward(pos_p, R_p, q_p)[..., FK_BODIES, :]
    with torch.no_grad():
        bp_gt = fk.forward(pos, R, q)[..., FK_BODIES, :]
    L_fk = F.mse_loss(bp_pred, bp_gt)

    L_contact = F.binary_cross_entropy_with_logits(
        contact_logits, batch["contact"])

    parts = {"q": L_q, "qdot": 0.1 * L_qdot, "base": L_base,
             "fk": L_fk, "contact": 0.2 * L_contact}
    if slip_weight > 0:
        # penalize horizontal foot velocity during GT contact (anti-slide);
        # addresses the mean-regression slip signature (pred 0.123 vs gt 0.049 m/s)
        vel = (bp_pred[:, 1:, :2] - bp_pred[:, :-1, :2]).norm(dim=-1) * 50
        mask = (batch["contact"][:, 1:] > 0).to(vel.dtype)
        parts["slip"] = slip_weight * (vel.pow(2) * mask).sum() / mask.sum().clamp_min(1)
    return sum(parts.values()), {k: v.detach() for k, v in parts.items()}, pred


def run_val(model, loader, stats, fk, device, slip_weight=0.0):
    model.eval()
    agg = {}
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            motion, contact, _ = model(batch["x"])
            total, parts, _ = loss_fn(batch, motion, contact, stats, fk,
                                      slip_weight=slip_weight)
            agg.setdefault("total", 0.0)
            agg["total"] += total.item()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v.item()
    n = len(loader)
    return {k: v / n for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=3000,
                    help="steps per epoch (windows overlap, subsampling is fine)")
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true",
                    help="load p1_ae.pt and continue (fresh cosine over remaining steps)")
    ap.add_argument("--slip-weight", type=float, default=0.0)
    ap.add_argument("--tag", default="",
                    help="suffix for checkpoint/history files (e.g. _l64)")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    md = MotionData()

    stats_path = OUT_DIR / "p1_stats.json"
    if stats_path.exists():
        stats = load_stats(stats_path)
        print(f"stats: loaded {stats_path}")
    else:
        t0 = time.time()
        stats = compute_stats(md)
        save_stats(stats, stats_path)
        print(f"stats: computed from train windows ({time.time()-t0:.0f}s)")

    train_ds = WindowDataset(md, "train", stride=1, stats=stats)
    val_ds = WindowDataset(md, "val", stride=25, max_windows=2000, stats=stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=2)

    model = MotionAE(in_dim=MODEL_DIM, latent_dim=args.latent,
                     window=WINDOW).to(device)
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(OUT_DIR / f"p1_ae{args.tag}.pt", map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        args.epochs += start_epoch
        print(f"resumed from epoch {ckpt['epoch']} (val total "
              f"{ckpt['val']['total']:.4f})")
    print(f"model: {model.num_params()/1e6:.2f}M params, latent R^{args.latent}, "
          f"train {len(train_ds)} windows, val {len(val_ds)}")

    fk = BDXFK().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_total = args.epochs * args.max_steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps_total)

    history, best_val = [], float("inf")
    hist_path = OUT_DIR / f"p1_history{args.tag}.json"
    if args.resume and hist_path.exists():
        history = json.loads(hist_path.read_text())
        # slip penalty changes the loss family — totals not comparable to
        # earlier history, so reset the best-checkpoint comparison
        best_val = float("inf") if args.slip_weight > 0 else \
            min(h["val"]["total"] for h in history)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run = time.time(), {}
        for step, batch in enumerate(train_loader):
            if step >= args.max_steps:
                break
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            motion, contact, _ = model(batch["x"])
            loss, parts, _ = loss_fn(batch, motion, contact, stats, fk,
                                     slip_weight=args.slip_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in [("total", loss.item())] + list(parts.items()):
                run[k] = run.get(k, 0.0) + float(v)
        train_avg = {k: v / args.max_steps for k, v in run.items()}
        val_avg = run_val(model, val_loader, stats, fk, device,
                          slip_weight=args.slip_weight)
        history.append({"epoch": epoch, "train": train_avg, "val": val_avg})
        marker = ""
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            torch.save({"model": model.state_dict(), "stats": stats_path.name,
                        "config": vars(args), "epoch": epoch,
                        "val": val_avg}, OUT_DIR / f"p1_ae{args.tag}.pt")
            marker = "  *best*"
        print(f"epoch {epoch}: train total {train_avg['total']:.4f} "
              f"(q {train_avg['q']:.4f} fk {train_avg['fk']:.5f}) | "
              f"val total {val_avg['total']:.4f} (q {val_avg['q']:.4f} "
              f"fk {val_avg['fk']:.5f} contact {val_avg['contact']:.4f}) "
              f"[{time.time()-t0:.0f}s]{marker}", flush=True)

    hist_path.write_text(json.dumps(history, indent=1))
    print(f"done. best val total {best_val:.4f}; "
          f"checkpoint -> {OUT_DIR / f'p1_ae{args.tag}.pt'}")


if __name__ == "__main__":
    main()
