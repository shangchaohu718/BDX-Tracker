"""P2 training: sparse student encoder distilled into the frozen P1 teacher.

Pipeline: sparse history (K=50) + sparse future (H=50, 29D each, canonical
frame anchored at the future window start) -> SparseEncoder -> c_s (R^64)
-> FROZEN teacher decoder -> full motion window.

Losses: the P1 5-family motion loss through the frozen decoder (primary) plus
latent alignment lam * MSE(c_s, c_r) with lam ramped 0 -> 0.5 -> 1 over epochs
(recon first; alignment once the student can actually decode).

Eval extras: per-dim R^2 of c_s vs frozen teacher c_r, and the student/teacher
output-error ratio mse(D(c_s), GT) / mse(D(c_r), GT).

Run: .venv/bin/python humanoidverse/scripts/train_p2.py [--epochs 8]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, WINDOW, MotionData,
                                           P2Dataset, compute_sparse_stats,
                                           load_stats, save_stats)
from humanoidverse.planner.fk import BDXFK, rotmat_to_rotvec
from humanoidverse.planner.model import (CommandedEncoder, MotionAE,
                                           SparseEncoder)
from humanoidverse.scripts.train_p1 import OUT_DIR, loss_fn

TEACHER = OUT_DIR / "p1_ae_l64.pt"


def r2_per_dim(pred, target):
    var = target.var(dim=0)
    mse = ((pred - target) ** 2).mean(dim=0)
    return 1 - mse / var.clamp_min(1e-8)


def build_cmd(batch, loco_mode):
    """Neck command (B,W,4); in P2.9a also append the (B,3) loco command."""
    if not loco_mode:
        return batch.get("cmd")
    return torch.cat([batch["cmd"],
                      batch["loco"][:, None, :].expand(-1, batch["cmd"].shape[1], -1)], -1)


def ablate(batch, mode):
    if mode == "no_head":
        batch["sp_fut"][..., 13:19] = 0
        batch["sp_hist"][..., 13:19] = 0
    elif mode == "no_contact":
        batch["sp_fut"][..., 27:29] = 0
        batch["sp_hist"][..., 27:29] = 0
    elif mode == "no_history":
        batch["sp_hist"][...] = 0
    return batch


@torch.no_grad()
def run_val(model, teacher, loader, stats, fk, device, lam, ablation="none",
            loco_mode=False):
    model.eval()
    agg, cs_all, cr_all = {}, [], []
    with torch.no_grad():
        for batch in loader:
            batch = ablate({k: v.to(device) if torch.is_tensor(v) else v
                            for k, v in batch.items()}, ablation)
            c_r = teacher.encode(batch["x"]).detach()
            if loco_mode:
                batch["sp_fut"][...] = 0
            c_s = model(batch["sp_hist"], batch["sp_fut"],
                        build_cmd(batch, loco_mode))
            motion_s, contact_s = teacher.decode(c_s)
            motion_r, contact_r = teacher.decode(c_r)
            total, parts, _ = loss_fn(batch, motion_s, contact_s, stats, fk)
            agg["total"] = agg.get("total", 0.0) + total.item()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            with torch.no_grad():
                _, parts_r, _ = loss_fn(batch, motion_r, contact_r, stats, fk)
            agg["teacher_q"] = agg.get("teacher_q", 0.0) + float(parts_r["q"])
            agg["teacher_fk"] = agg.get("teacher_fk", 0.0) + float(parts_r["fk"])
            agg["align"] = agg.get("align", 0.0) + \
                torch.nn.functional.mse_loss(c_s, c_r).item()
            cs_all.append(c_s.cpu())
            cr_all.append(c_r.cpu())
    n = len(loader)
    cs, cr = torch.cat(cs_all), torch.cat(cr_all)
    return {k: v / n for k, v in agg.items()}, r2_per_dim(cs, cr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--ablation", default="none",
                    choices=["none", "no_head", "no_contact", "no_history"],
                    help="mask sparse input dims (zero in normalized space = "
                         "set to train mean = no information)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--hist-noise", type=float, default=0.0,
                    help="Gaussian noise on sparse history (sigma in normalized "
                         "units) — simulates the gap between planner's own "
                         "history and sim-real feedback, improving AR robustness")
    ap.add_argument("--neck-weight", type=float, default=1.0,
                    help="upweight neck dims in the joint losses so the "
                         "student must read the head command (editability)")
    ap.add_argument("--cmd-dropout", type=float, default=0.0,
                    help="prob of masking HISTORY head dims per sample — the "
                         "future head command becomes the only head signal")
    ap.add_argument("--explicit-cmd", action="store_true",
                    help="P2.8: explicit joint-space neck command channel "
                         "(CommandedEncoder)")
    ap.add_argument("--loco-cmd", action="store_true",
                    help="P2.9a: add [vx,vy,vyaw] to the command and MASK the "
                         "future sparse input — command REPLACES the future "
                         "trajectory (true planner form)")
    ap.add_argument("--graft-neck", type=float, default=-1.0,
                    help="prob of NECK-ONLY graft (donor neck cmd+joints on host "
                         "body/loco). Default: follows --graft. Decoupling these "
                         "raises host-loco x donor-neck combo density — the "
                         "command-disentanglement lever (P2.10)")
    ap.add_argument("--graft", type=float, default=0.0,
                    help="prob of counterfactual neck graft per sample: donor "
                         "window's neck joints grafted onto this window's body; "
                         "input head fields recomputed by FK, target latent "
                         "re-encoded from the grafted full window")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    # v1.1 trains under the corrected representation (GPT ruling 2026-08-24)
    from humanoidverse.planner.dataset import set_preprocess_version
    set_preprocess_version("fixed_v1")
    md = MotionData()
    stats = load_stats(OUT_DIR / "p1_stats.json")

    sp_path = OUT_DIR / "p2_stats32.json"
    if sp_path.exists():
        sp_stats = {k: torch.tensor(v, dtype=torch.float32)
                    for k, v in json.loads(sp_path.read_text()).items()}
        print(f"sparse stats: loaded {sp_path}")
    else:
        t0 = time.time()
        sp_stats = compute_sparse_stats(md)
        sp_path.write_text(json.dumps({k: v.tolist() for k, v in sp_stats.items()}))
        print(f"sparse stats: computed ({time.time()-t0:.0f}s)")

    train_ds = P2Dataset(md, "train", stats=stats, sparse_stats=sp_stats)
    val_ds = P2Dataset(md, "val", stats=stats, sparse_stats=sp_stats,
                       max_windows=2000)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=2)

    tckpt = torch.load(TEACHER, map_location=device, weights_only=False)
    teacher = MotionAE(in_dim=MODEL_DIM,
                       latent_dim=tckpt["config"]["latent"]).to(device)
    teacher.load_state_dict(tckpt["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    enc_cls = CommandedEncoder if args.explicit_cmd else SparseEncoder
    model = enc_cls(latent_dim=args.latent,
                    cmd_dim=7 if args.loco_cmd else 4).to(device)
    print(f"student: {model.num_params()/1e6:.2f}M params, latent R^{args.latent}; "
          f"train {len(train_ds)} / val {len(val_ds)} windows")

    fk = BDXFK().to(device)
    sp_m = sp_stats["mean"].to(device)
    sp_s = sp_stats["std"].to(device)

    def augment(batch):
        """Command-controllability augmentations (P2.6), applied pre-teacher.
        v1.1: manifest-intervention samples (command_source==1) keep their
        requested-command/matched-target pairing — graft would re-break it."""
        src = batch.get("command_source")
        if args.loco_cmd:
            batch["sp_fut"][...] = 0   # command replaces the future trajectory
        if src is not None:
            nat = (src == 0)
            # mask natural samples through graft; intervention samples skip
            global _nat_mask
            _nat_mask = nat
        else:
            _nat_mask = None
        b = batch["x"].shape[0]
        if args.cmd_dropout > 0:
            keep = (torch.rand(b, device=device) >= args.cmd_dropout)
            batch["sp_hist"][..., 13:19] *= keep[:, None, None]
        gn = args.graft_neck if args.graft_neck >= 0 else args.graft
        if max(gn, args.graft) > 0:
            roll = torch.roll(torch.arange(b, device=device), 1)
            sel = (torch.rand(b, device=device) < gn)
            if _nat_mask is not None:
                sel = sel & _nat_mask                    # intervention: no graft
            sel = sel[:, None, None]
            # grafted raw q: this body + donor neck
            q_g = batch["q"].clone()
            q_g[..., 10:14] = torch.where(sel, batch["q"][roll][..., 10:14],
                                          q_g[..., 10:14])
            pos_g, R_all = fk.forward(batch["base_pos"], batch["base_R"], q_g,
                                      return_rot=True)
            rv = rotmat_to_rotvec(R_all[..., 12, :, :])
            rv_n = (rv - sp_m[:, 13:16]) / sp_s[:, 13:16]
            hp_g = pos_g[:, :, 12, :]  # grafted head position (canonical)
            hp_n = (hp_g - sp_m[:, 29:32]) / sp_s[:, 29:32]
            for key in ("sp_fut", "sp_hist"):
                batch[key][..., 13:16] = torch.where(sel, rv_n,
                                                     batch[key][..., 13:16])
                batch[key][..., 16:19] = torch.where(
                    sel, batch[key][roll][..., 16:19], batch[key][..., 16:19])
                batch[key][..., 29:32] = torch.where(
                    sel, hp_n, batch[key][..., 29:32])
            if "cmd" in batch:   # explicit command follows the donor neck
                batch["cmd"] = torch.where(sel, batch["cmd"][roll], batch["cmd"])
            if "loco" in batch and args.loco_cmd and args.graft > 0:
                # INTERVENTIONAL pair (P2.9c): donor's ENTIRE future (targets,
                # loco command) on the host history — same past x different
                # commanded future. Without this the loco command always
                # matched the host's own future and the planner ignored it
                # (vyaw gain measured 0.00, the reviewer's trap #4).
                sel_f = (torch.rand(b, device=device) < args.graft)
                if _nat_mask is not None:
                    sel_f = sel_f & _nat_mask            # intervention: no graft
                for key in ("loco", "x", "q", "qdot", "base_pos", "base_R",
                            "base_linvel", "base_angvel", "contact", "cmd"):
                    if key in batch:
                        m = sel_f.reshape(b, *([1] * (batch[key].dim() - 1)))
                        batch[key] = torch.where(m, batch[key][roll], batch[key])
            # targets + teacher input follow the graft (normalized x is affine
            # per dim, so grafting normalized values is equivalent)
            batch["x"][..., 10:14] = torch.where(sel, batch["x"][roll][..., 10:14],
                                                 batch["x"][..., 10:14])
            batch["x"][..., 24:28] = torch.where(sel, batch["x"][roll][..., 24:28],
                                                 batch["x"][..., 24:28])
            batch["q"][..., 10:14] = q_g[..., 10:14]
            batch["qdot"][..., 10:14] = torch.where(
                sel, batch["qdot"][roll][..., 10:14], batch["qdot"][..., 10:14])
        return batch

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.epochs * args.max_steps)

    history, best = [], float("inf")
    tag = args.tag or (f"_{args.ablation}" if args.ablation != "none" else "")
    for epoch in range(args.epochs):
        lam = 0.0 if epoch < 2 else (0.5 if epoch == 2 else 1.0)
        model.train()
        t0, run = time.time(), {}
        for step, batch in enumerate(train_loader):
            if step >= args.max_steps:
                break
            batch = ablate({k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                            for k, v in batch.items()}, args.ablation)
            batch = augment(batch)
            if args.hist_noise > 0:
                batch["sp_hist"] = batch["sp_hist"] + \
                    torch.randn_like(batch["sp_hist"]) * args.hist_noise
            c_r = teacher.encode(batch["x"]).detach()
            c_s = model(batch["sp_hist"], batch["sp_fut"],
                        build_cmd(batch, args.loco_cmd))
            motion, contact = teacher.decode(c_s)
            loss, parts, _ = loss_fn(batch, motion, contact, stats, fk,
                                     neck_weight=args.neck_weight)
            if lam > 0:
                loss = loss + lam * torch.nn.functional.mse_loss(c_s, c_r)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in [("total", loss.item())] + list(parts.items()):
                run[k] = run.get(k, 0.0) + float(v)
        train_avg = {k: v / args.max_steps for k, v in run.items()}
        val_avg, r2 = run_val(model, teacher, val_loader, stats, fk, device,
                              lam, ablation=args.ablation, loco_mode=args.loco_cmd)
        ratio = val_avg["q"] / val_avg["teacher_q"]
        history.append({"epoch": epoch, "lam": lam, "train": train_avg,
                        "val": val_avg, "r2_mean": float(r2.mean()),
                        "student_teacher_q_ratio": ratio})
        marker = ""
        if val_avg["total"] < best:
            best = val_avg["total"]
            torch.save({"model": model.state_dict(), "teacher": str(TEACHER),
                        "config": vars(args), "epoch": epoch, "val": val_avg,
                        "r2": r2.tolist(), "ablation": args.ablation},
                       OUT_DIR / f"p2_student{tag}.pt")
            marker = "  *best*"
        print(f"epoch {epoch} (lam {lam}): train total {train_avg['total']:.4f} "
              f"| val q {val_avg['q']:.4f} vs teacher {val_avg['teacher_q']:.4f} "
              f"(ratio {ratio:.2f}) | align {val_avg['align']:.4f} | "
              f"R2 mean {r2.mean():.3f} [{time.time()-t0:.0f}s]{marker}",
              flush=True)

    (OUT_DIR / f"p2_history{tag}.json").write_text(json.dumps(history, indent=1))
    print(f"done. best val {best:.4f}; -> {OUT_DIR / f'p2_student{tag}.pt'}")


if __name__ == "__main__":
    main()
