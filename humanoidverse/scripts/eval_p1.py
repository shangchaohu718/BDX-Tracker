"""P1 evaluation: reconstruction quality (layers 1-2) + latent diagnostics.

Layer 1 (reconstruction, held-out val windows):
  joint pos/vel error, base pos/angle error, FK foot/head error,
  contact accuracy/F1
Layer 2 (dynamic quality):
  joint acceleration error (finite diff, 50 Hz), mean horizontal foot speed
  during GT-contact frames (slip proxy, pred vs GT)
Latent diagnostics:
  per-dim Pearson correlation of c_r with forward speed and with the
  L/R foot-height alternation (gait phase proxy); t-SNE colored by category;
  latent interpolation demo (saved as npz for later playback).

Run: .venv/bin/python humanoidverse/scripts/eval_p1.py
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
from humanoidverse.planner.dataset import (MODEL_DIM, MotionData,
                                           WindowDataset, load_stats,
                                           sixd_to_mat)
from humanoidverse.planner.fk import BDXFK, HEAD, LFOOT, RFOOT
from humanoidverse.planner.model import MotionAE

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))
FK_BODIES = [LFOOT, RFOOT, HEAD]
DT = 1.0 / 50.0


@torch.no_grad()
def evaluate(model, loader, stats, fk, device):
    model.eval()
    mean = stats["mean"].to(device)
    std = stats["std"].to(device)
    sums = {}
    latents, fwd_speed, phase = [], [], []
    q_joint_sum = torch.zeros(14)   # per-joint |err| accumulation
    win = {"q": [], "foot": [], "head": [], "base_pos": [], "accel": []}

    for batch in loader:
        x = batch["x"].to(device)
        motion, contact_logits, c = model(x)
        latents.append(c.cpu())
        fwd_speed.append(batch["base_linvel"][..., 0].mean(-1).cpu())
        pred = motion * std + mean
        q_p, qdot_p = pred[..., 0:14], pred[..., 14:28]
        pos_p, R_p = pred[..., 28:31], sixd_to_mat(pred[..., 31:37])
        q, qdot = batch["q"].to(device), batch["qdot"].to(device)
        pos, R = batch["base_pos"].to(device), batch["base_R"].to(device)

        def add(k, v):
            sums[k] = sums.get(k, 0.0) + float(v)

        q_abs = (q_p - q).abs()                 # (B,W,14)
        add("q", q_abs.mean())
        q_joint_sum += q_abs.mean(dim=(0, 1)).cpu()
        win["q"].extend(q_abs.mean(dim=(1, 2)).cpu().tolist())
        add("qdot", (qdot_p - qdot).abs().mean())
        add("base_pos", (pos_p - pos).norm(dim=-1).mean())
        win["base_pos"].extend((pos_p - pos).norm(dim=-1).mean(dim=1).cpu().tolist())
        cos = ((R_p.transpose(-1, -2) @ R).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        add("base_ang", torch.acos(cos.clamp(-1, 1)).mean())

        bp_p = fk.forward(pos_p, R_p, q_p)[..., FK_BODIES, :]
        bp_g = fk.forward(pos, R, q)[..., FK_BODIES, :]
        foot_err = (bp_p[..., :2] - bp_g[..., :2]).norm(dim=-1)
        head_err = (bp_p[..., 2] - bp_g[..., 2]).norm(dim=-1)
        add("foot", foot_err.mean())
        add("head", head_err.mean())
        win["foot"].extend(foot_err.mean(dim=1).cpu().tolist())
        win["head"].extend(head_err.mean(dim=1).cpu().tolist())
        # gait phase proxy: L/R foot height alternation
        phase.append((bp_g[..., 0, 2] - bp_g[..., 1, 2]).mean(-1).cpu())

        ct_p = (contact_logits > 0).float()
        ct = batch["contact"].to(device)
        add("contact_acc", (ct_p == ct).float().mean())
        tp = ((ct_p == 1) & (ct == 1)).sum()
        fp = ((ct_p == 1) & (ct == 0)).sum()
        fn = ((ct_p == 0) & (ct == 1)).sum()
        prec, rec = tp / (tp + fp + 1e-6), tp / (tp + fn + 1e-6)
        add("contact_f1", 2 * prec * rec / (prec + rec + 1e-6))

        acc_p = (q_p[:, 2:] - 2 * q_p[:, 1:-1] + q_p[:, :-2]) / DT ** 2
        acc_g = (q[:, 2:] - 2 * q[:, 1:-1] + q[:, :-2]) / DT ** 2
        add("accel", (acc_p - acc_g).abs().mean())
        win["accel"].extend((acc_p - acc_g).abs().mean(dim=(1, 2)).cpu().tolist())

        # slip: mean horizontal foot speed on frames where GT says contact
        vel_p = (bp_p[:, 1:, :2] - bp_p[:, :-1, :2]).norm(dim=-1) / DT
        vel_g = (bp_g[:, 1:, :2] - bp_g[:, :-1, :2]).norm(dim=-1) / DT
        cm = (batch["contact"][:, 1:] > 0).to(device)
        add("slip_pred", (vel_p * cm).sum() / cm.sum())
        add("slip_gt", (vel_g * cm).sum() / cm.sum())

    n = len(loader)
    metrics = {k: v / n for k, v in sums.items()}
    return (metrics, torch.cat(latents).numpy(),
            torch.cat(fwd_speed).numpy(), torch.cat(phase).numpy(),
            (q_joint_sum / n).numpy(),
            {k: np.array(v) for k, v in win.items()})


def pearson(a, b):
    """a (N,D) latent dims, b (N,) scalar signal -> per-dim correlation (D,)."""
    a, b = a - a.mean(0), b - b.mean()
    return (a * b.unsqueeze(1)).sum(0) / (a.norm(dim=0) * b.norm() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-windows", type=int, default=4000)
    ap.add_argument("--checkpoint", default=None, help="default: canonical teacher")
    ap.add_argument("--no-diag", action="store_true",
                    help="skip t-SNE/interpolation (for ablation sweeps)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from humanoidverse.planner.registry import resolve_canonical
    if args.checkpoint is None:
        args.checkpoint = str(resolve_canonical(OUT_DIR)["teacher"])
    print(f"canonical: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MotionAE(in_dim=MODEL_DIM, latent_dim=ckpt["config"]["latent"]).to(device)
    model.load_state_dict(ckpt["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    fk = BDXFK().to(device)

    md = MotionData()
    val_ds = WindowDataset(md, "val", stride=25,
                           max_windows=args.max_windows, stats=stats)
    loader = DataLoader(val_ds, batch_size=128, num_workers=4)
    (metrics, lat, fwd, phase, q_joint,
     win) = evaluate(model, loader, stats, fk, device)
    cats = val_ds.categories[:len(lat)]
    tag = Path(args.checkpoint).stem.replace("p1_ae", "") or "_l32"

    print(f"P1 eval — {args.checkpoint} (epoch {ckpt['epoch']}), "
          f"{len(lat)} val windows")
    print("layer 1 (reconstruction):")
    print(f"  joint pos err     {metrics['q']*1000:7.1f} mrad")
    print(f"  joint vel err     {metrics['qdot']:7.3f} rad/s")
    print(f"  base pos err      {metrics['base_pos']*1000:7.1f} mm")
    print(f"  base ang err      {np.degrees(metrics['base_ang']):7.2f} deg")
    print(f"  FK foot err       {metrics['foot']*1000:7.1f} mm")
    print(f"  FK head err       {metrics['head']*1000:7.1f} mm")
    print(f"  contact acc/F1    {metrics['contact_acc']:.3f} / {metrics['contact_f1']:.3f}")
    print("layer 2 (dynamics):")
    print(f"  joint accel err   {metrics['accel']:7.1f} rad/s^2")
    print(f"  foot slip pred/gt {metrics['slip_pred']:.3f} / {metrics['slip_gt']:.3f} m/s")

    from humanoidverse.planner.fk import DOF_NAMES
    print("per-joint MAE (mrad):")
    order = np.argsort(-q_joint)
    for j in order:
        group = "neck" if j >= 10 else "leg"
        print(f"  [{group}] {DOF_NAMES[j]:<24} {q_joint[j]*1000:6.1f}")
    neck = q_joint[10:].mean() * 1000
    leg = q_joint[:10].mean() * 1000
    print(f"  neck mean {neck:.1f} vs leg mean {leg:.1f} mrad "
          f"(ratio {neck/leg:.2f})")

    print("per-category reconstruction:")
    cat_stats = {}
    for cat in sorted(set(cats)):
        sel = cats == cat
        cat_stats[cat] = {k: float(np.mean(v[sel]) * 1000)
                          for k, v in win.items() if k != "accel"}
        cat_stats[cat]["accel"] = float(np.mean(win["accel"][sel]))
        s = cat_stats[cat]
        print(f"  {cat:<12} q {s['q']:6.1f} mrad  foot {s['foot']:5.1f} mm  "
              f"head {s['head']:5.1f} mm  base {s['base_pos']:5.1f} mm  "
              f"accel {s['accel']:5.0f}")

    r_fwd = pearson(torch.from_numpy(lat), torch.from_numpy(fwd))
    r_pha = pearson(torch.from_numpy(lat), torch.from_numpy(phase))
    print("latent structure: max |corr| with forward speed "
          f"{r_fwd.abs().max():.2f} (dim {r_fwd.abs().argmax()}), "
          f"with phase proxy {r_pha.abs().max():.2f} (dim {r_pha.abs().argmax()})")

    (OUT_DIR / f"p1_eval{tag}.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "metrics": metrics,
         "per_joint_mrad": {DOF_NAMES[j]: float(q_joint[j] * 1000)
                            for j in range(14)},
         "per_category_mm": cat_stats,
         "n_val_windows": int(len(lat)), "epoch": ckpt["epoch"],
         "corr_fwd": r_fwd.tolist(), "corr_phase": r_pha.tolist()}, indent=1))

    if args.no_diag:
        return

    # ---- t-SNE ----
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        TSNE = None
    if TSNE is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = min(3000, len(lat))
        emb = TSNE(n_components=2, init="pca", perplexity=30,
                   random_state=0).fit_transform(lat[:n])
        cats_n = cats[:n]
        fig, ax = plt.subplots(figsize=(7, 6))
        for cat in sorted(set(cats_n)):
            sel = cats_n == cat
            ax.scatter(emb[sel, 0], emb[sel, 1], s=4, alpha=0.5, label=cat)
        ax.legend(markerscale=3)
        ax.set_title("teacher latent c_r (t-SNE, val windows)")
        fig.savefig(OUT_DIR / f"p1_tsne{tag}.png", dpi=140, bbox_inches="tight")
        print(f"t-SNE -> {OUT_DIR / f'p1_tsne{tag}.png'}")

    # ---- latent interpolation demo ----
    i_walk = int(np.where(cats == "locomotion")[0][0])
    i_gest = int(np.where(cats == "gesture")[0][0]) if (cats == "gesture").any() else 0
    with torch.no_grad():
        c = torch.from_numpy(lat[[i_walk, i_gest]]).to(device)
        steps = torch.linspace(0, 1, 7, device=device).unsqueeze(1)
        ci = (1 - steps) * c[0] + steps * c[1]
        motion, _ = model.decode(ci)
        q_i = (motion * stats["std"].to(device) + stats["mean"].to(device))[..., 0:14]
    np.savez(OUT_DIR / f"p1_interp{tag}.npz", q=q_i.cpu().numpy(),
             walk_idx=i_walk, gest_idx=i_gest)
    # continuity: joint-space step between consecutive interpolation alphas
    step = (q_i[1:] - q_i[:-1]).abs().mean().item()
    within = (q_i[:, 1:] - q_i[:, :-1]).abs().mean().item()
    print(f"interpolation demo -> {OUT_DIR / f'p1_interp{tag}.npz'} "
          f"(alpha-step |Δq| {step*1000:.1f} mrad vs within-window step "
          f"{within*1000:.1f} mrad; {'SMOOTH' if step < 3*within else 'JUMPY'})")


if __name__ == "__main__":
    main()
