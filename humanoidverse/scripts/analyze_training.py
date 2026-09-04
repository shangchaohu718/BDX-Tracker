"""Training analysis + plots for the BFM-zero BDX run.

Reads results/bfmzero-bdx-full/{humanoidverse_tracking_eval.csv,train_log.txt}
and writes analysis/ under the work dir:
  - completion_curve.png   completion median + p25/p75 vs steps  (THE metric)
  - mpjpe_curve.png        per-motion tracking error median vs steps
  - emd_curve.png          distribution emd vs steps
  - training_health.png    FPS / disc_loss / actor_loss / z_norm vs time
  - category_completion.png per-category completion at the latest eval (bar)
  - summary.txt            text digest
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else "results/bfmzero-bdx-full")
OUT = WORK / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def load_eval():
    rows = list(csv.DictReader(open(WORK / "humanoidverse_tracking_eval.csv")))
    by_t = defaultdict(lambda: defaultdict(list))
    for r in rows:
        t = int(float(r["timestep"]))
        for k in ("completion", "n_resets", "mpjpe_l", "emd", "obs_state_emd"):
            if r.get(k):
                by_t[t][k].append(float(r[k]))
        # category from motion_file like "v1_walk/xxx"
        mf = r.get("motion_file") or r.get("motion_id")
        if mf:
            by_t[t]["_cats"].append((str(mf).split("/")[0], float(r.get("completion") or 0)))
    return by_t


def load_train_log():
    rows = list(csv.DictReader(open(WORK / "train_log.txt")))
    out = {k: [] for k in ("timestep", "FPS", "disc_loss", "actor_loss", "z_norm", "fb_loss")}
    for r in rows:
        for k in out:
            if r.get(k):
                out[k].append(float(r[k]))
    return out


def main():
    lines = []
    by_t = load_eval()
    tl = load_train_log()
    if not by_t:
        print("no eval data yet")
        return

    ts = sorted(by_t)
    lines.append(f"评估点: {ts}")

    # 1) completion curve (median + quartiles)
    fig, ax = plt.subplots(figsize=(8, 5))
    meds = [med(by_t[t]["completion"]) for t in ts]
    p25 = [float(np.percentile(by_t[t]["completion"], 25)) if by_t[t]["completion"] else np.nan for t in ts]
    p75 = [float(np.percentile(by_t[t]["completion"], 75)) if by_t[t]["completion"] else np.nan for t in ts]
    ax.plot(ts, meds, "o-", label="median")
    ax.fill_between(ts, p25, p75, alpha=0.25, label="p25-p75")
    ax.set_xlabel("env steps"); ax.set_ylabel("completion (motion completion rate)")
    ax.set_title("Motion Completion Rate vs Training Steps"); ax.grid(alpha=0.3); ax.legend()
    fig.savefig(OUT / "completion_curve.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    lines.append(f"completion中位: " + ", ".join(f"{t/1e6:.0f}M:{m:.3f}" for t, m in zip(ts, meds) if not np.isnan(m)))

    # 2) mpjpe curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, [med(by_t[t]["mpjpe_l"]) / 1000 for t in ts], "s-", color="tab:red")
    ax.set_xlabel("env steps"); ax.set_ylabel("mpjpe_l median (m)")
    ax.set_title("Per-motion Tracking Error vs Steps"); ax.grid(alpha=0.3)
    fig.savefig(OUT / "mpjpe_curve.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # 3) emd curve
    if any(by_t[t]["emd"] for t in ts):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ts, [med(by_t[t]["emd"]) for t in ts], "^-", color="tab:green")
        ax.axhline(0.75, ls="--", c="gray", label="paper guideline 0.75")
        ax.set_xlabel("env steps"); ax.set_ylabel("emd median"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title("Distribution EMD vs Steps")
        fig.savefig(OUT / "emd_curve.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # 4) training health
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    axs[0, 0].plot(tl["timestep"], tl["FPS"]); axs[0, 0].set_title("FPS")
    axs[0, 1].plot(tl["timestep"], tl["disc_loss"]); axs[0, 1].set_title("disc_loss")
    axs[1, 0].plot(tl["timestep"], tl["actor_loss"]); axs[1, 0].set_title("actor_loss")
    axs[1, 1].plot(tl["timestep"], tl["z_norm"]); axs[1, 1].set_title("z_norm")
    for ax in axs.flat:
        ax.grid(alpha=0.3); ax.set_xlabel("steps")
    fig.tight_layout()
    fig.savefig(OUT / "training_health.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # 5) per-category completion at the LATEST eval
    last = ts[-1]
    cats = defaultdict(list)
    for mf, c in by_t[last]["_cats"]:
        cats[mf].append(c)
    if cats:
        fig, ax = plt.subplots(figsize=(9, 5))
        names = sorted(cats)
        vals = [med(cats[n]) for n in names]
        ax.bar(names, vals, color="tab:blue")
        ax.set_ylabel("completion median"); ax.set_title(f"Completion by category @ {last/1e6:.0f}M steps")
        ax.tick_params(axis="x", rotation=30); ax.grid(alpha=0.3, axis="y")
        fig.savefig(OUT / "category_completion.png", dpi=120, bbox_inches="tight"); plt.close(fig)
        lines.append("分类完成率@最新: " + ", ".join(f"{n}:{med(v):.3f}" for n, v in sorted(cats.items())))

    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n✓ plots -> {OUT}/")


if __name__ == "__main__":
    main()
