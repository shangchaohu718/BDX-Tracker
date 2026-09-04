"""Step-1 analysis of Run A (baseline) / Run B (scale_reg off) train logs.

Reads the two CSV train_log.txt files, aligns them on timesteps 8192..819200,
and prints:

  1. aligned key-metric table (rollout/used reward, F norm, Q_fb abs, actor
     loss components, actor grad, param update ratio)
  2. collapse onset markers (first timestep where rollout reward crosses -1/-2)
  3. Run-B-only diagnostics around the collapse (Q_fb actor delta, Q_fb-D corr,
     top/bottom reward spread)
  4. lagged cross-correlation: do actor-update-induced Q_fb increases at time t
     predict FUTURE rollout-reward changes (temporal causality) as opposed to
     same-batch correlation?

Observe-only: reads CSVs, writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN_A = Path("results/bfm-dynamics-baseline/train_log.txt")
RUN_B = Path("results/bfm-dynamics-scale-off/train_log.txt")
T_MAX = 819_200


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["timestep"] <= T_MAX].set_index("timestep").sort_index()


def fmt_row(label: str, a, b, width=12):
    def f(x):
        return f"{x:>{width}.4g}" if x is not None and np.isfinite(x) else " " * width
    return f"{label:<34}{f(a)}{f(b)}"


def main() -> None:
    a, b = load(RUN_A), load(RUN_B)
    common = sorted(set(a.index) & set(b.index))
    print(f"Run A rows (<= {T_MAX}): {len(a)}   Run B rows: {len(b)}   aligned: {len(common)}")
    print(f"Run A timesteps {a.index.min()}..{a.index.max()}   Run B {b.index.min()}..{b.index.max()}")

    # ---- 1. aligned table at a few milestones ----
    milestones = [t for t in (8192, 16384, 32768, 49152, 65536, 98304, 163840, 245760, 409600, 819200) if t in a.index and t in b.index]
    cols = [
        ("diag/reward_rollout_z", "rollout-z reward"),
        ("diag/reward_used_z", "used-z reward"),
        ("F_norm", "F norm"),
        ("Q_fb_abs", "|Q_fb|"),
        ("actor_loss_fb_component", "actor L: FB"),
        ("actor_loss_disc_component", "actor L: D"),
        ("actor_loss_aux_component", "actor L: Aux"),
        ("actor_grad_norm", "actor grad"),
        ("actor_param_update_ratio", "upd ratio"),
        ("mean_disc_reward", "mean disc rew"),
        ("mean_next_Q", "next Q"),
    ]
    for t in milestones:
        print(f"\n== step {t} ==                 {'Run A':>12}{'Run B':>12}")
        for col, label in cols:
            if col in a.columns and col in b.columns:
                print(fmt_row(label, a.loc[t, col], b.loc[t, col]))

    # ---- 2. collapse onset ----
    for name, df in (("A", a), ("B", b)):
        r = df["diag/reward_rollout_z"]
        for thresh in (-1.0, -2.0):
            cross = r.index[r < thresh]
            print(f"\nRun {name}: rollout reward first < {thresh}: {cross.min() if len(cross) else 'never'}")

    # ---- 3. Run B dynamics around collapse ----
    if "diag/Q_fb_actor_mean_delta" in b.columns:
        early = b[b.index <= 81920]  # first 10 rows ~ 82k
        print("\n-- Run B early diagnostics (per 8192-step row) --")
        show = b.loc[b.index <= 163840, [
            "diag/reward_rollout_z", "diag/Q_fb_actor_mean_delta", "diag/Q_fb_reward_corr",
            "diag/reward_Q_fb_top_minus_bottom10", "diag/actor_mean_action_delta", "Q_fb_abs",
        ]]
        print(show.to_string(float_format=lambda x: f"{x:8.4f}"))
        print(f"\nearly Q_fb_actor_delta mean: {early['diag/Q_fb_actor_mean_delta'].mean():.4f} "
              f"(frac>0: {(early['diag/Q_fb_actor_mean_delta'] > 0).mean():.2f})")
        print(f"early Q_fb-D corr mean: {early['diag/Q_fb_reward_corr'].mean():.4f}")

        # ---- 4. lagged causality: Q_fb delta / corr at t  vs  Δrollout reward t+k ----
        d = b[["diag/reward_rollout_z", "diag/Q_fb_actor_mean_delta", "diag/Q_fb_reward_corr"]].dropna()
        dr = d["diag/reward_rollout_z"].diff()  # change vs previous logged row
        print("\n-- lagged corr: X(t) vs Δrollout-reward(t+lag) --")
        for xcol in ("diag/Q_fb_actor_mean_delta", "diag/Q_fb_reward_corr"):
            row = []
            for lag in (0, 1, 2, 3, 5, 10):
                if lag == 0:
                    c = d[xcol].corr(dr)
                else:
                    c = d[xcol].iloc[:-lag].reset_index(drop=True).corr(dr.iloc[lag:].reset_index(drop=True))
                row.append(f"lag{lag}:{c:+.3f}")
            print(f"{xcol:<32}{'  '.join(row)}")
        # also: does ΔQ_fb_abs (proxy scale growth) lead rollout-reward decline?
        q = b["Q_fb_abs"].diff().dropna()
        rr = b["diag/reward_rollout_z"]
        for lag in (0, 1, 2, 5):
            c = q.iloc[:-lag].reset_index(drop=True).corr(rr.diff().iloc[lag:].reset_index(drop=True)) if lag else q.corr(rr.diff())
            print(f"Δ|Q_fb| vs Δrollout(t+{lag}): {c:+.3f}")

    # ---- 5. used-z vs rollout-z gap over time (Run A: is relabel 'helping' current D reward?) ----
    for name, df in (("A", a), ("B", b)):
        if {"diag/reward_used_z", "diag/reward_rollout_z"} <= set(df.columns):
            gap = (df["diag/reward_used_z"] - df["diag/reward_rollout_z"])
            print(f"\nRun {name} used-minus-rollout reward gap: mean {gap.mean():+.4f} "
                  f"@49k {gap.loc[49152] if 49152 in gap.index else float('nan'):+.4f} "
                  f"@last {gap.iloc[-1]:+.4f}")


if __name__ == "__main__":
    sys.exit(main())
