"""State-cluster bootstrap for Q1.5 D conditional-margin time curves (ruling 2026-08-24 night 10).

Recomputes per-window fine/coarse margins and 4-way retrieval hits on the SAME
32-window panel and 11 checkpoints per arm as probe_q15_d_curves.py (rng 606),
but stores PER-WINDOW values. Then, with windows as clusters (each window's
full checkpoint trajectory kept together):

  predefined periods (checkpoint steps):
    early  : 25k-75k    (25088, 50176, 75264)
    middle : 125k-225k  (125440, 150528, 175616, 200704, 225792)
    late   : 250k-300k  (250880, 275968)
  contrasts: middle_mean - early_mean, late_mean - middle_mean (per window
  aggregated to period means FIRST, then cluster bootstrap over windows)
  also: per-window slope (late - early), A/B paired per-window difference,
  coarse vs fine separate, D retrieval curves separate.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.bdx_state import bdx_motion_activity

SEQ = 8
DEV = torch.device("cuda")
PERIODS = {
    "early": [25088, 50176, 75264],
    "middle": [125440, 150528, 175616, 200704, 225792],
    "late": [250880, 275968],
}
N_BOOT = 10000


def build_panel():
    pool = torch.load("humanoidverse/data/bdx_expert_sim_pool_full.pt", map_location="cpu")
    eps = pool["episodes"]
    rng = np.random.default_rng(606)
    eligible = [i for i in range(len(eps)) if eps[i]["observation"]["state"].shape[0] > 160]
    q1_motions = rng.choice(eligible, 40, replace=False)
    for m in q1_motions:
        T = eps[m]["observation"]["state"].shape[0]
        for _ in range(4):
            rng.integers(10, T - 30)
    acts = []
    for m in rng.permutation(eligible)[:120]:
        T = eps[m]["observation"]["state"].shape[0]
        w = int(rng.integers(10, T - 30))
        acts.append((int(m), w, float(bdx_motion_activity(eps[m]["observation"]["state"][w:w + SEQ]).mean())))
    acts.sort(key=lambda x: x[2])
    return eps, acts[-16:] + acts[:16]


def main():
    eps, q15_panel = build_panel()
    arms = {"A": sorted(Path("results/gate1-screenA-combo-300k").glob("ckpt_*"), key=lambda p: int(p.name.split("_")[1])),
            "B": sorted(Path("results/gate1-screenB-refzalign-300k").glob("ckpt_*"), key=lambda p: int(p.name.split("_")[1]))}
    out = {}
    for arm, cks in arms.items():
        steps, fine, coarse, hits = [], [], [], []
        for ck in cks:
            step = int(ck.name.split("_")[1])
            model = load_model_from_checkpoint_dir(ck, device="cuda"); model.eval(); model._obs_normalizer.eval()
            with torch.no_grad():
                zs = torch.stack([
                    model.project_z(model._backward_map(model._obs_normalizer(
                        {k: eps[e]["observation"][k][w + 1:w + 1 + SEQ].to(DEV) for k in ("state", "privileged_state")}
                    )).mean(0, keepdim=True)).flatten() for e, w, _ in q15_panel])
                obs = tree_map(lambda *xs: torch.cat(xs), *[
                    model._obs_normalizer({k: eps[e]["observation"][k][w + SEQ - 1:w + SEQ].to(DEV) for k in ("state", "privileged_state")})
                    for e, w, _ in q15_panel])
                act = np.array([a for _, _, a in q15_panel]); med = np.median(act)
                dyn_m = act > med

                def score(o_idx, z):
                    o = tree_map(lambda x: x[o_idx:o_idx + 1], obs)
                    return float(model._discriminator.compute_logits(o, z.unsqueeze(0)).flatten())

                f_row, c_row, h_row = [], [], []
                for i in range(len(q15_panel)):
                    zi = zs[i]
                    same = np.where((act > med) == dyn_m[i])[0]
                    same_act_other = zs[torch.tensor(same[1])] if same.size > 1 else zi
                    opp = zs[torch.tensor(np.where((act > med) != dyn_m[i])[0][0])]
                    f_row.append(score(i, zi) - score(i, same_act_other))
                    c_row.append(score(i, zi) - score(i, opp))
                    uniq = []
                    for c in [zi, zs[torch.tensor(np.where(dyn_m)[0][0])], zs[torch.tensor(np.where(dyn_m)[0][1])], zs[torch.tensor(np.where(~dyn_m)[0][0])]]:
                        if not any(torch.allclose(c, u) for u in uniq):
                            uniq.append(c)
                    sc = [score(i, c) for c in uniq]
                    h_row.append(int(int(np.argmax(sc)) == 0))
            steps.append(step); fine.append(f_row); coarse.append(c_row); hits.append(h_row)
            del model; torch.cuda.empty_cache()
            print(f"{arm} {step} done", flush=True)
        out[arm] = {
            "steps": steps,
            "fine_margin_per_window": fine,       # [n_ckpt][n_window]
            "coarse_margin_per_window": coarse,
            "retrieval_hit_per_window": hits,
            "panel": [{"ep": e, "win": w, "activity": a} for e, w, a in q15_panel],
        }

    # ---- cluster bootstrap (windows are clusters) ----
    rng = np.random.default_rng(20260824)
    stats = {}
    for arm in out:
        steps = out[arm]["steps"]
        step_idx = {s: i for i, s in enumerate(steps)}
        stats[arm] = {}
        for metric in ("fine_margin_per_window", "coarse_margin_per_window", "retrieval_hit_per_window"):
            M = np.array(out[arm][metric])  # [T, W]
            per = {p: M[[step_idx[s] for s in ss], :].mean(0) for p, ss in PERIODS.items()}  # [W]
            obs_c = {f"{b}_minus_{a}": float(per[b].mean() - per[a].mean())
                     for a, b in (("early", "middle"), ("middle", "late"))}
            slope = per["late"] - per["early"]
            boots = {k: [] for k in obs_c}
            slope_boots = []
            W = M.shape[1]
            for _ in range(N_BOOT):
                idx = rng.integers(0, W, W)
                pb = {p: per[p][idx].mean() for p in per}
                for a, b in (("early", "middle"), ("middle", "late")):
                    boots[f"{b}_minus_{a}"].append(pb[b] - pb[a])
                slope_boots.append(slope[idx].mean())
            stats[arm][metric] = {
                "period_means": {p: float(per[p].mean()) for p in per},
                "contrasts": {k: {"est": obs_c[k],
                                  "ci95": [float(np.quantile(boots[k], .025)), float(np.quantile(boots[k], .975))]}
                              for k in obs_c},
                "slope_late_minus_early": {"mean": float(slope.mean()),
                                           "window_slope_quantiles": [float(q) for q in np.quantile(slope, (.25, .5, .75))],
                                           "bootstrap_ci95": [float(np.quantile(slope_boots, .025)), float(np.quantile(slope_boots, .975))],
                                           "frac_windows_negative": float((slope < 0).mean())},
            }

    # A/B paired difference on period means (same windows)
    paired = {}
    for metric in ("fine_margin_per_window", "coarse_margin_per_window", "retrieval_hit_per_window"):
        MA = np.array(out["A"][metric]); MB = np.array(out["B"][metric])
        steps = out["A"]["steps"]; step_idx = {s: i for i, s in enumerate(steps)}
        perA = {p: MA[[step_idx[s] for s in ss], :].mean(0) for p, ss in PERIODS.items()}
        perB = {p: MB[[step_idx[s] for s in ss], :].mean(0) for p, ss in PERIODS.items()}
        paired[metric] = {f"{p}_B_minus_A": float(perB[p].mean() - perA[p].mean()) for p in PERIODS}

    result = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "probe": "probe_q15_bootstrap.py (per-window recompute of probe_q15_d_curves Q1.5, same rng 606 panel)",
            "periods_predefined": PERIODS, "n_boot": N_BOOT, "cluster": "window (32 windows, distinct motions)",
            "status": "exploratory (bootstrap CIs, no multiplicity correction)",
        },
        "period_contrasts": stats,
        "paired_B_minus_A_period_means": paired,
        "per_window_curves": out,
    }
    p = Path("results/bfmzero-bdx-full/probe_q15_cluster_bootstrap.json")
    assert not p.exists()
    p.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"period_contrasts": stats, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
