"""Correction-2 probe (ruling 2026-08-24 night): sem_cos_fine with proper
uncertainty quantification on v1-armA-300k (current canonical Arm A).

For each of the 11 checkpoints:
  sem_cos_fine  = cos(a(s,z_i)-a(s,z_j), u_i-u_j) over dyn pairs
  permutation null = same values with (i,j) partner labels permuted (1000 draws)
  state-cluster bootstrap CI (10k draws) over panel windows

Panel: v1 pool, rng-606 construction parity (40 motion draw + 120 activity
windows, top16+bottom16 by activity), same as probe_q2_actor_conditioning.

Also reports sem_cos_coarse (dyn vs fixed stand) with its own bootstrap.
Exploratory; per-checkpoint p-values raw (no Holm here — ledger-level job).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env
from humanoidverse.utils.bdx_state import bdx_motion_activity, build_bdx_actor_history

SEQ = 8
DEV = torch.device("cuda")
POOL = "humanoidverse/data/bdx_expert_sim_pool_v1.pt"
N_PERM = 1000
N_BOOT = 10000


def build_panel():
    pool = torch.load(POOL, map_location="cpu")
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
    eps, panel = build_panel()
    cks = sorted(Path("results/v1-armA-300k").glob("ckpt_*"), key=lambda p: int(p.name.split("_")[1]))
    rng_perm = np.random.default_rng(808)
    rng_boot = np.random.default_rng(909)

    rows = []
    for ck in cks:
        step = int(ck.name.split("_")[1])
        model = load_model_from_checkpoint_dir(ck, device="cuda"); model.eval(); model._obs_normalizer.eval()
        with torch.no_grad():
            zs = torch.stack([
                model.project_z(model._backward_map(model._obs_normalizer(
                    {k: eps[e]["observation"][k][w + 1:w + 1 + SEQ].to(DEV) for k in ("state", "privileged_state")}
                )).mean(0, keepdim=True)).flatten() for e, w, _ in panel])
            obs_single = tree_map(lambda *xs: torch.cat(xs), *[
                model._obs_normalizer({
                    **{k: eps[e]["observation"][k][w + SEQ - 1:w + SEQ].to(DEV) for k in ("state", "privileged_state", "last_action")},
                    "history_actor": build_bdx_actor_history(
                        eps[e]["observation"]["state"], eps[e]["observation"]["last_action"], w + SEQ - 1).unsqueeze(0).to(DEV)})
                for e, w, _ in panel])
            raw_dof = torch.stack([eps[e]["observation"]["state"][w + SEQ][0:14].to(DEV) for e, w, _ in panel])
            dof_std = raw_dof.std(0, keepdim=True).clamp_min(1e-6)
            u = (raw_dof - raw_dof.mean(0, keepdim=True)) / dof_std
            act_lv = np.array([a for _, _, a in panel]); med = np.median(act_lv)
            dyn_idx = np.where(act_lv > med)[0]; stand_idx = np.where(act_lv <= med)[0]

            def a_of(i, z):
                o = tree_map(lambda x: x[i:i + 1], obs_single)
                return model.act(o, z.unsqueeze(0), mean=True).flatten()

            # per-pair values with window ids for cluster bootstrap
            pair_val, pair_i, pair_j = [], [], []
            coarse_val, coarse_i = [], []
            z_stand = zs[torch.tensor(stand_idx[0])]
            for ii, i in enumerate(dyn_idx):
                ai = a_of(i, zs[i]); as_ = a_of(i, z_stand)
                coarse_val.append(float(torch.nn.functional.cosine_similarity(ai - as_, u[i] - u[stand_idx[0]], dim=0)))
                coarse_i.append(int(i))
                for j in dyn_idx:
                    if j == i:
                        continue
                    da = ai - a_of(i, zs[j])
                    pair_val.append(float(torch.nn.functional.cosine_similarity(da, u[i] - u[j], dim=0)))
                    pair_i.append(int(i)); pair_j.append(int(j))

            pair_val = np.array(pair_val); pair_i = np.array(pair_i); pair_j = np.array(pair_j)
            coarse_val = np.array(coarse_val); coarse_i = np.array(coarse_i)
            obs_mean = float(pair_val.mean())
            coarse_mean = float(coarse_val.mean())

            # permutation null: permute partner j labels among pairs (breaks pairing, keeps marginals)
            null_means = np.empty(N_PERM)
            for p in range(N_PERM):
                jp = rng_perm.permutation(pair_j)
                null_vals = []
                for k in range(len(pair_val)):
                    null_vals.append(float(torch.nn.functional.cosine_similarity(
                        a_of(pair_i[k], zs[pair_i[k]]) - a_of(pair_i[k], zs[jp[k]]),
                        u[pair_i[k]] - u[jp[k]], dim=0)))
                null_means[p] = np.mean(null_vals)
            p_perm = float((null_means >= obs_mean).mean())

            # state-cluster bootstrap over dyn windows: resample dyn windows, keep their pairs
            uniq = np.unique(pair_i)
            boot = np.empty(N_BOOT)
            for b in range(N_BOOT):
                sel = rng_boot.choice(uniq, size=len(uniq), replace=True)
                vals = []
                for s in sel:
                    vals.extend(pair_val[(pair_i == s) | (pair_j == s)].tolist())
                boot[b] = np.mean(vals)
            ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]

            # coarse bootstrap (simple: windows are independent samples)
            cboot = np.array([coarse_val[rng_boot.integers(0, len(coarse_val), len(coarse_val))].mean() for _ in range(2000)])

        rows.append({
            "step": step,
            "sem_cos_fine_mean": obs_mean,
            "sem_cos_fine_bootstrap_ci95": ci,
            "sem_cos_fine_perm_p (H0: pairing permuted)": p_perm,
            "sem_cos_coarse_mean": coarse_mean,
            "sem_cos_coarse_bootstrap_ci95": [float(np.quantile(cboot, 0.025)), float(np.quantile(cboot, 0.975))],
        })
        print(f"v1-A {step}: {rows[-1]}", flush=True)
        del model; torch.cuda.empty_cache()

    result = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "probe": "probe_q21_semcos_ci_v1a.py",
            "artifact_class": "tracker_dataset_v1",
            "model": "results/v1-armA-300k (all 11 ckpts)",
            "panel": "v1 pool rng-606 parity construction, 32 windows, 16 dyn/16 stand",
            "null": f"partner-label permutation, {N_PERM} draws",
            "ci": f"state-cluster bootstrap over dyn windows, {N_BOOT} draws",
            "status": "exploratory; raw p-values, Holm at ledger level",
        },
        "per_checkpoint": rows,
    }
    p = Path("results/bfmzero-bdx-full/probe_q21_semcos_ci_v1a.json")
    assert not p.exists()
    p.write_text(json.dumps(result, indent=2) + "\n")
    print("saved")


if __name__ == "__main__":
    main()
