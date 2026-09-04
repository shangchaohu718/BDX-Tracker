"""Q2 actor conditioning audit (ruling 2026-08-24 night 10).

Q2.1 first-action valid-z swap, ALL 11 checkpoints per arm, same rng-606
32-window panel as probe_q15_d_curves/bootstrap:
  coarse response  ||a(s,z_dyn) - a(s,z_stand)||
  fine response    ||a(s,z_i) - a(s,z_j)|| (same-activity pair, state fixed)
  valid-z secant   ||Δa|| / ||Δz||  (both z from real source windows)
  semantic direction consistency
                   cos( a(s,z_i)-a(s,z_j) , u_i - u_j )  with u = expert
                   last_action at window end; null = shuffled pairing.

Q2.3 same-state multi-step trajectory separation at 4 representative
checkpoints (50k/150k/225k/275k), 6 reset states x 8 z-sources, 100 steps:
  matched-vs-wrong tracking advantage, corr(D_policy, D_source),
  trajectory retrieval, fall/action-rate.

Actor, B, normalizer and D all from the SAME checkpoint. Deterministic actions.
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
from humanoidverse.utils.bdx_state import bdx_motion_activity

SEQ = 8
DEV = torch.device("cuda")
HORIZON = 100
N_RESETS = 6
Q23_CKPTS = {50000: 50176, 150000: 150528, 225000: 225792, 275000: 275968}
N_Z = 8


def build_panel():
    """Replicate the rng-606 panel construction of probe_q15_d_curves."""
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


def to_dev(d, dev):
    return tree_map(lambda x: x.to(dev), d)


def main():
    eps, panel = build_panel()
    arms = {"A": sorted(Path("results/gate1-screenA-combo-300k").glob("ckpt_*"), key=lambda p: int(p.name.split("_")[1])),
            "B": sorted(Path("results/gate1-screenB-refzalign-300k").glob("ckpt_*"), key=lambda p: int(p.name.split("_")[1]))}
    rng_null = np.random.default_rng(707)

    q21, q23 = {}, {}

    # fixed env for rollouts (arm-independent)
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, DEV)

    for arm, cks in arms.items():
        rows21, rows23 = [], []
        for ck in cks:
            step = int(ck.name.split("_")[1])
            model = load_model_from_checkpoint_dir(ck, device="cuda"); model.eval(); model._obs_normalizer.eval()
            with torch.no_grad():
                # ---- encodings & single-state obs for the panel ----
                zs = torch.stack([
                    model.project_z(model._backward_map(model._obs_normalizer(
                        {k: eps[e]["observation"][k][w + 1:w + 1 + SEQ].to(DEV) for k in ("state", "privileged_state")}
                    )).mean(0, keepdim=True)).flatten() for e, w, _ in panel])
                from humanoidverse.utils.bdx_state import build_bdx_actor_history
                obs_single = tree_map(lambda *xs: torch.cat(xs), *[
                    model._obs_normalizer({
                        **{k: eps[e]["observation"][k][w + SEQ - 1:w + SEQ].to(DEV) for k in ("state", "privileged_state", "last_action")},
                        "history_actor": build_bdx_actor_history(
                            eps[e]["observation"]["state"], eps[e]["observation"]["last_action"], w + SEQ - 1).unsqueeze(0).to(DEV)})
                    for e, w, _ in panel])
                # semantic direction reference: next-frame dof_pos difference
                # (pool last_action is all-zero — injected-state pool has no
                # expert actions). Panel z-scored per dof dim for balance.
                raw_dof = torch.stack([eps[e]["observation"]["state"][w + SEQ][0:14].to(DEV) for e, w, _ in panel])
                dof_std = raw_dof.std(0, keepdim=True).clamp_min(1e-6)
                u = (raw_dof - raw_dof.mean(0, keepdim=True)) / dof_std
                act_lv = np.array([a for _, _, a in panel]); med = np.median(act_lv)
                dyn_idx = np.where(act_lv > med)[0]; stand_idx = np.where(act_lv <= med)[0]

                def a_of(i, z):
                    o = tree_map(lambda x: x[i:i + 1], obs_single)
                    return model.act(o, z.unsqueeze(0), mean=True).flatten()

                z_stand = zs[torch.tensor(stand_idx[0])]
                # ---- Q2.1 metrics ----
                coarse_resp, fine_resp, secant, sem_fine, sem_fine_null, sem_coarse = [], [], [], [], [], []
                for i in dyn_idx:
                    ai = a_of(i, zs[i]); as_ = a_of(i, z_stand)
                    coarse_resp.append(float((ai - as_).norm()))
                    for j in dyn_idx:
                        if j == i:
                            continue
                        aj = a_of(i, zs[j])
                        da, dz = ai - aj, zs[i] - zs[j]
                        fine_resp.append(float(da.norm()))
                        secant.append(float(da.norm() / (dz.norm() + 1e-8)))
                        v_src = u[i] - u[j]
                        sem_fine.append(float(torch.nn.functional.cosine_similarity(da, v_src, dim=0)))
                    # coarse semantic: dyn-vs-stand action direction vs expert source diff
                    v_src_c = u[i] - u[stand_idx[0]]
                    sem_coarse.append(float(torch.nn.functional.cosine_similarity(ai - as_, v_src_c, dim=0)))
                # null: same cos distribution but shuffled source-pair partners
                dacts = [a_of(int(i), zs[i]) for i in dyn_idx]
                for i in range(len(dyn_idx)):
                    for j in range(len(dyn_idx)):
                        if i == j:
                            continue
                        k = rng_null.integers(0, len(dyn_idx))
                        v_src = u[dyn_idx[i]] - u[dyn_idx[k]]
                        sem_fine_null.append(float(torch.nn.functional.cosine_similarity(dacts[i] - a_of(int(dyn_idx[i]), zs[dyn_idx[j]]), v_src, dim=0)))
                rows21.append({
                    "step": step,
                    "coarse_resp_mean": float(np.mean(coarse_resp)),
                    "fine_resp_mean": float(np.mean(fine_resp)),
                    "secant_fine_mean": float(np.mean(secant)),
                    "sem_cos_fine_mean": float(np.mean(sem_fine)),
                    "sem_cos_fine_null_mean": float(np.mean(sem_fine_null)),
                    "sem_cos_coarse_mean": float(np.mean(sem_coarse)),
                })
            print(f"{arm} {step} Q2.1 done: {rows21[-1]}", flush=True)

            # ---- Q2.3 (only at representative checkpoints) ----
            if step not in Q23_CKPTS.values():
                del model; torch.cuda.empty_cache()
                continue
            # 8 z-sources: 6 dyn + 2 stand from panel
            z_src_idx = list(dyn_idx[:6]) + list(stand_idx[:2])
            with torch.no_grad():
                zs_src = [zs[torch.tensor(k)] for k in z_src_idx]
                src_trk = []  # normalized privileged source trajectories
                for k in z_src_idx:
                    e, w, _ = panel[k]
                    src_trk.append(model._obs_normalizer(
                        {"privileged_state": eps[e]["observation"]["privileged_state"][w:w + HORIZON].to(DEV)})["privileged_state"])
                L = min(s.shape[0] for s in src_trk)
                # policy rollouts: resets x z-sources
                trajs = np.zeros((N_RESETS, len(z_src_idx), HORIZON, 253), dtype=np.float32)
                act_rates = np.zeros((N_RESETS, len(z_src_idx)))
                term_steps = np.zeros((N_RESETS, len(z_src_idx)), dtype=int)
                for r in range(N_RESETS):
                    for zi, z in enumerate(zs_src):
                        torch.manual_seed(7000 + r); np.random.seed(7000 + r)
                        obs, _ = wrapped.reset(to_numpy=False)
                        o = to_dev(obs, DEV)
                        prev = None; rates = []
                        for h in range(HORIZON):
                            a = model.act(o, z.expand(o["state"].shape[0], -1), mean=True)
                            if prev is not None:
                                rates.append(float((a - prev).norm()))
                            prev = a
                            o_n = model._obs_normalizer(o)
                            trajs[r, zi, h] = o_n["privileged_state"][0].cpu().numpy()
                            o2, _, term, _, _ = wrapped.step(a, to_numpy=False)
                            o = to_dev(o2, DEV)
                            if term.any():
                                term_steps[r, zi] = h + 1
                                break
                        act_rates[r, zi] = float(np.mean(rates)) if rates else 0.0
                        # hold last valid frame after early termination (zeros would corrupt distances)
                        t_end = term_steps[r, zi]
                        if 0 < t_end < HORIZON:
                            trajs[r, zi, t_end:] = trajs[r, zi, t_end - 1]
                # tracking error matrix err[r, zi -> src k]
                src_arr = torch.stack([s[:L] for s in src_trk])  # [K, L, 253]
                tr = torch.from_numpy(trajs[:, :, :L, :]).to(DEV)  # [R,K,L,D]
                err = ((tr.unsqueeze(2) - src_arr.unsqueeze(0).unsqueeze(0)).norm(dim=-1)).mean(-1)  # [R, z_query, z_src]
                err = err.cpu().numpy()
                matched_adv = float(np.mean([err[r, i, i] - np.mean([err[r, j, i] for j in range(len(z_src_idx)) if j != i]) for r in range(N_RESETS) for i in range(len(z_src_idx))]))
                traj_ret = float(np.mean([int(np.argmin(err[r, :, i]) == i) for r in range(N_RESETS) for i in range(len(z_src_idx))]))
                # corr(D_policy, D_source): pairwise policy trajectory distances vs source distances
                K = len(z_src_idx)
                Dp = np.zeros((K, K)); Ds = np.zeros((K, K))
                for r in range(N_RESETS):
                    Dp += np.array([[np.linalg.norm(trajs[r, a_, :L] - trajs[r, b_, :L]) for b_ in range(K)] for a_ in range(K)])
                Dp /= N_RESETS
                Ds = np.array([[np.linalg.norm((src_trk[a_][:L] - src_trk[b_][:L]).cpu().numpy()) for b_ in range(K)] for a_ in range(K)])
                iu = np.triu_indices(K, 1)
                corr = float(np.corrcoef(Dp[iu], Ds[iu])[0, 1]) if np.std(Dp[iu]) > 0 and np.std(Ds[iu]) > 0 else float("nan")
                term_q = [int(q) for q in np.quantile(term_steps, (0.25, 0.5, 0.75, 1.0))]
                rows23.append({
                    "step": step,
                    "matched_vs_wrong_tracking_advantage": matched_adv,
                    "trajectory_retrieval_at1": traj_ret, "chance": 1 / K,
                    "corr_Dpolicy_Dsource_pearson": corr,
                    "mean_action_rate": float(act_rates.mean()),
                    "frac_terminated_before_100": float((term_steps < HORIZON).mean()),
                    "term_step_quartiles": term_q,
                })
            print(f"{arm} {step} Q2.3 done: {rows23[-1]}", flush=True)
            del model; torch.cuda.empty_cache()
        q21[arm] = rows21; q23[arm] = rows23

    result = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "probe": "probe_q2_actor_conditioning.py",
            "q21": "first-action valid-z swap, all 11 ckpts, rng-606 32-window panel, deterministic act",
            "q23": f"multi-step separation @ {sorted(Q23_CKPTS.values())}, {N_RESETS} resets x 8 z (6 dyn+2 stand), {HORIZON} steps, seeds 7000+r",
            "pairing": "same reset seeds across arms (cross-arm paired)",
            "status": "exploratory; no multiplicity correction",
        },
        "Q21_first_action_swap": q21,
        "Q23_trajectory_separation": q23,
    }
    p = Path("results/bfmzero-bdx-full/probe_q2_actor_conditioning.json")
    assert not p.exists()
    p.write_text(json.dumps(result, indent=2) + "\n")
    print("saved")


if __name__ == "__main__":
    main()
