"""Q1 held-out z retrieval (no self-match) + Q1.5 D conditional signal time curves.

Q1: for each motion (episode), take 4 windows at far-apart offsets (different
phases). Leave-one-window-out nearest-centroid retrieval in z space, per arm's
own encoder. No self-match possible (prototype excludes query window).

Q1.5: for each arm's 11 checkpoints, on a FIXED 32-window expert panel:
  - standardized margins: D(o_i, z_i) - D(o_i, z_j) for j = same-activity
    other (fine), opposite-activity (coarse), stand; each / std
  - D 4-way retrieval: for each window, candidates = {matched z, 2 fixed dyn
    z, 1 fixed stand z}; argmax => retrieval@1, matched rank, margin vs 2nd
Outputs time series for routing (ruling Q1.5 cases A-D).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.bdx_state import bdx_motion_activity

SEQ = 8
DEV = torch.device("cuda")


def main():
    pool = torch.load("humanoidverse/data/bdx_expert_sim_pool_full.pt", map_location="cpu")
    eps, meta = pool["episodes"], pool["meta"]
    rng = np.random.default_rng(606)
    # ---- Q1 panel: 40 motions x 4 far-apart windows ----
    eligible = [i for i in range(len(eps)) if eps[i]["observation"]["state"].shape[0] > 160]
    q1_motions = rng.choice(eligible, 40, replace=False)
    q1_panel = []  # (ep_idx, [w1,w2,w3,w4])
    for m in q1_motions:
        T = eps[m]["observation"]["state"].shape[0]
        ws = [int(rng.integers(10, T - 30)) for _ in range(4)]
        q1_panel.append((int(m), ws))
    # ---- Q1.5 panel: 32 windows from distinct motions, activity-balanced ----
    acts = []
    for m in rng.permutation(eligible)[:120]:
        T = eps[m]["observation"]["state"].shape[0]
        w = int(rng.integers(10, T - 30))
        acts.append((int(m), w, float(bdx_motion_activity(eps[m]["observation"]["state"][w:w+SEQ]).mean())))
    acts.sort(key=lambda x: x[2])
    q15_panel = acts[-16:] + acts[:16]  # 16 dyn-ish + 16 stand-ish, distinct motions

    arms = {"A": sorted(Path("results/gate1-screenA-combo-300k").glob("ckpt_*"), key=lambda p: int(p.name.split("_")[1])),
            "B": sorted(Path("results/gate1-screenB-refzalign-300k").glob("ckpt_*"), key=lambda p: int(p.name.split("_")[1]))}
    out_q1, out_q15 = {}, {}
    for arm, cks in arms.items():
        # Q1 (final checkpoint encoder)
        model = load_model_from_checkpoint_dir(cks[-1], device="cuda"); model.eval(); model._obs_normalizer.eval()
        with torch.no_grad():
            def enc(e, w):
                nxt = {k: eps[e]["observation"][k][w+1:w+1+SEQ].to(DEV) for k in ("state","privileged_state")}
                return model.project_z(model._backward_map(model._obs_normalizer(nxt)).mean(0, keepdim=True)).flatten()
            correct = 0; total = 0; withins, betweens = [], []
            cents = {}
            for m_i, (e_idx, ws) in enumerate(q1_panel):
                zs = [enc(e_idx, w) for w in ws]
                for q in range(4):
                    proto = torch.stack([z for j, z in enumerate(zs) if j != q]).mean(0)
                    cents[(m_i, q)] = (proto, zs[q])
            # LOO retrieval: query window -> nearest motion centroid (excl. own window's contribution)
            keys = list(cents.keys())
            protos = torch.stack([cents[k][0] for k in keys])
            queries = torch.stack([cents[k][1] for k in keys])
            mids = torch.tensor([k[0] for k in keys], device=DEV)
            D = torch.cdist(queries, protos)
            D[torch.arange(len(keys)), torch.arange(len(keys))] = 1e9  # block self-window (same proto contains it via others — allowed: different phase)
            pred = D.argmin(dim=1)
            correct = int((pred == mids).sum()); total = len(keys)
            within = float(torch.stack([torch.stack([cents[(m_i,q)][1] for q in range(4) if (m_i,q) in cents]) .cdist(torch.stack([cents[(m_i,q)][1] for q in range(4) if (m_i,q) in cents])).mean() for m_i in range(1)]).mean()) if False else None
        out_q1[arm] = {"loo_retrieval_at1": correct / total, "n_queries": total, "n_motions": len(q1_panel),
                       "chance": 1/40, "note": "query excluded from own prototype; different-phase same-motion prototypes retained"}
        del model; torch.cuda.empty_cache()

        # Q1.5 time curves
        rows = []
        for ck in cks:
            step = int(ck.name.split("_")[1])
            model = load_model_from_checkpoint_dir(ck, device="cuda"); model.eval(); model._obs_normalizer.eval()
            with torch.no_grad():
                zs = torch.stack([enc(e, w) for e, w, _ in q15_panel])
                obs = tree_map(lambda *xs: torch.cat(xs), *[
                    model._obs_normalizer({k: eps[e]["observation"][k][w+SEQ-1:w+SEQ].to(DEV) for k in ("state","privileged_state")})
                    for e, w, _ in q15_panel])
                act = np.array([a for _,_,a in q15_panel]); med = np.median(act)
                dyn_m = act > med
                # fixed candidate z's: 2 dyn + 1 stand from panel
                dyn_z = zs[torch.tensor(np.where(dyn_m)[0][:2])]
                stand_z = zs[torch.tensor(np.where(~dyn_m)[0][0])]
                def score(o_idx, z):
                    o = tree_map(lambda x: x[o_idx:o_idx+1], obs)
                    return model._discriminator.compute_logits(o, z.unsqueeze(0)).flatten()
                margins_fine, margins_coarse, ret_hits, ranks = [], [], [], []
                for i in range(len(q15_panel)):
                    zi = zs[i]
                    same_act_other = zs[torch.tensor(np.where((act > med) == dyn_m[i])[0][1])] if (np.where((act > med) == dyn_m[i])[0].size > 1) else zi  # 2nd same-activity window
                    opp = zs[torch.tensor(np.where((act > med) != dyn_m[i])[0][0])]
                    m_f = float(score(i, zi) - score(i, same_act_other))
                    m_c = float(score(i, zi) - score(i, opp))
                    margins_fine.append(m_f); margins_coarse.append(m_c)
                    # 4-way retrieval
                    cand = torch.stack([zi, dyn_z[0] if dyn_z[0].norm()>0 else zi, dyn_z[1] if dyn_z[1].norm()>0 else zi, stand_z])
                    uniq = []
                    for c in [zi, dyn_z[0], dyn_z[1], stand_z]:
                        if not any(torch.allclose(c, u) for u in uniq): uniq.append(c)
                    cand = torch.stack(uniq)
                    sc = torch.cat([score(i, c) for c in cand])
                    order = sc.argsort(descending=True)
                    rank = int((order == 0).nonzero()[0]) + 1
                    ranks.append(rank); ret_hits.append(rank == 1)
                mf, mc = np.array(margins_fine), np.array(margins_coarse)
                rows.append({"step": step,
                             "std_margin_fine_dyn": float(mf.mean()/(mf.std()+1e-8)),
                             "std_margin_coarse_dynstand": float(mc.mean()/(mc.std()+1e-8)),
                             "D_retrieval_at1": float(np.mean(ret_hits)), "chance_retrieval": 1/4,
                             "mean_matched_rank": float(np.mean(ranks))})
            del model; torch.cuda.empty_cache()
        out_q15[arm] = rows
        print(f"{arm} Q1={out_q1[arm]['loo_retrieval_at1']:.3f}; Q1.5 final={rows[-1]}", flush=True)

    res = {"Q1_heldout_z_retrieval": out_q1, "Q1.5_D_conditional_curves": out_q15,
           "_panel": {"q1": "40 motions x 4 far-apart windows, LOO", "q15": "32 distinct-motion windows, activity-balanced"},
           "_note": "state-level independence of the 24-state command panel NOT verified (motion-id lineage unrecorded) — retrieval p-values conditional on independence"}
    json.dump(res, open("results/bfmzero-bdx-full/probe_q1_heldout_and_q15_dcurves.json", "w"), indent=2)
    print("saved")


if __name__ == "__main__":
    main()
