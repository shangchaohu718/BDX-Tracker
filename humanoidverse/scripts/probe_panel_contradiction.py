"""Localize the v2-vs-Q2.3 contradiction on Arm A (LEGACY-1462-POOL artifact).

v2 (probe_command_response_v2): 4-way z from fixed episodes/windows
  dyn1=(1345,32) dyn2=(833,48) dyn3=(427,42) stand1=(253,36) — all EARLY phase;
  advantage defined only for src1: err(dyn2->src1) - err(dyn1->src1) => +1.63.
Q2.3: 8-way z (6 dyn panel windows, arbitrary phase + 2 stand), advantage
  averaged over ALL queries => negative everywhere.

Factors tested from one rollout matrix per phase condition (early/late):
  F1 query set    : src1-only vs all-query mean
  F2 candidates   : 4-way (dyn1,dyn2,dyn3,stand1) vs same (fixed)
  F3 z phase      : w0 in [20,60] vs late window (T-300)
Rollouts: 6 resets (seeds 5000+i, v2 parity), 150 steps, deterministic.
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

DEV = torch.device("cuda")
HORIZON = 150
N_RESETS = 6
Z_SPECS = {"dyn1": (1345, 32), "dyn2": (833, 48), "dyn3": (427, 42), "stand1": (253, 36)}
SEQ = 8


def to_dev(d, dev):
    return tree_map(lambda x: x.to(dev), d)


def main():
    pool = torch.load("humanoidverse/data/bdx_expert_sim_pool_full.pt", map_location="cpu")
    eps = pool["episodes"]
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, DEV)
    out = {}
    for arm, ck in {"A": "results/gate1-screenA-combo-300k/checkpoint",
                    "B": "results/gate1-screenB-refzalign-300k/checkpoint"}.items():
        model = load_model_from_checkpoint_dir(Path(ck), device="cuda")
        model.eval(); model._obs_normalizer.eval()
        res = {}
        for phase in ("early", "late"):
            specs = {}
            for name, (e, w) in Z_SPECS.items():
                T = eps[e]["observation"]["state"].shape[0]
                specs[name] = (e, w if phase == "early" else max(SEQ + 2, T - 300))
            with torch.no_grad():
                zs, srcs = {}, {}
                for name, (e, w) in specs.items():
                    nxt = {k: eps[e]["observation"][k][w + 1:w + 1 + SEQ].to(DEV) for k in ("state", "privileged_state")}
                    zs[name] = model.project_z(model._backward_map(model._obs_normalizer(nxt)).mean(0, keepdim=True)).flatten()
                    srcs[name] = model._obs_normalizer(
                        {"privileged_state": eps[e]["observation"]["privileged_state"][w:w + HORIZON].to(DEV)})["privileged_state"]
                L = min(s.shape[0] for s in srcs.values())
                names = list(zs)
                K = len(names)
                # err[r, query_z, src]
                err = np.zeros((N_RESETS, K, K))
                for r in range(N_RESETS):
                    trajs = []
                    for name in names:
                        torch.manual_seed(5000 + r); np.random.seed(5000 + r)
                        obs, _ = wrapped.reset(to_numpy=False)
                        o = to_dev(obs, DEV)
                        tr = []
                        for h in range(HORIZON):
                            a = model.act(o, zs[name].expand(o["state"].shape[0], -1), mean=True)
                            o_n = model._obs_normalizer(o)
                            tr.append(o_n["privileged_state"][0].cpu().numpy())
                            o2, _, term, _, _ = wrapped.step(a, to_numpy=False)
                            o = to_dev(o2, DEV)
                            if term.any():
                                break
                        Lr = min(len(tr), L)
                        trajs.append(np.concatenate([np.array(tr[:Lr]), np.repeat(np.array(tr[max(0, Lr - 1)]), L - Lr, axis=0)]) if Lr < L else np.array(tr[:L]))
                    trq = torch.from_numpy(np.stack(trajs)).to(DEV)  # [K, L, D]
                    sr = torch.stack([srcs[n][:L] for n in names])    # [K, L, D]
                    err[r] = ((trq.unsqueeze(1) - sr.unsqueeze(0)).norm(dim=-1)).mean(-1).cpu().numpy()
                i1, i2 = names.index("dyn1"), names.index("dyn2")
                # F1: v2 definition (src1 only)
                f1 = float(np.mean(err[:, i2, i1] - err[:, i1, i1]))
                # all-query mean advantage (matched err minus mean wrong err)
                f_all = float(np.mean([err[r, i, i] - np.mean([err[r, j, i] for j in range(K) if j != i])
                                       for r in range(N_RESETS) for i in range(K)]))
                # dyn-only candidates advantage (exclude stand from wrong set)
                dyn_ids = [names.index(n) for n in names if n != "stand1"]
                f_dynonly = float(np.mean([err[r, i, i] - np.mean([err[r, j, i] for j in dyn_ids if j != i])
                                           for r in range(N_RESETS) for i in dyn_ids]))
                res[phase] = {
                    "v2_definition_advantage_dyn2_minus_dyn1_to_src1": f1,
                    "all_query_mean_advantage_4way": f_all,
                    "dyn_only_candidate_advantage": f_dynonly,
                    "err_matrix_mean_reset": err.mean(0).round(3).tolist(),
                }
            print(arm, phase, res[phase], flush=True)
        out[arm] = res
        del model; torch.cuda.empty_cache()

    p = Path("results/bfmzero-bdx-full/probe_panel_contradiction.json")
    assert not p.exists()
    p.write_text(json.dumps({
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "probe": "probe_panel_contradiction.py",
            "artifact_class": "LEGACY-1462-POOL",
            "setup": f"{N_RESETS} resets (seeds 5000+i, v2 parity), {HORIZON} steps, deterministic, final 300k ckpt",
            "factors": "F1 query set (src1-only vs all) x F3 z phase (early [20,60] vs late T-300); candidates fixed 4-way",
        },
    }, indent=2) + "\n" + json.dumps(out, indent=2) + "\n")
    print("saved")


if __name__ == "__main__":
    main()
