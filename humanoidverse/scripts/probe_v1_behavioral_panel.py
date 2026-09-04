"""Behavioral response panel on tracker_dataset_v1 Arm A (300k).

Successor of probe_command_response_v2 with the night-12/13 fixes:
  - AGGREGATE matched-vs-wrong advantage (all queries) as primary, v2-style
    single-pair kept only as reference (F1 lesson: single-pair is unstable
    and non-discriminative).
  - motion lineage recorded AT RESET TIME (inner.motion_ids) — no replay
    reconstruction needed.
  - z sources from the v1 pool (dyn/dyn/dyn/stand), activity-stratified.
Outputs results/bfmzero-bdx-full/probe_v1_behavioral_panel_armA.json
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

SEQ, N_STATES, HORIZON = 8, 24, 150
STAND = ("stand", "pose", "idle")
DEV = torch.device("cuda")


def to_dev(d, dev):
    return tree_map(lambda x: x.to(dev), d)


def main():
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, DEV)
    inner = wrapped._env
    pool = torch.load("humanoidverse/data/bdx_expert_sim_pool_v1.pt", map_location="cpu")
    eps, meta = pool["episodes"], pool["meta"]
    # preload the FULL motion set into the runtime lib (train.py does this via
    # kbuf; without it the CPU build(1) lib holds ONE motion and every reset
    # lands on motion 0 — see correction_manifest_v1_panel.json)
    from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
    load_expert_trajectories_from_motion_lib(inner, FBcprAuxAgentConfig(**cfg["agent"]),
                                             device="cpu", base_ang_vel_frame="body")

    # z sources: activity-stratified deterministic pick from v1 pool
    acts = np.array([float(bdx_motion_activity(e["observation"]["state"][:SEQ * 4]).mean()) for e in eps])
    order = np.argsort(acts)
    rng = np.random.default_rng(909)
    dyn_ids = list(rng.choice(order[-len(order) // 2:], 3, replace=False))
    stand_ids = list(rng.choice(order[:len(order) // 2], 1, replace=False))
    z_specs = {f"dyn{i+1}": (int(e), int(rng.integers(20, 60))) for i, e in enumerate(dyn_ids)}
    z_specs["stand1"] = (int(stand_ids[0]), int(rng.integers(20, 60)))

    model = load_model_from_checkpoint_dir(Path("results/v1-armA-300k/checkpoint"), device="cuda")
    model.eval(); model._obs_normalizer.eval()

    def sim_src(e_idx, w):
        src = eps[e_idx]["observation"]["privileged_state"][w:min(w + HORIZON, 10**9)].to(DEV)
        return src

    per_state, lineage = [], []
    names = list(z_specs)
    K = len(names)
    err_all = np.zeros((N_STATES, K, K))
    for s_i in range(N_STATES):
        torch.manual_seed(5000 + s_i); np.random.seed(5000 + s_i)
        wrapped.reset(to_numpy=False)
        lineage.append({"state": s_i, "motion_id": int(inner.motion_ids[0]),
                        "motion_time": float(inner.motion_start_times[0])})
        run = {}
        with torch.no_grad():
            zs, srcs = {}, {}
            for nm, (e_idx, w0) in z_specs.items():
                nxt = {k: eps[e_idx]["observation"][k][w0 + 1:w0 + 1 + SEQ].to(DEV) for k in ("state", "privileged_state")}
                zs[nm] = model.project_z(model._backward_map(model._obs_normalizer(nxt)).mean(0, keepdim=True)).flatten()
                srcs[nm] = sim_src(e_idx, w0)
            L = min(s.shape[0] for s in srcs.values())
            trajs = {}
            for nm in names:
                torch.manual_seed(5000 + s_i); np.random.seed(5000 + s_i)
                obs, _ = wrapped.reset(to_numpy=False)
                o = to_dev(obs, DEV)
                tr = []
                for h in range(HORIZON):
                    a = model.act(o, zs[nm].expand(o["state"].shape[0], -1), mean=True)
                    o_n = model._obs_normalizer(o)
                    tr.append(o_n["privileged_state"][0])
                    o2, _, term, _, _ = wrapped.step(a, to_numpy=False)
                    o = to_dev(o2, DEV)
                    if term.any():
                        break
                Lr = min(len(tr), L)
                arr = torch.stack(tr[:Lr])
                if Lr < L:
                    arr = torch.cat([arr, arr[-1].expand(L - Lr, -1)])
                trajs[nm] = arr
            trq = torch.stack([trajs[nm] for nm in names])
            sr = torch.stack([srcs[nm][:L] for nm in names])
            err = ((trq.unsqueeze(1) - sr.unsqueeze(0)).norm(dim=-1)).mean(-1).cpu().numpy()
            err_all[s_i] = err
            i1, i2 = names.index("dyn1"), names.index("dyn2")
            run = {
                "v2_singlepair_dyn2_minus_dyn1_to_src1": float(err[i2, i1] - err[i1, i1]),
            }
        per_state.append({"state": s_i, **run})

    # aggregate metrics
    adv = [err_all[r, i, i] - np.mean([err_all[r, j, i] for j in range(K) if j != i])
           for r in range(N_STATES) for i in range(K)]
    dyn_idx = [names.index(n) for n in names if n != "stand1"]
    adv_dyn = [err_all[r, i, i] - np.mean([err_all[r, j, i] for j in dyn_idx if j != i])
               for r in range(N_STATES) for i in dyn_idx]
    ret = [int(np.argmin(err_all[r, :, i]) == i) for r in range(N_STATES) for i in range(K)]
    rngb = np.random.default_rng(20260825)
    def boot(v, f=np.mean):
        v = np.array(v, dtype=float)
        bs = [f(v[rngb.integers(0, len(v), len(v))]) for _ in range(10000)]
        return {"mean": float(v.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    # state-cluster bootstrap for retrieval (states as clusters)
    ret_arr = np.array(ret, dtype=float).reshape(N_STATES, K)
    clb = [ret_arr[rngb.integers(0, N_STATES, N_STATES)].mean() for _ in range(10000)]

    result = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "probe": "probe_v1_behavioral_panel.py",
            "artifact_class": "tracker_dataset_v1",
            "model": "results/v1-armA-300k (canonical Arm A on v1, seed 4728)",
            "setup": f"{N_STATES} resets (seeds 5000+i), 4-way z (3 dyn + 1 stand, v1 pool), {HORIZON} steps, deterministic",
            "lineage_recorded_at_reset": True,
            "primary_definition": "aggregate matched-vs-wrong (F1 lesson); v2 single-pair kept as reference only",
            "status": "exploratory single-arm (v1-A); B arm not yet trained on v1",
        },
        "z_source_specs": {k: list(v) for k, v in z_specs.items()},
        "metrics": {
            "aggregate_matched_vs_wrong_advantage_4way": boot(adv),
            "aggregate_advantage_dyn_candidates_only": boot(adv_dyn),
            "trajectory_retrieval_at1": {"mean": float(np.mean(ret)), "chance": 1 / K,
                                          "state_cluster_bootstrap_ci95": [float(np.quantile(clb, .025)), float(np.quantile(clb, .975))]},
            "v2_singlepair_reference": boot([p["v2_singlepair_dyn2_minus_dyn1_to_src1"] for p in per_state]),
        },
        "reset_lineage": lineage,
        "per_state": per_state,
        "err_matrix_mean": err_all.mean(0).round(3).tolist(),
    }
    out = Path("results/bfmzero-bdx-full/probe_v1_behavioral_panel_armA.json")
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
