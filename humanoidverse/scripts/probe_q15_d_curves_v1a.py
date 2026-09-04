"""Correction-3 probe: v1-A D coarse/fine conditional time curves with
PREDEFINED period contrasts (ruling: no raw-peak reading).

Same construction as probe_q15_d_curves but on tracker_dataset_v1 pool +
v1-armA-300k checkpoints. Periods predefined:
  early  25k-75k   middle 125k-225k   late 250k-300k
Report per-period mean of the curve + window-level state-cluster bootstrap CI.
Exploratory.
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
POOL = "humanoidverse/data/bdx_expert_sim_pool_v1.pt"
PERIODS = {"early": (25000, 75000), "middle": (125000, 225000), "late": (250000, 300000)}


def main():
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
        acts.append((int(m), w, float(bdx_motion_activity(eps[m]["observation"]["state"][w:w+SEQ]).mean())))
    acts.sort(key=lambda x: x[2])
    panel = acts[-16:] + acts[:16]

    cks = sorted(Path("results/v1-armA-300k").glob("ckpt_*"), key=lambda p: int(p.name.split("_")[1]))
    rng_boot = np.random.default_rng(5150)
    rows = []
    for ck in cks:
        step = int(ck.name.split("_")[1])
        model = load_model_from_checkpoint_dir(ck, device="cuda"); model.eval(); model._obs_normalizer.eval()
        with torch.no_grad():
            def enc(e, w):
                nxt = {k: eps[e]["observation"][k][w+1:w+1+SEQ].to(DEV) for k in ("state", "privileged_state")}
                return model.project_z(model._backward_map(model._obs_normalizer(nxt)).mean(0, keepdim=True)).flatten()
            zs = torch.stack([enc(e, w) for e, w, _ in panel])
            obs = tree_map(lambda *xs: torch.cat(xs), *[
                model._obs_normalizer({k: eps[e]["observation"][k][w+SEQ-1:w+SEQ].to(DEV) for k in ("state", "privileged_state")})
                for e, w, _ in panel])
            act = np.array([a for _, _, a in panel]); med = np.median(act)
            dyn_m = act > med
            dyn_z = zs[torch.tensor(np.where(dyn_m)[0][:2])]
            stand_z = zs[torch.tensor(np.where(~dyn_m)[0][0])]

            def score(o_idx, z):
                o = tree_map(lambda x: x[o_idx:o_idx+1], obs)
                return float(model._discriminator.compute_logits(o, z.unsqueeze(0)).flatten()[0])

            m_fine, m_coarse, ret_hits = [], [], []
            for i in range(len(panel)):
                zi = zs[i]
                same_idx = np.where((act > med) == dyn_m[i])[0]
                same_act_other = zs[torch.tensor(same_idx[1])] if same_idx.size > 1 else zi
                opp_idx = np.where((act > med) != dyn_m[i])[0]
                opp = zs[torch.tensor(opp_idx[0])]
                m_fine.append(score(i, zi) - score(i, same_act_other))
                m_coarse.append(score(i, zi) - score(i, opp))
                uniq = []
                for c in [zi, dyn_z[0], dyn_z[1], stand_z]:
                    if not any(torch.allclose(c, u) for u in uniq):
                        uniq.append(c)
                sc = [score(i, c) for c in uniq]
                ret_hits.append(int(np.argmax(sc) == 0))
            mf, mc = np.array(m_fine), np.array(m_coarse)
            rows.append({
                "step": step,
                "std_margin_fine_dyn": float(mf.mean() / (mf.std() + 1e-8)),
                "std_margin_coarse_dynstand": float(mc.mean() / (mc.std() + 1e-8)),
                "D_retrieval_at1": float(np.mean(ret_hits)),
                "_raw_fine": mf.tolist(), "_raw_coarse": mc.tolist(), "_raw_ret": ret_hits,
            })
        print(f"v1-A {step}: fine={rows[-1]['std_margin_fine_dyn']:.3f} coarse={rows[-1]['std_margin_coarse_dynstand']:.3f} ret={rows[-1]['D_retrieval_at1']:.3f}", flush=True)
        del model; torch.cuda.empty_cache()

    # predefined period contrasts with window-level cluster bootstrap
    def period_stats(key):
        out = {}
        for pname, (lo, hi) in PERIODS.items():
            sel = [r for r in rows if lo <= r["step"] <= hi]
            if not sel:
                continue
            raw = np.array([r[f"_raw_{key}"] for r in sel])  # [n_ckpt_obs, n_win]
            flat = raw.flatten()
            # cluster bootstrap: resample windows jointly across ckpt-obs rows
            n_win = raw.shape[1]
            boots = []
            for _ in range(2000):
                wsel = rng_boot.integers(0, n_win, n_win)
                boots.append(float(raw[:, wsel].mean()))
            out[pname] = {"mean": float(raw.mean()),
                          "bootstrap_ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                          "steps": [r["step"] for r in sel]}
        return out

    curves = [{k: v for k, v in r.items() if not k.startswith("_raw")} for r in rows]
    result = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "probe": "probe_q15_d_curves_v1a.py",
            "artifact_class": "tracker_dataset_v1",
            "model": "results/v1-armA-300k, all 11 ckpts",
            "panel": "v1 pool rng-606 parity, 32 distinct-motion windows, activity-balanced",
            "periods_predefined": PERIODS,
            "status": "exploratory; descriptive curve + predefined-period contrasts only",
        },
        "curves": curves,
        "period_contrasts": {"margin_fine_dyn": period_stats("fine"), "margin_coarse_dynstand": period_stats("coarse")},
    }
    p = Path("results/bfmzero-bdx-full/probe_q15_d_curves_v1a.json")
    assert not p.exists()
    p.write_text(json.dumps(result, indent=2) + "\n")
    print("saved")


if __name__ == "__main__":
    main()
