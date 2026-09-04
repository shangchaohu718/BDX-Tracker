"""Command-response panel: same reset state, crossed z (final ruling).

For each of N_S fixed reset states s_i (unaligned env resets, paired seeds):
run (s_i, dyn_z1), (s_i, dyn_z2), (s_i, stand_z) for BOTH arms with identical
reset/DR/noise and z source identities. Records raw qpos/qvel checksum per
state to verify cross-arm pairing.

Deliverables:
  g_i = activity(s_i, dyn_z1) - activity(s_i, stand_z); E[g^B - g^A] CI
  |a_policy - a_source| absolute calibration error (stratified dyn/stand)
  matched-vs-permuted tracking: err(s_i, z2 -> src1) - err(s_i, z1 -> src1)
    (positive => z-specific tracking)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env
from humanoidverse.utils.bdx_state import bdx_motion_activity

SEQ, N_STATES, HORIZON = 8, 24, 150
STAND = ("stand", "pose", "idle")


def to_dev(d, dev):
    return tree_map(lambda x: x.to(dev), d)


def main():
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, torch.device("cuda"))
    inner = wrapped._env
    agent_cfg = FBcprAuxAgentConfig(**cfg["agent"])
    kbuf = load_expert_trajectories_from_motion_lib(inner, agent_cfg, device="cpu", base_ang_vel_frame="body")
    pool = torch.load("humanoidverse/data/bdx_expert_sim_pool_full.pt", map_location="cpu")
    eps, meta = pool["episodes"], pool["meta"]
    names = [str(x) for x in kbuf.file_names]
    starts = kbuf.start_idx[:, 0].long()
    ep_motion = kbuf.storage["motion_id"][starts].long()
    m2e = {}
    for i in range(len(names)):
        m2e.setdefault(int(ep_motion[i]), i)
    fam = lambda i: 1 if any(s in names[m2e[int(meta[i]["motion_id"])]].lower() for s in STAND) else 0

    rng = np.random.default_rng(404)
    dyn_pool = [i for i in range(len(eps)) if fam(i) == 0 and eps[i]["observation"]["state"].shape[0] > SEQ + HORIZON + 40]
    stand_pool = [i for i in range(len(eps)) if fam(i) == 1 and eps[i]["observation"]["state"].shape[0] > SEQ + HORIZON + 40]
    _picks_d = rng.choice(dyn_pool, 3, replace=False)
    z_specs = {"dyn1": (int(_picks_d[0]), int(rng.integers(20, 60))),
               "dyn2": (int(_picks_d[1]), int(rng.integers(20, 60))),
               "dyn3": (int(_picks_d[2]), int(rng.integers(20, 60))),
               "stand1": (int(rng.choice(stand_pool)), int(rng.integers(20, 60)))}

    def sim_src(e_idx, w):
        src = eps[e_idx]["observation"]["privileged_state"][w:min(w + HORIZON, 10**9)].cuda()
        sact = float(bdx_motion_activity(eps[e_idx]["observation"]["state"][w:w + SEQ]).mean())
        return src, sact

    arms = {"A": "results/gate1-screenA-combo-300k/checkpoint", "B": "results/gate1-screenB-refzalign-300k/checkpoint"}
    dev = torch.device("cuda")
    all_results, checksums = {}, {}
    for arm, ck in arms.items():
        model = load_model_from_checkpoint_dir(Path(ck), device="cuda")
        model.eval(); model._obs_normalizer.eval()
        per_state = []
        for s_i in range(N_STATES):
            torch.manual_seed(5000 + s_i); np.random.seed(5000 + s_i)
            wrapped.reset(to_numpy=False)
            qpos0 = inner.simulator.data.qpos.copy()
            qvel0 = inner.simulator.data.qvel.copy()
            if arm == "A":
                checksums[s_i] = {
                    "qpos_sha": hashlib.sha256(np.ascontiguousarray(qpos0).tobytes()).hexdigest()[:16],
                    "qvel_sha": hashlib.sha256(np.ascontiguousarray(qvel0).tobytes()).hexdigest()[:16],
                }
            obs0 = to_dev(wrapped._get_g1env_observation(to_numpy=False) if False else wrapped.reset(to_numpy=False)[0], dev)
            # re-reset deterministic per z-run: same seed => same reset+noise stream
            eps_run = {}
            for zname, (e_idx, w0) in z_specs.items():
                torch.manual_seed(5000 + s_i); np.random.seed(5000 + s_i)
                obs, _ = wrapped.reset(to_numpy=False)
                o = to_dev(obs, dev)
                with torch.no_grad():
                    nxt = {k: eps[e_idx]["observation"][k][w0 + 1:w0 + 1 + SEQ].to(dev) for k in ("state", "privileged_state")}
                    z_ep = model.project_z(model._backward_map(model._obs_normalizer(nxt)).mean(0, keepdim=True))
                src1, sact1 = sim_src(*z_specs["dyn1"])
                src_own, sact_own = sim_src(e_idx, w0)
                dvs, trk_own, trk_src1 = [], [], []
                with torch.no_grad():
                    for h in range(HORIZON):
                        a = model.act(o, z_ep.expand(o["state"].shape[0], -1), mean=True)
                        dvs.append(float(bdx_motion_activity(o["state"])[0]))
                        o_n = model._obs_normalizer(o)
                        trk_own.append(float((o_n["privileged_state"][0] - src_own[min(h, src_own.shape[0] - 1)]).norm()))
                        trk_src1.append(float((o_n["privileged_state"][0] - src1[min(h, src1.shape[0] - 1)]).norm()))
                        o2, _, term, _, _ = wrapped.step(a, to_numpy=False)
                        o = to_dev(o2, dev)
                        if term.any(): break
                eps_run[zname] = {
                    "activity_mean": float(np.mean(dvs)),
                    "abs_calib_err": float(abs(np.mean(dvs) - sact_own)),
                    "tracking_own_mean": float(np.mean(trk_own)),
                    "tracking_to_dyn1_mean": float(np.mean(trk_src1)),
                    "source_activity": sact_own,
                }
            g = eps_run["dyn1"]["activity_mean"] - eps_run["stand1"]["activity_mean"]
            # z-specific tracking: err under dyn2 to src1 minus err under dyn1 to src1 (positive = specific)
            zspec_w = eps_run["dyn2"]["tracking_to_dyn1_mean"] - eps_run["dyn1"]["tracking_to_dyn1_mean"]
            zspec_o = eps_run["stand1"]["tracking_to_dyn1_mean"] - eps_run["dyn1"]["tracking_to_dyn1_mean"]
            # retrieval top-1: which z-run tracks dyn1 best (should be dyn1 for command-specific tracker)
            cand = {k: eps_run[k]["tracking_to_dyn1_mean"] for k in ("dyn1", "dyn2", "dyn3", "stand1")}
            top1 = min(cand, key=cand.get) == "dyn1"
            per_state.append({"state": s_i, "g_dyn_minus_stand": g, "z_specific_tracking": zspec_w,
                              "z_specific_vs_stand": zspec_o, "retrieval_top1": top1, "runs": eps_run})
        all_results[arm] = per_state
        del model; torch.cuda.empty_cache()
        print(f"arm {arm} done", flush=True)

    def paired_ci(key):
        d = np.array([b[key] - a[key] for a, b in zip(all_results["A"], all_results["B"])])
        boots = [d[np.random.randint(0, len(d), len(d))].mean() for _ in range(10000)]
        return {"n": len(d), "delta": float(d.mean()), "lo": float(np.quantile(boots, .025)), "hi": float(np.quantile(boots, .975))}

    out = {
        "_provenance": {"created_at": datetime.now().isoformat(), "probe": "probe_command_response.py v2 (24 states x 4-way z)",
                        "states": N_STATES, "horizon": HORIZON, "deterministic_action": True,
                        "reset_checksums": checksums, "z_specs": {k: list(v) for k, v in z_specs.items()},
                        "bootstrap": "paired_state, 10000 reps"},
        "per_arm_summary": {
            arm: {
                "mean_g_dyn_minus_stand": float(np.mean([s["g_dyn_minus_stand"] for s in all_results[arm]])),
                "mean_z_specific_tracking": float(np.mean([s["z_specific_tracking"] for s in all_results[arm]])),
                "mean_z_specific_vs_stand": float(np.mean([s["z_specific_vs_stand"] for s in all_results[arm]])),
                "retrieval_top1_rate": float(np.mean([s["retrieval_top1"] for s in all_results[arm]])),
                "z_specific_tracking_state_std": float(np.std([s["z_specific_tracking"] for s in all_results[arm]])),
                "abs_calib_err_dyn1": float(np.mean([s["runs"]["dyn1"]["abs_calib_err"] for s in all_results[arm]])),
                "abs_calib_err_stand1": float(np.mean([s["runs"]["stand1"]["abs_calib_err"] for s in all_results[arm]])),
            } for arm in ("A", "B")},
        "contrasts": {
            "g_dyn_minus_stand_B_minus_A": paired_ci("g_dyn_minus_stand"),
            "z_specific_tracking_B_minus_A": paired_ci("z_specific_tracking"),
            "z_specific_vs_stand_B_minus_A": paired_ci("z_specific_vs_stand"),
        },
        "raw": all_results,
    }
    p = Path("results/bfmzero-bdx-full/probe_command_response_v2.json")
    assert not p.exists()
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("per_arm_summary", "contrasts")}, indent=2))


if __name__ == "__main__":
    main()
