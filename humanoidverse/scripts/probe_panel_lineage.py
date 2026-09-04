"""Reconstruct motion/clip lineage for the 24 behavioral-panel reset states.

The v2 panel states are motion-sampled env resets (torch RNG: multinomial
motion id + rand phase + init noise, seed 5000+s_i). We replay each reset and
record motion id/name, phase, plus qpos/qvel checksums to verify faithful
reconstruction against probe_command_response_v2.json. On top of the lineage
we run motion-cluster permutation (sign flip) tests and clip-cluster
bootstrap CIs for the existing per-state behavioral metrics.

Ruling (2026-08-24 night 10): existing binomial/McNemar results remain
"conditional on state-level independence, exploratory".
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env

N_STATES = 24


def main():
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, torch.device("cuda"))
    inner = wrapped._env
    ref = json.loads(Path("results/bfmzero-bdx-full/probe_command_response_v2.json").read_text())
    ref_ck = ref["_provenance"]["reset_checksums"]
    raw_A = ref["raw"]["A"]

    from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
    agent_cfg = FBcprAuxAgentConfig(**cfg["agent"])
    kbuf = load_expert_trajectories_from_motion_lib(inner, agent_cfg, device="cpu", base_ang_vel_frame="body")
    # same id->name mapping convention as probe_command_response_v2 (m2e identity fallback)
    starts = kbuf.start_idx[:, 0].long()
    ep_motion = kbuf.storage["motion_id"][starts].long()
    m2e = {}
    for i in range(len(kbuf.file_names)):
        m2e.setdefault(int(ep_motion[i]), i)
    names = [str(x) for x in kbuf.file_names]

    lineage = []
    for s_i in range(N_STATES):
        torch.manual_seed(5000 + s_i)
        np.random.seed(5000 + s_i)
        wrapped.reset(to_numpy=False)
        qpos0 = inner.simulator.data.qpos.copy()
        qvel0 = inner.simulator.data.qvel.copy()
        mid = int(inner.motion_ids[0])
        t0 = float(inner.motion_start_times[0])
        name = names[m2e.get(mid, mid)]
        sha_q = hashlib.sha256(np.ascontiguousarray(qpos0).tobytes()).hexdigest()[:16]
        # qvel sha is NOT reproducible even in-process (mj solver warm-start
        # residue in the first read after a state write) — qpos sha is the
        # lineage verification channel.
        ok = sha_q == ref_ck[str(s_i)]["qpos_sha"]
        lineage.append({
            "state": s_i, "motion_id": mid, "motion_name": name,
            "motion_time": t0, "checksum_match": bool(ok),
            "retrieval_top1_A": raw_A[s_i]["retrieval_top1"],
            "retrieval_top1_B": ref["raw"]["B"][s_i]["retrieval_top1"],
            "z_specific_tracking_A": raw_A[s_i]["z_specific_tracking"],
            "z_specific_tracking_B": ref["raw"]["B"][s_i]["z_specific_tracking"],
        })
        if not ok:
            print(f"WARNING: state {s_i} checksum mismatch — lineage reconstruction diverged")

    names = [l["motion_name"] for l in lineage]
    groups = {}
    for l in lineage:
        groups.setdefault(l["motion_name"], []).append(l["state"])
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"{len(groups)} unique motions over {N_STATES} states; multi-state motions: {len(multi)}")
    for k, v in multi.items():
        print(f"  {k}: states {v}")

    rng = np.random.default_rng(20260824)
    out_clusters = {
        "n_unique_motions": len(groups),
        "n_multi_state_motions": len(multi),
        "multi_state_groups": {k: v for k, v in multi.items()},
        "states_per_motion_histogram": {
            str(k): int(v) for k, v in zip(*np.unique([len(v) for v in groups.values()], return_counts=True))
        },
    }

    def cluster_test(metric, arm, stat_fn):
        """Motion-cluster sign-flip permutation for mean(metric); plus cluster bootstrap CI."""
        vals = np.array([l[metric.format(arm=arm)] for l in lineage], dtype=float)
        # binary metrics stored as bool -> float
        cl = [np.array([l[metric.format(arm=arm)] for l in lineage if l["motion_name"] == n], dtype=float)
              for n in groups]
        obs = float(np.mean(vals))
        # cluster permutation: shuffle within nothing; sign flip not valid for one-sided binary —
        # use cluster permutation against motion label randomization for retrieval,
        # cluster bootstrap for means.
        boots = []
        keys = list(groups)
        for _ in range(10000):
            sel = rng.choice(len(keys), len(keys), replace=True)
            sample = np.concatenate([cl[i] for i in sel])
            boots.append(float(sample.mean()))
        ci = {"lo": float(np.quantile(boots, .025)), "hi": float(np.quantile(boots, .975))}
        return {"mean": obs, "cluster_bootstrap_ci95": ci, "n_clusters": len(keys)}

    results = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "probe": "probe_panel_lineage.py",
            "replay": "torch.manual_seed(5000+s_i)+np.random.seed+reset on cuda (device-matched); motion id/time from inner.motion_ids after kbuf preload",
            "verification": "qpos sha256 vs probe_command_response_v2.json reset_checksums (qvel sha excluded: nondeterministic warm-start residue)",
            "all_checksums_match": all(l["checksum_match"] for l in lineage),
            "status": "exploratory; existing binomial/McNemar remain conditional on state-level independence",
        },
        "cluster_summary": out_clusters,
        "per_state": lineage,
        "metrics": {
            "retrieval_top1_A": cluster_test("retrieval_top1_{arm}", "A", None),
            "retrieval_top1_B": cluster_test("retrieval_top1_{arm}", "B", None),
            "z_specific_tracking_A": cluster_test("z_specific_tracking_{arm}", "A", None),
            "z_specific_tracking_B": cluster_test("z_specific_tracking_{arm}", "B", None),
        },
    }
    # motion-cluster permutation for the paired A-vs-B retrieval difference
    retA = {l["state"]: l["retrieval_top1_A"] for l in lineage}
    retB = {l["state"]: l["retrieval_top1_B"] for l in lineage}
    obs_diff = np.mean([retB[s] - retA[s] for s in retA])
    cl_keys = list(groups)
    perm = []
    for _ in range(100000):
        d = 0.0
        for k in cl_keys:
            sgn = -1.0 if rng.random() < 0.5 else 1.0
            for s in groups[k]:
                d += sgn * (retB[s] - retA[s])
        perm.append(d / N_STATES)
    p = float(np.mean(np.array(perm) <= obs_diff)) if obs_diff < 0 else float(np.mean(np.array(perm) >= obs_diff))
    results["metrics"]["retrieval_top1_B_minus_A_cluster_permutation"] = {
        "observed": float(obs_diff), "p_one_sided": p, "n_clusters": len(cl_keys), "n_perm": 100000,
    }
    p_out = Path("results/bfmzero-bdx-full/probe_panel_lineage.json")
    assert not p_out.exists()
    p_out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()
