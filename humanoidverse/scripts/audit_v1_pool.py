"""Coverage/activity audit for tracker_dataset_v1 expert-sim pool (ruling night 12).

Offline checks, all against the v1 artifacts:
  1. coverage: unique motion_ids in v1 pool vs episodes in v1 train pkl (per-episode
     scheme must be 1:1)
  2. activity distribution: pool windows vs kinematic buffer, dof_vel-based
     bdx_motion_activity quartiles
  3. composition: stand-vs-dynamic by name family, plus v1 semantic family mix
  4. legacy comparison: v1 pool activity/stand stats vs legacy_tracker_pool_v0 stats
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from humanoidverse.utils.bdx_state import bdx_motion_activity
from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env

V1_DATA = os.environ.get("BFM_POOL_DATA", "humanoidverse/data/bdx_14dof_train_v1.pkl")
V1_POOL = os.environ.get("BFM_COND_POOL", "humanoidverse/data/bdx_expert_sim_pool_v1.pt")
OUT = Path("results/bfmzero-bdx-full/v1_pool_coverage_activity_audit.json")
STAND = ("stand", "pose", "idle")


def main():
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, torch.device(os.environ.get("BFM_AUDIT_DEVICE", "cpu")))
    inner = wrapped._env
    kbuf = load_expert_trajectories_from_motion_lib(inner, FBcprAuxAgentConfig(**cfg["agent"]),
                                                    device="cpu", base_ang_vel_frame="body")
    pool = torch.load(V1_POOL, map_location="cpu")
    eps, meta = pool["episodes"], pool["meta"]
    names = [str(x) for x in kbuf.file_names]
    starts = kbuf.start_idx[:, 0].long()
    ep_motion = kbuf.storage["motion_id"][starts].long()
    m2e = {}
    for i in range(len(names)):
        m2e.setdefault(int(ep_motion[i]), i)

    pool_motions = {int(m["motion_id"]) for m in meta}
    kbuf_eps = int(kbuf.lengths.shape[0])
    coverage = {
        "kbuf_episodes": kbuf_eps, "pool_episodes": len(eps),
        "unique_pool_motion_ids": len(pool_motions),
        "coverage_fraction": len(pool_motions) / max(1, kbuf_eps),
        "per_episode_1to1": len(eps) == kbuf_eps and len(pool_motions) == kbuf_eps,
    }

    # activity: pool windows vs kinematic buffer (sampled)
    rng = np.random.default_rng(11)
    pool_act = []
    for i in rng.choice(len(eps), min(600, len(eps)), replace=False):
        st = eps[int(i)]["observation"]["state"]
        w = int(rng.integers(0, st.shape[0] - 8))
        pool_act.append(float(bdx_motion_activity(st[w:w + 8]).mean()))
    kbuf_act = []
    rows = kbuf.start_idx[:, 0].long(); lens = kbuf.lengths.long()
    for e in rng.choice(kbuf_eps, min(600, kbuf_eps), replace=False):
        s, L = int(rows[e]), int(lens[e])
        w = int(rng.integers(0, L - 8))
        st = kbuf.storage["observation"]["state"][s + w:s + w + 8]
        kbuf_act.append(float(bdx_motion_activity(st).mean()))

    fam = lambda n: "stand" if any(x in n.lower() for x in STAND) else "dynamic"
    pool_fam = Counter(fam(names[m2e[int(m["motion_id"])]]) for m in meta)
    # v1 semantic family via manifest keys
    man = json.load(open("results/bfmzero-bdx-full/tracker_dataset_v1_manifest.json"))
    sel_keys = set(man["selected_keys"])
    fam_of_key = {k: k.split("/")[0] for k in sel_keys}
    version_mix = Counter(fam_of_key.get(names[m2e[int(m["motion_id"])]].rsplit("/", 1)[0] + "/" + names[m2e[int(m["motion_id"])]].split("/")[-1], "?") for m in meta)

    result = {
        "_provenance": {"created_at": datetime.now().isoformat(),
                        "probe": "audit_v1_pool.py", "data": V1_DATA, "pool": V1_POOL,
                        "artifact_class": "tracker_dataset_v1"},
        "coverage": coverage,
        "activity": {
            "pool_activity_quartiles": [float(q) for q in np.quantile(pool_act, (.25, .5, .75))],
            "kbuf_activity_quartiles": [float(q) for q in np.quantile(kbuf_act, (.25, .5, .75))],
            "pool_stand_family_fraction": pool_fam.get("stand", 0) / len(meta),
            "legacy_reference": {"stand_family_fraction_legacy_pool": 0.31,  # from P0.4 audit (legacy)
                                 "note": "legacy pool oversampled stand 22%->31%"},
        },
        "pool_version_mix_top": version_mix.most_common(12),
        "stats": pool.get("stats", {}),
    }
    assert not OUT.exists()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
