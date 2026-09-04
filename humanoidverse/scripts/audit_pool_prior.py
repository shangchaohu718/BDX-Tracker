"""P0.4 pool prior audit: did the >=251-frame episode constraint and the
chunk-sampling scheme systematically bias the expert-sim pool's clip /
activity / stand-vs-dynamic distribution relative to the original dataset?

Checks (all offline):
  1. clip coverage: unique motion_ids in pool vs total episodes
  2. short-clip exclusion: how many dataset episodes cannot fit a 264-frame
     chunk (length < 266 frames) and what share of pool chunks they form
  3. activity: dof_vel L2 distribution, pool vs kinematic buffer
  4. stand vs dynamic composition by clip-name family
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from humanoidverse.utils.bdx_state import bdx_motion_activity

from humanoidverse.agents.envs.humanoidverse_isaac import (
    HumanoidVerseIsaacConfig,
    load_expert_trajectories_from_motion_lib,
)
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env

CHUNK = 264
STAND_MARKERS = ("stand", "pose", "idle")


def main():
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, torch.device("cuda"))
    inner = wrapped._env
    agent_cfg = FBcprAuxAgentConfig(**cfg["agent"])
    kbuf = load_expert_trajectories_from_motion_lib(inner, agent_cfg, device="cpu", base_ang_vel_frame="body")
    pool = torch.load("humanoidverse/data/bdx_expert_sim_pool_cpu.pt", map_location="cpu")
    meta, eps = pool["meta"], pool["episodes"]

    lengths = kbuf.lengths.long()
    n_eps = int(lengths.shape[0])
    names = [str(x) for x in kbuf.file_names] if getattr(kbuf, "file_names", None) else [f"ep{i}" for i in range(n_eps)]
    # motion id per episode from the buffer's motion_id rows
    starts = kbuf.start_idx[:, 0].long()
    ep_motion = kbuf.storage["motion_id"][starts].long()

    def family(name):
        n = name.lower()
        return "stand" if any(m in n for m in STAND_MARKERS) else "dynamic"

    ep_family = np.array([family(n) for n in names])

    # 1+2. coverage and short-clip exclusion
    pool_motions = np.array([m["motion_id"] for m in meta])
    short_mask = lengths < CHUNK + 2
    pool_motion_set = set(pool_motions.tolist())
    # map motion id -> episode index (first occurrence)
    motion_to_ep = {}
    for i in range(n_eps):
        motion_to_ep.setdefault(int(ep_motion[i]), i)
    pool_ep_idx = np.array([motion_to_ep.get(int(m), -1) for m in pool_motions])
    pool_short_share = float(short_mask[pool_ep_idx].float().mean()) if (pool_ep_idx >= 0).all() else float("nan")
    # activity of pool obs (raw, dof_vel block of state)
    pool_dv = torch.cat([bdx_motion_activity(e["observation"]["state"]) for e in eps])
    pool_act = pool_dv.norm(dim=-1).numpy()
    # kinematic reference (raw obs, same block)
    kin_state = kbuf.storage["observation"]["state"]
    kin_act = bdx_motion_activity(kin_state).numpy()

    # 4. composition
    kin_stand_frames = 0
    per_motion_frames = {}
    rows = []
    fam_act = {"stand": [], "dynamic": []}
    for i in range(n_eps):
        s, L = int(starts[i]), int(lengths[i])
        fam = ep_family[i]
        rows.append((int(ep_motion[i]), names[i], fam, L))
        fam_act[fam].append(kin_act[s:s + L].mean())
    # pool family share by frames
    pool_fam = np.array([ep_family[motion_to_ep[int(m)]] if int(m) in motion_to_ep else "unknown" for m in pool_motions])
    pool_frame_fam = np.concatenate([np.full(m["length"], f) for f, m in zip(pool_fam, meta)])

    result = {
        "dataset_episodes": n_eps,
        "dataset_total_frames": int(lengths.sum()),
        "short_episodes_lt266": int(short_mask.sum()),
        "short_episode_frame_share": float(lengths[short_mask].sum() / lengths.sum()),
        "pool_chunks": len(meta),
        "pool_frames": int(sum(m["length"] for m in meta)),
        "pool_unique_motions": len(pool_motion_set),
        "pool_short_clip_chunk_share": pool_short_share,
        "dataset_stand_episode_share": float((ep_family == "stand").mean()),
        "dataset_stand_frame_share": float(
            sum(L for (_, _, f, L) in rows if f == "stand") / sum(L for (_, _, _, L) in rows)
        ),
        "pool_stand_frame_share": float((pool_frame_fam == "stand").mean()),
        "activity_dataset_mean": float(kin_act.mean()),
        "activity_pool_mean": float(pool_act.mean()),
        "activity_dataset_p50": float(np.median(kin_act)),
        "activity_pool_p50": float(np.median(pool_act)),
        "activity_dataset_p90": float(np.quantile(kin_act, 0.9)),
        "activity_pool_p90": float(np.quantile(pool_act, 0.9)),
        "activity_standclips_mean": float(np.mean(fam_act["stand"])),
        "activity_dynamicclips_mean": float(np.mean(fam_act["dynamic"])),
    }
    out = Path("results/bfmzero-bdx-full/pool_prior_audit.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
