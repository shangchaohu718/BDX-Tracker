"""CPU mujoco exact-read determinism probe (single env).

Repeated same-state reads via inject_read_exact_lean must be EXACTLY equal on
all privileged channels (standard mj_forward computes cvel fully).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.scripts.generate_expert_sim_pool import (
    build_cpu_env, inject_read_exact_lean, OBS_KEYS,
)

PRIV_SLICES = {
    "root_height": (0, 1),
    "local_body_pos": (1, 49),
    "local_body_rot": (49, 151),
    "local_body_vel": (151, 202),
    "local_body_ang_vel": (202, 253),
}


def main():
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, torch.device("cuda"))
    inner = wrapped._env
    # reload all motions (the env itself loads only a 256-motion sample)
    agent_cfg = FBcprAuxAgentConfig(**cfg["agent"])
    kbuf = load_expert_trajectories_from_motion_lib(inner, agent_cfg, device="cpu", base_ang_vel_frame="body")
    num_motions = int(kbuf.storage["motion_id"].max().item()) + 1
    rng = np.random.default_rng(5)
    m_ids = torch.tensor([int(rng.integers(0, num_motions))], dtype=torch.long, device=inner.device)
    t0 = torch.tensor([float(rng.uniform(0.5, 3.0))], dtype=torch.float32, device=inner.device)

    wrapped.reset(to_numpy=False)
    reads = [inject_read_exact_lean(wrapped, m_ids, t0) for _ in range(3)]
    out = {"motion_id": int(m_ids[0]), "t0": float(t0[0])}
    for name, (lo, hi) in PRIV_SLICES.items():
        d12 = (reads[0]["privileged_state"][:, lo:hi] - reads[1]["privileged_state"][:, lo:hi]).abs().max().item()
        d13 = (reads[0]["privileged_state"][:, lo:hi] - reads[2]["privileged_state"][:, lo:hi]).abs().max().item()
        out[name] = {"read1_vs_2_max": d12, "read1_vs_3_max": d13}
    # also state channels
    s12 = (reads[0]["state"] - reads[1]["state"]).abs().max().item()
    out["state_read1_vs_2_max"] = s12
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())
