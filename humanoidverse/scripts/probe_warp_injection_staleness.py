"""Probe: are warp obs after state injection computed at the injected state?

Tests, all on the noise-free local_body_pos channel (privileged_state[1:49]):

  1. determinism  : inject the same states twice back-to-back; obs must match.
  2. one-step size: inject t vs t+dt; obs diff ~ one frame of motion.
  3. freshness    : frame-aligned injection vs kinematic buffer rows at the
                    same frames; must be ~1e-3 (CPU pairing level). If it is
                    ~one-step size instead, obs lag the injection (stale
                    kinematics buffers in the warp path).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from humanoidverse.scripts.generate_expert_sim_pool import build_warp_env, inject_round, OBS_KEYS

PRIV_POS = (1, 49)


def chan_diff(a, b):
    out = {}
    for name, (lo, hi) in PRIV_SLICES.items():
        d = (a["privileged_state"][:, lo:hi] - b["privileged_state"][:, lo:hi]).abs().flatten()
        out[name] = {"p50": d.median().item(), "p95": torch.quantile(d, 0.95).item()}
    return out


PRIV_SLICES = {
    "root_height": (0, 1),
    "local_body_pos": (1, 49),
    "local_body_rot": (49, 151),
    "local_body_vel": (151, 202),
    "local_body_ang_vel": (202, 253),
}


def main():
    cfg_path = Path("results/bfm-dynamics-baseline/config.json")
    cfg = json.loads(cfg_path.read_text())
    disable = "--no-noise" in sys.argv
    wrapped = build_warp_env(cfg, torch.device("cuda:0"), 256, disable_obs_noise=disable)
    inner = wrapped._env
    dt = float(inner.dt)
    n = 256
    print(f"disable_obs_noise={disable}")

    rng = np.random.default_rng(5)
    m_ids = torch.tensor(rng.integers(0, 256, size=n), dtype=torch.long, device=inner.device)
    t0 = torch.tensor(rng.uniform(0.5, 3.0, size=n), dtype=torch.float32, device=inner.device)

    wrapped.reset(to_numpy=False)

    # repeat the SAME injection 5 times; watch consecutive obs diffs
    obs_seq = []
    for _ in range(5):
        o, _ = inject_round(wrapped, m_ids, t0)
        obs_seq.append(o)
        # sim dof state after the step (refresh applied during it)
        dof_now = inner.simulator.dof_pos.detach().float().cpu()
        obs_seq[-1]["_dof"] = dof_now
    req = inner._motion_lib.get_motion_state(m_ids, t0)["dof_pos"].detach().float().cpu()
    consec = [chan_diff(obs_seq[i], obs_seq[i + 1]) for i in range(4)]
    dof_err = {
        "mean": (obs_seq[-1]["_dof"] - req).abs().mean().item(),
        "max": (obs_seq[-1]["_dof"] - req).abs().max().item(),
    }
    out = {}
    out["consecutive_same_state_injections"] = consec
    out["sim_dof_pos_vs_requested_after_step"] = dof_err
    # exact reader determinism: repeated same-state reads must be EXACTLY equal
    from humanoidverse.scripts.generate_expert_sim_pool import inject_read_exact
    ex1 = inject_read_exact(wrapped, m_ids, t0)
    ex2 = inject_read_exact(wrapped, m_ids, t0)
    exact_chans = {}
    for name, (lo, hi) in PRIV_SLICES.items():
        d = (ex1["privileged_state"][:, lo:hi] - ex2["privileged_state"][:, lo:hi]).abs().flatten()
        exact_chans[name] = {"max": d.max().item()}
    # raw cvel check: does mjw.forward recompute cvel from the injected qvel?
    import warp as wp
    def read_cvel():
        cv = wp.to_torch(inner.simulator.d.cvel)[:, inner.simulator.body_id]  # [N, nb, 6]
        return cv[:, :, :3].detach().float().cpu(), cv[:, :, 3:6].detach().float().cpu()
    _ = inject_read_exact(wrapped, m_ids, t0)
    ang1, lin1 = read_cvel()
    _ = inject_read_exact(wrapped, m_ids, t0)
    ang2, lin2 = read_cvel()
    lind = (lin1 - lin2).abs()  # [N, nb, 3]
    flat = lind.flatten()
    amax = int(flat.argmax())
    n_b = lin1.shape[1]
    env_i, rem = divmod(amax, 3 * n_b)
    body_i, _ = divmod(rem, 3)
    out["raw_cvel_after_exact_inject"] = {
        "ang_repeat_max": (ang1 - ang2).abs().max().item(),
        "lin_repeat_max": flat.max().item(),
        "lin_argmax_env_body": [int(env_i), int(body_i)],
        "lin_frac_gt_0.01": (flat > 0.01).float().mean().item(),
        "lin_frac_gt_1e-4": (flat > 1e-4).float().mean().item(),
        "lin1_at_argmax": [float(f"{x:.3f}") for x in lin1[env_i, body_i].tolist()],
        "lin2_at_argmax": [float(f"{x:.3f}") for x in lin2[env_i, body_i].tolist()],
        "motion_id_at_argmax_env": int(m_ids[env_i].item()),
        "t0_at_argmax_env": float(t0[env_i].item()),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())
