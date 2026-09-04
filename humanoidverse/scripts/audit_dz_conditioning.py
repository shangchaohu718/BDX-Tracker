"""P0.1+2 D-x-z conditioning audit on the rnorm-3M checkpoint discriminator.

Offline, no training. Three probes on the TRAINED D:

  1. N x N matrix S_ij = D(o_i, z_j) over stratified expert-sim pool samples
     (stand vs dynamic families): diagonal vs off-diagonal, per-block margins,
     top-1 retrieval, row-variance vs column-variance (column ~ 0 => D ignores z).
  2. z permutation test folded into (1): paired z vs permuted z score gaps.
  3. Conditional stillness test: zero-velocity standing obs vs mid-motion obs,
     each scored under stand-z and dynamic-z. Distinguishes:
        all-z-high  -> D unconditional AND does not punish stillness
        all-z-low   -> D punishes stillness (freeze has another driver)
        only-matching-z-high -> conditioning works; look at z sampling / RL
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.scripts.generate_expert_sim_pool import (
    build_cpu_env, inject_read_exact_lean, OBS_KEYS,
)

STAND_MARKERS = ("stand", "pose", "idle")
SEQ = 8
N_PER_FAM = 32


def family(name):
    n = name.lower()
    return "stand" if any(m in n for m in STAND_MARKERS) else "dynamic"


def encode_expert(model, next_obs, seq):
    b = model._backward_map(next_obs).detach()
    b = b.view(b.shape[0] // seq, seq, b.shape[-1]).mean(1)
    return torch.repeat_interleave(model.project_z(b), seq, dim=0)


def inject_custom(wrapped, ids, times, zero_vel):
    inner = wrapped._env
    res = inner._motion_lib.get_motion_state(ids, times)
    n = ids.shape[0]
    dof_states = torch.zeros(n, inner.num_dof, 2, device=inner.device)
    root_states = torch.zeros(n, 13, device=inner.device)
    dof_states[..., 0] = res["dof_pos"]
    root_states[:, :3] = res["root_pos"]
    root_states[:, 3:7] = res["root_rot"]
    if not zero_vel:
        dof_states[..., 1] = res["dof_vel"]
        root_states[:, 7:10] = res["root_vel"]
        root_states[:, 10:13] = res["root_ang_vel"]
    env_ids = torch.arange(n, device=inner.device)
    inner.reset_envs_idx(env_ids, target_states={
        "dof_states": dof_states, "root_states": root_states,
    })
    refresh_ids = inner.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
    if len(refresh_ids) > 0:
        inner.simulator.set_actor_root_state_tensor(refresh_ids, inner.target_robot_root_states)
        inner.simulator.set_dof_state_tensor(refresh_ids, inner.target_robot_dof_state)
        inner.need_to_refresh_envs[refresh_ids] = False
    import mujoco
    mujoco.mj_forward(inner.simulator.model, inner.simulator.data)
    inner._pre_compute_observations_callback()
    inner._compute_observations()
    obs = wrapped._get_g1env_observation(to_numpy=False)
    return {k: obs[k].detach().float().cpu() for k in OBS_KEYS}


def main():
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, torch.device("cuda"))
    inner = wrapped._env
    agent_cfg = FBcprAuxAgentConfig(**cfg["agent"])
    kbuf = load_expert_trajectories_from_motion_lib(inner, agent_cfg, device="cpu", base_ang_vel_frame="body")
    pool = torch.load("humanoidverse/data/bdx_expert_sim_pool_cpu.pt", map_location="cpu")
    eps, meta = pool["episodes"], pool["meta"]

    model = load_model_from_checkpoint_dir(Path(__import__("os").environ.get("DZ_CKPT", "results/bfm-dynamics-pool-rnorm-3m/checkpoint")), device="cuda")
    model.eval()
    model._obs_normalizer.eval()
    dev = torch.device("cuda")

    names = [str(x) for x in kbuf.file_names]
    starts = kbuf.start_idx[:, 0].long()
    ep_motion = kbuf.storage["motion_id"][starts].long()
    motion_to_ep = {}
    for i in range(len(names)):
        motion_to_ep.setdefault(int(ep_motion[i]), i)

    # stratified window sampling from pool episodes by family
    rng = np.random.default_rng(11)
    fam_of_ep = {}
    windows = []  # (ep_idx, w, family)
    for fam in ("stand", "dynamic"):
        cand = [i for i in range(len(eps))
                if family(names[motion_to_ep[int(meta[i]["motion_id"])]]) == fam and meta[i]["length"] > SEQ + 2]
        take = rng.choice(cand, size=min(N_PER_FAM, len(cand)), replace=False)
        for i in take:
            T = eps[i]["observation"]["state"].shape[0]
            windows.append((i, int(rng.integers(0, T - SEQ - 1)), fam))
    rng.shuffle(windows)

    obs_all = {k: torch.cat([eps[i]["observation"][k][w:w + SEQ] for i, w, _ in windows]) for k in OBS_KEYS}
    next_all = {k: torch.cat([eps[i]["observation"][k][w + 1:w + 1 + SEQ] for i, w, _ in windows]) for k in OBS_KEYS}
    fams = np.array([f for _, _, f in windows])

    with torch.no_grad():
        norm = lambda d: model._obs_normalizer(tree_map(lambda x: x.to(dev), d))
        obs_n = norm(obs_all)
        z_all = encode_expert(model, tree_map(lambda x: x.to(dev), next_all), SEQ)  # [2N*SEQ, dz]
        # one z per window (rows are window-major: window r occupies rows r*SEQ..r*SEQ+SEQ-1)
        z_win = z_all.view(-1, SEQ, z_all.shape[-1]).mean(dim=1)  # [2N, dz]
        obs_win = {k: v.view(-1, SEQ, v.shape[-1]).mean(dim=1) for k, v in obs_n.items()}  # window-mean obs

    # N x N matrix: score obs window i under z of window j
    with torch.no_grad():
        S = torch.zeros(len(windows), len(windows), device=dev)
        for j in range(len(windows)):
            zj = z_win[j:j + 1].expand(len(windows), -1)
            S[:, j] = model._discriminator.compute_logits(obs_win, zj).flatten()
    S = S.cpu().numpy()
    n = len(windows)
    diag = np.diag(S)
    off = S[~np.eye(n, dtype=bool)]
    fam_vec = fams

    def block(a, b):
        m = (fam_vec[:, None] == a) & (fam_vec[None, :] == b)
        np.fill_diagonal(m, False)
        return float(S[m].mean())

    col_var = S.var(axis=0).mean()  # variance across z for fixed obs
    row_var = S.var(axis=1).mean()  # variance across obs for fixed z
    top1 = float((S.argmax(axis=1) == np.arange(n)).mean())

    # conditional stillness test
    stand_ep = next(i for i in range(len(eps)) if family(names[motion_to_ep[int(meta[i]["motion_id"])]]) == "stand")
    dyn_ep = next(i for i in range(len(eps)) if family(names[motion_to_ep[int(meta[i]["motion_id"])]]) == "dynamic")
    ids_s = torch.tensor([meta[stand_ep]["motion_id"]], dtype=torch.long, device=inner.device)
    t_s = torch.tensor([meta[stand_ep]["t0"] + 4.0], dtype=torch.float32, device=inner.device)
    ids_d = torch.tensor([meta[dyn_ep]["motion_id"]], dtype=torch.long, device=inner.device)
    t_d = torch.tensor([meta[dyn_ep]["t0"] + 4.0], dtype=torch.float32, device=inner.device)
    wrapped.reset(to_numpy=False)
    still_s = inject_custom(wrapped, ids_s, t_s, zero_vel=True)   # frozen standing pose
    still_d = inject_custom(wrapped, ids_d, t_d, zero_vel=True)   # frozen mid-motion pose
    moving_d = inject_custom(wrapped, ids_d, t_d, zero_vel=False) # real moving expert frame
    z_stand = z_win[fam_vec == "stand"].mean(0, keepdim=True)
    z_dyn = z_win[fam_vec == "dynamic"].mean(0, keepdim=True)
    with torch.no_grad():
        def score(o):
            on = model._obs_normalizer(tree_map(lambda x: x.to(dev), o))
            return {
                "stand_z": model._discriminator.compute_logits(on, z_stand.expand(on["state"].shape[0], -1)).flatten()[0].item(),
                "dyn_z": model._discriminator.compute_logits(on, z_dyn.expand(on["state"].shape[0], -1)).flatten()[0].item(),
            }
        stillness = {
            "frozen_stand_pose": score(still_s),
            "frozen_dyn_pose": score(still_d),
            "moving_dyn_frame": score(moving_d),
        }

    result = {
        "n_windows": n,
        "diag_mean": float(diag.mean()), "offdiag_mean": float(off.mean()),
        "diag_minus_off": float(diag.mean() - off.mean()),
        "top1_retrieval": top1,
        "row_var_mean": float(row_var), "col_var_mean": float(col_var),
        "col_over_row_var": float(col_var / max(row_var, 1e-12)),
        "block_dyn_obs_dyn_z": block("dynamic", "dynamic"),
        "block_dyn_obs_stand_z": block("dynamic", "stand"),
        "block_stand_obs_dyn_z": block("stand", "dynamic"),
        "block_stand_obs_stand_z": block("stand", "stand"),
        "conditional_stillness_scores": stillness,
    }
    out = Path("results/bfmzero-bdx-full/dz_conditioning_audit.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
