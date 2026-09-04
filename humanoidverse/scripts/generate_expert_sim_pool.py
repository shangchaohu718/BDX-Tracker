"""Generate an expert-sim observation pool: expert motion states injected into
the MuJoCo (warp) simulator and read back through the SAME observation builder
the policy rollouts use.

Purpose: replace the kinematic-pipeline expert positives fed to the
discriminator (domain-separation root cause, see BFM_ZERO_DEBUG_HANDOFF) with
sim-pipeline positives, so D can only judge behavior, not observation source.

Mechanism per frame: reset_envs_idx(target_states=motion state at time t) sets
need_to_refresh; the following step() advances physics, THEN applies the
injected state, THEN computes observations -> obs live exactly at the injected
state. Termination-driven resets inside step() overwrite the injection for the
affected envs; those frames are detected via reset_buf and dropped (episodes
are split at contaminated frames).

Output: torch.save dict {episodes, meta, stats}; episodes follow the
TrajectoryDictBuffer format of load_expert_trajectories_from_motion_lib
(observation keys: state / last_action / privileged_state).

Run with the same env-var hygiene as the training launcher (see main()).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("BFM_ZERO_HARD_LIMIT_RESET", "0")
os.environ.setdefault("BFM_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import (
    HumanoidVerseIsaacConfig,
    load_expert_trajectories_from_motion_lib,
)
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig

OBS_KEYS = ("state", "last_action", "privileged_state")
MIN_EP_LEN = 17  # seq_length 8 + next + safety margin


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_warp_env(cfg, device, num_envs, disable_obs_noise=False):
    """Same construction path as the training launcher (mujoco_warp overrides)."""
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["device"] = str(device)
    env_cfg["lafan_tail_path"] = os.environ.get("BFM_POOL_DATA", str(Path("humanoidverse/data/bdx_14dof_train.pkl").resolve()))
    env_cfg["disable_obs_noise"] = bool(disable_obs_noise)
    overrides = list(env_cfg["hydra_overrides"])
    overrides += [
        "env.config.max_episode_length_s=10000",
        "env.config.headless=True",
        "robot.motion.smooth_angular_velocity=false",
    ]
    env_cfg["hydra_overrides"] = overrides
    wrapped, _ = HumanoidVerseIsaacConfig(**env_cfg).build(num_envs)
    return wrapped


def build_cpu_env(cfg, device):
    """CPU mujoco env (standard mj_forward computes cvel fully — the warp
    build's forward leaves cvel's linear part stale). num_envs must be 1
    (repo init bug for N>1 on this path)."""
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["device"] = str(device)
    env_cfg["lafan_tail_path"] = os.environ.get("BFM_POOL_DATA", str(Path("humanoidverse/data/bdx_14dof_train.pkl").resolve()))
    env_cfg["disable_obs_noise"] = False  # match training obs path
    overrides = [o for o in env_cfg["hydra_overrides"] if not o.startswith("simulator.config.sim.")]
    overrides += [
        "env.config.max_episode_length_s=10000",
        "env.config.headless=True",
        "simulator=mujoco",
        "robot.motion.smooth_angular_velocity=false",
    ]
    env_cfg["hydra_overrides"] = overrides
    wrapped, _ = HumanoidVerseIsaacConfig(**env_cfg).build(1)
    return wrapped


def inject_read_exact_lean(wrapped, ids, times, settle_reps=2):
    """CPU lean exact read: write states, mj_forward, compute obs.

    The first read after a state change carries a small solver warm-start
    residue in the velocity channels (probe_cpu_injection_determinism:
    read2 == read3 exactly); settle_reps=2 discards the first read.
    Skips reset_envs_idx's per-frame task/DR resets (the forward-based read
    does not use dynamics parameters). For the single-env CPU env.
    """
    inner = wrapped._env
    sim = inner.simulator
    n = ids.shape[0]
    obs = None
    for rep in range(max(1, settle_reps)):
        res = inner._motion_lib.get_motion_state(ids, times)
        dof_states = torch.zeros(n, inner.num_dof, 2, device=inner.device)
        root_states = torch.zeros(n, 13, device=inner.device)
        dof_states[..., 0] = res["dof_pos"]
        dof_states[..., 1] = res["dof_vel"]
        root_states[:, :3] = res["root_pos"]
        root_states[:, 3:7] = res["root_rot"]
        root_states[:, 7:10] = res["root_vel"]
        root_states[:, 10:13] = res["root_ang_vel"]
        env_ids = torch.arange(n, device=inner.device)
        inner.reset_envs_idx(env_ids, target_states={
            "dof_states": dof_states, "root_states": root_states,
        })
        refresh_ids = inner.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        if len(refresh_ids) > 0:
            sim.set_actor_root_state_tensor(refresh_ids, inner.target_robot_root_states)
            sim.set_dof_state_tensor(refresh_ids, inner.target_robot_dof_state)
            inner.need_to_refresh_envs[refresh_ids] = False
        import mujoco
        mujoco.mj_forward(sim.model, sim.data)
        inner._pre_compute_observations_callback()
        inner._compute_observations()
        raw = wrapped._get_g1env_observation(to_numpy=False)
        obs = {k: raw[k].detach().float().cpu() for k in OBS_KEYS}
    return obs


def inject_read_exact(wrapped, ids, times):
    """Inject states and read obs WITHOUT any physics step.

    Applies the refresh that step() would apply, then runs mjw.forward
    (prepare_sim) so every derived tensor (xpos/xquat/cvel) is recomputed
    from the injected qpos/qvel exactly, then computes observations through
    the same _compute_observations path the policy uses.

    Rationale: warp body-velocity buffers are only updated during physics
    steps, so an inject+step read returns stale, DR-randomized velocities
    (probe: probe_warp_injection_staleness — vel channels never converge).
    The forward-based read is exact and deterministic.
    """
    inner = wrapped._env
    n = ids.shape[0]
    res = inner._motion_lib.get_motion_state(ids, times)
    dof_states = torch.zeros(n, inner.num_dof, 2, device=inner.device)
    root_states = torch.zeros(n, 13, device=inner.device)
    dof_states[..., 0] = res["dof_pos"]
    dof_states[..., 1] = res["dof_vel"]
    root_states[:, :3] = res["root_pos"]
    root_states[:, 3:7] = res["root_rot"]
    root_states[:, 7:10] = res["root_vel"]
    root_states[:, 10:13] = res["root_ang_vel"]
    env_ids = torch.arange(n, device=inner.device)
    inner.reset_envs_idx(env_ids, target_states={
        "dof_states": dof_states, "root_states": root_states,
    })
    # apply the pending refresh (normally done inside step's post-physics)
    refresh_ids = inner.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
    if len(refresh_ids) > 0:
        inner.simulator.set_actor_root_state_tensor(refresh_ids, inner.target_robot_root_states)
        inner.simulator.set_dof_state_tensor(refresh_ids, inner.target_robot_dof_state)
        inner.need_to_refresh_envs[refresh_ids] = False
    inner.simulator.prepare_sim()  # mjw.forward: recompute derived tensors
    # this mjwarp build's forward leaves d.cvel's linear part stale (probe:
    # probe_warp_injection_staleness); step1 runs the physics-path velocity
    # kernels without integrating the state.
    sim = inner.simulator
    if getattr(sim, "d", None) is not None and hasattr(sim.d, "cvel"):
        import mujoco_warp as mjw
        mjw.step1(sim.m, sim.d)
    else:
        import mujoco
        mujoco.mj_forward(sim.model, sim.data)
    inner._pre_compute_observations_callback()
    inner._compute_observations()
    obs = wrapped._get_g1env_observation(to_numpy=False)
    obs = {k: obs[k].detach().float().cpu() for k in OBS_KEYS}
    return obs


def inject_round(wrapped, ids, times, settle_reps=1):
    """Inject per-env motion states, step once, return (obs dict, reset mask).

    reset mask = envs whose injection was overwritten by a termination-driven
    reset inside step() (those obs are NOT at the injected state).

    settle_reps=2 re-injects the same state before reading obs: the first
    read after a state change carries a ~8e-3 warm-start residue (probe:
    probe_warp_injection_staleness), repeated injection converges to ~5e-5.
    Use settle_reps=2 when the obs are stored (pool generation); keep 1 for
    cheap probes.
    """
    inner = wrapped._env
    n = ids.shape[0]
    obs, reset_mask = None, None
    for rep in range(max(1, settle_reps)):
        res = inner._motion_lib.get_motion_state(ids, times)
        dof_states = torch.zeros(n, inner.num_dof, 2, device=inner.device)
        root_states = torch.zeros(n, 13, device=inner.device)
        dof_states[..., 0] = res["dof_pos"]
        dof_states[..., 1] = res["dof_vel"]
        root_states[:, :3] = res["root_pos"]
        root_states[:, 3:7] = res["root_rot"]
        root_states[:, 7:10] = res["root_vel"]
        root_states[:, 10:13] = res["root_ang_vel"]
        env_ids = torch.arange(n, device=inner.device)
        inner.reset_envs_idx(env_ids, target_states={
            "dof_states": dof_states, "root_states": root_states,
        })
        action = torch.zeros(n, inner.num_dof, device=inner.device)
        obs, _, _, _, _ = wrapped.step(action, to_numpy=False)
        reset_mask = inner.reset_buf.bool().clone()
    obs = {k: obs[k].detach().float().cpu() for k in OBS_KEYS}
    return obs, reset_mask


def clean_segments(bad: np.ndarray, min_len: int):
    """Split a chunk's frame indices into contiguous clean segments."""
    segs, start = [], 0
    for i, b in enumerate(bad):
        if b:
            if i - start >= min_len:
                segs.append((start, i))
            start = i + 1
    if len(bad) - start >= min_len:
        segs.append((start, len(bad)))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/bfm-dynamics-baseline")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--backend", choices=["warp", "cpu"], default="warp")
    ap.add_argument("--sampling", choices=["uniform", "per-episode"], default="uniform",
                    help="per-episode: every dataset episode exactly once (full coverage)")
    ap.add_argument("--chunks", type=int, default=3200)
    ap.add_argument("--chunk-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default="humanoidverse/data/bdx_expert_sim_pool.pt")
    args = ap.parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    cfg = json.loads((Path(args.run_dir) / "config.json").read_text())

    wrapped = (
        build_warp_env(cfg, device, args.num_envs)
        if args.backend == "warp" else build_cpu_env(cfg, device)
    )
    inner = wrapped._env
    motion_lib = inner._motion_lib
    dt = float(inner.dt)

    num_motions = len(motion_lib._motion_data_keys)
    print(f"motions={num_motions} dt={dt}", flush=True)

    # kinematic buffer only for length bookkeeping (same episode set as training)
    agent_cfg = FBcprAuxAgentConfig(**cfg["agent"])
    kbuf = load_expert_trajectories_from_motion_lib(inner, agent_cfg, device="cpu", base_ang_vel_frame="body")
    ep_lengths = kbuf.lengths.float()
    num_eps = int(ep_lengths.shape[0])
    print(f"kinematic episodes={num_eps}", flush=True)

    rng = np.random.default_rng(args.seed + 1)
    if args.sampling == "per-episode":
        chunk_eps = np.arange(num_eps)
        rng.shuffle(chunk_eps)
        args.chunks = num_eps
    else:
        chunk_eps = rng.integers(0, num_eps, size=args.chunks)
    # buffer episode -> motion id via start rows
    start_rows = kbuf.start_idx[:, 0].long()
    first_rows = kbuf.storage["motion_id"][start_rows].long()
    chunk_motions = first_rows[torch.from_numpy(chunk_eps).long()].numpy()
    chunk_offsets = []
    for e in chunk_eps:
        L = float(ep_lengths[e]) * dt  # seconds
        max_off = max(0.0, L - (args.chunk_len + 2) * dt)
        chunk_offsets.append(rng.uniform(0, max_off))
    chunk_offsets = np.asarray(chunk_offsets, dtype=np.float64)

    wrapped.reset(to_numpy=False)
    episodes, meta = [], []
    n_bad_frames = n_dropped_chunks = 0
    import time
    t_start = time.time()
    batch = args.num_envs if args.backend == "warp" else 1
    reader = inject_round if args.backend == "warp" else None
    for b0 in range(0, args.chunks, batch):
        n_b = min(batch, args.chunks - b0)
        m_ids = chunk_motions[b0:b0 + n_b].copy()
        offs = chunk_offsets[b0:b0 + n_b].copy()
        if args.backend == "warp":
            if n_b < args.num_envs:
                m_ids = np.concatenate([m_ids, np.full(args.num_envs - n_b, m_ids[-1])])
                offs = np.concatenate([offs, np.full(args.num_envs - n_b, offs[-1])])
        ids_all = torch.tensor(m_ids, dtype=torch.long, device=inner.device)
        frames = []  # per k: obs dict [n_env, ...] cpu
        bads = []    # per k: bool [n_env]
        for k in range(args.chunk_len):
            times = torch.tensor(offs + k * dt, dtype=torch.float32, device=inner.device)
            if args.backend == "warp":
                obs, reset_mask = inject_round(wrapped, ids_all, times, settle_reps=2)
                bads.append(reset_mask.cpu().numpy())
            else:
                obs = inject_read_exact_lean(wrapped, ids_all, times, settle_reps=2)
                bads.append(np.zeros(len(m_ids), dtype=bool))
            frames.append(obs)
        # chunk-major: [k][env] -> [env][k]
        for i in range(n_b):
            bad = np.array([b[i] for b in bads], dtype=bool)
            n_bad_frames += int(bad.sum())
            segs = clean_segments(bad, MIN_EP_LEN)
            if not segs:
                n_dropped_chunks += 1
                continue
            for s, e in segs:
                T = e - s
                episodes.append({
                    "observation": {key: torch.stack([frames[k][key][i] for k in range(s, e)]) for key in OBS_KEYS},
                    "terminated": torch.zeros(T, dtype=torch.bool),
                    "truncated": torch.zeros(T, dtype=torch.bool),
                    "motion_id": torch.full((T,), int(ids_all[i].item()), dtype=torch.long),
                })
                meta.append({
                    "motion_id": int(ids_all[i].item()),
                    "t0": float(offs[i] + s * dt),
                    "length": T,
                })
        done = min(b0 + batch, args.chunks)
        fps_done = (done * args.chunk_len) / max(1e-9, time.time() - t_start)
        print(f"chunks {done}/{args.chunks}: episodes={len(episodes)} bad_frames={n_bad_frames} "
              f"dropped={n_dropped_chunks} fps={fps_done:.1f}", flush=True)

    total_frames = sum(m["length"] for m in meta)
    stats = {
        "chunks": args.chunks, "chunk_len": args.chunk_len, "dt": dt,
        "backend": args.backend,
        "num_envs": args.num_envs, "seed": args.seed,
        "episodes": len(episodes), "total_frames": total_frames,
        "bad_frames": n_bad_frames, "dropped_chunks": n_dropped_chunks,
        "source_config": str(Path(args.run_dir) / "config.json"),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"episodes": episodes, "meta": meta, "stats": stats}, out)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
