"""Shared machinery for the Tracker fixed-action Demo v0 (2026-08-24).

Implements the audited engineering contracts, reusing the exact paths already
proven by probe_aligned_2x2.py / generate_expert_sim_pool.py / train.py's
[BFM-REF-Z-ALIGN] rollout block:

- expert observations for z encoding come ONLY from the MuJoCo-injected
  expert-sim pool (bdx_expert_sim_pool_full.pt, 1462 episodes x 264 frames);
- z = project_z(backward_map(pool_window[w0:w0+seq]).mean over seq frames),
  checkpoint's own normalizer + B encoder, z HELD for the whole action
  (matches B-arm training: "hold z until the next reset");
- aligned reset = motion-lib physical state at the source frame through the
  env's own reset_envs_idx path, then mj_forward, then the wrapper's
  observation builder (history zero-filled per env reset semantics);
- rollouts step the CPU mujoco env with deterministic mean actions;
- state contract "bdx-state-v2" [dof_pos 14, dof_vel 14, gravity 3, ang_vel 3],
  history contract "newest-first" maintained by the wrapper's Handler.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("BFM_ZERO_HARD_LIMIT_RESET", "0")
os.environ.setdefault("BFM_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env

REPO = Path(__file__).resolve().parents[2]
ENV_CONFIG = REPO / "results/bfm-dynamics-baseline/config.json"
TRAIN_PKL = REPO / "humanoidverse/data/bdx_14dof_train.pkl"
POOL_PATH = REPO / "humanoidverse/data/bdx_expert_sim_pool_full.pt"
DT = 0.02

# arms -> checkpoint dir (final checkpoints, step 275968)
ARMS = {
    "A": REPO / "results/gate1-screenA-combo-300k/checkpoint",
    "B": REPO / "results/gate1-screenB-refzalign-300k/checkpoint",
}


def load_demo_model(arm_or_dir, device="cuda"):
    d = ARMS[arm_or_dir] if arm_or_dir in ARMS else Path(arm_or_dir)
    model = load_model_from_checkpoint_dir(d, device=device)
    model.eval()
    model._obs_normalizer.eval()
    return model


def _build_cpu_env_opts(cfg, device, disable_obs_noise):
    """build_cpu_env with a noise switch (one HumanoidVerse per process)."""
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["device"] = str(device)
    env_cfg["lafan_tail_path"] = str(TRAIN_PKL.resolve())
    env_cfg["disable_obs_noise"] = bool(disable_obs_noise)
    overrides = [o for o in env_cfg["hydra_overrides"] if not o.startswith("simulator.config.sim.")]
    overrides += [
        "env.config.max_episode_length_s=10000",
        "env.config.headless=True",
        "simulator=mujoco",
        "robot.motion.smooth_angular_velocity=false",
    ]
    env_cfg["hydra_overrides"] = overrides
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig

    wrapped, _ = HumanoidVerseIsaacConfig(**env_cfg).build(1)
    return wrapped


class DemoEnv:
    """CPU mujoco env + expert-sim pool + motion-name mapping."""

    def __init__(self, device="cuda", disable_obs_noise=False):
        cfg = json.loads(ENV_CONFIG.read_text())
        self.disable_obs_noise = bool(disable_obs_noise)
        self.wrapped = _build_cpu_env_opts(cfg, torch.device(device), disable_obs_noise)
        self.inner = self.wrapped._env
        self.motion_lib = self.inner._motion_lib
        # num_envs=1 build loads a 1-motion random subset; demo needs every
        # motion resident so pool/global motion ids are valid for
        # get_motion_state and aligned resets (see load_motions_for_training).
        self.motion_lib.load_all_motions()
        self.dt = float(self.inner.dt)
        self.device = torch.device(device)
        pool = torch.load(POOL_PATH, map_location="cpu", weights_only=False)
        self.pool = pool
        self.episodes = pool["episodes"]
        self.pool_meta = pool["meta"]
        self.row_by_motion = {int(m["motion_id"]): i for i, m in enumerate(pool["meta"])}
        self.names = list(joblib.load(TRAIN_PKL).keys())  # motion_id = insertion order
        self.seq_len = int(cfg["agent"]["model"]["seq_length"])

    def name_of(self, motion_id: int) -> str:
        return self.names[int(motion_id)]

    def motion_id_of(self, name: str) -> int:
        return self.names.index(name)

    def episode_row(self, motion_id: int) -> int:
        return self.row_by_motion[int(motion_id)]


def encode_z(model, env: DemoEnv, motion_id: int, w0: int):
    """z from the expert-sim pool window [w0, w0+seq) of this motion's episode,
    encoded with the checkpoint's own normalizer + B (train.py B-arm protocol).
    """
    row = env.episode_row(motion_id)
    ep_obs = env.episodes[row]["observation"]
    assert w0 + env.seq_len <= ep_obs["state"].shape[0], "window past episode end"
    win = {k: v[w0 : w0 + env.seq_len].to(model.cfg.device) for k, v in ep_obs.items()}
    with torch.no_grad():
        e = model.backward_map(win)  # normalizes internally, eval-mode stats
        z = model.project_z(e.view(1, env.seq_len, -1).mean(dim=1))
    return z


def aligned_reset(env: DemoEnv, motion_id: int, w0: int, seed: int):
    """Env's own reset path, then inject the source window's physical state.
    Returns the policy-ready observation (torch, on env device)."""
    inner = env.inner
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.wrapped.reset(to_numpy=False)
    t_abs = float(env.pool_meta[env.episode_row(motion_id)]["t0"]) + w0 * DT
    res = inner._motion_lib.get_motion_state(
        torch.tensor([int(motion_id)], dtype=torch.long, device=inner.device),
        torch.tensor([t_abs], dtype=torch.float32, device=inner.device),
    )
    n = 1
    dof_states = torch.zeros(n, inner.num_dof, 2, device=inner.device)
    root_states = torch.zeros(n, 13, device=inner.device)
    dof_states[..., 0] = res["dof_pos"]
    dof_states[..., 1] = res["dof_vel"]
    root_states[:, :3] = res["root_pos"]
    root_states[:, 3:7] = res["root_rot"]
    root_states[:, 7:10] = res["root_vel"]
    root_states[:, 10:13] = res["root_ang_vel"]
    env_ids = torch.arange(n, device=inner.device)
    inner.reset_envs_idx(env_ids, target_states={"dof_states": dof_states, "root_states": root_states})
    ids = inner.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
    inner.simulator.set_actor_root_state_tensor(ids, inner.target_robot_root_states)
    inner.simulator.set_dof_state_tensor(ids, inner.target_robot_dof_state)
    inner.need_to_refresh_envs[ids] = False
    import mujoco

    mujoco.mj_forward(inner.simulator.model, inner.simulator.data)
    inner._pre_compute_observations_callback()
    inner._compute_observations()
    obs = env.wrapped._get_g1env_observation(to_numpy=False)
    return obs


def activity(state: torch.Tensor) -> np.ndarray:
    """BDX state activity = ||dof_vel|| (bdx-state-v2: dof_vel at [14:28])."""
    return state[..., 14:28].float().norm(dim=-1).cpu().numpy()


FALL_HEIGHT = 0.15  # BDX stand ~0.30, deepest legit crouch ~0.22; below this = fallen
FALL_GRAVITY_TILT = 0.7  # ||projected_gravity[:2]|| above this = tipped over
FALL_SUSTAIN_STEPS = 25  # transient launch tilt at reset must not count as a fall


def fall_criterion(base_height: float, state_row: torch.Tensor) -> bool:
    gtilt = float(state_row[28:31].float().norm()) ** 2 - float(state_row[30]) ** 2
    # ||g_xy||^2 = ||g||^2 - g_z^2 ; state = [dof14, vel14, gravity3, angvel3]
    return base_height < FALL_HEIGHT or gtilt > FALL_GRAVITY_TILT ** 2


@dataclass
class EpisodeLog:
    steps: int = 0
    actions: list = field(default_factory=list)
    tracking: list = field(default_factory=list)
    activity: list = field(default_factory=list)
    base_height: list = field(default_factory=list)
    torque: list = field(default_factory=list)
    fallen_step: int | None = None
    terminated: bool = False
    trunc_reason: str = "done"

    def summary(self, src_activity: float | None = None):
        a = np.asarray(self.actions)
        return {
            "steps": self.steps,
            "fall": self.terminated or self.fallen_step is not None,
            "fall_step": self.fallen_step,
            "completion": (self.fallen_step / self.steps) if (self.fallen_step is not None and self.steps) else 1.0,
            "termination_reason": self.trunc_reason,
            "tracking_mean": float(np.mean(self.tracking)) if self.tracking else None,
            "tracking_final10": float(np.mean(self.tracking[-10:])) if self.tracking else None,
            "activity_mean": float(np.mean(self.activity)) if self.activity else None,
            "source_relative_activity": (
                float(np.mean(self.activity) - src_activity)
                if self.activity and src_activity is not None else None
            ),
            "action_abs_mean": float(a.mean()) if a.size else None,
            "action_rate": float(np.abs(np.diff(a, axis=0)).mean()) if a.shape[0] > 1 else None,
            "action_saturation_frac": float((np.abs(a) > 0.99).mean()) if a.size else None,
            "torque_abs_mean": float(np.mean(np.abs(self.torque))) if self.torque else None,
            "base_height_min": float(np.min(self.base_height)) if self.base_height else None,
        }


def read_torque(env: DemoEnv) -> np.ndarray:
    d = env.inner.simulator.data
    return np.asarray(d.qfrc_actuator[: env.inner.num_dof], dtype=np.float64)


def rollout_action(
    env: DemoEnv,
    model,
    z: torch.Tensor,
    obs: dict,
    horizon: int,
    src_priv: torch.Tensor | None = None,
    record: bool = True,
    abort_on_fall: bool = True,
):
    """Deterministic mean-action rollout. src_priv: normalized source-window
    privileged_state frames for matched tracking error (same length as horizon,
    last frame held if shorter)."""
    log = EpisodeLog()
    src_act_norm = None
    if src_priv is not None:
        src = src_priv.to(env.device)
    o = obs
    fall_run = 0  # consecutive fallen steps
    with torch.no_grad():
        for h in range(horizon):
            a = model.act(o, z.expand(o["state"].shape[0], -1), mean=True)
            if record:
                log.actions.append(a.detach().float().cpu().numpy()[0])
            if src_priv is not None:
                with torch.no_grad():
                    o_n = model._obs_normalizer(o)
                    ref = src[min(h, src.shape[0] - 1)].unsqueeze(0)
                    log.tracking.append(float((o_n["privileged_state"][0] - ref[0]).norm()))
                log.activity.append(float(activity(o["state"])[0]))
                h_now = float(np.asarray(env.inner.simulator.data.qpos[2]))
                log.base_height.append(h_now)
                log.torque.append(read_torque(env))
                if fall_criterion(h_now, o["state"][0]):
                    fall_run += 1
                    if log.fallen_step is None and fall_run >= FALL_SUSTAIN_STEPS:
                        log.fallen_step = h - FALL_SUSTAIN_STEPS + 1
                        log.trunc_reason = "fall_sustained"
                else:
                    fall_run = 0
            o2, _, term, trunc, _ = env.wrapped.step(a, to_numpy=False)
            o = o2
            log.steps = h + 1
            if term.any():
                log.terminated = True
                log.trunc_reason = "fall"
                if abort_on_fall:
                    break
            if trunc.any():
                log.trunc_reason = "timeout"
                break
            if not np.isfinite(np.asarray(o["state"].float().cpu())).all():
                log.trunc_reason = "nonfinite_obs"
                break
    return log


def source_activity(env: DemoEnv, motion_id: int, w0: int, n: int) -> float:
    st = env.episodes[env.episode_row(motion_id)]["observation"]["state"]
    return float(activity(st[w0 : min(w0 + n, st.shape[0])]).mean())


def sha256_file(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def z_checksum(z: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(z.detach().cpu().numpy().tobytes()).hexdigest()[:16]
