import os

os.environ.setdefault("MUJOCO_GL", "egl")  # EGL for offscreen mp4; glfw for the live viewer (--no-headless)
os.environ["OMP_NUM_THREADS"] = "1"

from datetime import datetime
from pathlib import Path
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
import json
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
import torch
from humanoidverse.utils.helpers import export_meta_policy_as_onnx
from humanoidverse.utils.helpers import get_backward_observation
import joblib
import mediapy as media
import numpy as np
from torch.utils._pytree import tree_map

import humanoidverse
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent


class NativeMuJoCoRenderer:
    """Offscreen renderer drawing qpos directly on the env's MuJoCo model (any robot).

    Unlike IsaacRendererWithMuJoco (which builds a separate G1 env), this renders
    the actual simulation model, so it works for BDX and other non-G1 robots.
    """

    def __init__(self, model, render_size: int = 256):
        import mujoco as mj

        self._mj = mj
        self.model = model
        self.data = mj.MjData(model)
        self.renderer = mj.Renderer(model, height=render_size, width=render_size)
        self.cam = mj.MjvCamera()
        self.cam.lookat[:] = [0.0, 0.0, 0.25]
        self.cam.distance = 1.4
        self.cam.elevation = -15
        self.cam.azimuth = -130

    def render_qpos(self, qpos):
        q = np.asarray(qpos).ravel()
        self.data.qpos[:] = q
        self._mj.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()


def main(model_folder: Path, data_path: Path | None = None, headless: bool = True, device="cuda", simulator: str = "isaacsim", save_mp4: bool=False, disable_dr: bool = False, disable_obs_noise: bool = True, motion_list: list[int] = [25], episode_len: int = 0, playback_speed: float = 1.0, z_mode: str = "closedloop"):
    # motion_list: motion ids to evaluate (default [25])
    # episode_len: number of control steps to render (0 = full motion length, capped at z.shape[0])

    # motion_list: motion ids to evaluate (default [25])
    
    model_folder = Path(model_folder)

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    model_name = "model"
    model_name = model.__class__.__name__
    with open(model_folder / "config.json", "r") as f:
        config = json.load(f)

    use_root_height_obs = config["env"].get("root_height_obs", False)

    if data_path is not None:
        config["env"]["lafan_tail_path"] = str(Path(data_path).resolve())
    elif not Path(config["env"].get("lafan_tail_path", "")).exists():
        default_path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof.pkl"
        if default_path.exists():
            config["env"]["lafan_tail_path"] = str(default_path)
        else:
            config["env"]["lafan_tail_path"] = "data/lafan_29dof.pkl"
    # import ipdb; ipdb.set_trace()
    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(f"env.config.headless={headless}")
    # drop simulator-internal overrides recorded at training time that only exist on the
    # training backend (e.g. mujoco_warp's trim_collision/nconmax) — they break hydra when
    # evaluating on a different backend
    config["env"]["hydra_overrides"] = [
        o for o in config["env"]["hydra_overrides"] if not o.startswith("simulator.config.sim.")
    ]
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise

    # Outputs under model_folder/tracking_inference (sibling of exported/)
    output_dir = model_folder / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Disable AMP autocast before export so the traced graph is fp32 (Mish on bf16 is rejected
    # by stock ONNXRuntime). cfg is frozen Pydantic -> model_copy. Without this, every inference
    # run silently overwrites a valid fp32 ONNX with a broken bf16 one. See export_onnx.py.
    model.cfg = model.cfg.model_copy(update={"amp": False})
    try:
        export_meta_policy_as_onnx(
            model,
            output_dir,
            f"{model_name}.onnx",
            {"actor_obs": torch.randn(1, model._actor.input_filter.output_space.shape[0] + model.cfg.archi.z_dim)},
            z_dim=model.cfg.archi.z_dim,
            history=('history_actor' in model.cfg.archi.actor.input_filter.key),
            use_29dof=True,
        )
        print(f"Exported model to {output_dir}/{model_name}.onnx")
    except Exception as e:
        print(f"Skipping ONNX export due to error: {e}")
        print("Continuing with evaluation...")

    def tracking_inference(obs) -> torch.Tensor:
        # z smoothing window forced to 1 (per-frame z, no averaging). The checkpoint's
        # seq_length (8) caused a trailing-window blend that blurred fast transitions
        # (turn apex, crouch bottom); reverting to 1 gives the actor the exact per-frame
        # backward_map(target) latent, which is the FB-faithful open-loop usage.
        seq_length = 1
        z = model.backward_map(obs)
        for step in range(z.shape[0]):
            end_idx = min(step + seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return model.project_z(z)

    # rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env
    print("="*80)
    print(env.config.simulator)
    print("-"*80)
    
    output_dir = model_folder / "tracking_inference"

    for MOTION_ID in motion_list:
        env.set_is_evaluating(MOTION_ID)
        # MOTION_ID is the pkl index (start_idx) of the motion to load into env 0.
        # IMPORTANT: read the clip name from the motion ACTUALLY loaded into env 0
        # (curr_motion_keys), NOT _motion_data_keys[MOTION_ID] — that array is the
        # full sorted key list and the index can point elsewhere after eval reload.
        loaded_keys = getattr(env._motion_lib, "curr_motion_keys", None)
        clip_name = str(loaded_keys[0]) if loaded_keys is not None else f"motion_{MOTION_ID}"
        print(f"\n{'='*80}\nMotion {MOTION_ID}: {clip_name}\n{'='*80}")
        # we visulize the first env
        obs, obs_dict = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs)

        expert_qpos = np.concatenate([
            obs_dict["ref_body_pos"][:,0].cpu().numpy(),
            np.roll(obs_dict["ref_body_rots"][:,0].cpu().numpy(),1,axis=-1),
            obs_dict["dof_pos"].cpu().numpy()
        ], axis=-1)

        z = tracking_inference(tree_map(lambda x: x[1:], obs))
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(z.cpu().numpy(), output_dir / f"zs_{MOTION_ID}__{simulator}.pkl")
        print(f"Saved zs_{MOTION_ID}__{simulator}.pkl")

        observation, info = wrapped_env.reset(to_numpy=False)

        # Root state: pos(3) + quat(4) + lin_vel(3) + ang_vel(3). Isaac expects quat as wxyz; motion lib uses xyzw.
        ref_body_rots = obs_dict["ref_body_rots"][0, 0].clone()
        if simulator == "isaacsim":
            ref_body_rots = ref_body_rots[[3, 0, 1, 2]]  # xyzw -> wxyz for correct humanoid facing in Isaac
        ref_root_init_state = torch.cat(
                [
                    obs_dict["ref_body_pos"][0, 0],
                    ref_body_rots,
                    obs_dict["ref_body_vels"][0, 0],
                    obs_dict["ref_body_angular_vels"][0, 0],
                ]
            )
        dof_init_state = torch.zeros_like(wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0])
        dof_init_state[..., 0] = obs_dict["dof_pos"][0]
        dof_init_state[..., 1] = obs_dict["ref_dof_vel"][0]
        target_states = {
            "dof_states": dof_init_state,
            "root_states": torch.stack([ref_root_init_state.clone() for i in range(num_envs)])
        }
        env_ids = torch.arange(num_envs, dtype=torch.long)
        observation, info = wrapped_env._env.reset_envs_idx(env_ids, target_states=target_states)
        observation_new, reward, terminated, truncated, info = wrapped_env.step(torch.zeros((num_envs, wrapped_env.action_space.shape[-1]), dtype=torch.float32), to_numpy=False)
        observation = wrapped_env._get_g1env_observation(to_numpy=False)
        qpos, qvel = wrapped_env._get_qpos_qvel(to_numpy=True)
        joint_pos = [wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy()]

        # Visualization length: default (0) = full motion length (number of latent vectors z).
        full_motion_len = z.shape[0]
        ep_len = min(episode_len, full_motion_len) if episode_len and episode_len > 0 else full_motion_len
        print(f"Saving video for tracking ({ep_len} steps)")
        if save_mp4:
            _non_g1 = getattr(wrapped_env._env.config.robot, "has_upper_body_dof", None) is False
            if simulator in ("mujoco", "mujoco_warp") and _non_g1:
                # non-G1 robot on a mujoco backend: render on the actual sim model
                _native = NativeMuJoCoRenderer(wrapped_env._env.simulator.model, render_size=256)
                expert_video = [_native.render_qpos(q) for q in expert_qpos[: 1 + ep_len]]
                frames = [_native.render_qpos(wrapped_env._env.simulator.data.qpos.copy())]
            else:
                rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
                # Only render 1 + ep_len frames (same as frames list), not the full motion
                expert_video = rgb_renderer.from_qpos(expert_qpos[: 1 + ep_len])
                frames = [rgb_renderer.render(wrapped_env._env, 0)[0]]

        print(f"Running tracking inference for {ep_len} steps")
        import time as _time
        _realtime_dt = (1.0 / 50 / max(playback_speed, 1e-6)) if (not headless and playback_speed > 0) else 0.0  # >1 = fast-forward, 0 = unthrottled
        _step_t0 = _time.perf_counter()
        # Measurement policy: LOG, DON'T INTERFERE. The rollout always runs the full
        # ep_len — nothing early-stops it. We record two independent signals per frame:
        #   - physical fall: root height < 0.3m (reported only, never breaks the loop)
        #   - tracking loss: env motion-far termination (mean per-body error > 0.5m,
        #     legged_robot_motions.py:142-143) — also reported only.
        # MPJPE is the mean per-body tracking error over ALL frames (no truncation).
        init_ref_root_h = float(obs_dict["ref_body_pos"][0, 0, 2])  # reference root height at frame 0
        root_h_hist = []  # per-frame live root height (physical-fall log)
        dif_norm_hist = []  # per-frame mean per-body tracking error (env's canonical dif_global_body_pos)
        env_terminated_hist = []  # per-frame motion-far termination flag (tracking-loss log)
        _dbg_first_n = int(os.environ.get("BDX_DBG_RESET_N", "0"))
        for i in range(ep_len):
            _zi = int(os.environ.get("BDX_Z_STRIDE", "1"))
            action = model.act(observation, z[(i // _zi) % len(z)].repeat(num_envs, 1), mean=True)
            observation, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)
            if _realtime_dt:
                # elapsed-aware throttle: total step time (compute + sleep) is capped at the
                # target so the requested speed factor is the TRUE playback speed
                _remain = _realtime_dt - (_time.perf_counter() - _step_t0)
                if _remain > 0:
                    _time.sleep(_remain)
                _step_t0 = _time.perf_counter()
            if _dbg_first_n and i < _dbg_first_n:
                _a = action.detach().cpu().numpy().flatten()
                print(f"    [act] step={i} act[min={_a.min():+.2f} max={_a.max():+.2f} mean={_a.mean():+.2f}]", flush=True)
                _t = bool(terminated[0].cpu().item()) if terminated.ndim > 0 else bool(terminated.cpu().item())
                _c = bool(truncated[0].cpu().item()) if truncated.ndim > 0 else bool(truncated.cpu().item())
                _ep_len_buf = int(wrapped_env._env.episode_length_buf[0].cpu().item()) if hasattr(wrapped_env._env, "episode_length_buf") else -1
                _err = float(torch.norm(wrapped_env._env.dif_global_body_pos[0], dim=-1).max().item())
                _pg = wrapped_env._env.projected_gravity[0].cpu().tolist()
                _cf = torch.norm(wrapped_env._env.simulator.contact_forces[0, wrapped_env._env.termination_contact_indices, :3], dim=-1)
                _cfmax = float(_cf.max().item())
                print(f"    [dbg] step={i} term={_t} trunc={_c} max_body_err={_err:.3f}m grav_xyz=[{_pg[0]:+.2f},{_pg[1]:+.2f},{_pg[2]:+.2f}] base_contact_max={_cfmax:.1f}N", flush=True)
            # --- log only, no early stop ---
            env_terminated = (bool(terminated[0].cpu().item()) if terminated.ndim > 0 else bool(terminated.cpu().item())) or (
                bool(truncated[0].cpu().item()) if truncated.ndim > 0 else bool(truncated.cpu().item())
            )
            env_terminated_hist.append(env_terminated)
            root_h_hist.append(float(wrapped_env._env.simulator._rigid_body_pos[0, 0, 2].cpu().item()))
            # env already computes the aligned (31 vs 31) per-body tracking error each step
            dif_global = wrapped_env._env.dif_global_body_pos[0]  # [num_bodies_total, 3]
            dif_norm_hist.append(float(torch.norm(dif_global, dim=-1).mean().item()))
            joint_pos.append(wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy())
            if save_mp4:
                if "_native" in dir():
                    frames.append(_native.render_qpos(wrapped_env._env.simulator.data.qpos.copy()))
                else:
                    frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])

        joint_pos = np.stack(joint_pos, axis=0).squeeze(1)
        # mean per-body tracking error over the FULL clip (no truncation by any signal)
        mpjpe = float(np.mean(dif_norm_hist)) if dif_norm_hist else float("nan")
        # physical-fall summary: first frame root height dropped below 0.3m, if ever
        fell_frame = next((i for i, h in enumerate(root_h_hist) if h < 0.3), None)
        # tracking-loss summary: first frame the env motion-far termination fired, if ever
        lost_track_frame = next((i for i, t in enumerate(env_terminated_hist) if t), None)
        min_root_h = float(min(root_h_hist)) if root_h_hist else float("nan")
        # --- completion metrics (the honest quality measure): how much of the episode
        # runs without ANY reset. Both terminated (motion-far / fall) and truncated
        # trigger a gymnasium auto-reset, which shows up in the video as a snap back.
        n_resets = int(sum(env_terminated_hist))
        first_reset = lost_track_frame
        completion = (first_reset / ep_len) if first_reset is not None else 1.0
        stats = {
            "clip": clip_name,
            "motion_id": MOTION_ID,
            "simulator": simulator,
            "ep_len": ep_len,
            "init_ref_root_h": init_ref_root_h,
            "min_root_h": min_root_h,
            "fell_frame": fell_frame,  # None = never physically fell
            "lost_track_frame": lost_track_frame,  # None = never lost tracking
            "n_resets": n_resets,
            "completion": round(completion, 3),  # fraction of episode before first reset (1.0 = full)
            "mpjpe_mean": mpjpe,
        }
        print(f"  STATS {stats}")

        if save_mp4:
            new_frames = []
            for a, b in zip(expert_video, frames):
                new_frames.append(np.concatenate([a, b], axis=1))
            # Tag with simulator + MMDDHH timestamp so runs across backends AND time don't clobber.
            _ts = datetime.now().strftime("%m%d%H")
            _safe_name = str(clip_name).replace("/", "_")  # dataset clip ids contain "/" (e.g. "v0/angry_no")
            video_path = output_dir / f"tracking_{_safe_name}__{simulator}__{_ts}.mp4"
            media.write_video(str(video_path), new_frames, fps=50)
            print(f"Saved video for tracking: {video_path}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
