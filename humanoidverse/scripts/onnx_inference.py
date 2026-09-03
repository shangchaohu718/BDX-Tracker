#!/usr/bin/env python
"""ONNX inference for BFM-Zero tracking — torch-free actor, env machinery for obs.

Drives the sim with actions from an exported ONNX actor (exported/FBcprAuxModel.onnx)
instead of the torch model. Everything else mirrors tracking_inference.py: the env
builds proprioception/history observations, z comes precomputed per clip
(tracking_inference/zs_<id>__mujoco.pkl, open-loop from the reference motion).

This validates the DEPLOYMENT artifact end-to-end: if this rollout matches the
torch rollout, the ONNX actor is behaviorally identical on the real obs stream
(not just on random inputs).

Usage (G1 750M, motion 25, live window):
    MUJOCO_GL=glfw uv run python -m humanoidverse.scripts.onnx_inference \
        --model_folder remote_checkpoint_750M \
        --motion_list 25 --episode_len 500 --no-headless
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen default; MUJOCO_GL=glfw + --no-headless for a live window
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_folder", type=Path, required=True,
                    help="checkpoint dir containing config.json + tracking_inference/zs_*.pkl")
    ap.add_argument("--onnx", type=Path, default=None,
                    help="ONNX actor (default: <model_folder>/exported/FBcprAuxModel.onnx)")
    ap.add_argument("--motion_list", type=int, nargs="+", default=[25])
    ap.add_argument("--episode_len", type=int, default=0, help="0 = full motion")
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.add_argument("--disable_dr", action="store_true")
    args = ap.parse_args()

    import onnxruntime as ort
    import torch
    import joblib
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
    from humanoidverse.utils.helpers import get_backward_observation

    model_folder = args.model_folder
    onnx_path = args.onnx or model_folder / "exported" / "FBcprAuxModel.onnx"
    assert onnx_path.exists(), f"no ONNX at {onnx_path} (run tracking_inference once to export)"

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    total_dim = inp.shape[1]
    z_dim = None  # resolved from zs pkl below

    import json
    config = json.load(open(model_folder / "config.json"))
    use_root_height_obs = config["env"].get("root_height_obs", False)

    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(f"env.config.headless={args.headless}")
    config["env"]["hydra_overrides"].append("simulator=mujoco")
    config["env"]["disable_domain_randomization"] = args.disable_dr
    config["env"]["disable_obs_noise"] = True

    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env

    num_envs = 1
    action_dim = env.num_actions if hasattr(env, "num_actions") else None

    for MOTION_ID in args.motion_list:
        env.set_is_evaluating(MOTION_ID)
        loaded_keys = getattr(env._motion_lib, "curr_motion_keys", None)
        clip_name = str(loaded_keys[0]) if loaded_keys is not None else f"motion_{MOTION_ID}"
        print(f"\n{'='*80}\nMotion {MOTION_ID}: {clip_name}\n{'='*80}")

        zs_path = model_folder / "tracking_inference" / f"zs_{MOTION_ID}__mujoco.pkl"
        z_all = joblib.load(zs_path)
        device = env.device
        z_all = torch.tensor(np.asarray(z_all), dtype=torch.float32, device=device)
        if z_dim is None:
            z_dim = z_all.shape[-1]
            assert total_dim - z_dim + z_dim == total_dim  # sanity
            actor_obs_dim = total_dim - z_dim
            print(f"ONNX input dim {total_dim} = actor_obs {actor_obs_dim} + z {z_dim}")

        obs, obs_dict = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs)

        # same init as tracking_inference: ref frame 0 as robot init state
        dof_init_state = torch.zeros_like(wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0])
        dof_init_state[..., 0] = obs_dict["dof_pos"][0]
        dof_init_state[..., 1] = obs_dict["ref_dof_vel"][0]
        ref_body_rots = obs_dict["ref_body_rots"][0, 0].clone()
        ref_root_init_state = torch.cat([
            obs_dict["ref_body_pos"][0, 0],
            ref_body_rots,
            obs_dict["ref_body_vels"][0, 0],
            obs_dict["ref_body_angular_vels"][0, 0],
        ])
        target_states = {
            "dof_states": dof_init_state,
            "root_states": torch.stack([ref_root_init_state.clone()]),
        }
        env_ids = torch.arange(num_envs, dtype=torch.long)
        observation, info = wrapped_env._env.reset_envs_idx(env_ids, target_states=target_states)
        observation_new, reward, terminated, truncated, info = wrapped_env.step(
            torch.zeros((num_envs, wrapped_env.action_space.shape[-1]), dtype=torch.float32), to_numpy=False)
        observation = wrapped_env._get_g1env_observation(to_numpy=False)

        full_len = z_all.shape[0]
        ep_len = min(args.episode_len, full_len) if args.episode_len and args.episode_len > 0 else full_len
        print(f"Running ONNX tracking inference for {ep_len} steps")

        root_h_hist, dif_norm_hist = [], []
        dump = os.environ.get("OBS_DUMP")  # torch-path ground truth for runtime diffing
        dump_rows = [] if dump else None
        for i in range(ep_len):
            # flat actor_obs in the SAME order the export wrapper slices it:
            # [state, last_action, history_actor] — then z is appended.
            actor_obs = torch.cat([
                observation["state"],
                observation["last_action"],
                observation["history_actor"],
            ], dim=-1)
            if dump:
                qpos = env.simulator.data.qpos.copy()
                qvel = env.simulator.data.qvel.copy()
                dump_rows.append({
                    "i": i, "qpos": qpos, "qvel": qvel,
                    "state": observation["state"][0].cpu().numpy(),
                    "last_action": observation["last_action"][0].cpu().numpy(),
                    "history": observation["history_actor"][0].cpu().numpy(),
                    "torque": env.simulator.data.ctrl.copy(),
                })
            x = torch.cat([actor_obs, z_all[i % full_len].unsqueeze(0)], dim=-1).cpu().numpy()
            action = torch.from_numpy(sess.run(None, {inp.name: x})[0])
            if dump:
                dump_rows[-1]["action"] = action[0].cpu().numpy()
            observation, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)
            root_h_hist.append(float(env.simulator._rigid_body_pos[0, 0, 2].cpu().item()))
            dif = env.dif_global_body_pos[0]
            dif_norm_hist.append(float(torch.norm(dif, dim=-1).mean().item()))
        if dump:
            np.savez(dump, **{k: np.stack([r[k] for r in dump_rows]) for k in dump_rows[0]})
            print(f"Dumped {len(dump_rows)} steps -> {dump}")

        stats = {
            "clip": clip_name,
            "motion_id": MOTION_ID,
            "onnx": str(onnx_path),
            "ep_len": ep_len,
            "min_root_h": min(root_h_hist),
            "mpjpe_mean": float(np.mean(dif_norm_hist)),
        }
        print(f"  STATS {stats}")


if __name__ == "__main__":
    main()