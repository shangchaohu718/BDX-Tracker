#!/usr/bin/env python3
# live_inference.py — REAL-TIME local inference in a live MuJoCo GLFW viewer, with keyboard control.
#
# Opens a mujoco.viewer.launch_passive window, paces the control loop to wall-clock dt, and lets
# you switch motion clips at the keyboard:
#   N      -> next motion clip (advance motion_id by 1)
#   P      -> previous motion clip
#   R      -> restart current clip from frame 0
#   Q/Esc  -> quit
# The active motion restarts immediately on a switch.
#
# Usage:
#   .venv/bin/python -m humanoidverse.scripts.live_inference \
#       --model_folder remote_checkpoint_301M --motion_id 25
import os
os.environ.setdefault("MUJOCO_GL", "glfw")   # live window needs glfw, NOT egl (offscreen)
os.environ["OMP_NUM_THREADS"] = "1"

import argparse, json, time
from pathlib import Path
import mujoco.viewer as mjv
import numpy as np
import torch
from torch.utils._pytree import tree_map

import humanoidverse
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.utils.helpers import get_backward_observation

if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent

# GLFW key codes (subset). mujoco's key_callback receives the raw GLFW key int.
GLFW_N, GLFW_P, GLFW_R, GLFW_Q, GLFW_ESCAPE = 78, 80, 82, 81, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_folder", required=True, type=Path)
    ap.add_argument("--motion_id", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--z_mode", choices=["closedloop", "openloop"], default="closedloop")
    args = ap.parse_args()

    model_folder = args.model_folder
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=args.device)
    model.to(args.device); model.eval()
    model.cfg = model.cfg.model_copy(update={"amp": False})  # fp32 inference

    with open(model_folder / "config.json") as f:
        config = json.load(f)
    use_root_height_obs = config["env"].get("root_height_obs", False)
    default_pkl = HUMANOIDVERSE_DIR / "data" / "lafan_29dof.pkl"
    if not Path(config["env"].get("lafan_tail_path", "")).exists():
        config["env"]["lafan_tail_path"] = str(default_pkl) if default_pkl.exists() else "data/lafan_29dof.pkl"

    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append("env.config.headless=True")
    config["env"]["hydra_overrides"].append("simulator=mujoco")
    config["env"]["disable_domain_randomization"] = True
    config["env"]["disable_obs_noise"] = True

    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env
    sim = env.simulator
    dt = float(env.dt)
    # NOTE: _motion_lengths only holds the on-demand-loaded eval motion (len 1). The FULL clip
    # list is _motion_data_keys (862 for the clipped eval pkl) -- use that for next/prev cycling.
    all_keys = getattr(env._motion_lib, "_motion_data_keys", None)
    num_motions = len(all_keys) if all_keys is not None else len(env._motion_lib._motion_lengths)
    print(f"simulator: mujoco | env.dt={dt:.4f}s ({1/dt:.1f} fps) | {num_motions} motions available")

    # shared state between the viewer key thread and the main loop
    state = {"motion_id": args.motion_id, "restart": True, "quit": False}

    def key_callback(key):
        # called by mujoco's GLFW thread on key press
        if key == GLFW_N:
            state["motion_id"] = (state["motion_id"] + 1) % num_motions
            state["restart"] = True
            print(f"[N] next -> motion {state['motion_id']}")
        elif key == GLFW_P:
            state["motion_id"] = (state["motion_id"] - 1) % num_motions
            state["restart"] = True
            print(f"[P] prev -> motion {state['motion_id']}")
        elif key == GLFW_R:
            state["restart"] = True
            print("[R] restart current clip")
        elif key in (GLFW_Q, GLFW_ESCAPE):
            state["quit"] = True
            print("[Q] quitting...")

    # open the LIVE viewer WITH a key callback (sim.setup_viewer doesn't pass one)
    sim.viewer = mjv.launch_passive(sim.model, sim.data, key_callback=key_callback)
    print("Live viewer opened. Keys:  N=next  P=prev  R=restart  Q/Esc=quit")

    def compute_z(obs):
        z = model.backward_map(obs)
        if args.z_mode == "openloop":
            for step in range(z.shape[0]):
                z[step] = z[step:].mean(dim=0)
        return model.project_z(z)

    def init_to_ref(obs_dict):
        ref_root = torch.cat([
            obs_dict["ref_body_pos"][0, 0], obs_dict["ref_body_rots"][0, 0],
            obs_dict["ref_body_vels"][0, 0], obs_dict["ref_body_angular_vels"][0, 0],
        ])
        dof_init = torch.zeros_like(sim.dof_state.view(1, -1, 2)[0])
        dof_init[..., 0] = obs_dict["dof_pos"][0]
        dof_init[..., 1] = obs_dict["ref_dof_vel"][0]
        target_states = {"dof_states": dof_init, "root_states": torch.stack([ref_root.clone()])}
        env_ids = torch.arange(1, dtype=torch.long)
        wrapped_env._env.reset_envs_idx(env_ids, target_states=target_states)
        wrapped_env.step(torch.zeros((1, wrapped_env.action_space.shape[-1]), dtype=torch.float32),
                         to_numpy=False)
        return wrapped_env._get_g1env_observation(to_numpy=False)

    try:
        while not state["quit"]:
            if sim.viewer is None:
                break
            if state["restart"]:
                state["restart"] = False
                mid = state["motion_id"]
                env.set_is_evaluating(mid)
                loaded = getattr(env._motion_lib, "curr_motion_keys", None)
                clip = str(loaded[0]) if loaded is not None else f"motion_{mid}"
                print(f"  >> motion {mid}: {clip}")
                obs, obs_dict = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs)
                z = compute_z(tree_map(lambda x: x[1:], obs))
                n_steps = z.shape[0]
                observation = init_to_ref(obs_dict)

            for i in range(n_steps):
                if state["quit"] or sim.viewer is None or state["restart"]:
                    break
                loop_start = time.monotonic()
                action = model.act(observation, z[i % len(z)].repeat(1, 1), mean=True)
                observation, _, _, _, _ = wrapped_env.step(action, to_numpy=False)
                elapsed = time.monotonic() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
            else:
                # clip finished naturally -> loop the same motion
                state["restart"] = True
    except KeyboardInterrupt:
        print("\nstopped by user.")
    finally:
        if sim.viewer is not None:
            try: sim.viewer.close()
            except Exception: pass


if __name__ == "__main__":
    main()