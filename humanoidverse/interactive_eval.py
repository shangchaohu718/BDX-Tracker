# [INTERACTIVE-EVAL] Local GUI tool: browse ALL clips of the eval corpus, click one,
# and watch the policy's tracking inference played back in real time next to the
# reference motion (robot | ghost reference, both rendered from the same mujoco model).
#
# Usage (local, needs a display):
#   BFM_ZERO_ROBOT=bdx BFM_ZERO_PROFILE=bdx3 BFM_ZERO_MOTION_PKL=humanoidverse/data/bdx_walkexport_clipped.pkl \
#   uv run python -m humanoidverse.interactive_eval --model-folder remote_checkpoint_144M \
#       --data-path humanoidverse/data/bdx_walkexport.pkl
#
# The motion ids are the SORTED index into the eval pkl's key list (same convention as
# tracking_inference --motion-list).

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering into the Tk panels
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

import humanoidverse
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.helpers import get_backward_observation

HUMANIDVERSE_DIR = Path(humanoidverse.__file__).parent

import tkinter as tk
from PIL import Image, ImageTk


class InteractiveEval:
    def __init__(self, model_folder: Path, data_path: Path | None, device="cuda", render_size=384):
        import mujoco  # after MUJOCO_GL is set

        self.mujoco = mujoco
        self.render_size = render_size

        model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
        model.to(device)
        model.eval()
        self.model = model
        self.device = device

        with open(model_folder / "config.json") as f:
            config = json.load(f)
        use_root_height_obs = config["env"].get("root_height_obs", False)
        if data_path is not None:
            config["env"]["lafan_tail_path"] = str(Path(data_path).resolve())
        config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
        config["env"]["hydra_overrides"].append("env.config.headless=True")
        # classic mujoco backend: profile default may be mujoco_warp (no .model/.data,
        # no per-frame render) — we need the single-env classic sim for live rendering.
        config["env"]["hydra_overrides"].append("simulator=mujoco")
        config["env"]["disable_domain_randomization"] = True
        config["env"]["disable_obs_noise"] = True

        env_cfg = HumanoidVerseIsaacConfig(**config["env"])
        self.wrapped_env, _ = env_cfg.build(num_envs=1)
        self.env = self.wrapped_env._env
        self.sim = self.env.simulator
        self.use_root_height_obs = use_root_height_obs

        # clip list = the eval corpus, sorted keys == motion ids (same as --motion-list)
        self.clip_keys = sorted(self.env._motion_lib._motion_data_keys)
        print(f"[interactive_eval] {len(self.clip_keys)} clips in eval corpus")

        # reference ghost: own MjData + Renderer on the same model
        self.ref_data = mujoco.MjData(self.sim.model)
        self.ref_renderer = mujoco.Renderer(self.sim.model, height=render_size, width=render_size)

        # playback state
        self.z = None
        self.expert_qpos = None
        self.observation = None
        self.frame = 0
        self.ep_len = 0
        self.playing = False
        self._last_step_wall = 0.0
        self.dif_hist = []
        self._gen = 0          # playback generation: stale after()-chains abort on mismatch
        self._start_after = None  # debounced click-to-play timer

        self._build_gui()

    # ------------------------------------------------------------------ GUI
    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("BDX interactive tracking eval")

        left = tk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(left, text="Filter:").pack(anchor=tk.W)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._refresh_list())
        tk.Entry(left, textvariable=self.search_var).pack(fill=tk.X)
        self.listbox = tk.Listbox(left, width=48, height=34, exportselection=False)
        self.listbox.pack(fill=tk.Y, expand=True)
        self.listbox.bind("<Double-Button-1>", lambda e: self.play_selected())
        # single click on a clip starts it right away (stopping any current playback)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)
        tk.Button(left, text="Play selected", command=self.play_selected).pack(fill=tk.X)
        self.stop_btn = tk.Button(left, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X)
        speed_frame = tk.Frame(left)
        speed_frame.pack(fill=tk.X)
        tk.Label(speed_frame, text="speed:").pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="2x")
        tk.OptionMenu(speed_frame, self.speed_var, "1x", "2x", "4x", "8x", "max").pack(side=tk.LEFT, fill=tk.X, expand=True)

        right = tk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        panels = tk.Frame(right)
        panels.pack()
        self.robot_img = self._placeholder(panels, "policy (robot)")
        self.ref_img = self._placeholder(panels, "reference (ghost)")
        self.status = tk.Label(right, text="select a clip and press Play", anchor=tk.W, justify=tk.LEFT, font=("Monospace", 9))
        self.status.pack(fill=tk.X)

        self._refresh_list()

    def _placeholder(self, parent, title):
        frame = tk.Frame(parent)
        frame.pack(side=tk.LEFT, padx=4)
        tk.Label(frame, text=title).pack()
        lbl = tk.Label(frame)
        lbl.pack()
        return lbl

    def _refresh_list(self):
        q = self.search_var.get().lower()
        self.listbox.delete(0, tk.END)
        self.matches = [(i, k) for i, k in enumerate(self.clip_keys) if q in k.lower()]
        for _, k in self.matches[:2000]:
            self.listbox.insert(tk.END, k)

    # ------------------------------------------------------------- playback
    def _on_list_select(self, _event):
        # debounce: <<ListboxSelect>> can fire several times per click; collapse into
        # one delayed start, so rapid clicks/storms restart playback at most once
        if self._start_after is not None:
            self.root.after_cancel(self._start_after)
        self._start_after = self.root.after(80, self.play_selected)

    def play_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        if self.playing:  # a new selection interrupts whatever is currently playing
            self.stop()
            self.root.update_idletasks()
        motion_id = self.matches[sel[0]][0]
        self.play(motion_id)

    def play(self, motion_id: int):
        env = self.env
        env.set_is_evaluating(motion_id)
        clip = str(env._motion_lib.curr_motion_keys[0])
        print(f"[interactive_eval] motion {motion_id}: {clip}")

        obs, obs_dict = get_backward_observation(env, 0, use_root_height_obs=self.use_root_height_obs)

        # z: per-frame backward_map(target), no smoothing (FB-faithful open loop)
        z = self.model.backward_map(tree_map(lambda x: x[1:], obs))
        self.z = self.model.project_z(z)
        self.expert_qpos = np.concatenate([
            obs_dict["ref_body_pos"][:, 0].cpu().numpy(),
            np.roll(obs_dict["ref_body_rots"][:, 0].cpu().numpy(), 1, axis=-1),
            obs_dict["dof_pos"].cpu().numpy(),
        ], axis=-1)

        # reset robot onto the reference start state (same recipe as tracking_inference)
        self.wrapped_env.reset(to_numpy=False)
        ref_body_rots = obs_dict["ref_body_rots"][0, 0].clone()
        ref_root_init_state = torch.cat([
            obs_dict["ref_body_pos"][0, 0],
            ref_body_rots,
            obs_dict["ref_body_vels"][0, 0],
            obs_dict["ref_body_angular_vels"][0, 0],
        ])
        dof_init_state = torch.zeros_like(self.sim.dof_state.view(1, -1, 2)[0])
        dof_init_state[..., 0] = obs_dict["dof_pos"][0]
        dof_init_state[..., 1] = obs_dict["ref_dof_vel"][0]
        target_states = {
            "dof_states": dof_init_state,
            "root_states": torch.stack([ref_root_init_state.clone()]),
        }
        self.observation, _ = self.wrapped_env._env.reset_envs_idx(torch.arange(1, dtype=torch.long), target_states=target_states)
        self.observation = self.wrapped_env._get_g1env_observation(to_numpy=False)

        self.frame = 0
        self.ep_len = self.z.shape[0]
        self.dif_hist = []
        self.clip_name = clip
        self.playing = True
        self.stop_btn.config(state=tk.NORMAL)
        self._last_step_wall = time.time()
        self._gen += 1
        self.root.after(1, lambda: self._step(self._gen))

    def stop(self):
        self.playing = False
        self.stop_btn.config(state=tk.DISABLED)
        self._set_status("stopped")

    def _step(self, gen=None):
        # abort stale chains: a new play() bumped the generation, this callback is
        # from the previous clip's after()-chain (running two chains would step the
        # env twice per frame)
        if not self.playing or (gen is not None and gen != self._gen):
            return
        i = self.frame
        action = self.model.act(self.observation, self.z[i % len(self.z)].repeat(1, 1), mean=True)
        self.observation, _, terminated, truncated, _ = self.wrapped_env.step(action, to_numpy=False)
        self.frame += 1

        dif = float(torch.norm(self.env.dif_global_body_pos[0], dim=-1).mean().item())
        root_h = float(self.sim._rigid_body_pos[0, 0, 2].cpu().item())
        ref_h = float(self.expert_qpos[min(i, len(self.expert_qpos) - 1), 2])
        self.dif_hist.append(dif)

        robot_rgb = self.sim.render()  # offscreen render of the live sim state
        ref_rgb = self._render_ref(min(i, len(self.expert_qpos) - 1))
        self._show(self.robot_img, robot_rgb)
        self._show(self.ref_img, ref_rgb)
        self._set_status(
            f"clip: {self.clip_name}\n"
            f"frame {i + 1}/{self.ep_len}   ({(i + 1) * self.env.dt:.2f}s)\n"
            f"tracking err (mean body): {dif:.3f} m   avg so far: {np.mean(self.dif_hist):.3f} m\n"
            f"root height: robot {root_h:.3f} m | ref {ref_h:.3f} m"
        )

        if i + 1 >= self.ep_len:
            self.stop()
            self._set_status(f"DONE  {self.clip_name}  avg err {np.mean(self.dif_hist):.3f} m")
            return
        # pacing: real-time dt divided by the speed factor ("max" = no intentional sleep)
        speed = self.speed_var.get()
        if speed == "max":
            wait_ms = 1
        else:
            now = time.time()
            budget = self.env.dt / float(speed[:-1])
            wait_ms = max(1, int((budget - (now - self._last_step_wall)) * 1000))
            self._last_step_wall = time.time()
        self.root.after(wait_ms, lambda: self._step(gen))

    def _render_ref(self, i):
        self.ref_data.qpos[:] = self.expert_qpos[i]
        self.mujoco.mj_forward(self.sim.model, self.ref_data)
        self.ref_renderer.update_scene(self.ref_data, camera=-1)
        return self.ref_renderer.render()

    def _show(self, label, rgb):
        img = Image.fromarray(rgb)
        img = img.resize((self.render_size, self.render_size))
        label._photo = ImageTk.PhotoImage(img)
        label.config(image=label._photo)

    def _set_status(self, text):
        self.status.config(text=text)

    def run(self):
        self.root.mainloop()
        self.wrapped_env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=HUMANIDVERSE_DIR / "data" / "bdx_walkexport.pkl")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    InteractiveEval(args.model_folder, args.data_path, device=args.device).run()


if __name__ == "__main__":
    main()