"""Record a pool-trained checkpoint on ONE pinned motion to mp4 (user review).

Play env with render_mode="rgb_array"; hard-pin via fixed set_motion_index;
terminations dropped so a failure plays out for the full clip (a termination
reset would _resample_command -> random reassignment mid-video).

Run from bdx_rl_mjlab repo root with ITS venv, MUJOCO_GL=egl:
  MUJOCO_GL=egl .venv/bin/python <this> --task ... --checkpoint model_1999.pt \
      --motion-index 1 --out video.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import torch


def load_actor(path: str, device: str = "cpu"):
  ck = torch.load(path, map_location="cpu", weights_only=False)
  sd = ck["actor_state_dict"]
  mean = sd["obs_normalizer._mean"].float().to(device)
  std_raw = sd["obs_normalizer._std"].float().to(device)
  std = torch.where(std_raw > 1e-8, std_raw, torch.ones_like(std_raw))
  layers = [(sd[f"mlp.{i}.weight"].float().to(device),
             sd[f"mlp.{i}.bias"].float().to(device)) for i in (0, 2, 4, 6)]

  @torch.no_grad()
  def act(obs):
    x = (obs - mean) / std
    for w, b in layers[:-1]:
      x = torch.nn.functional.elu(x @ w.T + b)
    w, b = layers[-1]
    return x @ w.T + b

  return act, int(ck["iter"])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--motion-index", type=int, required=True)
  ap.add_argument("--steps", type=int, default=0,
                  help="0 = full clip (time_step_total - 2)")
  ap.add_argument("--every", type=int, default=1, help="render every Nth step")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  import tracking_bdx.config.bdx_v4  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  assert not Path(args.out).exists(), f"refusing overwrite {args.out}"
  torch.manual_seed(0)
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = 1
  cfg.terminations = {}  # show failures for the full clip; no mid-video resample
  cfg.viewer.height, cfg.viewer.width = 480, 854
  cfg.viewer.max_extra_envs = 0  # single-robot focus
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array")
  mc = env.command_manager.get_term("motion")
  robot = env.scene["robot"]
  act, it = load_actor(args.checkpoint)

  name = Path(str(mc.cfg.motion_files[args.motion_index])).name
  T = int(mc.motion_pool[args.motion_index].time_step_total)
  steps = args.steps or (T - 2)
  print(f"motion {args.motion_index} = {name}, {T} frames, recording {steps} steps")

  env.reset()
  mc.set_motion_index(args.motion_index)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  env.command_manager.compute(dt=0.0)
  obs = env.observation_manager.compute(update_history=True)

  frames, zs = [], []
  for step in range(steps):
    a = act(obs["actor"])
    zs.append(float(robot.data.root_link_pose_w[0, 2]))
    obs, _, _, _, _ = env.step(a)
    if step % args.every == 0:
      frames.append(env.render())
  fps = round(1.0 / env.step_dt / args.every)
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(args.out, frames, fps=fps, quality=8, macro_block_size=1)
  env.close()
  print(f"saved {args.out}: {len(frames)} fr @ {fps} fps, "
        f"min_z={min(zs):.3f} mean_z={sum(zs)/len(zs):.3f}")


if __name__ == "__main__":
  main()
