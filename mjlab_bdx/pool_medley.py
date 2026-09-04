"""One checkpoint performs MANY motions back-to-back — the generality demo.

For each motion index: hard-pin (fixed set_motion_index), teleport, run the
full clip collecting RGB frames; concatenate everything into one mp4.
Terminations dropped (pool_record semantics) so nothing cuts a segment.

Usage (from bdx_rl_mjlab root, MUJOCO_GL=egl):
  .venv/bin/python pool_medley.py --task <pool task> --checkpoint model_1999.pt \
      --motion-indices 0,6,9,22,27,34,43,46 --out medley.mp4
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
  ap.add_argument("--motion-indices", required=True,
                  help="comma-separated pool indices, played in order")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  import tracking_bdx.config.bdx_v4  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  assert not Path(args.out).exists(), f"refusing overwrite {args.out}"
  idxs = [int(x) for x in args.motion_indices.split(",")]
  torch.manual_seed(0)
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = 1
  cfg.terminations = {}
  cfg.viewer.height, cfg.viewer.width = 480, 854
  cfg.viewer.max_extra_envs = 0
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array")
  mc = env.command_manager.get_term("motion")
  robot = env.scene["robot"]
  act, it = load_actor(args.checkpoint)

  fps = round(1.0 / env.step_dt)
  frames, order, per = [], [], []
  for mi in idxs:
    name = Path(str(mc.cfg.motion_files[mi])).name
    T = int(mc.motion_pool[mi].time_step_total)
    env.reset()
    mc.set_motion_index(mi)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    env.command_manager.compute(dt=0.0)
    obs = env.observation_manager.compute(update_history=True)
    zs = []
    for _ in range(T - 2):
      a = act(obs["actor"])
      zs.append(float(robot.data.root_link_pose_w[0, 2]))
      obs, _, _, _, _ = env.step(a)
      frames.append(env.render())
    order.append(f"{mi}:{name}")
    per.append((name, min(zs), sum(zs) / len(zs)))
    print(f"[{mi}] {name}: {T - 2} steps, min_z={min(zs):.3f}", flush=True)

  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(args.out, frames, fps=fps, quality=8, macro_block_size=1)
  env.close()
  print("order:", " -> ".join(order))
  print(f"saved {args.out}: {len(frames)} fr @ {fps} fps "
        f"({len(frames)/fps:.1f}s, {len(idxs)} motions)")


if __name__ == "__main__":
  main()
