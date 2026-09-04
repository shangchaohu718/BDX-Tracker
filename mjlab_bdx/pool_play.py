"""Interactive viser viewer for a POOL-trained checkpoint, with motion pinning.

The stock `play` entry works for pool tasks but assigns a RANDOM motion on every
reset. This wrapper pins one motion (monkeypatching _assign_motions) so resets
and clip-end resamples always replay the chosen clip — the honest way to watch
"the policy tracking motion X".

Usage (from bdx_rl_mjlab root):
  MUJOCO_GL=egl .venv/bin/python pool_play.py \
      --task Mjlab-Tracking-Flat-BDX-V4-Wiggle-Pool-8-V2 \
      --checkpoint .../model_1999.pt \
      --motion-index 1            # -1 = stock random behavior
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--motion-index", type=int, default=-1,
                  help="-1 = random per reset (stock behavior)")
  ap.add_argument("--num-envs", type=int, default=1)
  ap.add_argument("--port", type=int, default=8080)
  args = ap.parse_args()

  import tracking_bdx.config.bdx_v4  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.viewer import ViserPlayViewer

  device = "cpu"
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task)
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
              map_location=device)
  policy = runner.get_inference_policy(device=device)

  mc = env.unwrapped.command_manager.get_term("motion")
  names = [Path(str(f)).name for f in mc.cfg.motion_files]
  if args.motion_index >= 0:
    pin = args.motion_index
    mc.set_motion_index(pin)

    def _pin_assign(env_ids, _mc=mc, _pin=pin):
      # Teleport on EVERY (re)assign: play-path resets write qpos0 (z~0.35),
      # and crouch-family references (dance/gesture z~0.23) violate the z
      # band at birth -> instant-termination reset loop (robot frozen at 0,
      # ref frozen at frame 1). set_motion_index teleports onto the ref
      # frame, which is what the eval harness has always done.
      _mc.set_motion_index(_pin)

    mc._assign_motions = _pin_assign
    print(f"[pool_play] pinned motion {pin} = {names[pin]} "
          f"(resets & clip-end replays stay on it)")
  else:
    print("[pool_play] random motion per reset. Pool contents:")
    for i, n in enumerate(names):
      print(f"  [{i}] {n}")

  ViserPlayViewer(env, policy).run()
  env.close()


if __name__ == "__main__":
  main()
