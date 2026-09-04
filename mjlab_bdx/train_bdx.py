"""Train BDX side_step_4 tracking with mjlab + RSL-RL PPO.

Usage:
  .venv-mjlab/bin/python mjlab_bdx/train_bdx.py [--num-envs 1024]
      [--max-iterations 5000] [--log-root results/mjlab_bdx/logs]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bdx_tracking  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--num-envs", type=int, default=1024)
  ap.add_argument("--max-iterations", type=int, default=None)
  ap.add_argument("--log-root", default=None)
  args = ap.parse_args()

  bdx_tracking.register()

  from mjlab.scripts.train import TrainConfig, launch_training
  cfg = TrainConfig.from_task(bdx_tracking.TASK_ID)
  cfg.env.scene.num_envs = args.num_envs
  if args.max_iterations is not None:
    cfg.agent.max_iterations = args.max_iterations
  if args.log_root is not None:
    cfg.log_root = args.log_root
  launch_training(bdx_tracking.TASK_ID, cfg)


if __name__ == "__main__":
  main()
