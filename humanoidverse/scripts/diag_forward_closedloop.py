"""[DIAGNOSTIC — NOT PART OF FROZEN SMOKE] Forward-control calibration trial.

User ruling (2026-09-07) froze a 12-trial backward smoke; its mechanical
outcome was 12/12 FAIL with net steady velocity ~0. Post-hoc decomposition
of the recorded trials showed the official deployment runtime's auto-reset
regime (episode_length_buf wrap every ~22-30 steps) teleports the robot
+0.10-0.18 m at each reset (mean +6.5 m/s over reset steps), cancelling a
genuine non-reset backward drift of -0.24..-0.35 m/s.

This diagnostic runs ONE forward trial (vx=+0.30, seed 0) through the
IDENTICAL harness/chain/adapter to calibrate the instrument: can it show
NET locomotion in the forward direction, where drift and reset teleports
share sign? Result feeds the limitation/dislosure text only — it does not
modify the frozen smoke, its thresholds, or the mechanical envelope.

Run: BDX_PLANNER_DATA=<diagFwd overlay> .venv/bin/python \
  humanoidverse/scripts/diag_forward_closedloop.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1].parents[0]))

import humanoidverse.scripts.smoke_backward_closedloop as smoke


def main():
    smoke.LEVELS = [0.30]          # forward control, within capability_v12
    smoke.SEEDS = [0]              # single trial — calibration, not a gate
    print("### DIAGNOSTIC forward-control trial vx=+0.30 (gates do not apply)")
    smoke.main()


if __name__ == "__main__":
    main()
