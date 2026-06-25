"""[BFM-DIAG-NAN] Validate Arm B (deterministic hard-joint-limit reset) through the
REAL env step path. Sets env 0 to a state with a joint JUST past its hard limit
(left_ankle_roll at +0.45 vs limit +0.26), then runs env.step() with the reset
ACTIVE and checks: (1) no NaN escapes to the observation, (2) the past-limit env
is reset. Run with BFM_ZERO_HARD_LIMIT_RESET unset (=on) and =0 (=off, baseline)."""

import os
import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from humanoidverse.scripts.diff_sim_io import build_env  # noqa: E402

BACKEND = os.environ.get("TEST_BACKEND", "mujoco_warp")
N = 16
env = build_env(BACKEND, N)
base_env = env.base_env
sim = base_env.simulator
device = base_env.device
all_ids = torch.arange(N, device=device)

# standing root + a single joint (left_ankle_roll = dof index 5) pushed just past its hi limit (+0.26).
hard = sim.hard_dof_pos_limits  # [29,2]
print(f"[{BACKEND}] BFM_ZERO_HARD_LIMIT_RESET={os.environ.get('BFM_ZERO_HARD_LIMIT_RESET','<unset=on>')}")
print(f"[{BACKEND}] left_ankle_roll hard limit = [{hard[5,0]:.3f}, {hard[5,1]:.3f}]")

dof_pos = base_env.default_dof_pos[0].clone().repeat(N, 1)
dof_pos[:, 5] = hard[5, 1] + 0.21  # EXACT captured violation: left_ankle_roll at 0.473 vs +0.262
dof_vel = torch.zeros(N, base_env.num_dof, device=device)
dof_state = torch.stack([dof_pos, dof_vel], dim=-1)
# root: standing upright at origin
root = torch.zeros(N, 13, device=device)
root[:, 2] = 0.8; root[:, 6] = 1.0  # z=0.8, quat w=1 (xyzw)

sim.set_actor_root_state_tensor(all_ids, root.clone())
sim.set_dof_state_tensor(all_ids, dof_state.clone())
print(f"[{BACKEND}] set env 0+ dof_pos[left_ankle_roll] = {dof_pos[0,5].item():.3f} (PAST limit {hard[5,1].item():.3f})")

# Run the REAL control step (action = 0 -> default-pose target). This goes through
# _physics_step (4 substeps) then _post_physics_step -> _check_termination (our reset).
zero_action = torch.zeros(N, base_env.dim_actions, device=device)
obs, rew, terminated, truncated, info = env.step(zero_action, to_numpy=False)

# Detect reset NOT via reset_buf (that is CONSUMED inside step() by reset_envs_idx,
# so reading it post-step is always 0) but via the perturbed joint snapping back to its
# DEFAULT pose. With reset ON, the past-limit env is reset -> dof_pos[5] ~= default_dof_pos[5].
# With reset OFF, physics/PD evolves the joint -> dof_pos[5] stays far from default.
obs_finite = True
for v in obs.values():
    if torch.is_tensor(v):
        obs_finite = obs_finite and torch.isfinite(v).all()
post = sim.dof_pos[0, 5].item()
default_v = base_env.default_dof_pos[0, 5].item()
snapped_to_default = abs(post - default_v) < 0.05  # reset replaces state with default pose
print(f"[{BACKEND}] post-step dof_pos[left_ankle_roll][0] = {post:.3f} "
      f"(default={default_v:.3f}; set was {dof_pos[0,5].item():.3f})")
print(f"[{BACKEND}] OBSERVATION finite across all keys: {bool(obs_finite)}")
print(f"[{BACKEND}] snapped to default pose (=> reset fired): {snapped_to_default}")
print(f"[{BACKEND}] reward finite: {torch.isfinite(rew).all().item()}")
on = os.environ.get("BFM_ZERO_HARD_LIMIT_RESET", "1") != "0"
expect_snap = on  # reset ON -> expect snap; OFF -> expect NO snap (physics evolves)
verdict = "PASS" if (obs_finite and torch.isfinite(rew).all() and snapped_to_default == expect_snap) else "CHECK"
print(f"[{BACKEND}] VERDICT: {verdict} "
      f"(reset {'ON' if on else 'OFF'}: expect snap_to_default={expect_snap}, got={snapped_to_default})")