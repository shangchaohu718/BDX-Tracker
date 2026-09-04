"""Single source of truth for the BDX 14-DOF observation-state contract.

The wrapper (humanoidverse_isaac._get_g1env_observation) concatenates:
    [dof_pos(14), dof_vel(14), projected_gravity(3), base_ang_vel(3)] -> 34

ALL activity/motion metrics and conditional-pairing splits MUST go through
these functions (external ruling 2026-08-24: a previous `state[:, 20:34]`
convention was a contract error that polluted activity labels across probes,
L_cond pairing and diagnostics — see handoff section 18).
"""

import torch

DOF_POS = slice(0, 14)
DOF_VEL = slice(14, 28)
GRAVITY = slice(28, 31)
BASE_ANG_VEL = slice(31, 34)


def split_bdx_state(state: torch.Tensor):
    """Split a (..., 34) BDX state tensor into its four blocks."""
    assert state.shape[-1] == 34, f"BDX state must end in 34 dims, got {state.shape[-1]}"
    return (
        state[..., DOF_POS],
        state[..., DOF_VEL],
        state[..., GRAVITY],
        state[..., BASE_ANG_VEL],
    )


def bdx_motion_activity(state: torch.Tensor) -> torch.Tensor:
    """Joint-space motion activity = L2 norm of the dof_vel block."""
    return split_bdx_state(state)[1].norm(dim=-1)


def build_bdx_actor_history(obs_rows: torch.Tensor, action_rows: torch.Tensor, t: int) -> torch.Tensor:
    """Build the 192-dim history_actor vector matching the env contract.

    HistoryHandler.add writes the NEWEST value at index 0 (history[:, 1:]
    shifts right), so query[:, :4] is newest-first: [t, t-1, t-2, t-3].
    The obs layout uses _get_obs_long_history: sorted key order
    (actions, base_ang_vel, dof_pos, dof_vel, projected_gravity), each key's
    4 frames flattened row-major, then concatenated.

    obs_rows/action_rows: [T, 34]/[T, 14] RAW pool observation rows.
    Requires t >= 3 (callers must restrict to positions with a full prefix —
    no padding simulation; episode starts in the env are zero-filled, which
    cannot be reconstructed from pool data, so we refuse t < 3 instead).
    """
    assert t >= 3, "build_bdx_actor_history requires a full 4-frame prefix (t >= 3)"
    frames = [t, t - 1, t - 2, t - 3]  # newest-first, matching handler contract
    blocks = []
    for i in frames:
        blocks.append(action_rows[i])                     # actions (14)
    for i in frames:
        blocks.append(obs_rows[i][31:34])                 # base_ang_vel (3)
    for i in frames:
        blocks.append(obs_rows[i][0:14])                  # dof_pos (14)
    for i in frames:
        blocks.append(obs_rows[i][14:28])                 # dof_vel (14)
    for i in frames:
        blocks.append(obs_rows[i][28:31])                 # projected_gravity (3)
    return torch.cat(blocks)
