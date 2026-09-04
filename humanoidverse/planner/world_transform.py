"""Production canonical<->world frame transforms — single implementation
consumed by export_planner_refs, demos, closed_loop_test and the contract
tests (audit 2026-08-24: tests must not re-derive the math themselves)."""
import numpy as np


def canonical_Rz(psi):
    """The canonicalization rotation used by dataset.canonical_sparse:
    Rz = [[c,s,0],[-s,c,0],[0,0,1]] — row-vector v @ Rz.T applies R(-psi)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]], np.float32)


def pos_canonical_to_world(pos_c, anchor, psi):
    """Inverse of (pos_w - anchor) @ Rz.T : returns pos_c @ Rz + anchor."""
    return pos_c @ canonical_Rz(psi) + anchor


def rot_canonical_to_world(R_c, psi):
    """Inverse of Rz @ R_w (quat canonicalization): Rz.T @ R_c."""
    return canonical_Rz(psi).T @ R_c
