"""Planner representation contract tests — the audit's one-line-test lesson.

Covers the failure modes found in the 2026-08-21 external audit:
  1. 6D rotation round-trip (identity / random SO(3))
  2. sparse stats contact index (27:29, not the last two dims)
  3. canonical_sparse yaw-invariance of ALL fields incl. base quaternion
  4. world<->canonical round-trip for position AND rotation

Run: .venv/bin/python humanoidverse/tests/test_planner_contracts.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from scipy.spatial.transform import Rotation as sRot

from humanoidverse.planner.dataset import (mat_to_6d, sixd_to_mat,
                                           canonical_sparse,
                                           set_preprocess_version)
set_preprocess_version("fixed_v1")  # contract tests verify the CORRECT math


def test_6d_roundtrip_identity():
    R = torch.eye(3)
    back = sixd_to_mat(mat_to_6d(R))
    assert (back - R).abs().max() < 1e-5, f"identity round-trip err {(back-R).abs().max()}"


def test_6d_roundtrip_random():
    rng = np.random.default_rng(0)
    Rs = torch.from_numpy(sRot.random(200, random_state=0).as_matrix()).float()
    back = sixd_to_mat(mat_to_6d(Rs))
    err = (back - Rs).abs().max()
    assert err < 1e-4, f"random round-trip err {err}"


def test_stats_contact_index():
    import json
    stats = json.loads((Path(__file__).parents[1] / "data" /
                        "bdx_planner_v2combo" / "p2_stats32.json").read_text())
    mean = np.array(stats["mean"])   # (50, 32)
    # contact @ 27:29 must be identity-normalized (0 mean, 1 std)
    assert np.allclose(mean[:, 27:29], 0.0), "contact not at identity normalization"
    # head_pos dims must NOT be all-zero-mean (they carry real offsets)
    assert np.abs(mean[:, 29:32]).max() > 1e-4, "head_pos unexpectedly zero"


def _fake_frames(yaw_deg):
    """Minimal frames dict with a known world yaw."""
    n = 10
    q = sRot.from_euler("z", np.radians(yaw_deg)).as_quat()  # xyzw
    bq = np.tile(np.roll(q, 1), (n, 1)).astype(np.float32)  # wxyz
    zax = np.array([0, 0, 1], np.float32)
    def rot(v):
        return v  # positions/vels zero — only quat matters here
    return {
        "base_pos": np.zeros((n, 3), np.float32),
        "base_quat": bq,
        "base_linvel": np.zeros((n, 3), np.float32),
        "base_angvel": np.zeros((n, 3), np.float32),
        "head_rotvec": np.zeros((n, 3), np.float32),
        "head_angvel": np.zeros((n, 3), np.float32),
        "foot_pos": np.zeros((n, 2, 3), np.float32),
        "foot_yaw": np.zeros((n, 2), np.float32),
        "contact": np.zeros((n, 2), np.float32),
        "head_pos": np.zeros((n, 3), np.float32),
        "q": np.zeros((n, 14), np.float32),
        "qdot": np.zeros((n, 14), np.float32),
    }


def test_canonical_quat_yaw_invariance():
    """Same motion, different world yaw, psi anchored to EACH window's own
    yaw (as training does) -> identical canonical tokens (equivariance)."""
    a = canonical_sparse(_fake_frames(0.0), 0.0).numpy()
    for yaw in (90.0, -140.0, 177.0):
        b = canonical_sparse(_fake_frames(yaw), np.radians(yaw)).numpy()
        assert np.abs(a[:, 3:7] - b[:, 3:7]).max() < 1e-4, \
            f"canonical quat not yaw-equivariant at world yaw {yaw}"


def _rich_frames(seed=0):
    """Non-zero positions/velocities/contacts across all sparse fields."""
    rng = np.random.default_rng(seed)
    n = 12
    bq = sRot.from_rotvec(rng.normal(size=(n, 3)) * 0.3).as_quat()  # xyzw
    return {
        "base_pos": rng.normal(size=(n, 3)).astype(np.float32) * 0.3,
        "base_quat": np.roll(bq, 1, axis=-1).astype(np.float32),
        "base_linvel": rng.normal(size=(n, 3)).astype(np.float32) * 0.4,
        "base_angvel": rng.normal(size=(n, 3)).astype(np.float32) * 0.5,
        "head_rotvec": (rng.normal(size=(n, 3)) * 0.2).astype(np.float32),
        "head_angvel": rng.normal(size=(n, 3)).astype(np.float32) * 0.3,
        "foot_pos": rng.normal(size=(n, 2, 3)).astype(np.float32) * 0.2,
        "foot_yaw": rng.normal(size=(n, 2)).astype(np.float32) * 0.5,
        "contact": (rng.random((n, 2)) > 0.5).astype(np.float32),
        "head_pos": rng.normal(size=(n, 3)).astype(np.float32) * 0.1,
        "q": rng.normal(size=(n, 14)).astype(np.float32) * 0.1,
        "qdot": rng.normal(size=(n, 14)).astype(np.float32) * 0.2,
    }


def test_production_canonical_equivariance_all_fields():
    """GPT closure item: rotate the WORLD, psi anchored per-window — the
    PRODUCTION canonical_sparse() must return identical tokens (all 32 dims
    except none: contact invariant, everything else equivariant)."""
    f0 = _rich_frames(0)
    a = canonical_sparse(f0, 0.0).numpy()
    for dyaw in (45.0, -170.0):
        f = _rich_frames(0)
        c, sn = np.cos(np.radians(dyaw)), np.sin(np.radians(dyaw))
        Rz = np.array([[c, sn, 0], [-sn, c, 0], [0, 0, 1]], np.float32)
        g = dict(f)
        for k in ("base_pos", "base_linvel", "base_angvel", "head_pos", "head_angvel"):
            g[k] = f[k] @ Rz      # world rotation +delta (row-vec form)
        R_w = sRot.from_quat(np.roll(f["base_quat"], -1, axis=-1)).as_matrix()
        g["base_quat"] = np.roll(sRot.from_matrix(Rz.T[None] @ R_w).as_quat(), 1, -1).astype(np.float32)
        hr = sRot.from_rotvec(f["head_rotvec"])
        g["head_rotvec"] = sRot.from_matrix(Rz.T[None] @ hr.as_matrix()).as_rotvec().astype(np.float32)
        g["foot_pos"] = (f["foot_pos"].reshape(-1, 3) @ Rz).reshape(f["foot_pos"].shape)
        g["foot_yaw"] = f["foot_yaw"] + np.radians(dyaw)
        b = canonical_sparse(g, np.radians(dyaw)).numpy()
        # all dims must match: positions/vels rotated then canonicalized back,
        # quat de-yawed, yaw fields re-wrapped, contact invariant
        # (wrap-safe compare for yaw-like dims 25:27 via cos-sin? keep simple:
        #  strict compare — foot_yaw wrap can differ by 2π, allow that)
        diff = np.abs(a - b)
        diff[:, 25:27] = np.abs(np.mod(a[:, 25:27] - b[:, 25:27] + np.pi, 2*np.pi) - np.pi)
        worst = diff.max()
        assert worst < 1e-3, f"dims not equivariant: max {worst} at {np.unravel_index(diff.argmax(), diff.shape)}"


def test_production_export_rotation_inverse():
    """The FIXED export chain (Rz.T) must undo the production canonical
    rotation exactly, on non-trivial rotations."""
    from humanoidverse.planner.world_transform import canonical_Rz, rot_canonical_to_world, pos_canonical_to_world
    psi = np.radians(137.0)
    Rw = sRot.from_rotvec([0.1, -0.2, np.radians(137)]).as_matrix()
    Rc = canonical_Rz(psi) @ Rw                       # production canonicalization
    Rw_back = rot_canonical_to_world(Rc, psi)         # PRODUCTION inverse fn
    err = np.abs(Rw_back - Rw).max()
    assert err < 1e-4, f"production export inverse err {err}"
    # position chain through the production fn
    pos_w = np.array([[0.3, -0.2, 0.28]], np.float32)
    anchor = np.array([[1.0, 2.0, 0.0]], np.float32)
    pos_c = (pos_w - anchor) @ canonical_Rz(psi).T
    back = pos_canonical_to_world(pos_c, anchor, psi)
    assert np.abs(back - pos_w).max() < 1e-5


def test_command_source_routing():
    """v1.1 contract: intervention clips feed manifest commands; natural clips
    feed GT-derived; source flag visible per-sample."""
    import os
    from humanoidverse.planner.dataset import MotionData, P2Dataset, load_stats
    data_dir = os.environ.get("BDX_PLANNER_DATA")
    md = MotionData()
    if md.cmd_source is None:
        print("  (npz without cmd_source — routing inert, skip)")
        return
    stats = load_stats(os.path.join(data_dir, "p1_stats.json")) if data_dir else load_stats()
    import json
    sp = {k: torch.tensor(v) for k, v in json.loads(
        open(os.path.join(data_dir, "p2_stats32.json")).read()).items()} if data_dir else {}
    ds = P2Dataset(md, "train", stats=stats, sparse_stats=sp or None, max_windows=50)
    seen = {0: 0, 1: 0}
    for i in range(len(ds)):
        it = ds[i]
        seen[it["command_source"]] += 1
        if it["command_source"] == 1:
            # intervention: loco must equal manifest physical / scales
            ci = ds.windows[i][0]
            cv = md.cmd_override[ci]
            exp = torch.tensor([cv[4]/1.0, cv[5]/0.5, cv[6]/1.5], dtype=torch.float32)
            assert torch.allclose(it["loco"], exp, atol=1e-5), (
                f"intervention loco mismatch clip {ci}")
            assert torch.allclose(it["cmd"][0], torch.tensor(np.asarray(cv[:4])/1.7, dtype=torch.float32), atol=1e-4)
    assert seen[1] > 0, "no intervention samples in window sample"
    print(f"  sources seen: {seen}")



if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"✓ {fn.__name__}")
    print(f"\n{len(fns)} contract tests passed")


def test_yaw_unwrap_crossing():
    """GPT P1-2: 179 -> -179 must be +2deg continuous, not -358."""
    from humanoidverse.planner.angles import unwrap_yaw, angle_diff
    seq = np.radians([179.0, -179.0])
    unwrapped = np.degrees(unwrap_yaw(seq))
    step = unwrapped[1] - unwrapped[0]
    assert abs(step - 2.0) < 1e-6, f"179->-179 gave {step}deg, expected ~2"
    assert abs(angle_diff(np.radians(-179.0), np.radians(179.0)) - np.radians(2.0)) < 1e-6
    # reverse crossing
    seq2 = np.radians([-179.0, 179.0])
    step2 = np.degrees(unwrap_yaw(seq2))[1] - np.degrees(unwrap_yaw(seq2))[0]
    assert abs(step2 + 2.0) < 1e-6


