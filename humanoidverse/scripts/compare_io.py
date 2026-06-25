"""[BFM-DIAG-NAN] Compare two sim-IO npz dumps (one per backend), aligned by
physics substep index, and localize the first divergence / any NaN.

Usage:
    uv run python -m humanoidverse.scripts.compare_io --a /tmp/io_mjw.npz --b /tmp/io_isaac.npz

Convention notes baked into the comparison:
- dof_pos / dof_vel: JOINT space, same ordering on both backends (preserve_order
  via config dof_names). Compared directly — the primary "money" quantities.
- base_pos / base_quat / base_lin_vel: world frame on both. Compared directly.
- base_ang_vel: mjw stores BODY frame, isaacsim WORLD frame. Rather than depend on
  the quat_rotate convention (itself under test), compare the frame-INVARIANT
  magnitude (|ang vel| = rotation rate, identical in any frame) as primary, and
  ALSO a quat-converted body-frame comparison as secondary (reported, not gated).
- contact_forces: world-frame net force on both; compare per-body magnitude
  (PhysX vs MuJoCo contact models differ, so expect noise — flag only large/NaN).
"""

import argparse
import os

import numpy as np


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("nan")
    both_finite = np.isfinite(a) & np.isfinite(b)
    if not both_finite.any():
        return float("nan")
    return float(np.max(np.abs(a[both_finite] - b[both_finite])))


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest rotation angle [deg] between two xyzw quats, batched [S,N,4].
    Robust to sign ambiguity (q and -q same rotation)."""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    dot = np.abs(np.sum(a * b, axis=-1))  # [S,N]
    dot = np.clip(dot, 0.0, 1.0)
    ang = 2.0 * np.arccos(dot)
    both_finite = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    if not both_finite.any():
        return float("nan")
    return float(np.degrees(np.max(ang[both_finite])))


def _first_diverging(diff_per_step: np.ndarray, tol: float) -> int:
    """diff_per_step: [S] max-diff per substep. Return first index > tol, or -1."""
    over = np.where(diff_per_step > tol)[0]
    return int(over[0]) if len(over) else -1


def _any_nan(arr: np.ndarray) -> bool:
    return bool(np.any(~np.isfinite(arr)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="first npz (e.g. mujoco_warp)")
    p.add_argument("--b", required=True, help="second npz (e.g. isaacsim)")
    p.add_argument("--tol", type=float, default=1e-3, help="divergence tolerance (abs units)")
    args = p.parse_args()

    A = dict(np.load(args.a, allow_pickle=True))
    B = dict(np.load(args.b, allow_pickle=True))
    name_a, name_b = os.path.basename(args.a), os.path.basename(args.b)
    print(f"Comparing  A={name_a}  vs  B={name_b}  (tol={args.tol})\n")

    meta_a = A["_meta"]; meta_b = B["_meta"]
    S = int(min(A["dof_pos"].shape[0], B["dof_pos"].shape[0]))
    print(f"substeps: A={A['dof_pos'].shape[0]}  B={B['dof_pos'].shape[0]}  -> compare S={S}")
    print(f"meta A: num_envs={meta_a[0]} dof={meta_a[1]} decim={meta_a[2]} dt={meta_a[4]:.5f}")
    print(f"meta B: num_envs={meta_b[0]} dof={meta_b[1]} decim={meta_b[2]} dt={meta_b[4]:.5f}\n")

    # ---- per-quantity step-0 + first-divergence table ----
    def per_step_diff(a, b):
        a = a[:S].astype(np.float64)
        b = b[:S].astype(np.float64)
        if a.shape != b.shape:
            return np.full(S, np.nan)  # structurally incompatible (e.g. body count)
        # max over envs + dof/vec for each substep -> [S]
        # NaNs excluded pairwise
        d = np.full(S, np.nan)
        for s in range(S):
            a_s, b_s = a[s], b[s]
            both = np.isfinite(a_s) & np.isfinite(b_s)
            d[s] = np.max(np.abs(a_s[both] - b_s[both])) if both.any() else np.nan
        return d

    rows = []

    for key, label, kind in [
        ("dof_pos", "dof_pos", "direct"),
        ("dof_vel", "dof_vel", "direct"),
        ("base_pos", "base_pos [m]", "direct"),
        ("base_lin_vel", "base_lin_vel [m/s]", "direct"),
        ("base_ang_vel", "base_ang_vel |.| [rad/s]", "magnitude"),
    ]:
        a = A[key][:S]; b = B[key][:S]
        if kind == "magnitude":
            a = np.linalg.norm(a, axis=-1)
            b = np.linalg.norm(b, axis=-1)
        d = per_step_diff(a, b)
        rows.append((label, d[0], _first_diverging(d, args.tol), _any_nan(a), _any_nan(b)))

    # quaternion (angle metric)
    qa = A["base_quat_xyzw"][:S]; qb = B["base_quat_xyzw"][:S]
    ang = np.array([_quat_angle_deg(qa[s], qb[s]) for s in range(S)])
    rows.append(("base_quat [deg]", ang[0], _first_diverging(ang, args.tol), _any_nan(qa), _any_nan(qb)))

    # contact forces magnitude per body (structurally comparable only if body counts match)
    ca = np.linalg.norm(A["contact_forces"][:S], axis=-1)
    cb = np.linalg.norm(B["contact_forces"][:S], axis=-1)
    cf_shape_ok = ca.shape == cb.shape
    d = per_step_diff(ca, cb)
    rows.append(("contact_forces |F| [N]", d[0], _first_diverging(d, args.tol), _any_nan(ca), _any_nan(cb)))
    if not cf_shape_ok:
        print(f"  [NOTE] contact_forces body count differs: A={ca.shape} B={cb.shape} — "
              f"comparing on the common shape is unsafe; this row is informational only.\n")

    # torques (only directly comparable for the explicit-torque path; isaacsim stores
    # the would-be torque too, so still informative)
    ta = A["torques"][:S]; tb = B["torques"][:S]
    d = per_step_diff(ta, tb)
    rows.append(("torques [N*m] (ref-only)", d[0], _first_diverging(d, args.tol), _any_nan(ta), _any_nan(tb)))

    print(f"{'quantity':28s} {'substep0':>12s} {'1st-div@tol':>12s} {'NaN(A)':>7s} {'NaN(B)':>7s}")
    print("-" * 70)
    for label, s0, first, na, nb in rows:
        s0_s = f"{s0:.3e}" if np.isfinite(s0) else "nan"
        first_s = f"{first}" if first >= 0 else "none"
        print(f"{label:28s} {s0_s:>12s} {first_s:>12s} {str(na):>7s} {str(nb):>7s}")
    print()

    # ---- substep-0 dof_pos side-by-side (env 0) — eyeball joint ordering ----
    print("substep-0 dof_pos (env 0):")
    dp_a = A["dof_pos"][0, 0]
    dp_b = B["dof_pos"][0, 0]
    if dp_a.shape == dp_b.shape:
        for i, (xa, xb) in enumerate(zip(dp_a, dp_b)):
            flag = "  <-- DIFF" if abs(float(xa) - float(xb)) > args.tol else ""
            print(f"  dof[{i:2d}]  A={xa:+.4f}  B={xb:+.4f}{flag}")
    else:
        print(f"  SHAPE MISMATCH: A={dp_a.shape} B={dp_b.shape}")

    # ---- verdict ----
    print("\n=== VERDICT ===")
    # substep-0 dof/pos/quat diffs beyond noise => read/write convention bug
    s0_keys = ["dof_pos", "base_pos", "base_quat [deg]"]
    bad_s0 = []
    for label, s0, first, na, nb in rows:
        if label in s0_keys and np.isfinite(s0) and s0 > args.tol:
            bad_s0.append((label, s0))
    if bad_s0:
        print("STEP-0 (identical start state) MISMATCH -> convention/read-write bug:")
        for label, s0 in bad_s0:
            print(f"   {label}: diff={s0:.3e}")
    else:
        print("Step-0 agrees (within tol) -> no obvious read/write convention bug.")
    nan_rows = [(l, na, nb) for (l, _, _, na, nb) in rows if na or nb]
    if nan_rows:
        print("NaN detected in at least one backend:")
        for l, na, nb in nan_rows:
            print(f"   {l}: NaN(A)={na} NaN(B)={nb}")


if __name__ == "__main__":
    main()