"""fixed_dance_reference_specialist — build a CANONICAL-sequence variant pkl.

The _cf trainer takes a single-clip pkl whose frame order IS the canonical
performance sequence (phase 0 = sequence start). Variants (ruling, from
fds_ss4_phase_scan.py section 5):
  A  original clip as recorded (0->291)            — the original pkl itself
  B  rotated 49->291->0->48 with a C1 Hermite blend (+-5 frames) at the
     291->0 junction — blend applied to DOF ONLY; root_trans_offset /
     root_rot / pose_aa are NOT blended (junction root jump uncorrected;
     B is expected to lose the comparison — if data ever favors B, root
     blending must be designed before use)
  C  anchor49 suffix [49:292] (243 frames, phase0==anchor, no seam, no blend)

All four per-frame arrays (dof, pose_aa, root_trans_offset, root_rot) are
reordered consistently; fps and clip metadata are preserved. Clip name gets a
_variant suffix so motion-lib keys / report provenance cannot collide with the
original.

Usage:
  .venv/bin/python humanoidverse/scripts/fds_build_canonical_variant.py \
      --variant C --pkl humanoidverse/data/fds_re_dancegen_side_step_4.pkl \
      --out humanoidverse/data/fds_ss4_canonical_C.pkl
"""

from __future__ import annotations

import argparse
import copy

import joblib
import numpy as np

ANCHOR = 49
PER_FRAME_KEYS = ("dof", "pose_aa", "root_trans_offset", "root_rot")
BLEND_W = 5  # frames each side of the junction (must match the phase scan's B)


def smooth_hermite_np(q_lo, d_lo, q_hi, d_hi, n):
    """n C1 cubic-Hermite interior samples between (q_lo,d_lo) and (q_hi,d_hi);
    numpy port of fds_ss4_phase_scan.smooth_hermite (d_* are per-frame slopes)."""
    s = np.linspace(0.0, 1.0, n + 2)[1:-1, None]
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return h00 * q_lo[None] + h10 * d_lo[None] + h01 * q_hi[None] + h11 * d_hi[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["A", "B", "C"], required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = joblib.load(args.pkl)
    assert len(data) == 1, "single-clip pkl expected"
    name, entry = next(iter(data.items()))
    T = int(np.asarray(entry["dof"]).shape[0])
    out_name = f"{name}_var{args.variant}"

    if args.variant == "A":
        order = list(range(T))
    elif args.variant == "C":
        order = list(range(ANCHOR, T))
    else:  # B
        order = list(range(ANCHOR, T)) + list(range(0, ANCHOR))

    new_entry = {}
    for k, v in entry.items():
        if k in PER_FRAME_KEYS:
            arr = np.asarray(v)
            assert arr.shape[0] == T, f"{k}: {arr.shape[0]} != {T}"
            new_entry[k] = arr[order].copy()
        else:
            new_entry[k] = copy.deepcopy(v)

    if args.variant == "B":
        # C1 Hermite blend on DOF ONLY at the junction (frames J-W : J+W of the
        # rotated sequence), anchored on the ORIGINAL frames just outside the
        # window — mirrors fds_ss4_phase_scan.build_variant("B") exactly.
        q = np.asarray(entry["dof"], dtype=np.float64)
        dq_cent = np.zeros_like(q)
        dq_cent[1:-1] = (q[2:] - q[:-2]) / 2.0
        dq_cent[0], dq_cent[-1] = dq_cent[1], dq_cent[-2]
        J = T - ANCHOR  # junction: rotated[J-1] = orig[T-1], rotated[J] = orig[0]
        W = BLEND_W
        lo_orig = ANCHOR + (J - W - 1)
        hi_orig = (J + W) - (T - ANCHOR)
        span = (J + W) - (J - W - 1)
        blended = smooth_hermite_np(
            np.asarray(new_entry["dof"][J - W - 1], dtype=np.float64),
            dq_cent[lo_orig] * span,
            np.asarray(new_entry["dof"][J + W], dtype=np.float64),
            dq_cent[hi_orig] * span,
            2 * W)
        new_entry["dof"][J - W:J + W] = blended.astype(new_entry["dof"].dtype)

    joblib.dump({out_name: new_entry}, args.out)
    n = len(order)
    print(f"[variant {args.variant}] {name} -> {out_name}: {T} -> {n} frames "
          f"({n / float(entry['fps']):.2f}s), start_orig_frame={order[0]}, "
          f"written {args.out}")


if __name__ == "__main__":
    main()
