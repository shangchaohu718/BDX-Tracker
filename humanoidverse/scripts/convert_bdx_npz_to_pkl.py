#!/usr/bin/env python
"""Convert BDX motion data into BFM-Zero pkl format.

Two input modes:
  --npz            a single segmented motion_library.npz (18 clips, 162000 fr)
                   [NEW: field names body_pos/body_rot, split by length_starts]
  --npz-dir        a directory of per-clip NPZs (legacy: body_pos_w/body_quat_w)

BFM-Zero's MotionLibBase.load_motion_with_skeleton consumes:
  - pose_aa          (N, num_bodies, 3)  LOCAL axis-angle rotations (parent-relative)
  - root_trans_offset (N, 3)             root world translation (absolute)
  - fps              int

Both NPZ layouts store GLOBAL body_pos / body_rot (wxyz). This converter inverts
the kinematics: computes each body's local rotation relative to its parent from
the global quaternions, using the SAME BDX skeleton tree (SkeletonTree.from_mjcf)
that BFM-Zero will load.

Verified: NPZ body axis order == MJCF body order == SkeletonTree node_names, so
NO reordering is needed. Quaternion convention is (w,x,y,z) in both layouts
(FK round-trip gate: 0.0002mm on motion_library.npz, 0.00mm on legacy per-clip).

Usage (segmented library — preferred):
    python -m humanoidverse.scripts.convert_bdx_npz_to_pkl \
        --npz humanoidverse/data/bdx\\ motions/motion_library.npz \
        --skeleton humanoidverse/data/robots/bdx/bdx_v4.xml \
        --out-dir humanoidverse/data [--clip-seconds 10]

Usage (legacy per-clip dir):
    python -m humanoidverse.scripts.convert_bdx_npz_to_pkl \
        --npz-dir <bdx_rl_mjlab>/policy_assets/bdx_v4/npz \
        --skeleton humanoidverse/data/robots/bdx/bdx_v4.xml \
        --out-dir humanoidverse/data [--clip-seconds 2]
"""
import argparse
import os
import glob
import json
import numpy as np
import joblib
from scipy.spatial.transform import Rotation

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree

# Names for the 18 segments in motion_library.npz (order matches _motion_num_frames).
# Falls back to f"clip{i}" if meta_json names differ / are absent.
_DEFAULT_SEGMENT_NAMES = [
    "walk_seed0", "walk_seed1", "walk_seed2", "walk_seed3", "walk_seed4",
    "stand_seed0", "stand_seed1", "stand_seed2", "stand_seed3", "stand_seed4",
    "angry_no", "angry_yes", "attention", "bored_snore",
    "laugh_giggle", "relax", "relax_v2", "shy_yes",
]


def convert_arrays(body_pos_w, body_quat_w, joint_pos, skeleton):
    """Shared global→local conversion. Takes (N,B,3)/(N,B,4)/(N,14) arrays,
    returns a BFM-Zero clip dict (without fps; caller sets it).

    body_quat_w is (w,x,y,z) — MuJoCo native. FK round-trip is the gate.
    """
    N, B, _ = body_pos_w.shape
    assert B == len(skeleton.node_names), (
        f"NPZ has {B} bodies, skeleton has {len(skeleton.node_names)}"
    )
    parents = [int(p) for p in skeleton.parent_indices]  # -1 for root

    # Global quaternions: NPZ is (w,x,y,z); scipy wants (x,y,z,w), so roll.
    quat_xyzw = np.roll(body_quat_w, shift=-1, axis=-1)        # (N,17,4) wxyz->xyzw
    R_global = Rotation.from_quat(quat_xyzw.reshape(-1, 4))     # flat (N*B,)

    # LOCAL (parent-relative) rotations in flat (N*B,) space.
    # local_i = inv(R_parent_i) * R_global_i ; root local = its global.
    local_quats_xyzw = np.zeros((N * B, 4), dtype=np.float64)
    for i in range(B):
        p = parents[i]
        col_i = np.arange(N) * B + i
        if p == -1:
            R_local_i = R_global[col_i]
        else:
            col_p = np.arange(N) * B + p
            R_local_i = R_global[col_p].inv() * R_global[col_i]
        local_quats_xyzw[col_i] = R_local_i.as_quat()          # xyzw

    # pose_aa = axis-angle (rotvec) of the LOCAL rotations
    R_local_all = Rotation.from_quat(local_quats_xyzw)
    pose_aa = R_local_all.as_rotvec().reshape(N, B, 3).astype(np.float32)  # (N, B, 3)

    # root_trans_offset = absolute world translation of the base (body 0)
    root_trans_offset = body_pos_w[:, 0, :].astype(np.float32)             # (N, 3)
    # root_rot = base global rotation, wxyz (informational; redundant with pose_aa[:,0])
    root_rot = body_quat_w[:, 0, :].astype(np.float32)

    return {
        "pose_aa": pose_aa,                       # (N, 17, 3) — CONSUMED
        "root_trans_offset": root_trans_offset,   # (N, 3)     — CONSUMED
        "dof": joint_pos.astype(np.float32),      # (N, 14)    — informational
        "root_rot": root_rot,                     # (N, 4) wxyz — informational
        "smpl_joints": np.zeros((N, 24, 3), dtype=np.float32),  # vestigial schema parity
    }


def _new_field(d, *candidates):
    """Read a field that may be renamed across NPZ versions."""
    for c in candidates:
        if c in d:
            return d[c]
    raise KeyError(f"none of {candidates} present in NPZ; keys={list(d.keys())}")


def npz_to_clip_entry(npz_path, skeleton, fps_override=None):
    """Convert one per-clip NPZ file (legacy layout) into a BFM-Zero clip dict."""
    d = np.load(npz_path, allow_pickle=True)
    body_pos_w = np.asarray(_new_field(d, "body_pos_w", "body_pos"), dtype=np.float32)
    body_quat_w = np.asarray(_new_field(d, "body_quat_w", "body_rot"), dtype=np.float32)
    joint_pos = np.asarray(d["joint_pos"], dtype=np.float32)
    entry = convert_arrays(body_pos_w, body_quat_w, joint_pos, skeleton)
    entry["fps"] = fps_override if fps_override is not None else int(round(float(d["fps"])))
    return entry


def segmented_npz_to_clips(npz_path, skeleton, fps_override=None):
    """Split one segmented motion_library.npz into per-clip BFM-Zero entries.

    Returns list of (name, entry) in segment order. Uses length_starts /
    _motion_num_frames for splitting; names come from meta_json if present,
    else from _DEFAULT_SEGMENT_NAMES.
    """
    d = dict(np.load(npz_path, allow_pickle=True))
    body_pos = np.asarray(_new_field(d, "body_pos", "body_pos_w"), dtype=np.float32)
    body_rot = np.asarray(_new_field(d, "body_rot", "body_quat_w"), dtype=np.float32)
    joint_pos = np.asarray(d["joint_pos"], dtype=np.float32)
    fps = fps_override if fps_override is not None else int(round(float(d["fps"])))

    num_frames = np.asarray(d["_motion_num_frames"]).reshape(-1)
    starts = np.asarray(d["length_starts"]).reshape(-1)
    n_clips = len(num_frames)
    assert len(starts) == n_clips
    assert starts[-1] + num_frames[-1] == body_pos.shape[0], (
        "segment indices don't cover the full array"
    )

    # segment names from meta_json if available
    names = list(_DEFAULT_SEGMENT_NAMES)
    try:
        meta = json.loads(str(d["meta_json"]))
        meta_names = [c.get("name") for c in meta.get("clips", [])]
        if len(meta_names) == n_clips and all(meta_names):
            names = meta_names
    except Exception:
        pass
    # pad/truncate name list to match clip count
    if len(names) < n_clips:
        names += [f"clip{i}" for i in range(len(names), n_clips)]
    elif len(names) > n_clips:
        names = names[:n_clips]

    out = []
    for i in range(n_clips):
        s = int(starts[i]); e = s + int(num_frames[i])
        entry = convert_arrays(body_pos[s:e], body_rot[s:e], joint_pos[s:e], skeleton)
        entry["fps"] = fps
        out.append((names[i], entry))
    return out


def clip_windows(entry, clip_frames, name):
    """Split one clip entry into non-overlapping fixed-length windows.

    Returns list of (subname, sub-entry). Trailing < clip_frames dropped.
    """
    N = entry["pose_aa"].shape[0]
    if N < clip_frames:
        return []  # too short to yield a full window
    n_windows = N // clip_frames
    out = []
    for w in range(n_windows):
        s, e = w * clip_frames, (w + 1) * clip_frames
        sub = {}
        for k, v in entry.items():
            sub[k] = v if k == "fps" else v[s:e]
        out.append((f"{name}_clip{w}", sub))
    return out


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", help="single segmented motion_library.npz (18 clips)")
    src.add_argument("--npz-dir", help="directory of per-clip NPZs (legacy)")
    ap.add_argument("--skeleton", required=True, help="path to bdx_v4.xml")
    ap.add_argument("--out-dir", default="humanoidverse/data")
    ap.add_argument("--clip-seconds", type=float, default=10.0,
                    help="window length for the _clipped training pkl (seconds)")
    ap.add_argument("--fps", type=int, default=None, help="override fps (default: read from NPZ)")
    args = ap.parse_args()

    skeleton = SkeletonTree.from_mjcf(args.skeleton)
    print(f"Skeleton: {len(skeleton.node_names)} bodies from {args.skeleton}")

    # ---- collect whole clips ----
    whole = {}  # name -> entry
    if args.npz:
        print(f"Segmented library: {args.npz}")
        for name, entry in segmented_npz_to_clips(args.npz, skeleton, fps_override=args.fps):
            whole[name] = entry
        base_fps = args.fps if args.fps else int(round(float(
            dict(np.load(args.npz, allow_pickle=True))["fps"])))
    else:
        npz_files = sorted(glob.glob(os.path.join(args.npz_dir, "**", "*.npz"), recursive=True))
        print(f"Found {len(npz_files)} per-clip NPZ files")
        assert len(npz_files) > 0, f"no NPZ under {args.npz_dir}"
        for f in npz_files:
            name = os.path.splitext(os.path.basename(f))[0]
            whole[name] = npz_to_clip_entry(f, skeleton, fps_override=args.fps)
        base_fps = args.fps if args.fps else int(round(float(
            np.load(npz_files[0], allow_pickle=True)["fps"])))

    clip_frames = max(1, int(round(args.clip_seconds * base_fps)))
    print(f"Window: {clip_frames} frames @ {base_fps} fps = {clip_frames/base_fps:.2f}s")

    # ---- windowed training clips ----
    windowed = {}
    for name, entry in whole.items():
        for sub_name, sub_entry in clip_windows(entry, clip_frames, name):
            windowed[sub_name] = sub_entry

    os.makedirs(args.out_dir, exist_ok=True)
    eval_path = os.path.join(args.out_dir, "bdx_14dof.pkl")
    train_path = os.path.join(args.out_dir, "bdx_14dof_clipped.pkl")
    joblib.dump(whole, eval_path, compress=3)
    joblib.dump(windowed, train_path, compress=3)

    tot_frames_whole = sum(v["pose_aa"].shape[0] for v in whole.values())
    tot_frames_clip = sum(v["pose_aa"].shape[0] for v in windowed.values())
    # per-kind summary
    def kind_of(nm):
        if nm.startswith("walk"): return "walk"
        if nm.startswith("stand"): return "stand"
        return "gesture"
    from collections import Counter
    wk = Counter(kind_of(n) for n in windowed)
    print(f"\nWrote {eval_path}: {len(whole)} clips, {tot_frames_whole} frames "
          f"({tot_frames_whole/base_fps/60:.1f} min)")
    print(f"Wrote {train_path}: {len(windowed)} windows, {tot_frames_clip} frames "
          f"({tot_frames_clip/base_fps/60:.1f} min)")
    print(f"  window mix: walk={wk['walk']} stand={wk['stand']} gesture={wk['gesture']}")
    print("\nNOTE: run validate_motion_pkl.py --fk-check before training.")


if __name__ == "__main__":
    main()
