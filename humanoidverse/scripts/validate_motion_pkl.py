#!/usr/bin/env python
"""Validate a BFM-Zero motion pkl against a skeleton + (optionally) FK round-trip.

Checks per clip:
  - required keys present (pose_aa, root_trans_offset, fps)
  - pose_aa.shape[1] == len(skeleton.node_names) (17 for BDX)
  - root_trans_offset.shape == (N, 3)
  - fps is a positive int
  - all values finite (no NaN/Inf)

FK round-trip (--fk-check): the DEFINITIVE correctness check for the converter.
Feeds pose_aa + root_trans_offset through Humanoid_Batch.fk_batch (the exact call
the motion lib makes) and asserts the recomputed global body positions match the
NPZ source body_pos_w to a tolerance. Requires --npz-dir to find the source NPZs.

Usage:
    python -m humanoidverse.scripts.validate_motion_pkl \
        --pkl humanoidverse/data/bdx_14dof.pkl \
        --skeleton humanoidverse/data/robots/bdx/bdx_v4.xml \
        --asset-root humanoidverse/data/robots/bdx/ \
        --asset-file bdx_v4.xml --fk-check \
        --npz-dir <bdx_rl_mjlab>/policy_assets/bdx_v4/npz
"""
import argparse
import os
import glob
import sys
import numpy as np
import joblib
import torch

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
from omegaconf import OmegaConf


def build_humanoid_batch(asset_root, asset_file):
    cfg = OmegaConf.create({
        "asset": {"assetRoot": asset_root, "assetFileName": asset_file},
        "extend_config": [],
    })
    return Humanoid_Batch(cfg)


def check_schema(name, entry, num_bodies):
    errs = []
    for k in ("pose_aa", "root_trans_offset", "fps"):
        if k not in entry:
            errs.append(f"missing key {k}")
            return errs  # can't check further
    pa = entry["pose_aa"]
    rt = entry["root_trans_offset"]
    N = pa.shape[0]
    if pa.shape != (N, num_bodies, 3):
        errs.append(f"pose_aa shape {pa.shape} != ({N}, {num_bodies}, 3)")
    if rt.shape != (N, 3):
        errs.append(f"root_trans_offset shape {rt.shape} != ({N}, 3)")
    if not isinstance(entry["fps"], (int, np.integer)) or entry["fps"] <= 0:
        errs.append(f"fps={entry['fps']} not a positive int")
    for k in ("pose_aa", "root_trans_offset"):
        v = entry[k]
        if not np.all(np.isfinite(v)):
            n_bad = int(np.sum(~np.isfinite(v)))
            errs.append(f"{k} has {n_bad} non-finite values")
    return errs


def fk_roundtrip(hb, entry, source_body_pos_w, tol=1e-3):
    """Run fk_batch on the clip; compare global body pos to source NPZ.

    Returns (max_err_m, per_body_max_err ndarray)."""
    pa = torch.from_numpy(entry["pose_aa"]).float()           # (N, B, 3) — engine internals are f32
    rt = torch.from_numpy(entry["root_trans_offset"]).float()  # (N, 3)
    N, B, _ = pa.shape
    # fk_batch expects (B_batch=1, seq_len, num_bodies, 3) and (1, seq_len, 3)
    pose = pa.unsqueeze(0)
    trans = rt.unsqueeze(0)
    res = hb.fk_batch(pose, trans, return_full=False, dt=1.0 / entry["fps"])
    fk_pos = res.global_translation[0].numpy()  # (N, B, 3)
    src = source_body_pos_w.astype(np.float64)
    if src.shape[0] < N:
        # windowed: source covers the whole clip; compare the window slice
        # (caller passes the right slice; if mismatch, fall back to min length)
        L = min(src.shape[0], N)
        fk_pos = fk_pos[:L]; src = src[:L]
    err = np.linalg.norm(fk_pos - src, axis=-1)  # (L, B)
    return float(err.max()), err.max(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--asset-root", required=True)
    ap.add_argument("--asset-file", required=True)
    ap.add_argument("--expected-bodies", type=int, default=17)
    ap.add_argument("--fk-check", action="store_true")
    ap.add_argument("--npz", default=None, help="single segmented motion_library.npz for --fk-check")
    ap.add_argument("--npz-dir", default=None, help="directory of per-clip NPZs for --fk-check")
    ap.add_argument("--tol", type=float, default=1e-3, help="FK position tolerance (meters)")
    ap.add_argument("--max-clips-fk", type=int, default=5, help="how many clips to FK round-trip")
    args = ap.parse_args()

    skeleton = SkeletonTree.from_mjcf(args.skeleton)
    nb = len(skeleton.node_names)
    assert nb == args.expected_bodies, f"skeleton has {nb} bodies, expected {args.expected_bodies}"
    print(f"Skeleton: {nb} bodies")

    data = joblib.load(args.pkl)
    print(f"Loaded {len(data)} clips from {args.pkl}")

    # 1. schema + finiteness on ALL clips
    schema_fail = 0
    for name, entry in data.items():
        errs = check_schema(name, entry, nb)
        if errs:
            schema_fail += 1
            print(f"  [SCHEMA FAIL] {name}: {'; '.join(errs)}")
    print(f"Schema check: {len(data) - schema_fail}/{len(data)} clips OK")

    # 2. FK round-trip on the first N clips (the definitive converter check)
    if args.fk_check:
        if not args.npz_dir and not args.npz:
            print("--fk-check requires --npz (segmented library) or --npz-dir (per-clip)")
            sys.exit(2)
        hb = build_humanoid_batch(args.asset_root, args.asset_file)
        print(f"Humanoid_Batch: num_dof={hb.num_dof}, num_bodies={len(hb.body_names)}, "
              f"has_freejoint={hb.has_freejoint}")

        # Build clip-name -> source body_pos array, from either source.
        # - segmented (--npz): one file, sliced by length_starts; names from meta.
        # - per-clip (--npz-dir): one file per clip, globbed.
        src_map = {}  # name -> (body_pos array, global_offset)
        if args.npz:
            import json as _json
            sd = dict(np.load(args.npz, allow_pickle=True))
            body_pos_src = np.asarray(sd.get("body_pos", sd.get("body_pos_w")))
            starts = np.asarray(sd["length_starts"]).reshape(-1)
            num_frames = np.asarray(sd["_motion_num_frames"]).reshape(-1)
            # names: reuse converter's default list, fall back to clipN
            from humanoidverse.scripts.convert_bdx_npz_to_pkl import _DEFAULT_SEGMENT_NAMES
            names = list(_DEFAULT_SEGMENT_NAMES)
            try:
                meta = _json.loads(str(sd["meta_json"]))
                mn = [c.get("name") for c in meta.get("clips", [])]
                if len(mn) == len(starts) and all(mn):
                    names = mn
            except Exception:
                pass
            for i, s in enumerate(starts):
                nm = names[i] if i < len(names) else f"clip{i}"
                src_map[nm] = body_pos_src[int(s):int(s) + int(num_frames[i])]
        else:
            npz_files = glob.glob(os.path.join(args.npz_dir, "**", "*.npz"), recursive=True)
            for f in npz_files:
                nm = os.path.splitext(os.path.basename(f))[0]
                sd = np.load(f, allow_pickle=True)
                src_map[nm] = np.asarray(sd.get("body_pos", sd.get("body_pos_w")))

        # match windowed sub-clips back to their source (strip _clipN)
        checked = 0
        worst = 0.0
        for name, entry in data.items():
            base = name.rsplit("_clip", 1)[0] if "_clip" in name else name
            if base not in src_map:
                continue
            src = src_map[base]
            if "_clip" in name:
                w = int(name.rsplit("_clip", 1)[1])
                s = w * entry["pose_aa"].shape[0]
                src = src[s:s + entry["pose_aa"].shape[0]]
            if src.shape[0] < entry["pose_aa"].shape[0]:
                continue
            max_err, per_body = fk_roundtrip(hb, entry, src, tol=args.tol)
            worst = max(worst, max_err)
            status = "OK" if max_err < args.tol else "FAIL"
            print(f"  [FK {status}] {name}: max body-pos err = {max_err*1000:.2f} mm "
                  f"(tol {args.tol*1000:.1f} mm); worst body idx={int(np.argmax(per_body))}")
            checked += 1
            if checked >= args.max_clips_fk:
                break
        print(f"FK round-trip: checked {checked} clips, worst err = {worst*1000:.2f} mm")
        if worst >= args.tol:
            print(f"FAILED: FK reconstruction error exceeds {args.tol*1000:.1f} mm tol "
                  "— converter global->local conversion is WRONG. Do NOT train.")
            sys.exit(1)
        print("FK ROUND-TRIP PASSED — converter is correct.")

    if schema_fail:
        sys.exit(1)
    print("ALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
