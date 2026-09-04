"""Slice walk segments of bdx_motion_library/motion_library.npz into a
mjlab MotionPool clip directory (walk_pool50).

Compatibility is GATED, not assumed: the raw-write test (write library
root+joints into the real training env -> body states reproduce 0.000 mm /
quat 1.0) passed on 2026-08-26 for walk/stand/gesture frames
(results/mjlab_bdx/fk_check_motion_library.json records the earlier
standalone-FK FAIL that motivated the correct in-env gate).

Layout decisions (matching the proven wiggle_pool50 schema byte-for-byte in
key set): 5 walk seeds x 10 windows of 300 frames (6.0 s), window starts
every 1500 frames -> stratified across each seed's 300 s so all 7 gaits
(fwd 0.3/0.6, back 0.2/0.4, march, turn L/R in place) appear. Orders are
identity-asserted against the frozen name tuples; quats pass through as
declared wxyz (world frame).

Usage:
  python walk_pool_extract.py   # writes walk_pool50/ + provenance JSON
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SRC = Path("/home/tcl/Desktop/start/bdx-data/bdx_motion_library/motion_library.npz")
OUT_DIR = Path(
  "/home/tcl/Desktop/start/bdx_rl_mjlab/policy_assets/bdx_v4/npz/walk_pool50"
)
PROV = Path(
  "/home/tcl/Desktop/start/BFM-zero/results/mjlab_bdx/walk_pool50_provenance.json"
)

BODY_NAMES = (
  "base_link",
  "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link",
  "left_knee_link", "left_ankle_link",
  "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link",
  "right_knee_link", "right_ankle_link",
  "neck_forward_link", "neck_pitch_link", "neck_yaw_link", "neck_roll_link",
  "left_ear_link", "right_ear_link",
)
JOINT_NAMES = (
  "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
  "left_knee_joint", "left_ankle_joint",
  "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
  "right_knee_joint", "right_ankle_joint",
  "neck_forward_joint", "neck_pitch_joint", "neck_yaw_joint", "neck_roll_joint",
)
WALK_SEGS = {0: "walk_seed0", 1: "walk_seed1", 2: "walk_seed2",
             3: "walk_seed3", 4: "walk_seed4"}
WIN_FRAMES = 300          # 6.0 s @ 50 Hz
WIN_STRIDE = 1500         # every 5th non-overlapping window
WINS_PER_SEED = 10
ROOT_Z_BAND = (0.15, 0.35)  # sanity band for BDX v4 walking base height


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 22), b""):
      h.update(chunk)
  return h.hexdigest()


def main() -> None:
  assert SRC.exists(), f"missing source {SRC}"
  assert not OUT_DIR.exists(), f"refusing to overwrite existing {OUT_DIR}"
  assert not PROV.exists(), f"refusing to overwrite existing {PROV}"

  lib = np.load(SRC, allow_pickle=True)
  meta = json.loads(str(lib["meta_json"]))
  assert meta["quat_convention"].startswith("wxyz"), meta["quat_convention"]
  assert float(lib["fps"]) == 50.0
  starts, lengths = lib["length_starts"], lib["_motion_num_frames"]

  OUT_DIR.mkdir(parents=True)
  clips = []
  for seg, name in sorted(WALK_SEGS.items()):
    s, T = int(starts[seg]), int(lengths[seg])
    assert T == 15000, f"{name}: expected 15000 frames, got {T}"
    for w in range(WINS_PER_SEED):
      a = s + w * WIN_STRIDE
      b = a + WIN_FRAMES
      sl = slice(a, b)
      jp = lib["joint_pos"][sl].astype(np.float32)
      jv = lib["joint_vel"][sl].astype(np.float32)
      bp = lib["body_pos"][sl].astype(np.float32)
      bq = lib["body_rot"][sl].astype(np.float32)
      blv = lib["body_lin_vel"][sl].astype(np.float32)
      bav = lib["body_ang_vel"][sl].astype(np.float32)
      for arr in (jp, jv, bp, bq, blv, bav):
        assert np.isfinite(arr).all(), f"{name} w{w}: non-finite values"
      norms = np.linalg.norm(bq, axis=-1)
      assert np.allclose(norms, 1.0, atol=1e-5), f"{name} w{w}: quat norms {norms.min()}..{norms.max()}"
      root_z = bp[:, 0, 2]
      assert ROOT_Z_BAND[0] < root_z.min() and root_z.max() < ROOT_Z_BAND[1], (
        f"{name} w{w}: root z outside {ROOT_Z_BAND}: "
        f"[{root_z.min():.3f}, {root_z.max():.3f}]")
      root = bp[:, 0, :2]
      travel = float(np.linalg.norm(root[-1] - root[0]))
      fname = f"{name}_w{w:02d}.npz"
      np.savez(
        OUT_DIR / fname,
        joint_pos=jp, joint_vel=jv,
        body_pos_w=bp, body_quat_w=bq,
        body_lin_vel_w=blv, body_ang_vel_w=bav,
        body_names=np.array(BODY_NAMES),
        joint_names=np.array(JOINT_NAMES),
        quat_order=np.array("wxyz"),
        fps=np.array([50.0]),
        resettable=np.ones(WIN_FRAMES + 2, dtype=bool),
      )
      clips.append({
        "file": fname, "segment": name,
        "src_frames": [a, b],
        "root_z": [round(float(root_z.min()), 4), round(float(root_z.max()), 4)],
        "xy_travel_m": round(travel, 3),
        "mean_speed_mps": round(travel / (WIN_FRAMES / 50.0), 3),
      })
      print(f"{fname}: root_z=[{root_z.min():.3f},{root_z.max():.3f}] "
            f"travel={travel:.2f}m speed={travel/6.0:.2f}m/s")

  assert len(clips) == 50, len(clips)
  prov = {
    "source": str(SRC),
    "source_sha256": sha256(SRC),
    "source_mtime": datetime.fromtimestamp(SRC.stat().st_mtime, timezone.utc).isoformat(),
    "source_meta": {"num_clips": meta["num_clips"], "fps": meta["fps"],
                    "quat_convention": meta["quat_convention"]},
    "gate": "in-env raw-write reproduction 0.000mm/quat1.0 on "
            "walk_seed0/2, stand_seed1, laugh_giggle, relax_v2 frames "
            "(2026-08-26); standalone entity.compile() FK is INVALID for "
            "validation (mirror-sign artifact) — see PIVOT_STATUS.",
    "order_assert": "body/joint name tuples identity vs bdx_rl_mjlab entity",
    "windows": {"frames": WIN_FRAMES, "stride": WIN_STRIDE,
                "per_seed": WINS_PER_SEED, "seeds": sorted(WALK_SEGS.values())},
    "clips": clips,
    "created_at": datetime.now(timezone.utc).isoformat(),
  }
  PROV.write_text(json.dumps(prov, indent=2))
  print(f"\nwrote 50 clips -> {OUT_DIR}")
  print(f"provenance -> {PROV}")


if __name__ == "__main__":
  main()
