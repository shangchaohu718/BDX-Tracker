"""Slice stand + gesture segments of bdx_motion_library/motion_library.npz
into mjlab MotionPool clip directories (stand_pool10, gesture_pool24) for
the Mixed-134 all-family pool.

Same gates and schema as the proven walk_pool_extract.py (2026-08-26):
compatibility is GATED by the in-env raw-write test (which passed on
stand_seed1 / laugh_giggle / relax_v2 frames — recorded in
walk_pool50_provenance.json), orders identity-asserted vs the frozen name
tuples, quats pass through as declared wxyz world.

Windowing:
  stand:   5 seeds x 2 windows x 300 fr (6 s), stride 6000 -> static-posture
           family is low-diversity; 2 windows/seed suffice.
  gesture: 8 gestures x 3 windows x 200 fr (4 s), stride 500 -> each 30 s
           recording loops its gesture 3-40x, so 4 s windows always contain
           full expression cycles except relax_v2 (9.2 s cycle) whose
           windows are partial-cycle — acceptable: tracking task does not
           require complete cycles, and provenance records cycle lengths.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SRC = Path("/home/tcl/Desktop/start/bdx-data/bdx_motion_library/motion_library.npz")
OUT_BASE = Path("/home/tcl/Desktop/start/bdx_rl_mjlab/policy_assets/bdx_v4/npz")
PROV = Path(
  "/home/tcl/Desktop/start/BFM-zero/results/mjlab_bdx/stand_gesture_provenance.json"
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
STAND_SEGS = {5: "stand_seed0", 6: "stand_seed1", 7: "stand_seed2",
              8: "stand_seed3", 9: "stand_seed4"}
GESTURE_SEGS = {10: "angry_no", 11: "angry_yes", 12: "attention",
                13: "bored_snore", 14: "laugh_giggle", 15: "relax",
                16: "relax_v2", 17: "shy_yes"}
STAND_WIN, STAND_STRIDE, STAND_PER = 300, 6000, 2
GEST_WIN, GEST_STRIDE, GEST_PER = 200, 500, 3
ROOT_Z_BAND = (0.15, 0.35)


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 22), b""):
      h.update(chunk)
  return h.hexdigest()


def write_window(lib, a: int, b: int, out: Path, name: str) -> dict:
  sl = slice(a, b)
  jp = lib["joint_pos"][sl].astype(np.float32)
  jv = lib["joint_vel"][sl].astype(np.float32)
  bp = lib["body_pos"][sl].astype(np.float32)
  bq = lib["body_rot"][sl].astype(np.float32)
  blv = lib["body_lin_vel"][sl].astype(np.float32)
  bav = lib["body_ang_vel"][sl].astype(np.float32)
  n = b - a
  for arr in (jp, jv, bp, bq, blv, bav):
    assert np.isfinite(arr).all(), f"{name}: non-finite values"
  norms = np.linalg.norm(bq, axis=-1)
  assert np.allclose(norms, 1.0, atol=1e-5), f"{name}: quat norms"
  root_z = bp[:, 0, 2]
  assert ROOT_Z_BAND[0] < root_z.min() and root_z.max() < ROOT_Z_BAND[1], (
    f"{name}: root z [{root_z.min():.3f},{root_z.max():.3f}] outside {ROOT_Z_BAND}")
  root = bp[:, 0, :2]
  travel = float(np.linalg.norm(root[-1] - root[0]))
  out.parent.mkdir(parents=True, exist_ok=True)
  assert not out.exists(), f"refusing to overwrite {out}"
  np.savez(
    out,
    joint_pos=jp, joint_vel=jv,
    body_pos_w=bp, body_quat_w=bq,
    body_lin_vel_w=blv, body_ang_vel_w=bav,
    body_names=np.array(BODY_NAMES),
    joint_names=np.array(JOINT_NAMES),
    quat_order=np.array("wxyz"),
    fps=np.array([50.0]),
    resettable=np.ones(n + 2, dtype=bool),
  )
  return {"file": out.name, "src_frames": [a, b],
          "root_z": [round(float(root_z.min()), 4), round(float(root_z.max()), 4)],
          "xy_travel_m": round(travel, 3)}


def main() -> None:
  assert SRC.exists(), f"missing {SRC}"
  assert not PROV.exists(), f"refusing to overwrite {PROV}"
  lib = np.load(SRC, allow_pickle=True)
  meta = json.loads(str(lib["meta_json"]))
  assert meta["quat_convention"].startswith("wxyz")
  assert float(lib["fps"]) == 50.0
  starts, lengths = lib["length_starts"], lib["_motion_num_frames"]

  clips = []
  for seg, name in sorted(STAND_SEGS.items()):
    s, T = int(starts[seg]), int(lengths[seg])
    assert T == 15000, f"{name}: expected 15000, got {T}"
    for w in range(STAND_PER):
      a = s + w * STAND_STRIDE
      rec = write_window(lib, a, a + STAND_WIN,
                         OUT_BASE / "stand_pool10" / f"{name}_w{w:02d}.npz",
                         f"{name}_w{w}")
      rec["segment"] = name
      clips.append(rec)
  for seg, name in sorted(GESTURE_SEGS.items()):
    s, T = int(starts[seg]), int(lengths[seg])
    assert T == 1500, f"{name}: expected 1500, got {T}"
    for w in range(GEST_PER):
      a = s + w * GEST_STRIDE
      rec = write_window(lib, a, a + GEST_WIN,
                         OUT_BASE / "gesture_pool24" / f"gesture_{name}_w{w:02d}.npz",
                         f"gesture_{name}_w{w}")
      rec["segment"] = name
      clips.append(rec)

  assert len(clips) == 34, len(clips)
  prov = {
    "source": str(SRC), "source_sha256": sha256(SRC),
    "source_mtime": datetime.fromtimestamp(
      SRC.stat().st_mtime, timezone.utc).isoformat(),
    "gate": "in-env raw-write reproduction 2026-08-26 covered stand_seed1 / "
            "laugh_giggle / relax_v2 frames (see walk_pool50_provenance.json)",
    "windows": {"stand": {"frames": STAND_WIN, "stride": STAND_STRIDE,
                          "per_seed": STAND_PER},
                "gesture": {"frames": GEST_WIN, "stride": GEST_STRIDE,
                            "per_gesture": GEST_PER,
                            "note": "relax_v2 cycle 9.2s > 4s window = "
                                    "partial-cycle windows, by design"}},
    "clips": clips,
    "created_at": datetime.now(timezone.utc).isoformat(),
  }
  PROV.write_text(json.dumps(prov, indent=2))
  print(f"wrote {len(clips)} clips (10 stand + 24 gesture)")
  print(f"provenance -> {PROV}")


if __name__ == "__main__":
  main()
