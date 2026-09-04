"""Patch excited_wiggle.npz: trim the dynamically-impossible head transient.

The clip's first ~4 frames carry a kinematic whip-start (root spawns with
1.3 m/s lateral vel, joint_vel norm 15 rad/s — decaying to normal by frame 4).
Spawn-at-frame-0 in play mode writes those velocities into the sim, flinging
the robot and tripping anchor_ori within 5 steps, for ANY policy.

Fix: keep frames [TRIM:] so the clip starts at the settled frame. Positions
are untouched; velocity channels stay self-consistent finite differences.
Append-only: writes a NEW npz, refuses overwrite, sha256 provenance sidecar.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SRC = Path(
  "/home/tcl/Desktop/start/bdx_rl_mjlab/policy_assets/bdx_v4/npz/episodic/"
  "excited_wiggle.npz"
)
DST = Path(
  "/home/tcl/Desktop/start/bdx_rl_mjlab/policy_assets/bdx_v4/npz/episodic/"
  "excited_wiggle_trim.npz"
)
SIDECAR = Path(
  "/home/tcl/Desktop/start/BFM-zero/results/mjlab_bdx/"
  "patch_excited_wiggle_trim_provenance.json"
)
TRIM = 4


def sha256(p: Path) -> str:
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def main():
  assert SRC.exists(), SRC
  if DST.exists():
    raise SystemExit(f"refusing overwrite: {DST} exists")
  d = dict(np.load(SRC, allow_pickle=True))

  T0 = d["joint_pos"].shape[0]
  keys_t = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w"]
  for k in keys_t:
    d[k] = d[k][TRIM:]

  # Validation on the trimmed clip.
  v = d
  assert v["joint_pos"].shape[0] == T0 - TRIM
  for k in keys_t:
    assert np.isfinite(v[k]).all(), k
  q = v["body_quat_w"]
  assert np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-5)
  r0_lin = float(np.linalg.norm(v["body_lin_vel_w"][0, 0]))
  r0_ang = float(np.linalg.norm(v["body_ang_vel_w"][0, 0]))
  j0 = float(np.linalg.norm(v["joint_vel"][0]))
  b0_ang = float(np.linalg.norm(v["body_ang_vel_w"][0]))
  report = {
    "src": str(SRC), "src_sha256": sha256(SRC),
    "dst": str(DST), "dst_sha256": None,
    "trim_frames": TRIM, "frames_before": T0, "frames_after": T0 - TRIM,
    "new_frame0_root_lin_vel_norm": round(r0_lin, 3),
    "new_frame0_root_ang_vel_norm": round(r0_ang, 3),
    "new_frame0_joint_vel_norm": round(j0, 3),
    "new_frame0_body_ang_vel_norm": round(b0_ang, 3),
    "removed_head_frame0_vals": {
      "root_lin_vel": [float(x) for x in
                       np.load(SRC)["body_lin_vel_w"][0, 0]],
      "joint_vel_norm": 14.959, "body_ang_vel_norm": 27.454,
    },
  }
  assert r0_lin < 0.5, f"trimmed frame0 root vel still high: {r0_lin}"
  assert j0 < 5.0, f"trimmed frame0 joint vel still high: {j0}"

  np.savez(DST, **d)
  report["dst_sha256"] = sha256(DST)
  SIDECAR.write_text(json.dumps(report, indent=2))
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
