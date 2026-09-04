"""Ingest a dataset/v3_planner motion npz into bdx_rl_mjlab's episodic dir.

Runs under bdx_rl_mjlab's venv. Verifies by NAME against the live Entity
(the body-ordering trap), permutes if needed, reshapes fps to (1,) for the
auto-registration reader, validates physics sanity, and writes with
no-overwrite + a provenance sidecar.

Usage:
  .venv/bin/python ingest_npz.py \
    --src /home/tcl/Desktop/start/dataset/v3_planner/clips/re_dancegen_excited_wiggle_0.npz \
    --name excited_wiggle
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BDXRL_ROOT = Path("/home/tcl/Desktop/start/bdx_rl_mjlab")
BFM_RESULTS = Path("/home/tcl/Desktop/start/BFM-zero/results/mjlab_bdx")
GESTURE_DIR = BDXRL_ROOT / "policy_assets" / "bdx_v4" / "npz" / "episodic"
MOTION_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
               "body_lin_vel_w", "body_ang_vel_w")


def perm_by_name(src_names, dst_names, desc):
  src, dst = [str(x) for x in src_names], [str(x) for x in dst_names]
  assert set(src) == set(dst), (
    f"{desc} name-set mismatch: only-src={sorted(set(src)-set(dst))} "
    f"only-dst={sorted(set(dst)-set(src))}")
  return np.array([src.index(n) for n in dst], dtype=np.int64)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", required=True)
  ap.add_argument("--name", required=True, help="task stem, e.g. excited_wiggle")
  args = ap.parse_args()
  src = Path(args.src).resolve()
  out = GESTURE_DIR / f"{args.name}.npz"
  assert src.exists(), src
  assert not out.exists(), f"refusing to overwrite {out}"

  d = np.load(src, allow_pickle=False)
  missing = [k for k in MOTION_KEYS if k not in d.files]
  assert not missing, f"missing motion keys: {missing}"
  assert "body_names" in d.files and "joint_names" in d.files, "not self-describing"

  from mjlab.entity import Entity
  from robots.bdx_v4 import get_bdx_v4_robot_cfg
  entity = Entity(get_bdx_v4_robot_cfg())
  bperm = perm_by_name(d["body_names"], entity.body_names, "body")
  jperm = perm_by_name(d["joint_names"], entity.joint_names, "joint")

  pack = {k: d[k] for k in MOTION_KEYS}
  if not (bperm == np.arange(len(bperm))).all():
    for k in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
      pack[k] = pack[k][:, bperm]
  if not (jperm == np.arange(len(jperm))).all():
    for k in ("joint_pos", "joint_vel"):
      pack[k] = pack[k][:, jperm]
  fps = float(np.asarray(d["fps"]).reshape(-1)[0])
  pack["fps"] = np.array([fps])
  for k in ("body_names", "joint_names"):
    if k in d.files:
      pack[k] = d[k]
  if "resettable" in d.files:
    pack["resettable"] = d["resettable"]

  # physics sanity
  T = pack["joint_pos"].shape[0]
  qn = np.linalg.norm(pack["body_quat_w"], axis=-1)
  assert np.allclose(qn, 1.0, atol=1e-4), f"quat norms {qn.min()}..{qn.max()}"
  assert all(np.isfinite(pack[k]).all() for k in MOTION_KEYS), "non-finite"
  rz = pack["body_pos_w"][:, 0, 2]
  assert (rz > 0.15).all(), f"root z dips to {rz.min()}"

  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, **pack)

  sha = hashlib.sha256(src.read_bytes()).hexdigest()
  prov = {
    "src": str(src), "src_sha256": sha, "out": str(out),
    "name": args.name, "frames": int(T), "fps": fps,
    "duration_s": round(T / fps, 3),
    "body_perm_identity": bool((bperm == np.arange(len(bperm))).all()),
    "joint_perm_identity": bool((jperm == np.arange(len(jperm))).all()),
    "root_z_range": [float(rz.min()), float(rz.max())],
    "task_id": f"Mjlab-Tracking-Flat-BDX-V4-Gesture-"
               + "-".join(p.capitalize() for p in args.name.split("_")),
    "ingested_at": datetime.now(timezone.utc).isoformat(),
  }
  sidecar = BFM_RESULTS / f"ingest_{args.name}_provenance.json"
  assert not sidecar.exists(), sidecar
  sidecar.write_text(json.dumps(prov, indent=2))
  print(json.dumps(prov, indent=2))


if __name__ == "__main__":
  main()
