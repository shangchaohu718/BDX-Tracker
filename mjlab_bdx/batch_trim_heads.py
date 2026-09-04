"""Batch adaptive head-trim for the v3_planner wiggle cohort (50 clips).

Principle: a spawn state is viable iff it lies INSIDE the distribution of
mid-clip states that RSI training already exercises every day. The head
burst is 2-25x the steady envelope; the trim point is the first frame where
all three velocity channels re-enter the steady envelope and STAY there.

Detection (per clip, thresholds recorded in provenance):
  steady stats over frames [STEADY_SKIP:]  (skip the transient itself)
  pass(t) := rv[t] <= BAND*rv_p75  and  jv[t] <= BAND*jv_p75  and
             bav[t] <= BAND*bav_p75
  t0 := first t s.t. pass holds on [t, t+HOLD)      (BAND=1.1, HOLD=8)
  guard: t0 <= MAX_TRIM (20), else flag for manual review, no output.

Provenance per clip: src sha256, t0, thresholds, per-channel values at t0,
frames before/after, dst sha256. Velocity channels are NOT recomputed —
they are exact central differences of positions (cohort-audit fact), and a
contiguous slice preserves that; consistency is re-verified post-trim.

Outputs: results/mjlab_bdx/pool_v1/<name>.trim<h>.npz + provenance JSONs
+ summary index. Append-only; refuses overwrite.
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import numpy as np

CLIPS = sorted(glob.glob(
  "/home/tcl/Desktop/start/dataset/v3_planner/clips/"
  "re_dancegen_excited_wiggle_*.npz"))
OUT_DIR = Path("/home/tcl/Desktop/start/BFM-zero/results/mjlab_bdx/pool_v1")
BAND = 1.25
HOLD = 5
STEADY_SKIP = 10
MAX_TRIM = 25


def sha256(p: Path) -> str:
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def channels(d):
  rv = np.linalg.norm(d["body_lin_vel_w"][:, 0, :], axis=-1)
  jv = np.linalg.norm(d["joint_vel"].reshape(len(d["joint_vel"]), -1), axis=-1)
  bav = np.linalg.norm(d["body_ang_vel_w"].reshape(len(d["body_ang_vel_w"]), -1),
                       axis=-1)
  return rv, jv, bav


def find_t0(rv, jv, bav):
  sl = slice(STEADY_SKIP, None)
  thr = {k: BAND * float(np.percentile(v[sl], 75))
         for k, v in (("rv", rv), ("jv", jv), ("bav", bav))}
  ok = (rv <= thr["rv"]) & (jv <= thr["jv"]) & (bav <= thr["bav"])
  T = len(rv)
  for t in range(0, MAX_TRIM + 1):
    if t + HOLD <= T and bool(ok[t:t + HOLD].all()):
      return t, thr
  return None, thr


def main():
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  index = []
  for p in CLIPS:
    p = Path(p)
    d = dict(np.load(p, allow_pickle=True))
    rv, jv, bav = channels(d)
    t0, thr = find_t0(rv, jv, bav)
    T = len(rv)
    entry = {
      "src": str(p), "src_sha256": sha256(p),
      "band": BAND, "hold": HOLD, "steady_skip": STEADY_SKIP,
      "thresholds": {k: round(v, 3) for k, v in thr.items()},
      "f0_values": {"rv": round(float(rv[0]), 3), "jv": round(float(jv[0]), 3),
                    "bav": round(float(bav[0]), 3)},
      "frames_before": T,
    }
    if t0 is None:
      entry["status"] = "FLAG_MANUAL_REVIEW_no_settled_window"
      index.append(entry)
      print(f"[FLAG] {p.name}: no settled window within {MAX_TRIM} frames")
      continue
    entry["trim_frames"] = t0
    entry["t0_values"] = {"rv": round(float(rv[t0]), 3),
                          "jv": round(float(jv[t0]), 3),
                          "bav": round(float(bav[t0]), 3)}
    entry["reason"] = ("head velocity burst outside 1.1x steady p75 envelope; "
                       "first frame back inside envelope with 8-frame hold")
    keys = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w"]
    out = {k: d[k][t0:] for k in keys}
    for k in ("body_names", "joint_names", "fps", "resettable"):
      if k in d:
        out[k] = d[k]
    # normalize fps to (1,) — bdx_rl_mjlab registration reads fps[0]
    out["fps"] = np.atleast_1d(np.asarray(out["fps"]).astype(float))
    # post-trim velocity consistency re-check (central diff on interior)
    fps_arr = np.atleast_1d(out["fps"]).astype(float)  # source is 0-d scalar
    jp = out["joint_pos"]
    jd = (jp[2:] - jp[:-2]) * 0.5 * fps_arr[0]
    resid = float(np.abs(out["joint_vel"][1:-1] - jd).max())
    assert resid < 1e-5, f"velocity consistency broken: {resid}"  # float32 eps
    entry["post_trim_vel_central_diff_max_residual"] = resid
    dst = OUT_DIR / f"{p.stem}.trim{t0}.npz"
    assert not dst.exists(), f"refusing overwrite {dst}"
    np.savez(dst, **out)
    entry["dst"] = str(dst)
    entry["dst_sha256"] = sha256(dst)
    entry["frames_after"] = int(T - t0)
    entry["status"] = "ok"
    index.append(entry)
    print(f"[ok] {p.name}: trim {t0} -> {T - t0} frames "
          f"(f0 rv {rv[0]:.2f}->{rv[t0]:.2f}, jv {jv[0]:.1f}->{jv[t0]:.1f})")
  idx_path = OUT_DIR / "trim_index.json"
  assert not idx_path.exists()
  idx_path.write_text(json.dumps(index, indent=2))
  print("saved", idx_path)


if __name__ == "__main__":
  main()
