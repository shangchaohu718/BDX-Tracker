"""dataset_contract v1 — receiving-side ingestion gate for mjlab pool tracking.

Layering (agreed with user 2026-08-27): CONSUMER-side interface. Thresholds
encode training-stack properties and live HERE, versioned and pinned to the
calibration config below. Data producers only owe offline-unrecoverable
metadata (command labels, provenance ids); content problems are handled by
rejection at this gate, never by changing the frozen producer chains.

Calibration pin (v1):
  z-termination 0.08   (Wiggle/Walk-Pool-V2 family, bad_anchor_pos_z_only)
  action scale flat 0.25 (gesture/deploy-matched contract)
  DROOP_Z 0.20          (static probe: untrained-policy sag root z)

Gates and their tuition:
  F1 schema keys            (format, hard)
  F2 name bijection         (bare-index corruption tuition, hard)
  F3 quat unit + finite     (P0 quat-convention incident, hard)
  F4 fps recorded           (format, hard)
  S1 non-degeneracy         (box_step round-18 tuition: standing clip
                             impersonating a motion; hard only on total
                             degeneracy, else record)
  S2 root_z band            (family param, hard; walk band from extraction)
  S3 joint-limit margin     (FDS 100k zero-tolerance tuition: record+flag,
                             NOT reject — tolerance adjudication pending)
  S4 static feasibility     (GRADED record-only in v1: its hard-gate evidence
                             was RETRACTED 2026-08-27 10:05 — five walk
                             flights were num_envs=1 invalid. Re-grade after
                             walk_pool50_v2_ne1024 rerun adjudicates.)
  S5 provenance             (record: file sha256 + stats)

Usage (bdx_rl_mjlab venv):
  python dataset_contract.py --pool <dir> --family {walk,dance} --out <json>
Output refuse-overwrites (append-only discipline).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CONTRACT_VERSION = "dataset_contract_v1"
CALIBRATION_PIN = {
    "z_termination": 0.08,
    "action_scale": "flat 0.25 (gesture/deploy-matched)",
    "droop_z": 0.20,
    "pinned": "Wiggle-Pool-50-V2 recipe, 2026-08-27",
}
REQUIRED_KEYS = (
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w",
)
ROOT_Z_BAND = {"walk": (0.15, 0.35), "dance": (0.12, 0.32)}
DEGENERATE_EXCURSION_RAD = 0.02   # per-joint p99-p01 below this on ALL joints...
DEGENERATE_TRAVEL_M = 0.005       # ...AND root xy travel below this => statue
LIMIT_MARGIN_FLAG_RAD = 0.05      # S3 flag threshold (record, not reject)
STATIC_GAP_FLAG = 0.08            # S4: median_root_z - DROOP_Z above this => risk note


def entity_reference() -> tuple[list[str], list[str], dict[str, tuple[float, float]]]:
    """Live Entity (the thing the task indexes with) + joint position limits.

    Reading jnt_range off the compiled model is a static-model property and is
    NOT subject to the standalone-compile FK trap (that trap is about
    evaluating kinematics of large-angle joint states).
    """
    from robots.bdx_v4 import get_bdx_v4_robot_cfg
    from mjlab.entity import Entity
    e = Entity(get_bdx_v4_robot_cfg())
    body_names, joint_names = list(e.body_names), list(e.joint_names)
    comp = e.compile()
    m = comp.model if hasattr(comp, "model") else comp
    jr = m.jnt_range
    limits = {}
    for i in range(jr.shape[0]):
        nm = m.joint(i).name
        if nm in joint_names:
            limits[nm] = (float(jr[i, 0]), float(jr[i, 1]))
    return body_names, joint_names, limits


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_clip(path: Path, family: str, body_ref: list[str], joint_ref: list[str],
               limits: dict[str, tuple[float, float]]) -> dict:
    r: dict = {"clip": path.name, "verdict": "PASS", "reject_reasons": [], "flags": []}
    d = dict(np.load(path, allow_pickle=False))

    # F1 schema
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        r["verdict"] = "REJECT"
        r["reject_reasons"].append(f"F1 missing keys: {missing}")
        return r

    # F2 name bijection (only when manifests stored; absence recorded, not fatal,
    # because MotionLoader reorders by entity names anyway — but silent drift
    # between stored and entity names is exactly the bare-index trap).
    if "body_names" in d and "joint_names" in d:
        bn, jn = [str(x) for x in d["body_names"]], [str(x) for x in d["joint_names"]]
        if set(bn) != set(body_ref):
            r["verdict"] = "REJECT"
            r["reject_reasons"].append(f"F2 body name-set mismatch: "
                                       f"only-clip={sorted(set(bn)-set(body_ref))} "
                                       f"only-entity={sorted(set(body_ref)-set(bn))}")
        if set(jn) != set(joint_ref):
            r["verdict"] = "REJECT"
            r["reject_reasons"].append(f"F2 joint name-set mismatch: "
                                       f"only-clip={sorted(set(jn)-set(joint_ref))} "
                                       f"only-entity={sorted(set(joint_ref)-set(jn))}")
    else:
        r["flags"].append("F2 name manifests absent (recorded; loader reorders by name)")

    jp = d["joint_pos"]
    T = jp.shape[0]

    # F3 finite + quat unit (body_quat_w is wxyz per repo contract)
    for k in REQUIRED_KEYS:
        if not np.isfinite(d[k]).all():
            r["verdict"] = "REJECT"
            r["reject_reasons"].append(f"F3 non-finite values in {k}")
    qn = np.linalg.norm(d["body_quat_w"].astype(np.float64), axis=-1)
    if qn.size and (np.abs(qn - 1.0) > 1e-4).any():
        r["verdict"] = "REJECT"
        r["reject_reasons"].append(f"F3 non-unit quats: |min/max norm - 1| = "
                                   f"{abs(qn.min()-1):.2e}/{abs(qn.max()-1):.2e}")

    # F4 fps
    fps = float(np.atleast_1d(d["fps"])[0]) if "fps" in d else 0.0
    if fps <= 0:
        r["verdict"] = "REJECT"
        r["reject_reasons"].append("F4 fps missing or <= 0")

    # S1 non-degeneracy: per-joint excursion + root travel
    exc = np.percentile(jp, 99, axis=0) - np.percentile(jp, 1, axis=0)
    root_xy = d["body_pos_w"][:, 0, :2]
    travel = float(np.linalg.norm(root_xy.max(0) - root_xy.min(0)))
    r["excursion_rad"] = {
        joint_ref[i] if i < len(joint_ref) and len(jp.shape) == 2 else str(i):
        round(float(exc[i]), 4) for i in range(len(exc))}
    r["root_travel_m"] = round(travel, 4)
    if bool((exc < DEGENERATE_EXCURSION_RAD).all()) and travel < DEGENERATE_TRAVEL_M:
        r["verdict"] = "REJECT"
        r["reject_reasons"].append(
            f"S1 degenerate: all-joint excursion < {DEGENERATE_EXCURSION_RAD} rad "
            f"and travel {travel:.4f} m < {DEGENERATE_TRAVEL_M} m (statue clip)")

    # S2 root_z band (family)
    root_z = d["body_pos_w"][:, 0, 2]
    lo, hi = ROOT_Z_BAND[family]
    zmin, zmax = float(root_z.min()), float(root_z.max())
    r["root_z_range"] = [round(zmin, 4), round(zmax, 4)]
    if zmin < lo or zmax > hi:
        r["verdict"] = "REJECT"
        r["reject_reasons"].append(
            f"S2 root_z [{zmin:.3f},{zmax:.3f}] outside {family} band ({lo},{hi})")

    # S3 joint-limit margins (record + flag; FDS tolerance unresolved)
    margins = {}
    for i, jn in enumerate(joint_ref):
        if jn in limits and (limits[jn][1] - limits[jn][0]) > 1e-6:
            lo_j, hi_j = limits[jn]
            margins[jn] = round(float(min((jp[:, i] - lo_j).min(),
                                          (hi_j - jp[:, i]).min())), 4)
    r["limit_margin_rad_min"] = min(margins.values()) if margins else None
    tight = {k: v for k, v in margins.items() if v < LIMIT_MARGIN_FLAG_RAD}
    if tight:
        r["flags"].append(f"S3 joints within {LIMIT_MARGIN_FLAG_RAD} rad of a hard "
                          f"limit: {tight}")

    # S4 static feasibility (GRADED record-only in v1 — see module docstring)
    med_z, init_z = float(np.median(root_z)), float(root_z[0])
    gap = med_z - CALIBRATION_PIN["droop_z"]
    r["static_feasibility"] = {
        "median_root_z": round(med_z, 4), "initial_root_z": round(init_z, 4),
        "gap_vs_droop": round(gap, 4),
        "status": "RISK_GRADED" if gap > STATIC_GAP_FLAG else "ok_record_only",
    }

    # S5 provenance
    r["frames"] = int(T)
    r["duration_s"] = round(T / fps, 2) if fps else None
    r["sha256"] = sha256(path)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--family", choices=list(ROOT_Z_BAND), required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    assert not args.out.exists(), f"refusing to overwrite {args.out}"

    body_ref, joint_ref, limits = entity_reference()
    clips = sorted(args.pool.glob("*.npz"))
    assert clips, f"no npz under {args.pool}"

    results = [check_clip(p, args.family, body_ref, joint_ref, limits) for p in clips]
    n_rej = sum(1 for x in results if x["verdict"] == "REJECT")
    report = {
        "contract_version": CONTRACT_VERSION,
        "calibration_pin": CALIBRATION_PIN,
        "pool": str(args.pool),
        "family": args.family,
        "gate_status_notes": {
            "S3": "record+flag only (FDS zero-tolerance unresolved)",
            "S4": "GRADED record-only (hard-gate evidence retracted "
                  "2026-08-27 10:05; re-grade after walk_pool50_v2_ne1024)",
        },
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "summary": {"clips": len(results), "pass": len(results) - n_rej,
                    "reject": n_rej},
        "clips": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in
                      ("contract_version", "pool", "family", "summary")}, indent=1))
    for x in results:
        if x["verdict"] == "REJECT":
            print(f"REJECT {x['clip']}: {x['reject_reasons']}")
        elif x["flags"]:
            print(f"FLAG {x['clip']}: {x['flags']}")


if __name__ == "__main__":
    main()
