"""Real-stepping candidate screening over v3_planner dance library (ruling
2026-08-25 evening) + corrected body/contact contract.

CONTRACT (fixes 3rd bare-index incident):
  body index resolved via body_names.index("left_ankle_link"/"right_ankle_link")
  every record carries body_name, resolved_index, source_body_order_hash,
  robot_body_order_hash
  contact = ankle_z < CONTACT_Z (FIXED absolute threshold, NOT per-clip min).
  CONTACT_Z calibrated from box_step_15 planted mode: both ankles constant
  0.0797 when planted; threshold = 0.0797 + 0.015 = 0.095.

Screening criteria (preregistered):
  real-step candidate := foot xy path > 0.05 m (either foot)
                      AND >= 1 clear contact switch (L or R)
                      AND no double flight (never both feet off)
                      AND no reference joint-limit violation
  score = 0.4*norm(activity) + 0.2*norm(root drift) + 0.2*norm(q gap) + 0.2*norm(dq gap)
  prefer LOWER activity / drift / gap; pick best-scoring candidate for full gate.
  If no candidate passes -> box_step_15 fallback confirmed as final dance.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/home/tcl/Desktop/start/dataset/v3_planner/clips")
OUT = Path("results/fds/stepping_candidate_screening.json")
CONTACT_Z = 0.095
FPS = 50.0


def main():
    rcfg = yaml.safe_load(open("humanoidverse/config/robot/bdx/bdx_14dof.yaml"))["robot"]
    lo = np.array(rcfg["dof_pos_lower_limit_list"]); hi = np.array(rcfg["dof_pos_upper_limit_list"])
    robot_names = rcfg["dof_names"]
    robot_hash = hashlib.sha256("|".join(robot_names).encode()).hexdigest()[:16]

    rows = []
    for f in sorted(ROOT.glob("re_dancegen_*.npz")):
        npz = dict(np.load(f, allow_pickle=True))
        bn = [str(x) for x in npz["body_names"]]
        src_hash = hashlib.sha256("|".join(bn).encode()).hexdigest()[:16]
        li = bn.index("left_ankle_link"); ri = bn.index("right_ankle_link")
        jn = [str(x) for x in npz["joint_names"]]
        jorder = np.array([jn.index(n) for n in robot_names])
        jp = npz["joint_pos"][:, jorder].astype(np.float64)
        jv = npz["joint_vel"][:, jorder].astype(np.float64)
        fz = npz["body_pos_w"][:, [li, ri], 2]
        fxy = npz["body_pos_w"][:, [li, ri], :2]
        contact = fz < CONTACT_Z
        ct = contact.astype(int)
        switches = int((np.diff(ct, axis=0) != 0).any(axis=1).sum())
        double_flight = bool((~contact).all(axis=1).any())
        path = [float(np.linalg.norm(np.diff(fxy[:, i], axis=0), axis=1).sum()) for i in range(2)]
        root = npz["body_pos_w"][:, 0, :]
        drift = float(np.linalg.norm(root[-1, :2] - root[0, :2]))
        act = float(np.abs(jv).mean())
        qgap = float(np.linalg.norm(jp[-1] - jp[0])); dqp = float(np.linalg.norm(jv[-1] - jv[0]))
        lim_viol = int(((jp < lo) | (jp > hi)).sum())
        family = f.stem.replace("re_dancegen_", "").rsplit("_", 1)[0]
        is_cand = (max(path) > 0.05) and (switches >= 1) and (not double_flight) and (lim_viol == 0)
        rows.append({
            "clip": f.stem, "family": family, "frames": int(jp.shape[0]),
            "foot_xy_path_L_R": [round(path[0], 4), round(path[1], 4)],
            "contact_switches": switches, "double_flight": double_flight,
            "contact_fraction_L_R": [round(float(contact[:, 0].mean()), 3), round(float(contact[:, 1].mean()), 3)],
            "activity_mean_abs_dq": round(act, 3), "root_xy_drift": round(drift, 4),
            "q_gap_start_end": round(qgap, 4), "dq_gap_start_end": round(dqp, 4),
            "ref_limit_viol_frames": lim_viol,
            "is_real_step_candidate": is_cand,
            "_contract": {"body_names": ["left_ankle_link", "right_ankle_link"],
                          "resolved_index": [li, ri], "source_body_order_hash": src_hash,
                          "robot_body_order_hash": robot_hash,
                          "contact_def": f"ankle_z < {CONTACT_Z} (absolute; planted mode 0.0797 + 0.015)"},
        })

    cands = [r for r in rows if r["is_real_step_candidate"]]
    stats = {
        "n_clips": len(rows), "n_families": len({r["family"] for r in rows}),
        "n_candidates": len(cands),
        "families_with_candidates": sorted({r["family"] for r in cands}),
        "double_flight_clips": sum(1 for r in rows if r["double_flight"]),
    }
    best = None
    if cands:
        A = np.array([r["activity_mean_abs_dq"] for r in cands])
        D = np.array([r["root_xy_drift"] for r in cands])
        Q = np.array([r["q_gap_start_end"] for r in cands])
        V = np.array([r["dq_gap_start_end"] for r in cands])
        def nz(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-9)
        score = 0.4 * nz(A) + 0.2 * nz(D) + 0.2 * nz(Q) + 0.2 * nz(V)
        order = np.argsort(score)
        ranked = [{"clip": cands[i]["clip"], "score": round(float(score[i]), 4),
                   **{k: cands[i][k] for k in ("family", "foot_xy_path_L_R", "contact_switches",
                                                "activity_mean_abs_dq", "root_xy_drift",
                                                "q_gap_start_end", "dq_gap_start_end", "frames")}}
                  for i in order]
        best = ranked[0]

    out = {
        "_provenance": {"created_at": datetime.now().isoformat(),
                        "script": "fds_screen_stepping_candidates.py",
                        "contact_contract": "absolute ankle_z < 0.095 (calibrated planted mode 0.0797+0.015); body index via body_names.index",
                        "preregistered_criteria": "xy path>0.05m & >=1 switch & no double flight & no limit viol; score lower=better"},
        "stats": stats, "ranked_candidates": ([] if not cands else ranked[:10]), "best_candidate": best,
        "fallback": None if cands else "box_step_15 (no real-step candidate passed screening)",
        "all_clips": rows,
    }
    assert not OUT.exists()
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "all_clips"}, indent=2)[:2500])
    print("saved", OUT)


if __name__ == "__main__":
    main()
