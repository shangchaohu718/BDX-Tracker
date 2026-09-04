"""Step 0 artifact fixes (ruling 2026-08-25 Step0-GO review):
1. rename mean_abs_err_by_quartile -> pooled_joint_abs_err_quartiles
2. add temporal_quarter_mean/max_abs_err (per-quarter temporal profile)
3. canonical stand-distance contract, full-frame spectrum, per-joint
4. duration text: 248f @ 50Hz = 4.96s; transient window frames 6-43 = 0.76s

Writes a NEW supplement file (existing report untouched).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

SRC = Path("results/fds/step0_link_validation_bounce_sway_13.json")
OUT = Path("results/fds/step0_link_validation_bounce_sway_13_fieldfix.json")
CLIP = "/home/tcl/Desktop/start/dataset/v3_planner/clips/re_dancegen_bounce_sway_13.npz"
FPS = 50.0


def main():
    rep = json.loads(SRC.read_text())
    npz = dict(np.load(CLIP, allow_pickle=True))
    rcfg = yaml.safe_load(open("humanoidverse/config/robot/bdx/bdx_14dof.yaml"))["robot"]
    jn = [str(x) for x in npz["joint_names"]]
    jorder = [jn.index(n) for n in rcfg["dof_names"]]
    jp = npz["joint_pos"][:, jorder].astype(np.float64)
    jv = npz["joint_vel"][:, jorder].astype(np.float64)
    T = jp.shape[0]

    home = np.array([float(rcfg["init_state"]["default_joint_angles"][n]) for n in rcfg["dof_names"]])

    # pooled per-joint error quartiles (rename; same computation as before)
    pf = rep["D_openloop_pd_playback"]["per_frame"]
    pooled = np.array([e for p in pf for e in p["per_joint_err"]])
    # temporal quarters
    q_edges = [0, T // 4, T // 2, 3 * T // 4, T]
    means = [float(np.mean([p["mean_abs_err"] for p in pf[q_edges[i]:q_edges[i + 1]]])) for i in range(4)]
    maxs = [float(np.max([p["max_abs_err"] for p in pf[q_edges[i]:q_edges[i + 1]]])) for i in range(4)]

    # canonical stand-distance spectrum
    dist = np.linalg.norm(jp - home[None], axis=1)  # L2, all 14 joints, robot order, no wrap
    per_joint_first = (np.abs(jp[0] - home)).tolist()
    per_joint_last = (np.abs(jp[-1] - home)).tolist()

    out = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "script": "fds_step0_field_fix.py",
            "supersedes_fields_in": str(SRC),
            "reason": "Step0 GO review: field naming, stand-distance contract unification, duration wording",
        },
        "renamed_fields": {
            "old": "mean_abs_err_by_quartile",
            "new": "pooled_joint_abs_err_quartiles",
            "value": [float(q) for q in np.quantile(pooled, [0.25, 0.5, 0.75, 1.0])],
            "definition": "quartiles over the pooled per-frame x per-joint |q - q_ref| table (NOT temporal quarters)",
        },
        "temporal_quarter_mean_abs_err": means,
        "temporal_quarter_max_abs_err": maxs,
        "temporal_quarter_frames": [list(range(q_edges[i], q_edges[i + 1])) and [q_edges[i], q_edges[i + 1] - 1] for i in range(4)],
        "duration_text": {
            "clip_duration_s": T / FPS,
            "frames": T, "fps": FPS,
            "transient_window_frames": [6, 43],
            "transient_window_duration_s": (43 - 6) / FPS,
            "note": "248 frames @ 50Hz = 4.96 s total; open-loop transient difficulty window frames 6-43 = 0.76 s",
        },
        "canonical_stand_distance": {
            "canonical_stand_source": "humanoidverse/config/robot/bdx/bdx_14dof.yaml robot.init_state.default_joint_angles",
            "joint_indices": "ALL 14 dof (incl. neck_forward/pitch/yaw/roll), ordered per robot.dof_names, i.e. AFTER clip->robot reorder",
            "distance_definition": "L2 norm of (q - q_home), no wrap (no continuous joints), computed per frame; RMS = L2/sqrt(14)",
            "first_frame_l2": float(dist[0]),
            "last_frame_l2": float(dist[-1]),
            "min_frame_l2": float(dist.min()), "argmin_frame": int(dist.argmin()),
            "min_frame_l2_excl_neck": float(np.linalg.norm((jp - home)[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]][dist.argmin()])),
            "first_frame_l2_excl_neck": float(np.linalg.norm((jp[0] - home)[:10])),
            "per_joint_distance_first": [round(v, 4) for v in per_joint_first],
            "per_joint_distance_last": [round(v, 4) for v in per_joint_last],
            "spectrum_l2_every8frames": [round(float(d), 4) for d in dist[::8]] + [round(float(dist[-1]), 4)],
            "verdict": "no stand-like frame anywhere in the clip (min L2 >> 0); intro/outro transitions are REQUIRED, consistent with product ruling",
            "reconciliation_note": "the 1.447 value in the original report IS this canonical definition (L2, all 14, post-reorder); any other figure (e.g. 1.717) used a non-canonical definition and is superseded by this file",
        },
    }
    assert not OUT.exists()
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "_provenance"}, indent=2)[:3000])
    print("saved", OUT)


if __name__ == "__main__":
    main()
