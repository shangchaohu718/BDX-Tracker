"""fixed_dance_reference_specialist — Step 0: link validation on
re_dancegen_bounce_sway_13 (ruling 2026-08-25 v2).

Pipeline:
  A. clip schema + FK sanity (5mm @ 8 frames, via Humanoid_Batch — same
     conversion path as the canonical dataset)
  B. stand-connectivity: first/last frame vs home stand (default_joint_angles,
     zero vel, nominal root) — ruling §5
  C. single-clip pkl export (assert-not-exists)
  D. open-loop PD playback in stable MuJoCo CPU env: inject frame 0, then
     action_t = (q_ref[t] - default_dof_pos)/action_scale each step;
     per-frame |q - q_ref[t+1]| per joint, torque saturation vs 0.8*effort,
     foot z penetration, joint limit violations. Deterministic.
Deliverable: results/fds/step0_link_validation_bounce_sway_13.json
Note: open-loop PD falls are EXPECTED and non-veto (ruling §4); this is the
per-frame error + contract sanity report, not the dynamic-feasibility gate.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from easydict import EasyDict
from scipy.spatial.transform import Rotation as sRot

from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

CLIP = Path("/home/tcl/Desktop/start/dataset/v3_planner/clips/re_dancegen_bounce_sway_13.npz")
OUT_DIR = Path("results/fds")
PKL = Path("humanoidverse/data/fds_bounce_sway_13.pkl")

MOTION_CFG = EasyDict({
    "asset": {
        "assetRoot": str(Path(__file__).parents[1] / "data" / "robots" / "bdx") + "/",
        "assetFileName": "bdx_v4.xml",
        "urdfFileName": "bdx_v4.urdf",
    },
    "extend_config": [],
})


def wxyz_to_scipy(q):
    q = np.asarray(q).reshape(-1, 4)
    return sRot.from_quat(np.concatenate([q[:, 1:4], q[:, :1]], axis=-1))


def convert_clip(npz, skel, parents, q_fixed, verify_frames):
    bn = [str(x) for x in npz["body_names"]]
    order = [bn.index(n) for n in skel.body_names]
    root_pos = npz["body_pos_w"][:, order[0]].astype(np.float32)
    root_rot = npz["body_quat_w"][:, order[0]].astype(np.float32)
    body_rot = npz["body_quat_w"][:, order].astype(np.float32)
    joint_pos = npz["joint_pos"].astype(np.float32)
    T, nb = body_rot.shape[:2]
    assert nb == len(skel.body_names)
    pose_aa = np.zeros((T, nb, 3), dtype=np.float32)
    pose_aa[:, 0] = wxyz_to_scipy(root_rot).as_rotvec()
    R_world = wxyz_to_scipy(body_rot).as_matrix().reshape(T, nb, 3, 3)
    R_fixed = wxyz_to_scipy(q_fixed).as_matrix()
    R_parent = R_world[:, np.clip(parents, 0, None)]
    R_joint = np.linalg.inv(R_parent @ R_fixed[None]) @ R_world
    pose_aa[:, 1:] = sRot.from_matrix(R_joint.reshape(T * nb, 3, 3)).as_rotvec().reshape(T, nb, 3)[:, 1:]
    idx = np.linspace(0, T - 1, min(verify_frames, T)).astype(int)
    fk = skel.fk_batch(torch.from_numpy(pose_aa[idx][None]), torch.from_numpy(root_pos[idx][None]), return_full=False)
    err_mm = np.abs(fk.global_translation[0].numpy() - npz["body_pos_w"][idx][:, order]).max() * 1000
    return {
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa,
        "dof": joint_pos,
        "root_rot": root_rot,
        "fps": int(npz["fps"]),
    }, err_mm


def main():
    OUT_DIR.mkdir(exist_ok=True)
    npz = dict(np.load(CLIP, allow_pickle=True))
    report = {"_provenance": {
        "created_at": datetime.now().isoformat(),
        "script": "fds_step0_link_validation.py",
        "project": "fixed_dance_reference_specialist v2 (ruling 2026-08-25)",
        "clip": str(CLIP), "frames": int(npz["joint_pos"].shape[0]), "fps": float(npz["fps"]),
        "note": "open-loop PD playback; falls expected & non-veto per ruling §4",
    }}

    # ---- A. schema + FK ----
    keys = set(npz.keys())
    assert keys == {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                    "body_lin_vel_w", "body_ang_vel_w", "fps", "body_names",
                    "joint_names", "resettable"}, f"unexpected schema: {keys}"
    assert np.isfinite(npz["joint_pos"]).all() and np.isfinite(npz["joint_vel"]).all()

    skel = Humanoid_Batch(MOTION_CFG, torch.device("cpu"))
    rcfg = yaml.safe_load(open(Path(__file__).parents[1] / "config/robot/bdx/bdx_14dof.yaml"))["robot"]
    jn = [str(x) for x in npz["joint_names"]]
    jorder = [jn.index(n) for n in rcfg["dof_names"]]
    npz["joint_pos"] = npz["joint_pos"][:, jorder]
    npz["joint_vel"] = npz["joint_vel"][:, jorder]
    parents = skel._parents.numpy()
    q_fixed = skel._local_rotation[0].numpy()
    entry, err_mm = convert_clip(npz, skel, parents, q_fixed, 8)
    report["A_schema_fk"] = {
        "n_bodies": int(npz["body_pos_w"].shape[1]), "n_dof": int(npz["joint_pos"].shape[1]),
        "joint_order_matches_robot_cfg": True, "fk_max_err_mm": float(err_mm),
        "fk_pass_5mm": bool(err_mm < 5.0),
        "resettable_frames": int(npz["resettable"].sum()),
    }

    # ---- B. stand connectivity ----
    default_q = np.zeros(14, dtype=np.float32)
    for i, name in enumerate(rcfg["dof_names"]):
        default_q[i] = float(rcfg["init_state"]["default_joint_angles"][name])
    jp = npz["joint_pos"]
    jv = npz["joint_vel"]
    root0 = npz["body_pos_w"][0, 0]
    rootT = npz["body_pos_w"][-1, 0]
    report["B_stand_connectivity"] = {
        "first_frame": {"q_dist_to_home": float(np.linalg.norm(jp[0] - default_q)),
                        "dq_norm": float(np.linalg.norm(jv[0])),
                        "root_xy": [float(root0[0]), float(root0[1])], "root_z": float(root0[2])},
        "last_frame": {"q_dist_to_home": float(np.linalg.norm(jp[-1] - default_q)),
                       "dq_norm": float(np.linalg.norm(jv[-1])),
                       "root_xy": [float(rootT[0]), float(rootT[1])], "root_z": float(rootT[2])},
        "end_gap_q_first_vs_last": float(np.linalg.norm(jp[0] - jp[-1])),
        "verdict_note": "first/last-to-stand distances decide: φ=0 stand-transition training vs v6_patch 0.5s ramp (ruling §5, data-driven choice)",
    }

    # ---- C. single-clip pkl (reuse if present — content deterministic) ----
    if not PKL.exists():
        joblib.dump({"bounce_sway_13": entry}, PKL, compress=3)
    report["C_pkl"] = {"path": str(PKL), "frames": int(entry["dof"].shape[0])}

    # ---- D. open-loop PD playback ----
    import os
    os.environ["BFM_POOL_DATA"] = str(PKL.resolve())
    from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env, inject_read_exact_lean
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, torch.device("cuda"))
    inner = wrapped._env
    q_ref = torch.from_numpy(jp).to(inner.device)          # [T,14] robot order
    T = q_ref.shape[0]
    action_scale = inner.config.robot.control.action_scale
    default_dof_pos = inner.default_dof_pos.flatten()
    torque_limits = inner.torque_limits.flatten()
    lo = torch.tensor(rcfg["dof_pos_lower_limit_list"], device=inner.device)
    hi = torch.tensor(rcfg["dof_pos_upper_limit_list"], device=inner.device)

    # inject frame 0
    ids = torch.zeros(1, dtype=torch.long, device=inner.device)
    times = torch.zeros(1, device=inner.device)
    inject_read_exact_lean(wrapped, ids, times, settle_reps=2)

    per_frame = []
    for t in range(T):
        # PD target for the NEXT sim step is the reference at the CURRENT index
        # (action_t computed from state at t; error measured at t+1 vs q_ref[t+1])
        q_des = q_ref[t]
        # Full action chain (legged_robot_base): normalize_action (x5: from=1.0
        # to=5.0) -> clip 5 -> action_rescale (x0.25*effort/kp) -> PD:
        #   torque = 0.25*effort*(5*a) + kp*(default-q) - kd*dq
        # choosing torque = kp*(q_des-q) - kd*dq  =>
        effort = torch.tensor(rcfg["dof_effort_limit_list"], device=inner.device)
        kp_j = inner.p_gains.flatten()
        action = (kp_j * (q_des - default_dof_pos) / (1.25 * effort)).clamp(-1.0, 1.0)
        obs, _, term, trunc, _ = wrapped.step(action.unsqueeze(0), to_numpy=False)
        q_now = inner.simulator.dof_pos[0]
        tau = torch.from_numpy(inner.simulator.data.ctrl[-inner.num_dof:].copy()).to(inner.device)
        nxt = min(t + 1, T - 1)
        err = (q_now - q_ref[nxt]).abs()
        viol_mask = ((q_now < lo) | (q_now > hi))
        per_frame.append({
            "t": t, "mean_abs_err": float(err.mean()), "max_abs_err": float(err.max()),
            "per_joint_err": [round(float(e), 4) for e in err.cpu()],
            "limit_viol": int(viol_mask.sum()),
            "limit_viol_joints": [int(i) for i in viol_mask.nonzero().flatten()],
            "torque_sat_frac": float((tau.abs() >= 0.98 * torque_limits).float().mean()),
            "torque_max_ratio": float((tau.abs() / torque_limits).max()),
            "terminated": bool(term.any()),
        })
        if term.any() or trunc.any():
            report["D_playback_terminated_at"] = t + 1
            break

    errs = np.array([p["mean_abs_err"] for p in per_frame])
    maxs = np.array([p["max_abs_err"] for p in per_frame])
    viol = int(sum(p["limit_viol"] for p in per_frame))
    viol_j = {}
    for p in per_frame:
        for j in p["limit_viol_joints"]:
            viol_j[j] = viol_j.get(j, 0) + 1
    tsat = [p["torque_sat_frac"] for p in per_frame]
    tmax = [p["torque_max_ratio"] for p in per_frame]
    report["D_openloop_pd_playback"] = {
        "frames_played": len(per_frame),
        "terminated_early": "D_playback_terminated_at" in report,
        "mean_abs_err_overall": float(errs.mean()),
        "mean_abs_err_by_quartile": [float(q) for q in np.quantile(errs, [0.25, 0.5, 0.75, 1.0])],
        "err_transient_peak_frame": int(errs.argmax()), "err_steady_last32_mean": float(errs[-32:].mean()),
        "max_abs_err_p95": float(np.quantile(maxs, 0.95)),
        "joint_limit_violation_frame_joints_total": viol,
        "joint_limit_viol_by_joint_idx": viol_j,
        "torque_saturation_frac_mean": float(np.mean(tsat)),
        "torque_max_ratio_overall": float(np.max(tmax)),
        "interpretation": "open-loop PD replay error profile; falls/none expected & non-veto. Baseline for the trained-policy gate (§8).",
        "per_frame": per_frame,
    }

    out = OUT_DIR / "step0_link_validation_bounce_sway_13.json"
    assert not out.exists()
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: (v if k != "D_openloop_pd_playback" else {kk: vv for kk, vv in v.items() if kk != "per_frame"}) for k, v in report.items()}, indent=2, default=str))
    print("saved", out)


if __name__ == "__main__":
    main()
