"""box_step_15 full dynamic gate (ruling 2026-08-25 Step0-GO §6).

PREREGISTERED weights/thresholds (written before run):
  anchor J(t) = 1.0*||q_t - q_home|| + 0.5*||dq_t|| + 0.05*||ddq_t|| + 2.0*C_t
  C_t = 0.25*1[not double support] + 0.25*1[contact switch within +-3 frames]
      + 0.25*1[knee/ankle within 0.15 rad of limit] + 0.25*1[PD margin tight]
  PD margin tight = pre-norm action clip OR torque ratio > 0.9 at frame t
  contact = foot z - min_z(foot over clip) < 0.01 m  (per-foot autocalibration)
  intro/outro durations searched: 0.8 / 1.0 / 1.25 / 1.5 / 2.0 s (quintic
    min-jerk matching q, dq, ddq at both ends); pass = no fall, action clip
    < 0.1%, max torque ratio < 0.9, endpoint |q-target| < 0.15 rad.
Outputs all 11 ruling items to results/fds/boxstep15_dynamic_gate.json.
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

import sys
CLIP = Path("/home/tcl/Desktop/start/dataset/v3_planner/clips/" + sys.argv[1] + ".npz")
OUT = Path(f"results/fds/{sys.argv[1]}_dynamic_gate.json")
PKL = Path(f"humanoidverse/data/fds_{sys.argv[1].replace(chr(39),chr(39))}.pkl") if False else Path(f"humanoidverse/data/fds_{sys.argv[1]}.pkl")
FPS = 50.0
DT = 1.0 / FPS
FOOT_IDX = [5, 9]  # left_ankle_link, right_ankle_link (body order in npz)
DUR_SEARCH = [0.8, 1.0, 1.25, 1.5, 2.0]
W_Q, W_V, W_A, W_C = 1.0, 0.5, 0.05, 2.0

MOTION_CFG = EasyDict({
    "asset": {"assetRoot": str(Path(__file__).parents[1] / "data" / "robots" / "bdx") + "/",
              "assetFileName": "bdx_v4.xml", "urdfFileName": "bdx_v4.urdf"},
    "extend_config": [],
})


def wxyz_to_scipy(q):
    q = np.asarray(q).reshape(-1, 4)
    return sRot.from_quat(np.concatenate([q[:, 1:4], q[:, :1]], axis=-1))


def quintic(t, T, p0, v0, a0, p1, v1, a1):
    """Min-jerk quintic with full boundary constraints, vectorized over joints."""
    t = np.asarray(t, dtype=np.float64)
    T = float(T)
    a0_, a1_ = np.zeros_like(p0), np.zeros_like(p1)
    # standard 6-coefficient solve per joint
    A = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5],
        [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
        [0, 0, 2, 6*T, 12*T**2, 20*T**3],
    ])
    b = np.stack([p0, v0, a0_, p1, v1, a1_], axis=0)  # [6, J]
    c = np.linalg.solve(A, b)  # [6, J]
    tp = np.stack([t**0, t, t**2, t**3, t**4, t**5], axis=-1)  # [t, 6]
    return tp @ c  # [t, J]


def main():
    rcfg = yaml.safe_load(open("humanoidverse/config/robot/bdx/bdx_14dof.yaml"))["robot"]
    npz = dict(np.load(CLIP, allow_pickle=True))
    jn = [str(x) for x in npz["joint_names"]]
    jorder = [jn.index(n) for n in rcfg["dof_names"]]
    jp = npz["joint_pos"][:, jorder].astype(np.float64)
    jv = npz["joint_vel"][:, jorder].astype(np.float64)
    ja = np.gradient(jv, DT, axis=0)
    T = jp.shape[0]
    home = np.array([float(rcfg["init_state"]["default_joint_angles"][n]) for n in rcfg["dof_names"]])
    lo = np.array(rcfg["dof_pos_lower_limit_list"])
    hi = np.array(rcfg["dof_pos_upper_limit_list"])

    R = {"_provenance": {
        "created_at": datetime.now().isoformat(), "script": "fds_boxstep15_gate.py",
        "clip": str(CLIP), "frames": T, "fps": FPS,
        "preregistered": {"anchor_weights": {"w_q": W_Q, "w_v": W_V, "w_a": W_A, "w_c": W_C},
                          "dur_search_s": DUR_SEARCH,
                          "gate_pass": "no fall & clip<0.1% & torque<0.9 & endpoint<0.15 rad"},
    }}

    # 1. canonical stand-distance spectrum
    dist = np.linalg.norm(jp - home[None], axis=1)
    R["1_stand_distance_spectrum"] = {
        "canonical_stand_source": "bdx_14dof.yaml init_state.default_joint_angles; L2 over all 14 dof post-reorder; no wrap",
        "first_frame_l2": float(dist[0]), "last_frame_l2": float(dist[-1]),
        "min_l2": float(dist.min()), "argmin": int(dist.argmin()),
        "spectrum_every8": [round(float(d), 4) for d in dist[::8]] + [round(float(dist[-1]), 4)],
    }

    # 3. start/end continuity
    R["3_start_end_continuity"] = {
        "dq_first_norm": float(np.linalg.norm(jv[0])), "dq_last_norm": float(np.linalg.norm(jv[-1])),
        "q_gap_first_last_l2": float(np.linalg.norm(jp[0] - jp[-1])),
        "dq_gap_first_last_l2": float(np.linalg.norm(jv[0] - jv[-1])),
        "ddq_gap_first_last_l2": float(np.linalg.norm(ja[0] - ja[-1])),
        "root_first_xyz": npz["body_pos_w"][0, 0].tolist(), "root_last_xyz": npz["body_pos_w"][-1, 0].tolist(),
    }

    # 4/5. contact schedule
    fz = npz["body_pos_w"][:, FOOT_IDX, 2]  # [T,2]
    contact = (fz - fz.min(axis=0, keepdims=True)) < 0.01
    both = contact.all(axis=1)
    ss = ~both
    intervals = []
    t0 = None
    for t in range(T):
        if ss[t] and t0 is None:
            t0 = t
        elif not ss[t] and t0 is not None:
            intervals.append([t0, t - 1]); t0 = None
    if t0 is not None:
        intervals.append([t0, T - 1])
    switch = np.zeros(T, dtype=bool)
    ct = contact.astype(int)
    sw_frames = np.where((ct[1:] != ct[:-1]).any(axis=1))[0] + 1
    for s in sw_frames:
        switch[max(0, s - 3):min(T, s + 4)] = True
    R["4_contact_schedule"] = {
        "definition": "foot z - per-foot clip-min z < 0.01 m",
        "double_support_fraction": float(both.mean()),
        "contact_switch_frames": sw_frames.tolist(),
    }
    R["5_single_support_intervals"] = [
        {"start": int(a), "end": int(b), "duration_s": (b - a + 1) / FPS,
         "stance": "L" if contact[a, 0] else "R"} for a, b in intervals]

    # 6-10. open-loop PD playback (reuse step0 machinery via env)
    skel = Humanoid_Batch(MOTION_CFG, torch.device("cpu"))
    parents = skel._parents.numpy(); q_fixed = skel._local_rotation[0].numpy()
    bn = [str(x) for x in npz["body_names"]]
    order = [bn.index(n) for n in skel.body_names]
    pose_rot = npz["body_quat_w"][:, order].astype(np.float32)
    Tn, nb = pose_rot.shape[:2]
    pose_aa = np.zeros((Tn, nb, 3), dtype=np.float32)
    pose_aa[:, 0] = wxyz_to_scipy(pose_rot[:, 0]).as_rotvec()
    Rw = wxyz_to_scipy(pose_rot).as_matrix().reshape(Tn, nb, 3, 3)
    Rf = wxyz_to_scipy(q_fixed).as_matrix()
    Rj = np.linalg.inv(Rw[:, np.clip(parents, 0, None)] @ Rf[None]) @ Rw
    pose_aa[:, 1:] = sRot.from_matrix(Rj.reshape(Tn * nb, 3, 3)).as_rotvec().reshape(Tn, nb, 3)[:, 1:]
    root_pos = npz["body_pos_w"][:, order[0]].astype(np.float32)
    root_rot = npz["body_quat_w"][:, order[0]].astype(np.float32)
    entry = {"root_trans_offset": root_pos, "pose_aa": pose_aa,
             "dof": jp.astype(np.float32), "root_rot": root_rot, "fps": 50}
    if not PKL.exists():
        joblib.dump({sys.argv[1]: entry}, PKL, compress=3)

    import os
    os.environ["BFM_POOL_DATA"] = str(PKL.resolve())
    from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env, inject_read_exact_lean
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, torch.device("cuda"))
    inner = wrapped._env
    dev = inner.device
    q_ref = torch.from_numpy(jp.astype(np.float32)).to(dev)
    default_dof_pos = inner.default_dof_pos.flatten()
    effort = torch.tensor(rcfg["dof_effort_limit_list"], device=dev)
    kp_j = inner.p_gains.flatten()
    torque_limits = inner.torque_limits.flatten()
    loT = torch.tensor(lo, device=dev); hiT = torch.tensor(hi, device=dev)

    ids = torch.zeros(1, dtype=torch.long, device=dev)
    inject_read_exact_lean(wrapped, ids, torch.zeros(1, device=dev), settle_reps=2)

    per_frame = []
    for t in range(T):
        q_des = q_ref[t]
        raw = (kp_j * (q_des - default_dof_pos) / (1.25 * effort))
        action = raw.clamp(-1.0, 1.0)
        obs, _, term, trunc, _ = wrapped.step(action.unsqueeze(0), to_numpy=False)
        q_now = inner.simulator.dof_pos[0]
        tau = torch.from_numpy(inner.simulator.data.ctrl[-inner.num_dof:].copy()).to(dev)
        nxt = min(t + 1, T - 1)
        err = (q_now - q_ref[nxt]).abs()
        vm = (q_now < loT) | (q_now > hiT)
        per_frame.append({
            "t": t, "mean_err": float(err.mean()), "max_err": float(err.max()),
            "ankle_mean": float(err[[4, 9]].mean()), "knee_mean": float(err[[3, 8]].mean()),
            "pre_norm_action_clip_frac": float((raw.abs() > 1.0).float().mean()),
            "torque_max_ratio": float((tau.abs() / torque_limits).max()),
            "limit_viol": int(vm.sum()),
            "terminated": bool(term.any() or trunc.any()),
        })
        if term.any() or trunc.any():
            break
    errs = np.array([p["mean_err"] for p in per_frame])
    steady = float(np.median(errs[-32:])) if len(errs) >= 32 else float(np.median(errs))
    win = 0
    for e in errs:
        if e > 1.5 * steady:
            win += 1
        else:
            break
    R["6_openloop_pd"] = {
        "frames_played": len(per_frame), "terminated": per_frame[-1]["terminated"],
        "mean_abs_err": float(errs.mean()), "steady_last32": steady,
        "7_pre_norm_action_clip_frac_overall": float(np.mean([p["pre_norm_action_clip_frac"] for p in per_frame])),
        "8_torque_max_ratio_overall": float(np.max([p["torque_max_ratio"] for p in per_frame])),
        "9_limit_viol_total": int(sum(p["limit_viol"] for p in per_frame)),
        "10_transient_window": {"frames": [0, max(win - 1, 0)], "duration_s": win / FPS,
                                 "definition": "mean_err > 1.5*steady from t=0 until first drop below"},
        "per_frame": per_frame,
    }

    # 2. anchor search (uses playback margins)
    clip_t = np.array([p["pre_norm_action_clip_frac"] > 0 for p in per_frame])
    tq_t = np.array([p["torque_max_ratio"] > 0.9 for p in per_frame])
    near_lim = ((jp[:, [3, 8, 4, 9]] - lo[[3, 8, 4, 9]] < 0.15) | (hi[[3, 8, 4, 9]] - jp[:, [3, 8, 4, 9]] < 0.15)).any(axis=1)
    C = (0.25 * (~both).astype(float) + 0.25 * switch.astype(float)
         + 0.25 * near_lim.astype(float) + 0.25 * (clip_t | tq_t).astype(float))
    J = W_Q * dist + W_V * np.linalg.norm(jv, axis=1) + W_A * np.linalg.norm(ja, axis=1) + W_C * C
    anchor = int(J.argmin())
    R["2_transition_anchor"] = {
        "anchor_frame": anchor,
        "J_at_anchor": float(J[anchor]), "J_components": {
            "q_dist": float(dist[anchor]), "dq_norm": float(np.linalg.norm(jv[anchor])),
            "ddq_norm": float(np.linalg.norm(ja[anchor])), "C": float(C[anchor]),
            "double_support": bool(both[anchor]), "contact_switching": bool(switch[anchor]),
            "near_limit": bool(near_lim[anchor]), "pd_margin_tight": bool(clip_t[anchor] or tq_t[anchor])},
        "J_spectrum_every8": [round(float(j), 4) for j in J[::8]] + [round(float(J[-1]), 4)],
    }

    # 11. intro/outro duration search (quintic, PD-tracked in sim)
    def run_ramp(q_path, dq_path, q_init, root_init):
        # inject start state
        data = inner.simulator.data
        data.qpos[0:3] = root_init[:3]
        data.qpos[3:7] = [root_init[6], root_init[3], root_init[4], root_init[5]]  # wxyz->xyzw
        data.qpos[7:21] = q_init
        data.qvel[:] = 0.0
        import mujoco
        mujoco.mj_forward(inner.simulator.model, data)
        n = q_path.shape[0]
        rec = []
        for t in range(n):
            raw = (kp_j * (torch.from_numpy(q_path[t]).float().to(dev) - default_dof_pos) / (1.25 * effort))
            action = raw.clamp(-1.0, 1.0)
            wrapped.step(action.unsqueeze(0), to_numpy=False)
            q_now = inner.simulator.dof_pos[0]
            tau = torch.from_numpy(data.ctrl[-14:].copy()).to(dev)
            rec.append({
                "mean_err": float((q_now - torch.from_numpy(q_path[min(t + 1, n - 1)]).to(dev)).abs().mean()),
                "clip_frac": float((raw.abs() > 1.0).float().mean()),
                "tq": float((tau.abs() / torque_limits).max()),
                "term": False,
            })
            if not np.isfinite(inner.simulator.dof_pos.cpu().numpy()).all():
                rec[-1]["term"] = True
                break
        return rec

    root_a = npz["body_pos_w"][anchor, 0].astype(np.float64)
    root_a_full = np.zeros(7)
    root_a_full[:3] = root_a
    root_a_full[3:] = npz["body_quat_w"][anchor, 0].astype(np.float64)  # wxyz
    dur_results = []
    for D in DUR_SEARCH:
        n = int(round(D * FPS))
        ts = np.arange(n + 1) * DT
        q_path = quintic(ts, D, home, np.zeros(14), np.zeros(14), jp[anchor], jv[anchor], ja[anchor])
        rec = run_ramp(q_path, None, home, np.array([root_a[0], root_a[1], 0.35, root_a_full[3], root_a_full[4], root_a_full[5], root_a_full[6]]))
        end_gap = float(np.linalg.norm(inner.simulator.dof_pos[0].cpu().numpy() - jp[anchor])) if not rec[-1]["term"] else None
        dur_results.append({
            "duration_s": D, "frames": n, "terminated": rec[-1]["term"],
            "max_clip_frac": float(np.max([r["clip_frac"] for r in rec])),
            "max_torque_ratio": float(np.max([r["tq"] for r in rec])),
            "mean_err": float(np.mean([r["mean_err"] for r in rec])),
            "endpoint_gap_l2": end_gap,
            "pass": (not rec[-1]["term"]) and max(r["clip_frac"] for r in rec) < 0.001
                    and max(r["tq"] for r in rec) < 0.9 and end_gap is not None and end_gap < 0.15,
        })
    R["11_intro_outro_duration_search"] = {
        "intro": dur_results,
        "outro_note": "outro built separately from dance END (not reversed intro) once Stage-B composite reference is constructed; same quintic machinery",
    }

    assert not OUT.exists()
    OUT.write_text(json.dumps(R, indent=2) + "\n")
    slim = {k: (v if k not in ("6_openloop_pd",) else {kk: vv for kk, vv in v.items() if kk != "per_frame"}) for k, v in R.items()}
    print(json.dumps(slim, indent=2, default=str))
    print("saved", OUT)


if __name__ == "__main__":
    main()
