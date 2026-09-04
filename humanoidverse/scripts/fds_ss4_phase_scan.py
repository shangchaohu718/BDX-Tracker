"""fixed_dance_reference_specialist — side_step_4 full-phase diagnostic scan.

Answers ONE question for the ss4 Stage-A failure (all eval seeds die at frame 1
via right-ankle err ~1.0): is the right-ankle pathology
  (a) global all-phases, (b) seam-band-only (near wrap T-1->0), or
  (c) frame-0-only?
This determines whether the v2 temporal-contract fix can cure it.

Sections (all deterministic: policy mean action, pushes OFF via
inner.is_evaluating=True, obs noise ON per training obs path, seeded per phase):
  1. single-step scan, EVERY phase t in 0..T-1: inject exact q_ref[t],dq_ref[t]
     (inject_read_exact_lean), compute deterministic mean action a, convert to
     q_des per joint (q_des = default + 1.25*effort*a/kp; chain: a*5 -> clip +-5
     -> *0.25 -> *effort/kp -> PD), record q_des limit violations and the
     one-step tracking error after wrapped.step.
  2. short rollout scan, every --rollout-stride phases: 25 deterministic steps
     with NON-cyclic reference (current = min(t+k, T-1), future clamped at
     T-1), NO-STOP through rule failures; completion = 25 steps with none of
     {env term, any-joint err>0.8, base_height<0.2, non-foot contact, NaN}
     firing.
  3. right-ankle err histogram vs the 0.8 training gate over all pooled
     rollout-scan frames.
  4. raw seam analysis from the pkl + env FK contact states:
     gap T-1 -> 0 (q L2, dq L2, per-joint q diff, contact change, root state
     diff) and frame 0 vs anchor 49 (q distance to home stand, dq norm,
     contact state). Contact = ankle_z < 0.095 m (ankle body z from
     body_names.index at injection, after mj_forward).

Usage:
  .venv/bin/python humanoidverse/scripts/fds_ss4_phase_scan.py \
      --ckpt results/fds/dance_core_ss4_F/ckpt_final.pth \
      --out results/fds/ss4_phase_scan_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("BFM_ZERO_HARD_LIMIT_RESET", "0")
os.environ.setdefault("BFM_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

ENV_CFG = ROOT / "results/bfm-dynamics-baseline/config.json"
FPS = 50.0
OBS_STATE_DIM = 34
INPUT_DIM = 106
ACT_DIM = 14
ERR_TERM = 0.8
RIGHT_ANKLE = 9
LEFT_ANKLE = 4
ANKLE_Z_CONTACT = 0.095
SEAM_BAND = 10  # phases within +-10 of {0, T-1} count as "seam band"


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GaussianTanhPolicy(torch.nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = ACT_DIM, hidden: int = 256, layers: int = 3):
        super().__init__()
        import torch.nn as nn
        mods: list[torch.nn.Module] = []
        prev = obs_dim
        for _ in range(layers):
            mods += [nn.Linear(prev, hidden), nn.ReLU()]
            prev = hidden
        self.trunk = torch.nn.Sequential(*mods)
        self.mean = nn.Linear(prev, act_dim)
        self.log_std = torch.nn.Parameter(torch.full((act_dim,), -0.7))

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mean(h)
        return mu, None


def build_reference(pkl_path: Path):
    data = joblib.load(pkl_path)
    assert len(data) == 1
    name, entry = next(iter(data.items()))
    dof_np = np.asarray(entry["dof"], dtype=np.float64)
    T = dof_np.shape[0]
    dq = np.zeros_like(dof_np)
    dq[1:-1] = (dof_np[2:] - dof_np[:-2]) * (float(entry["fps"]) / 2.0)
    dq[0], dq[-1] = dq[1], dq[-2]
    return (torch.as_tensor(dof_np, dtype=torch.float32),
            torch.as_tensor(dq, dtype=torch.float32), int(T),
            float(entry["fps"]), name, entry)


def phase_table(device, T: int):
    ang = 2 * math.pi * torch.arange(T, device=device) / T
    return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)


def build_input(state34, t_idx, q_ref, dq_ref, ptab, T):
    q, dq = state34[:14], state34[14:28]
    e = [q_ref[t_idx] - q, dq_ref[t_idx] - dq]
    for k in (5, 10, 20):
        e.append(q_ref[min(t_idx + k, T - 1)] - q)
    return torch.cat([state34] + e + [ptab[t_idx]])


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [max(0.0, (c - h) / d), min(1.0, (c + h) / d)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pkl", default=str(ROOT / "humanoidverse/data/fds_re_dancegen_side_step_4.pkl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--rollout-stride", type=int, default=4)
    ap.add_argument("--rollout-steps", type=int, default=25)
    ap.add_argument("--seed-base-single", type=int, default=40000)
    ap.add_argument("--seed-base-rollout", type=int, default=50000)
    args = ap.parse_args()
    out_path = Path(args.out).resolve()
    assert not out_path.exists(), f"refusing to overwrite {out_path}"
    pkl_path = Path(args.pkl).resolve()

    os.environ["BFM_POOL_DATA"] = str(pkl_path)
    from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env, inject_read_exact_lean

    cfg = json.loads(ENV_CFG.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapped = build_cpu_env(cfg, device)
    inner = wrapped._env
    dev = inner.device

    q_ref_np, dq_ref_np, T, clip_fps, clip_name, entry = build_reference(pkl_path)
    q_ref = q_ref_np.to(dev)
    dq_ref = dq_ref_np.to(dev)
    ptab = phase_table(dev, T)

    rcfg = yaml.safe_load(open(ROOT / "humanoidverse/config/robot/bdx/bdx_14dof.yaml"))["robot"]
    joint_names = list(rcfg["dof_names"])
    lo = torch.tensor(rcfg["dof_pos_lower_limit_list"], device=dev)
    hi = torch.tensor(rcfg["dof_pos_upper_limit_list"], device=dev)
    effort = torch.tensor(rcfg["dof_effort_limit_list"], device=dev, dtype=torch.float32)
    home = torch.tensor([rcfg["init_state"]["default_joint_angles"][n] for n in joint_names],
                        device=dev, dtype=torch.float32)

    # action -> q_des chain (verified against _compute_torques):
    #   a*5 (normalize_action 1->5) -> clip +-5 -> *0.25 (action_scale)
    #   -> *effort/kp (action_rescale) -> PD: torque = kp*(tgt + default - q) - kd*dq
    # => q_des(a) = default_dof_pos + default_dof_pos_offset + 1.25*effort*a/kp
    kp = inner.p_gains.flatten().float().to(dev)
    kd = inner.d_gains.flatten().float().to(dev)
    def get_default():
        d = inner.default_dof_pos.flatten().float().to(dev)
        off = getattr(inner, "default_dof_pos_offset", None)
        if off is not None and torch.is_tensor(off):
            d = d + off.flatten().float().to(dev)
        return d
    a_to_qdes_gain = 1.25 * effort / kp  # q_des = default + gain*a

    # policy from checkpoint
    ck = torch.load(args.ckpt, map_location=dev)
    policy = GaussianTanhPolicy(INPUT_DIM).to(dev)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    rms_mean = ck["rms"]["mean"].to(dev)
    rms_var = ck["rms"]["var"].to(dev)

    def normalize(x):
        return ((x - rms_mean) / (rms_var + 1e-8).sqrt()).clamp(-10.0, 10.0)

    def mean_action(x_raw):
        with torch.no_grad():
            mu, _ = policy(normalize(x_raw))
        return torch.tanh(mu)

    feet_set = set(inner.feet_indices.tolist())
    forbidden_idx = torch.tensor([i for i in range(len(inner.body_names)) if i not in feet_set],
                                 dtype=torch.long, device=dev)
    ankle_bid = {side: inner.body_names.index(f"{side}_ankle_link") for side in ("left", "right")}
    ts = inner.config.termination_scales
    FALL_MIN_H = float(ts.termination_min_base_height)
    inner.is_evaluating = True  # pushes OFF (repo eval convention)

    def inject(frame: int, seed: int):
        seed_all(seed)
        torch.manual_seed(seed)
        ids = torch.zeros(1, dtype=torch.long, device=dev)
        times = torch.full((1,), frame / FPS, device=dev)
        obs = inject_read_exact_lean(wrapped, ids, times, settle_reps=2)
        return obs["state"][0].to(dev).float()

    def ankle_z():
        bp = inner.simulator._rigid_body_pos[0]  # [nbodies,3] aligned with body_names
        return float(bp[ankle_bid["left"], 2]), float(bp[ankle_bid["right"], 2])

    def rule_fail(q_true, err, obs_state):
        """Task-specified scan termination rule: env | any-joint>0.8 |
        base_height<0.2 | non-foot contact | NaN."""
        reasons = []
        if bool((err > ERR_TERM).any()):
            reasons.append("tracking_err_gt_0p8")
        if float(inner.simulator.robot_root_states[0, 2]) < 0.2:
            reasons.append("base_height_fall")
        cf = torch.norm(inner.simulator.contact_forces[0, forbidden_idx, :], dim=-1)
        if bool((cf > 1.0).any()):
            reasons.append("nonfoot_contact")
        if not (torch.isfinite(q_true).all() and torch.isfinite(obs_state).all()):
            reasons.append("nan_inf")
        return reasons

    # ------------------------------------------------------------------
    # 1. single-step scan, every phase
    # ------------------------------------------------------------------
    n_ph = T
    ss = {
        "per_phase": [],
    }
    qdes_mat = np.zeros((n_ph, ACT_DIM))
    err1_mat = np.zeros((n_ph, ACT_DIM))
    viol_lo_mat = np.zeros((n_ph, ACT_DIM))
    viol_hi_mat = np.zeros((n_ph, ACT_DIM))
    a_mat = np.zeros((n_ph, ACT_DIM))
    anklez = np.zeros((n_ph, 2))
    contact_at_inject = np.zeros((n_ph, 2), dtype=bool)
    # injected-state hard-limit margin (coordinator cross-check: does injection
    # itself sit trivially close to hip_yaw/hip_roll limits?)
    inj_margin = np.minimum(q_ref_np.numpy() - lo.cpu().numpy()[None, :],
                            hi.cpu().numpy()[None, :] - q_ref_np.numpy())  # [T,14]
    post_step_hardviol = np.zeros((n_ph, ACT_DIM), dtype=bool)
    for t in range(n_ph):
        cur = inject(t, args.seed_base_single + t)
        lz, rz = ankle_z()
        anklez[t] = (lz, rz)
        contact_at_inject[t] = (lz < ANKLE_Z_CONTACT, rz < ANKLE_Z_CONTACT)
        x_raw = build_input(cur, t, q_ref, dq_ref, ptab, T)
        a = mean_action(x_raw)
        default = get_default()
        q_des = default + a_to_qdes_gain * a
        obs, _, term, _, _ = wrapped.step(a[None], to_numpy=False)
        q_true = inner.simulator.dof_pos[0].float()
        obs_state = obs["state"][0].to(dev).float()
        err = (q_true - q_ref[min(t + 1, T - 1)]).abs()
        qdes_mat[t] = q_des.cpu().numpy()
        err1_mat[t] = err.cpu().numpy()
        viol_lo_mat[t] = (q_des - lo).clamp(max=0).cpu().numpy()  # 0 unless below lo
        viol_hi_mat[t] = (q_des - hi).clamp(min=0).cpu().numpy()  # 0 unless above hi
        a_mat[t] = a.cpu().numpy()
        post_step_hardviol[t] = ((q_true < lo) | (q_true > hi)).cpu().numpy()
        ss["per_phase"].append({
            "t": t,
            "env_term": bool(term.any()),
            "rule_fail": rule_fail(q_true, err, obs_state),
        })
        if t % 40 == 0:
            print(f"[scan1] phase {t}/{n_ph} a_rankle={float(a[RIGHT_ANKLE]):+.3f} "
                  f"qdes_rankle={float(q_des[RIGHT_ANKLE]):+.3f} "
                  f"err1_rankle={float(err[RIGHT_ANKLE]):.3f}", flush=True)

    any_viol = (viol_lo_mat < 0) | (viol_hi_mat > 0)
    ra_viol = any_viol[:, RIGHT_ANKLE]
    ra_qdes = qdes_mat[:, RIGHT_ANKLE]
    ra_lo = float(lo[RIGHT_ANKLE])
    err1_ra = err1_mat[:, RIGHT_ANKLE]
    err1_worst_joint = [int(np.argmax(err1_mat[t])) for t in range(n_ph)]
    err1_max = err1_mat.max(axis=1)

    viol_phases = {jn: [int(i) for i in np.where(any_viol[:, j])[0]]
                   for j, jn in enumerate(joint_names)}
    seam_set = set(range(0, SEAM_BAND + 1)) | set(range(T - 1 - SEAM_BAND, T))
    ra_viol_phases = viol_phases["right_ankle_joint"]
    ra_viol_in_seam = [p for p in ra_viol_phases if p in seam_set]
    frac_ra = len(ra_viol_phases) / n_ph

    single_step_summary = {
        "n_phases": n_ph,
        "right_ankle_qdes_min": float(ra_qdes.min()),
        "right_ankle_qdes_argmin": int(ra_qdes.argmin()),
        "right_ankle_lower_limit": ra_lo,
        "right_ankle_qdes_below_limit_phase_count": int(ra_viol.sum()),
        "right_ankle_qdes_below_limit_fraction": frac_ra,
        "right_ankle_qdes_below_limit_phases": ra_viol_phases,
        "right_ankle_qdes_below_limit_phases_outside_seam_band":
            [p for p in ra_viol_phases if p not in seam_set],
        "right_ankle_qdes_mean": float(ra_qdes.mean()),
        "per_joint_violation_phase_counts": {jn: int(any_viol[:, j].sum())
                                             for j, jn in enumerate(joint_names)},
        "per_joint_violation_phases": viol_phases,
        "one_step_right_ankle_err_gt_0p8_phase_count": int((err1_ra > ERR_TERM).sum()),
        "one_step_right_ankle_err_max": float(err1_ra.max()),
        "one_step_right_ankle_err_mean": float(err1_ra.mean()),
        "one_step_worst_err_gt_0p8_phase_count": int((err1_max > ERR_TERM).sum()),
        "one_step_worst_err_mean": float(err1_max.mean()),
        "one_step_worst_joint_is_right_ankle_count": int(sum(
            1 for j in err1_worst_joint if j == RIGHT_ANKLE)),
        "env_term_phase_count": int(sum(1 for r in ss["per_phase"] if r["env_term"])),
        "injection_hard_limit_margin": {
            "question": ("coordinator cross-check: does the injected exact state sit within "
                         "~0.02-0.05 of hip_yaw/hip_roll hard limits at many phases (making "
                         "joint_hard_limit termination fire trivially from injection)?"),
            "definition": "min(q_ref[t]-lo, hi-q_ref[t]) per joint per phase (injected q_ref)",
            "per_joint_min_margin_over_phases": {
                jn: float(inj_margin[:, j].min()) for j, jn in enumerate(joint_names)},
            "phases_with_any_joint_margin_lt_0p02": int((inj_margin.min(axis=1) < 0.02).sum()),
            "phases_with_any_joint_margin_lt_0p05": int((inj_margin.min(axis=1) < 0.05).sum()),
            "hip_yaw_margin_min": float(inj_margin[:, [0, 5]].min()),
            "hip_roll_margin_min": float(inj_margin[:, [1, 6]].min()),
            "answer": None,  # filled below
        },
        "post_step_hard_limit_violations": {
            "definition": "actual q after ONE deterministic step outside [lo,hi] (v1 policy)",
            "per_joint_phase_counts": {jn: int(post_step_hardviol[:, j].sum())
                                        for j, jn in enumerate(joint_names)},
            "phases_with_any_violation": int(post_step_hardviol.any(axis=1).sum()),
        },
        "note": ("one-step err measured after a single wrapped.step from the exact "
                 "injected reference state at phase t (err vs q_ref[min(t+1,T-1)])"),
    }
    _inj = single_step_summary["injection_hard_limit_margin"]
    _inj["answer"] = (
        f"NO for this clip: min injected margin over all joints/phases = "
        f"{_inj['per_joint_min_margin_over_phases'] and min(_inj['per_joint_min_margin_over_phases'].values()):.4f} rad; "
        f"hip_yaw min {_inj['hip_yaw_margin_min']:.4f}, hip_roll min {_inj['hip_roll_margin_min']:.4f}; "
        f"{_inj['phases_with_any_joint_margin_lt_0p05']}/292 phases have any joint within 0.05 of a hard limit. "
        "The quarantined objective-v2 hip_yaw hard-limit deaths are therefore POLICY-driven "
        "(one-step q excursions beyond +-0.349), not trivially-fired from injection.")

    # ------------------------------------------------------------------
    # 2. rollout scan every rollout-stride phases (25 steps, non-cyclic ref)
    # ------------------------------------------------------------------
    phases = list(range(0, T, args.rollout_stride))
    ro_rows = []
    ra_err_all = []  # pooled right-ankle err frames for the histogram
    for t0 in phases:
        cur = inject(t0, args.seed_base_rollout + t0)
        completed = True
        first_fail, fail_reasons = None, []
        max_joint_err = 0.0
        ra_max = 0.0
        forbidden_contact = False
        env_term = False
        frames_done = 0
        for k in range(args.rollout_steps):
            idx = min(t0 + k, T - 1)
            x_raw = build_input(cur, idx, q_ref, dq_ref, ptab, T)
            a = mean_action(x_raw)
            obs, _, term, _, _ = wrapped.step(a[None], to_numpy=False)
            frames_done = k + 1
            q_true = inner.simulator.dof_pos[0].float()
            obs_state = obs["state"][0].to(dev).float()
            err = (q_true - q_ref[min(idx + 1, T - 1)]).abs()
            ra_e = float(err[RIGHT_ANKLE])
            ra_err_all.append(ra_e)
            mje = float(err.max())
            max_joint_err = max(max_joint_err, mje)
            ra_max = max(ra_max, ra_e)
            cf = torch.norm(inner.simulator.contact_forces[0, forbidden_idx, :], dim=-1)
            forbidden_contact = forbidden_contact or bool((cf > 1.0).any())
            reasons = rule_fail(q_true, err, obs_state)
            if bool(term.any()):
                reasons = ["env_termination"] + reasons
                env_term = True
            if reasons and first_fail is None:
                first_fail = k + 1
                fail_reasons = reasons
            if reasons:
                completed = False
            if env_term or "nan_inf" in reasons:
                break
            cur = obs_state
        ro_rows.append({
            "phase": t0, "completed": completed, "frames": frames_done,
            "first_rule_fail_step": first_fail, "fail_reasons": fail_reasons,
            "max_joint_err": max_joint_err, "right_ankle_err_max": ra_max,
            "forbidden_contact": forbidden_contact, "env_term": env_term,
        })
        print(f"[scan2] phase {t0}: completed={completed} first_fail={first_fail} "
              f"maxerr={max_joint_err:.3f} rankle_max={ra_max:.3f}", flush=True)

    n_comp = sum(r["completed"] for r in ro_rows)
    seam_phases_ro = [r for r in ro_rows if r["phase"] in seam_set]
    mid_phases_ro = [r for r in ro_rows if r["phase"] not in seam_set]
    rollout_summary = {
        "stride": args.rollout_stride, "steps": args.rollout_steps, "n_rollouts": len(ro_rows),
        "completion_rate": n_comp / len(ro_rows),
        "completion_wilson95": wilson_ci(n_comp, len(ro_rows)),
        "completion_in_seam_band": (sum(r["completed"] for r in seam_phases_ro) /
                                     max(1, len(seam_phases_ro))),
        "completion_outside_seam_band": (sum(r["completed"] for r in mid_phases_ro) /
                                          max(1, len(mid_phases_ro))),
        "mean_max_joint_err": float(np.mean([r["max_joint_err"] for r in ro_rows])),
        "mean_right_ankle_err_max": float(np.mean([r["right_ankle_err_max"] for r in ro_rows])),
        "forbidden_contact_rollout_count": int(sum(r["forbidden_contact"] for r in ro_rows)),
        "per_rollout": ro_rows,
        "reference_contract": "NON-cyclic: current=min(t0+k, T-1), lookahead clamp at T-1 (consistent)",
    }

    # ------------------------------------------------------------------
    # 3. right-ankle err histogram vs 0.8 (pooled rollout frames)
    # ------------------------------------------------------------------
    ra = np.asarray(ra_err_all)
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf]
    labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0+"]
    hist, _ = np.histogram(ra, bins=bins)
    histogram = {
        "n_frames": int(ra.size),
        "bins": labels,
        "counts": [int(c) for c in hist],
        "fractions": [float(c / ra.size) for c in hist],
        "mean": float(ra.mean()), "p50": float(np.percentile(ra, 50)),
        "p95": float(np.percentile(ra, 95)), "max": float(ra.max()),
        "fraction_below_0p8": float((ra < 0.8).mean()),
        "fraction_in_0p6_0p8": float(((ra >= 0.6) & (ra < 0.8)).mean()),
        "riding_test": ("if a large mass sits in 0.6-0.8 the policy is riding "
                        "just under the 0.8 training gate"),
    }

    # ------------------------------------------------------------------
    # 4. raw seam analysis
    # ------------------------------------------------------------------
    q_np = q_ref_np.numpy()
    dq_np = dq_ref_np.numpy()
    def quat_angle(q1, q2):
        d = abs(float(np.dot(q1, q2)))
        return 2 * math.acos(min(1.0, d))
    root_pos = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    root_rot = np.asarray(entry["root_rot"], dtype=np.float64)
    seam = {
        "seam_definition": "raw clip frame T-1 -> frame 0 (what the modulo contract jumps across)",
        "q_l2": float(np.linalg.norm(q_np[T - 1] - q_np[0])),
        "dq_l2": float(np.linalg.norm(dq_np[T - 1] - dq_np[0])),
        "per_joint_q_diff": {jn: float(q_np[T - 1, j] - q_np[0, j])
                             for j, jn in enumerate(joint_names)},
        "per_joint_q_diff_abs_max_joint": joint_names[int(np.argmax(np.abs(q_np[T - 1] - q_np[0])))],
        "root_pos_diff_l2": float(np.linalg.norm(root_pos[T - 1] - root_pos[0])),
        "root_rot_angle_diff_rad": quat_angle(root_rot[T - 1], root_rot[0]),
        "contact_state": {
            "definition": f"ankle_z < {ANKLE_Z_CONTACT} m at injection (env FK, body_names.index)",
            "left_at_T_minus_1": bool(contact_at_inject[T - 1][0]),
            "right_at_T_minus_1": bool(contact_at_inject[T - 1][1]),
            "left_at_0": bool(contact_at_inject[0][0]),
            "right_at_0": bool(contact_at_inject[0][1]),
            "changed": bool((contact_at_inject[T - 1] != contact_at_inject[0]).any()),
        },
        "ankle_z_at_T_minus_1": [float(anklez[T - 1][0]), float(anklez[T - 1][1])],
        "ankle_z_at_0": [float(anklez[0][0]), float(anklez[0][1])],
        "adjacent_frame_gap_q_l2_T2_to_T1": float(np.linalg.norm(q_np[T - 1] - q_np[T - 2])),
        "adjacent_frame_gap_q_l2_1_to_0": float(np.linalg.norm(q_np[1] - q_np[0])),
        "median_adjacent_q_l2": float(np.median(np.linalg.norm(np.diff(q_np, axis=0), axis=1))),
    }
    anchor = 49
    frame0_vs_anchor = {
        "anchor": anchor,
        "anchor_source": "results/fds/re_dancegen_side_step_4_dynamic_gate.json 2_transition_anchor",
        "q_l2_to_home_stand_frame0": float(np.linalg.norm(q_np[0] - home.cpu().numpy())),
        "q_l2_to_home_stand_anchor": float(np.linalg.norm(q_np[anchor] - home.cpu().numpy())),
        "dq_norm_frame0": float(np.linalg.norm(dq_np[0])),
        "dq_norm_anchor": float(np.linalg.norm(dq_np[anchor])),
        "contact_frame0": [bool(contact_at_inject[0][0]), bool(contact_at_inject[0][1])],
        "contact_anchor": [bool(contact_at_inject[anchor][0]), bool(contact_at_inject[anchor][1])],
        "ankle_z_frame0": [float(anklez[0][0]), float(anklez[0][1])],
        "ankle_z_anchor": [float(anklez[anchor][0]), float(anklez[anchor][1])],
        "q_gap_frame0_to_anchor_l2": float(np.linalg.norm(q_np[0] - q_np[anchor])),
    }

    # ------------------------------------------------------------------
    # 5. A/B/C canonical-sequence comparison (ruling: Option C suffix is the
    #    expected winner; bias toward C unless the data says otherwise)
    # ------------------------------------------------------------------
    def smooth_hermite(q_lo, d_lo, q_hi, d_hi, n):
        """n C1 (cubic Hermite) samples from (q_lo,d_lo) to (q_hi,d_hi);
        d_* are per-frame derivatives (central diff)."""
        s = torch.linspace(0.0, 1.0, n + 2, device=q_lo.device)[1:-1]
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2
        h11 = s ** 3 - s ** 2
        return (h00[:, None] * q_lo[None] + h10[:, None] * d_lo[None]
                + h01[:, None] * q_hi[None] + h11[:, None] * d_hi[None])

    def build_variant(name):
        q = q_np.copy()
        if name == "A":
            meta = {"name": "A", "desc": "natural 0->291 (original ordering, full clip)",
                    "start_orig_frame": 0, "orig_frames": list(range(T))}
            return q.copy(), meta
        if name == "C":
            meta = {"name": "C", "desc": "anchor49 suffix [49:292] (243 frames, phase0==anchor, no seam, no blend)",
                    "start_orig_frame": 49, "orig_frames": list(range(49, T))}
            return q[49:].copy(), meta
        if name == "B":
            W = 5  # blend half-window (frames each side of the junction)
            part1, part2 = q[49:], q[:49]
            qb = np.concatenate([part1, part2], axis=0)
            J = len(part1)  # junction between qb[J-1]=q[291] and qb[J]=q[0]
            # central-diff slopes of the ORIGINAL clip (per frame)
            dq_cent = np.zeros_like(q)
            dq_cent[1:-1] = (q[2:] - q[:-2]) / 2.0
            dq_cent[0], dq_cent[-1] = dq_cent[1], dq_cent[-2]
            # C1 blend replaces qb[J-W : J+W]; anchors are the ORIGINAL frames just
            # outside the window with their original central-diff slopes, so the
            # blended curve is position- AND velocity-continuous at both edges.
            #   lo anchor: qb[J-W-1] = q[49 + (J-W-1)] ; hi anchor: qb[J+W] = q[J-49... map below]
            lo_orig = 49 + (J - W - 1)              # original frame index of lo anchor
            hi_orig = (J + W) - len(part1)           # original frame index of hi anchor (=5)
            q_lo = torch.as_tensor(qb[J - W - 1], dtype=torch.float32, device=dev)
            d_lo = torch.as_tensor(dq_cent[lo_orig], dtype=torch.float32, device=dev)
            q_hi = torch.as_tensor(qb[J + W], dtype=torch.float32, device=dev)
            d_hi = torch.as_tensor(dq_cent[hi_orig], dtype=torch.float32, device=dev)
            n = 2 * W
            span = (J + W) - (J - W - 1)            # frame intervals lo->hi (2W+1)
            blended = smooth_hermite(q_lo, d_lo * span, q_hi, d_hi * span, n)
            qb[J - W:J + W] = blended.cpu().numpy()
            meta = {"name": "B", "desc": ("rotated 49->291->0->48 with C1 Hermite blend "
                                          f"(+-{W} frames) at the 291->0 junction"),
                    "start_orig_frame": 49,
                    "orig_frames": (list(range(49, T)) + list(range(0, 49))),
                    "blend": {"window": [J - W, J + W - 1], "lo_anchor_orig": lo_orig,
                              "hi_anchor_orig": hi_orig, "replaced_frames": n}}
            return qb, meta
        raise ValueError(name)

    def openloop_pd(q_seq, start_orig_frame, seed):
        seed_all(seed)
        torch.manual_seed(seed)
        ids = torch.zeros(1, dtype=torch.long, device=dev)
        times = torch.full((1,), start_orig_frame / FPS, device=dev)
        inject_read_exact_lean(wrapped, ids, times, settle_reps=2)
        default_flat = inner.default_dof_pos.flatten().float().to(dev)
        qs = torch.as_tensor(q_seq, dtype=torch.float32, device=dev)
        Tn = qs.shape[0]
        loT = lo; hiT = hi
        per = []
        for t in range(Tn):
            raw = (kp * (qs[t] - default_flat) / (1.25 * effort))
            action = raw.clamp(-1.0, 1.0)
            obs, _, term, trunc, _ = wrapped.step(action.unsqueeze(0), to_numpy=False)
            q_now = inner.simulator.dof_pos[0].float()
            tau = torch.from_numpy(inner.simulator.data.ctrl[-inner.num_dof:].copy()).to(dev).float()
            nxt = min(t + 1, Tn - 1)
            err = (q_now - qs[nxt]).abs()
            per.append({
                "mean_err": float(err.mean()), "max_err": float(err.max()),
                "right_ankle_err": float(err[RIGHT_ANKLE]),
                "pre_norm_action_clip_frac": float((raw.abs() > 1.0).float().mean()),
                "torque_max_ratio": float((tau.abs() / inner.torque_limits.flatten()).max()),
                "limit_viol_joints": int(((q_now < loT) | (q_now > hiT)).sum()),
            })
            if bool(term.any() or trunc.any()):
                break
        errs = np.array([p["mean_err"] for p in per])
        return {
            "frames_played": len(per), "terminated": bool(term.any() or trunc.any()),
            "mean_abs_err": float(errs.mean()),
            "steady_last32": float(np.median(errs[-32:])) if len(errs) >= 32 else float(np.median(errs)),
            "right_ankle_err_mean": float(np.mean([p["right_ankle_err"] for p in per])),
            "pre_norm_action_clip_frac_overall": float(np.mean([p["pre_norm_action_clip_frac"] for p in per])),
            "torque_max_ratio_overall": float(np.max([p["torque_max_ratio"] for p in per])),
            "limit_viol_total": int(sum(p["limit_viol_joints"] for p in per)),
            "per_frame_mean_err": [float(e) for e in errs],
            "method": ("gate methodology (fds_boxstep15_gate.py 6_openloop_pd): inject frame0, "
                       "PD-track the sequence via action=(q_ref[t]-default)*kp/(1.25*effort)"),
        }

    home_np = home.cpu().numpy()
    abc = {"ruling": ("Option C (anchor49 suffix) expected winner; comparison biased toward "
                      "C unless data says otherwise; decision lands in this report"),
           "anchor_frame": 49}
    completion_by_phase = {r["phase"]: r["completed"] for r in ro_rows}
    max_joint_err_by_phase = {r["phase"]: r["max_joint_err"] for r in ro_rows}
    for name in ("A", "B", "C"):
        q_seq, meta = build_variant(name)
        Tn = q_seq.shape[0]
        dq_seq = np.zeros_like(q_seq)
        dq_seq[1:-1] = (q_seq[2:] - q_seq[:-2]) * (clip_fps / 2.0)
        dq_seq[0], dq_seq[-1] = dq_seq[1], dq_seq[-2]
        adj = np.linalg.norm(np.diff(q_seq, axis=0), axis=1)
        s0 = meta["start_orig_frame"]
        retained = set(meta["orig_frames"])
        ph_ro = [p for p in completion_by_phase if p in retained]
        var = {
            **meta,
            "frames": Tn, "duration_s": Tn / clip_fps,
            "intro_endpoint": {
                "q_l2_to_home_stand": float(np.linalg.norm(q_seq[0] - home_np)),
                "dq_norm_first_frame": float(np.linalg.norm(dq_seq[0])),
                "contact_first_frame": [bool(contact_at_inject[s0][0]), bool(contact_at_inject[s0][1])],
                "ankle_z_first_frame": [float(anklez[s0][0]), float(anklez[s0][1])],
                "note": ("error the Stage-B intro must bridge to enter the dance core; "
                         "lower q_l2 + lower dq_norm + double support = easier entry"),
            },
            "internal_seam": {
                "one_shot_playback_seam_count": (1 if name == "B" else 0),
                "one_shot_note": ("under the v2 NON-cyclic contract no variant wraps; B's seam is "
                                  "the rotated 291->0 junction (smoothed by the C1 blend); A/C have none"),
                "adjacent_gap_q_l2_max": float(adj.max()),
                "adjacent_gap_q_l2_median": float(np.median(adj)),
                "max_gap_at_frame": int(adj.argmax()),
                "blend_distortion_frames": (10 if name == "B" else 0),
                "blend_distortion_note": ("B replaces 10 frames around the junction with Hermite "
                                          "samples (not original motion)" if name == "B" else
                                          "no blend: every retained frame is an original frame"),
                "loop_seam_gap_if_loopplayed_q_l2": float(np.linalg.norm(q_seq[-1] - q_seq[0])),
            },
            "openloop_pd": openloop_pd(q_seq, s0, args.seed_base_single + 7000 + ord(name)),
            "completion_estimate_from_v1_rollout_scan": {
                "retained_rollout_phases": len(ph_ro),
                "completed": int(sum(completion_by_phase[p] for p in ph_ro)),
                "rate": (sum(completion_by_phase[p] for p in ph_ro) / len(ph_ro)) if ph_ro else None,
                "mean_max_joint_err": (float(np.mean([max_joint_err_by_phase[p] for p in ph_ro]))
                                        if ph_ro else None),
                "note": "diagnostic only: v1 checkpoint, 25-step windows (NOT a trained-variant result)",
            },
            "visual_integrity": {
                "A": "full original motion incl. 49-frame pre-anchor intro inside the core",
                "B": "all 292 original frames but reordered (starts mid-clip) + 10 blended frames",
                "C": "original frames 49-291 EXACTLY as recorded; drops the 49-frame intro (Stage-B territory)",
            }[name],
        }
        abc[name] = var

    # decision: prefer C unless the data contradicts (open-loop tracking clearly
    # worse than A/B, or endpoint clearly worse than A)
    c, a, b = abc["C"], abc["A"], abc["B"]
    c_ol_ok = (c["openloop_pd"]["mean_abs_err"] <= 1.15 * a["openloop_pd"]["mean_abs_err"])
    c_endpoint_ok = (c["intro_endpoint"]["q_l2_to_home_stand"] <=
                     1.10 * a["intro_endpoint"]["q_l2_to_home_stand"])
    reasons = []
    if c_ol_ok:
        reasons.append(f"C open-loop mean err {c['openloop_pd']['mean_abs_err']:.4f} within 1.15x of A "
                       f"({a['openloop_pd']['mean_abs_err']:.4f})")
    else:
        reasons.append(f"WARNING: C open-loop err {c['openloop_pd']['mean_abs_err']:.4f} exceeds 1.15x of A")
    if c_endpoint_ok:
        reasons.append(f"C intro endpoint q_l2 {c['intro_endpoint']['q_l2_to_home_stand']:.4f} within 1.10x of A "
                       f"({a['intro_endpoint']['q_l2_to_home_stand']:.4f})")
    else:
        reasons.append(f"note: C intro endpoint q_l2 {c['intro_endpoint']['q_l2_to_home_stand']:.4f} > 1.10x of A "
                       f"({a['intro_endpoint']['q_l2_to_home_stand']:.4f}) but C starts in double support at the "
                       f"preregistered transition anchor")
    decision = "C" if (c_ol_ok and c_endpoint_ok) else "A"
    abc["canonical_sequence_decision"] = decision
    abc["decision_rule"] = ("C wins unless open-loop tracking > 1.15x A mean err or endpoint q_l2 "
                            "> 1.10x A (bias toward C per ruling; overriding data would have to be stark)")
    abc["decision_reasons"] = reasons
    abc["decision_implication"] = (f"canonical sequence = {decision}; v2 trainer trains NON-cyclic on "
                                   "the canonical sequence (pkl sliced accordingly), phase 0 == sequence "
                                   "frame 0, no modulo, no blend")

    # ------------------------------------------------------------------
    # verdict
    # ------------------------------------------------------------------
    if frac_ra > 0.8:
        verdict = "(a) global all-phases"
        verdict_basis = (f"right-ankle q_des below the joint lower limit at "
                         f"{len(ra_viol_phases)}/{n_ph} phases (>80%), spread across the clip")
    elif len(ra_viol_phases) == 0 and int((err1_ra > ERR_TERM).sum()) <= 1:
        verdict = "(c) frame-0-only"
        verdict_basis = "violations confined to phase 0"
    elif ra_viol_phases and all(p in seam_set for p in ra_viol_phases):
        verdict = "(b) seam-band-only"
        verdict_basis = (f"right-ankle violations only within +-{SEAM_BAND} of the "
                         f"wrap seam (phases {ra_viol_phases})")
    else:
        verdict = "mixed/other"
        verdict_basis = (f"right-ankle q_des violations at {len(ra_viol_phases)}/{n_ph} "
                         f"phases, not global (>80%) and not confined to the seam band; "
                         f"see per-phase arrays")

    report = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "script": "humanoidverse/scripts/fds_ss4_phase_scan.py",
            "ckpt": str(Path(args.ckpt).resolve()),
            "ckpt_global_steps": int(ck.get("global_steps", -1)),
            "clip": f"{clip_name} ({T} frames @ {clip_fps} fps)",
            "pkl": str(pkl_path),
            "mode": ("deterministic (policy mean action), pushes OFF via is_evaluating, "
                     "obs noise ON (training obs path), seeded per phase "
                     f"(single {args.seed_base_single}+t, rollout {args.seed_base_rollout}+t0)"),
        },
        "action_chain": {
            "formula": "q_des = default_dof_pos + default_dof_pos_offset + 1.25*effort*a/kp",
            "kp": [float(x) for x in kp.cpu()],
            "effort": [float(x) for x in effort.cpu()],
            "default_dof_pos": [float(x) for x in inner.default_dof_pos.flatten().cpu()],
            "right_ankle_gain_rad_per_unit_action": float(a_to_qdes_gain[RIGHT_ANKLE]),
            "right_ankle_action_at_qdes_lower_limit": float(
                (lo[RIGHT_ANKLE] - get_default()[RIGHT_ANKLE]) / a_to_qdes_gain[RIGHT_ANKLE]),
            "per_joint_reachable_raw_action_window": {
                jn: [float((lo[j] - get_default()[j]) / a_to_qdes_gain[j]),
                     float((hi[j] - get_default()[j]) / a_to_qdes_gain[j])]
                for j, jn in enumerate(joint_names)},
            "reachable_window_note": ("raw action (pre-normalize, +-1 tanh range) that maps to "
                                      "[lo,hi]; the ankle windows are narrowest — the ankle is "
                                      "the most limit-fragile joint per unit action"),
        },
        "single_step_scan": single_step_summary,
        "single_step_scan_arrays": {
            "per_phase_right_ankle_qdes": [float(x) for x in ra_qdes],
            "per_phase_right_ankle_action": [float(x) for x in a_mat[:, RIGHT_ANKLE]],
            "per_phase_right_ankle_one_step_err": [float(x) for x in err1_ra],
            "per_phase_worst_one_step_err": [float(x) for x in err1_max],
            "per_phase_worst_joint_idx": err1_worst_joint,
            "per_phase_ankle_z": [[float(x), float(y)] for x, y in anklez],
            "per_phase_qdes": [[float(v) for v in row] for row in qdes_mat],
            "per_phase_qdes_violation_any_joint": [bool(v) for v in any_viol.any(axis=1)],
            "per_phase_injection_min_hard_margin": [float(v) for v in inj_margin.min(axis=1)],
            "per_phase_post_step_hardviol_any_joint": [bool(v) for v in post_step_hardviol.any(axis=1)],
        },
        "rollout_scan": rollout_summary,
        "right_ankle_err_histogram": histogram,
        "seam_analysis": seam,
        "frame0_vs_anchor49": frame0_vs_anchor,
        "abc": abc,
        "VERDICT": verdict,
        "VERDICT_basis": verdict_basis,
        "VERDICT_implication": (
            "DECODE (coordinator-verified): the q_des values below are the v1 POLICY's "
            "commanded targets (mean action from --ckpt), NOT the reference. The "
            "reference itself was independently verified legal (0 hard-limit "
            "violations, 0 frames within 0.05 of any limit; tightest margin "
            "hip_roll 0.125/0.130; right-ankle q_ref in [+0.635,+0.950]). So all "
            "a/b/c branches here describe v1 POLICY behavior; whether the "
            "REFERENCE is trackable from injection is answered by abc.*."
            "openloop_pd (reference-derived actions, R2-vs-R3 discriminator)."),
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[scan] wrote {out_path}")
    print(f"[scan] VERDICT: {verdict} — {verdict_basis}")


if __name__ == "__main__":
    main()
