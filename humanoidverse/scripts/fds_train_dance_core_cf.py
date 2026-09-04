"""fixed_dance_reference_specialist — CONTRACT-FIX trainer (suffix "_cf";
NOT the quarantined objective-v2 line "dance_core_ss4_Fv2_*").

Chain of custody:
  baseline (dance_core_ss4_F, 0.1M, objective v1) -> 0/10 completion, all eval
  seeds die at FRAME 1 via right-ankle err ~1.0 (deterministic mean action
  a~-0.8 => q_des ~ -2.13 vs ankle lower limit -1.396);
  objective-v2 worst-joint penalty (QUARANTINED, dance_core_ss4_Fv2) -> failure
  MIGRATED to hip_yaw hard-limit crossing (0.437 rad) => reward is not the fix;
  THIS trainer fixes the TEMPORAL CONTRACT + adds a q_des physical clamp.

Changes vs fds_train_dance_core.py (baseline; everything else IDENTICAL —
same seed 4728, PPO hyperparams, nets, normalizer, obs layout, reward):
  1. NON-CYCLIC reference contract: the current index NEVER wraps. It advances
     +1 per env step and is CLAMPED at T-1; an episode that reaches T-1 ends
     with success=True (success-at-end). The future-lookahead clamp at T-1 is
     now CONSISTENT with the current-index clamp (the baseline was
     self-contradictory: current modulo T wraps 291->0, lookahead frozen).
     wrap_count is counted and asserted 0 every step.
  2. THREE-WAY TERMINATION SPLIT:
       terminated_success      idx reaches T-1, no catastrophic rule on that
                               step. NO death penalty (reward = normal step
                               reward + SUCCESS_BONUS=0.0; success bonus kept
                               at 0 so the reward is COMPLETELY unchanged).
       terminated_catastrophic the v1 evening set fires (env termination; any-
                               joint |q-q_ref|>0.8 kept as divergence line;
                               base-height fall <0.2; orientation fall
                               |pg_x|/|pg_y|>0.8; non-foot contact >1N;
                               NaN/Inf; joint hard limit). reward -5.
       truncated_time_limit    episode cap reached without success. NO death
                               penalty; GAE BOOTSTRAPS V(final observation).
     GAE: success & catastrophic are TRUE terminals (next-value mask = 0);
     time-limit truncation bootstraps V(final obs) (never the new episode obs).
     Priority on a step where several fire: catastrophic > success > truncate.
  3. q_des PHYSICAL CLAMP (residual form, FIXED margin 0.05 — never shrunk):
       q_des_raw = default_dof_pos + default_dof_pos_offset
                   + (1.25*effort/kp)*a          [v1 action chain, unchanged]
       delta_raw  = q_des_raw - q_ref[t]
       delta_safe = clip(delta_raw, lo+0.05-q_ref[t], hi-0.05-q_ref[t])
       q_des      = q_ref[t] + delta_safe        [always within lo+0.05, hi-0.05]
     commanded through the env by exact inversion
       a_env = (q_des - default - offset)/(1.25*effort/kp), clipped +-1
     (the +-1 clip is a no-op for this robot: q_des within limits maps inside
     +-0.99 for every joint). The env's own limits/termination/reward stay
     ACTIVE — the clamp only bounds the commanded PD target. Startup check:
     q_ref itself must lie within [lo+0.05, hi-0.05] for every frame (fail
     fast with a full violation report if not; ss4 canonical passes with 0).
     If hip_roll clamps persistently: diagnose reference-too-close-to-limit /
     wrong residual direction / PD compensation / reference retarget — do NOT
     shrink the safety margin.
  4. NON-CYCLIC BIAS MONITOR (observation only): episode length / success
     rate / tracking error / advantage mean&std BY RESET-PHASE BUCKET (later
     resets are naturally shorter). No fixed-length windows (deferred).
  5. REWARD: completely unchanged from baseline objective v1:
       r = -1.0*mean|q-q_ref[t+1]| -0.05*mean|a_env - a_env_prev|
           -5.0*1[catastrophic] +0.01*clamp(-pg_z,0,1)
     (no per-joint max, no logsumexp, no dense changes; the action-rate term
     uses the actually-commanded a_env, which equals v1's a whenever the
     clamp does not bind.)
  6. THROUGHPUT accounting unchanged, with honest None when no training
     happened (eval-only mode must not print fake steps/s).

Deterministic eval: inject frame 0 (== canonical sequence start), T frames,
mean action through the SAME clamp pipeline and the SAME pkl reference as
training. completion = success-at-end AND no rule fired anywhere (success-at-
end is a NECESSARY condition). Headline reports completion+Wilson, mean, p95,
worst-joint together.

Outputs (never overwrite; suffix on collision):
  results/fds/{tag}/ckpt_batch{N}.pth + ckpt_final.pth + batch_log.jsonl
  results/fds/{tag}_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("BFM_ZERO_HARD_LIMIT_RESET", "0")
os.environ.setdefault("BFM_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

ENV_CFG = ROOT / "results/bfm-dynamics-baseline/config.json"
OUT_BASE = ROOT / "results/fds"

FPS = 50.0
OBS_STATE_DIM = 34
INPUT_DIM = OBS_STATE_DIM + 5 * 14 + 2   # 106
ACT_DIM = 14

# reward weights — IDENTICAL to baseline objective v1 (ruling: reward unchanged)
W_TRACK, W_RATE, W_TERM, W_UPRIGHT = 1.0, 0.05, 5.0, 0.01
SUCCESS_BONUS = 0.0          # ruling: success gets NO -5; bonus preregistered at 0
ERR_TERM = 0.8               # any-joint |q - q_ref| divergence line [rad]
QDES_MARGIN = 0.05           # FIXED physical clamp margin (never auto-shrunk)
N_PHASE_BUCKETS = 8          # non-cyclic bias monitor

OUTCOMES = ("step", "success", "catastrophic", "truncate")


# --------------------------------------------------------------------------
# pure helpers (unit-tested by humanoidverse/scripts/fds_cf_unit_tests.py)
# --------------------------------------------------------------------------
def qdes_residual_clamp(a: torch.Tensor, q_ref_t: torch.Tensor,
                        default_total: torch.Tensor, gain: torch.Tensor,
                        lo: torch.Tensor, hi: torch.Tensor,
                        margin: float = QDES_MARGIN):
    """Residual-form q_des clamp. Returns (q_des, a_env, delta_raw, delta_safe,
    clamped_bool, preclamp_violation).

    Requires q_ref_t within [lo+margin, hi-margin] per joint (startup-checked);
    otherwise the clip bounds would be inverted (undefined)."""
    q_des_raw = default_total + gain * a
    delta_raw = q_des_raw - q_ref_t
    dlo = lo + margin - q_ref_t
    dhi = hi - margin - q_ref_t
    delta_safe = torch.minimum(torch.maximum(delta_raw, dlo), dhi)
    q_des = q_ref_t + delta_safe
    a_env = ((q_des - default_total) / gain).clamp(-1.0, 1.0)
    clamped = delta_safe != delta_raw
    viol = (delta_raw - delta_safe).abs()
    return q_des, a_env, delta_raw, delta_safe, clamped, viol


def classify_outcome(cat_reasons, reached_end: bool, steps_after: int, cap: int) -> str:
    """Three-way split. Priority: catastrophic > success > truncate."""
    if cat_reasons:
        return "catastrophic"
    if reached_end:
        return "success"
    if steps_after >= cap:
        return "truncate"
    return "step"


def compute_step_reward(track_err: float, rate_err: float, upright: float,
                        outcome: str) -> float:
    r = (-W_TRACK * track_err
         - W_RATE * rate_err
         - W_TERM * float(outcome == "catastrophic")
         + W_UPRIGHT * upright)
    if outcome == "success":
        r += SUCCESS_BONUS
    return r


def compute_gae(br: torch.Tensor, bv: torch.Tensor, nonterm: torch.Tensor,
                next_v_override: torch.Tensor, v_boot: float,
                gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
    """GAE with per-step next-value override for time-limit truncation.

    nonterm[j] = 0 for true terminals (success/catastrophic) -> no bootstrap;
    nonterm[j] = 1 for continuing steps AND truncation steps. For truncation
    steps next_v_override[j] carries V(final observation of the truncated
    episode) — never the new episode's obs. For j == N-1 with a continuing
    episode, v_boot (V(current input)) is used. next_v_override entries are
    NaN wherever unused."""
    N = br.shape[0]
    adv = torch.zeros_like(br)
    gae = 0.0
    for j in reversed(range(N)):
        if j + 1 < N:
            next_v = next_v_override[j] if not torch.isnan(next_v_override[j]) else bv[j + 1]
        else:
            next_v = next_v_override[j] if not torch.isnan(next_v_override[j]) else v_boot
        delta = br[j] + gamma * next_v * nonterm[j] - bv[j]
        gae = delta + gamma * lam * nonterm[j] * gae
        adv[j] = gae
    return adv


# --------------------------------------------------------------------------
# networks (identical to baseline)
# --------------------------------------------------------------------------
class RunningMeanStd:
    def __init__(self, dim: int, device: torch.device):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 1e-4

    def update(self, x: torch.Tensor):
        bm, bv = x.mean(0), x.var(0, unbiased=False)
        n = x.shape[0]
        delta = bm - self.mean
        tot = self.count + n
        self.mean = self.mean + delta * n / tot
        m2 = self.var * self.count + bv * n + delta**2 * self.count * n / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / (self.var + 1e-8).sqrt()).clamp(-10.0, 10.0)

    def state(self):
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count}


LOG_STD_INIT = -0.7  # std 0.50 (baseline retune, unchanged)


class GaussianTanhPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = ACT_DIM, hidden: int = 256, layers: int = 3):
        super().__init__()
        mods: list[nn.Module] = []
        prev = obs_dim
        for _ in range(layers):
            mods += [nn.Linear(prev, hidden), nn.ReLU()]
            prev = hidden
        self.trunk = nn.Sequential(*mods)
        self.mean = nn.Linear(prev, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), LOG_STD_INIT))

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mean(h)
        std = torch.exp(self.log_std.clamp(-2.0, 0.5))
        return mu, std.expand_as(mu)

    @staticmethod
    def logp_squashed(u: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        g = -0.5 * (((u - mu) / std) ** 2 + 2 * torch.log(std) + math.log(2 * math.pi))
        corr = torch.log(1.0 - torch.tanh(u) ** 2 + 1e-6)
        return g.sum(-1) - corr.sum(-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        mods: list[nn.Module] = []
        prev = obs_dim
        for _ in range(layers):
            mods += [nn.Linear(prev, hidden), nn.ReLU()]
            prev = hidden
        mods += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------
# rollout helpers
# --------------------------------------------------------------------------
def build_reference(device, pkl_path: Path):
    data = joblib.load(pkl_path)
    assert len(data) == 1, f"expected single-motion pkl, got keys {list(data.keys())}"
    name, entry = next(iter(data.items()))
    q_ref = torch.as_tensor(entry["dof"], dtype=torch.float32, device=device)
    T, n_dof = q_ref.shape
    assert n_dof == ACT_DIM, n_dof
    fps = float(entry["fps"])
    assert abs(fps - FPS) < 1e-6, f"clip fps {fps} != injection FPS {FPS}"
    dof_np = np.asarray(entry["dof"])
    dq = np.zeros_like(dof_np)
    dq[1:-1] = (dof_np[2:] - dof_np[:-2]) * (fps / 2.0)
    dq[0], dq[-1] = dq[1], dq[-2]
    dq_ref = torch.as_tensor(dq, dtype=torch.float32, device=device)
    return q_ref, dq_ref, int(T), fps, name


def build_input(state34: torch.Tensor, t_idx: int, q_ref, dq_ref, phase_sc, T: int) -> torch.Tensor:
    """[106] raw input. NON-CYCLIC: t_idx <= T-1 always (clamped upstream);
    lookahead indices clamped at T-1 — CONSISTENT with the current index."""
    assert 0 <= t_idx < T, f"reference index OOB: {t_idx}"
    q, dq = state34[:14], state34[14:28]
    e = [q_ref[t_idx] - q, dq_ref[t_idx] - dq]
    for k in (5, 10, 20):
        idx = min(t_idx + k, T - 1)
        assert 0 <= idx < T
        e.append(q_ref[idx] - q)
    return torch.cat([state34] + e + [phase_sc[t_idx]])


class PhaseTable:
    def __init__(self, device, T: int):
        ang = 2 * math.pi * torch.arange(T, device=device) / T
        self.tab = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)

    def __getitem__(self, t):
        return self.tab[t]


def inject_episode(wrapped, inner, frame: int):
    from humanoidverse.scripts.generate_expert_sim_pool import inject_read_exact_lean
    ids = torch.zeros(1, dtype=torch.long, device=inner.device)
    times = torch.full((1,), frame / FPS, device=inner.device)
    obs = inject_read_exact_lean(wrapped, ids, times, settle_reps=2)
    return obs["state"][0].to(inner.device).float()


def fail_reasons(inner, q_true, err, obs_state34, forbidden_idx, lo, hi,
                 min_h: float, gxy: float) -> list:
    """Catastrophic rule set — IDENTICAL to the baseline evening ruling."""
    reasons = []
    if bool((err > ERR_TERM).any()):
        reasons.append("tracking_err_gt_0p8")
    if float(inner.simulator.robot_root_states[0, 2]) < min_h:
        reasons.append("base_height_fall")
    pg = inner.projected_gravity[0]
    if abs(float(pg[0])) > gxy or abs(float(pg[1])) > gxy:
        reasons.append("orientation_fall")
    cf = torch.norm(inner.simulator.contact_forces[0, forbidden_idx, :], dim=-1)
    if bool((cf > 1.0).any()):
        reasons.append("nonfoot_contact")
    if not (torch.isfinite(q_true).all() and torch.isfinite(obs_state34).all()):
        reasons.append("nan_inf")
    if bool((q_true < lo).any() or (q_true > hi).any()):
        reasons.append("joint_hard_limit")
    return reasons


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [max(0.0, (c - h) / d), min(1.0, (c + h) / d)]


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_reference_band(q_ref, lo, hi, joint_names, margin: float = QDES_MARGIN):
    """Startup check: q_ref within [lo+margin, hi-margin] every frame.
    Returns violation list [(frame, joint, value, bound)]."""
    q = q_ref.cpu()
    lo_c, hi_c = lo.cpu(), hi.cpu()
    bad = (q < (lo_c + margin)) | (q > (hi_c - margin))
    viol = []
    for f, j in zip(*bad.nonzero(as_tuple=True)):
        v = float(q[f, j])
        bound = float(lo_c[j] + margin) if v < float(lo_c[j] + margin) else float(hi_c[j] - margin)
        viol.append({"frame": int(f), "joint": joint_names[int(j)],
                     "q_ref": v, "violated_bound": bound})
    return viol


def unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    i = 1
    while True:
        q = p.with_name(f"{p.name}_{i}")
        if not q.exists():
            return q
        i += 1


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["F"], default="F",
                    help="cf line is Fresh-only (P was a selection-phase control)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--steps-total", type=int, default=102400)
    ap.add_argument("--batch-steps", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=4728)
    ap.add_argument("--eval-seeds", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--ppo-epochs", type=int, default=10)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--pkl", required=True, help="single-clip CANONICAL-sequence pkl")
    ap.add_argument("--episode-cap", type=int, default=None,
                    help="episode step cap (default T). Values < T force time-limit "
                         "truncations — for unit tests only")
    ap.add_argument("--eval-ckpt", default=None)
    ap.add_argument("--eval-no-stop", action="store_true")
    ap.add_argument("--openloop-baseline-mean", type=float, default=None,
                    help="open-loop PD mean track err for THIS clip/variant "
                         "(from fds_ss4_phase_scan.py abc.*.openloop_pd)")
    ap.add_argument("--openloop-baseline-worst", type=float, default=None,
                    help="open-loop PD worst-joint err for THIS clip/variant")
    ap.add_argument("--openloop-baseline-source", type=str, default=None,
                    help="provenance string, e.g. ss4_phase_scan_report.json:C.openloop_pd")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="torch device for env+ nets; cpu avoids GPU contention "
                         "(probe 2026-08-25: cpu 9.8ms/inj 11.9ms/step vs contended "
                         "cuda 257/190ms — AniMoFormer job held the GPU)")
    return ap.parse_args()


def main():
    args = parse_args()
    pkl_path = Path(args.pkl).resolve()
    if args.eval_ckpt:
        out_dir = Path(args.eval_ckpt).resolve().parent  # provenance only
        report_path = OUT_BASE / f"{args.tag or 'cf_evalonly'}_evalonly_report.json"
    else:
        tag = args.tag or "dance_core_ss4_F_cf"
        out_dir = unique_path(OUT_BASE / tag)
        out_dir.mkdir(parents=True)
        report_path = OUT_BASE / f"{out_dir.name}_report.json"
    assert not report_path.exists()
    print(f"[cf] arm={args.arm} out_dir={out_dir} report={report_path} "
          f"pkl={pkl_path.name} eval_only={bool(args.eval_ckpt)} "
          f"no_stop={args.eval_no_stop}", flush=True)

    os.environ["BFM_POOL_DATA"] = str(pkl_path)
    from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env

    cfg = json.loads(ENV_CFG.read_text())
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    wrapped = build_cpu_env(cfg, device)
    inner = wrapped._env
    dev = inner.device

    q_ref, dq_ref, T, clip_fps, clip_name = build_reference(dev, pkl_path)
    phase_sc = PhaseTable(dev, T)
    episode_cap = args.episode_cap if args.episode_cap is not None else T
    print(f"[cf] clip={clip_name} frames={T} fps={clip_fps} episode_cap={episode_cap}",
          flush=True)

    rcfg = yaml.safe_load(open(ROOT / "humanoidverse/config/robot/bdx/bdx_14dof.yaml"))["robot"]
    joint_names = list(rcfg["dof_names"])
    lo = torch.tensor(rcfg["dof_pos_lower_limit_list"], device=dev)
    hi = torch.tensor(rcfg["dof_pos_upper_limit_list"], device=dev)
    effort = torch.tensor(rcfg["dof_effort_limit_list"], device=dev, dtype=torch.float32)
    torque_limits = inner.torque_limits.flatten()
    ts = inner.config.termination_scales
    FALL_MIN_H = float(ts.termination_min_base_height)
    FALL_GXY = float(ts.termination_gravity_x)
    feet_set = set(inner.feet_indices.tolist())
    forbidden_idx = torch.tensor([i for i in range(len(inner.body_names))
                                  if i not in feet_set], dtype=torch.long, device=dev)

    # ---- startup reference-band check (residual clamp precondition) ----
    band_viol = check_reference_band(q_ref, lo, hi, joint_names)
    if band_viol:
        print(f"[cf] FATAL: q_ref outside [lo+{QDES_MARGIN}, hi-{QDES_MARGIN}] "
              f"for {len(band_viol)} (frame,joint) entries — the residual clamp "
              "bounds would be inverted; needs a ruling (retarget reference), "
              "margin is FIXED and will not be shrunk:", flush=True)
        for v in band_viol[:50]:
            print(f"    frame {v['frame']:3d} {v['joint']:24s} q_ref={v['q_ref']:+.4f} "
                  f"bound={v['violated_bound']:+.4f}", flush=True)
        raise SystemExit(2)
    ref_margins = torch.minimum(q_ref - (lo + QDES_MARGIN),
                                (hi - QDES_MARGIN) - q_ref)
    rm_idx = int(ref_margins.flatten().argmin())
    rm_frame, rm_joint = divmod(rm_idx, q_ref.shape[1])
    print(f"[cf] reference band check: 0 violations (tightest margin to clamp band "
          f"= {float(ref_margins.min()):.4f} rad at frame {rm_frame} "
          f"joint {joint_names[rm_joint]}, both sides)", flush=True)

    # ---- nets (identical seeding) ----
    seed_all(args.seed)
    torch.manual_seed(args.seed)
    policy = GaussianTanhPolicy(INPUT_DIM).to(dev)
    value = ValueNet(INPUT_DIM).to(dev)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-4)
    rms = RunningMeanStd(INPUT_DIM, dev)
    phase_rng = np.random.default_rng(args.seed)
    train_mode = args.eval_ckpt is None

    if not train_mode:
        ck = torch.load(args.eval_ckpt, map_location=dev)
        policy.load_state_dict(ck["policy"])
        value.load_state_dict(ck["value"])
        rms.mean = ck["rms"]["mean"].to(dev)
        rms.var = ck["rms"]["var"].to(dev)
        rms.count = float(ck["rms"]["count"])
        print(f"[cf] eval-only: loaded {args.eval_ckpt} (steps {ck.get('global_steps')})",
              flush=True)

    def save_ckpt(name: str, batch: int, gsteps: int, extras: dict):
        torch.save({
            "batch": batch, "global_steps": gsteps, "arm": args.arm,
            "policy": policy.state_dict(), "value": value.state_dict(),
            "opt": opt.state_dict(), "rms": rms.state(), **extras,
        }, out_dir / name)

    def live_action_map():
        """default(+DR offset) and gain read LIVE from the env so the inversion
        matches the env's own PD chain under any per-episode randomization."""
        d = inner.default_dof_pos.flatten().float().to(dev)
        off = getattr(inner, "default_dof_pos_offset", None)
        if off is not None and torch.is_tensor(off):
            d = d + off.flatten().float().to(dev)
        gain = 1.25 * effort / inner.p_gains.flatten().float().to(dev)
        return d, gain

    # ---- monitors ----
    wrap_count = 0
    outcome_counts = {"success": 0, "catastrophic": 0, "truncate": 0}
    cat_reason_tally: dict[str, int] = {}
    clamp_stats = {"steps": 0, "clamped_by_joint": torch.zeros(ACT_DIM, device=dev),
                   "max_preclamp_viol": 0.0, "resid_raw_abs_sum": 0.0,
                   "resid_safe_abs_sum": 0.0, "qdes_out_of_band_steps": 0}
    buckets = {b: {"episodes": 0, "success": 0, "len_sum": 0, "err_sum": 0.0,
                   "adv_sum": 0.0, "adv_sq_sum": 0.0, "adv_n": 0}
               for b in range(N_PHASE_BUCKETS)}

    def bucket_of(frame: int) -> int:
        return min(N_PHASE_BUCKETS - 1, frame * N_PHASE_BUCKETS // T)

    # ---- rollout state ----
    pending_reset = True
    t_idx, steps_in_ep = 0, 0
    prev_a_env = torch.zeros(ACT_DIM, device=dev)
    cur_state = None
    ep_ret, ep_steps, ep_err = 0.0, 0, []
    ep_bucket, ep_frame = 0, 0

    n_batches = math.ceil(args.steps_total / args.batch_steps) if train_mode else 0
    batch_log = []
    logjl = open(out_dir / "batch_log.jsonl", "a") if train_mode else None
    global_steps = int(ck.get("global_steps", 0)) if not train_mode else 0
    collect_time_total = 0.0
    t_start = time.time()

    for batch in range(1, n_batches + 1):
        N = args.batch_steps
        bx = torch.zeros(N, INPUT_DIM, device=dev)
        bu = torch.zeros(N, ACT_DIM, device=dev)
        blp = torch.zeros(N, device=dev)
        br = torch.zeros(N, device=dev)
        b_nonterm = torch.zeros(N, device=dev)
        bv = torch.zeros(N, device=dev)
        b_next_v = torch.full((N,), float("nan"), device=dev)
        b_err = torch.zeros(N, device=dev)
        b_clip = torch.zeros(N, device=dev)
        b_step_bucket = torch.zeros(N, dtype=torch.long, device=dev)
        b_outcome = []
        reason_counts: dict[str, int] = {}
        ep_returns, ep_compls, ep_err_means = [], 0, []
        b_clamp_frac_joint = torch.zeros(ACT_DIM, device=dev)
        b_max_viol = 0.0

        t0 = time.time()
        i = 0
        while i < N:
            if pending_reset:
                frame = int(phase_rng.integers(0, T))
                cur_state = inject_episode(wrapped, inner, frame)
                t_idx, steps_in_ep = frame, 0
                ep_frame = frame
                ep_bucket = bucket_of(frame)
                prev_a_env = torch.zeros(ACT_DIM, device=dev)
                ep_ret, ep_steps, ep_err = 0.0, 0, []
                pending_reset = False
            x_raw = build_input(cur_state, t_idx, q_ref, dq_ref, phase_sc, T)
            if steps_in_ep == 0:
                assert torch.allclose(x_raw[-2:], phase_sc.tab[t_idx]), \
                    "phase channels misaligned with injected reset frame"
            with torch.no_grad():
                mu, std = policy(rms.normalize(x_raw))
                v = value(rms.normalize(x_raw))
                u = mu + std * torch.randn(ACT_DIM, device=dev)
                a = torch.tanh(u)
                logp = GaussianTanhPolicy.logp_squashed(u[None], mu, std)[0]
            default_total, gain = live_action_map()
            q_des, a_env, d_raw, d_safe, clamped, viol = qdes_residual_clamp(
                a, q_ref[t_idx], default_total, gain, lo, hi)
            obs, _, term, trunc_env, _ = wrapped.step(a_env[None], to_numpy=False)
            t_next = min(t_idx + 1, T - 1)          # NON-CYCLIC: clamp, never wrap
            wrap_count += int(t_next < t_idx)        # must stay 0 (asserted below)
            assert wrap_count == 0, "reference index wrapped (contract violation)"
            q_true = inner.simulator.dof_pos[0].float()
            obs_state = obs["state"][0].to(dev).float()
            err = (q_true - q_ref[t_next]).abs()
            track_err = err.mean()
            cat_reasons = []
            if bool(term.any()):
                cat_reasons.append("env_termination")
            cat_reasons += fail_reasons(inner, q_true, err, obs_state, forbidden_idx,
                                        lo, hi, FALL_MIN_H, FALL_GXY)
            reached_end = (t_next == T - 1)
            outcome = classify_outcome(cat_reasons, reached_end, steps_in_ep + 1, episode_cap)
            upright = torch.clamp(-inner.projected_gravity[0, 2], 0.0, 1.0)
            r = compute_step_reward(float(track_err),
                                    float((a_env - prev_a_env).abs().mean()),
                                    float(upright), outcome)
            bx[i], bu[i], blp[i], br[i], bv[i] = x_raw, u, logp, r, v
            b_nonterm[i] = 0.0 if outcome in ("success", "catastrophic") else 1.0
            if outcome == "truncate":
                # bootstrap from the FINAL observation of the truncated episode
                with torch.no_grad():
                    x_final = build_input(obs_state, t_next, q_ref, dq_ref, phase_sc, T)
                    b_next_v[i] = value(rms.normalize(x_final))
            b_err[i] = track_err
            b_clip[i] = float((u.abs() > 1).any())
            b_step_bucket[i] = ep_bucket
            b_outcome.append(outcome)
            b_clamp_frac_joint += clamped.float()
            b_max_viol = max(b_max_viol, float(viol.max()))
            clamp_stats["steps"] += 1
            clamp_stats["clamped_by_joint"] += clamped.float()
            clamp_stats["max_preclamp_viol"] = max(clamp_stats["max_preclamp_viol"],
                                                   float(viol.max()))
            clamp_stats["resid_raw_abs_sum"] += float(d_raw.abs().mean())
            clamp_stats["resid_safe_abs_sum"] += float(d_safe.abs().mean())
            if bool(((q_des < lo + QDES_MARGIN) | (q_des > hi - QDES_MARGIN)).any()):
                clamp_stats["qdes_out_of_band_steps"] += 1
            for rs in cat_reasons:
                reason_counts[rs] = reason_counts.get(rs, 0) + 1
            ep_ret += float(r)
            ep_steps += 1
            ep_err.append(float(track_err))
            prev_a_env, cur_state, t_idx = a_env, obs_state, t_next
            steps_in_ep += 1
            i += 1
            if outcome != "step":
                outcome_counts[outcome] += 1
                if outcome == "success":
                    ep_compls += 1
                b = buckets[ep_bucket]
                b["episodes"] += 1
                b["success"] += int(outcome == "success")
                b["len_sum"] += ep_steps
                b["err_sum"] += float(np.mean(ep_err))
                ep_returns.append(ep_ret)
                ep_err_means.append(float(np.mean(ep_err)))
                pending_reset = True
        collect_s = time.time() - t0
        collect_time_total += collect_s
        if batch == 1:
            assert torch.isfinite(bx).all(), "non-finite obs input channels on first batch"
            assert torch.isfinite(br).all() and torch.isfinite(bv).all(), \
                "non-finite rewards/values on first batch"

        # GAE(0.95) with the three-way split: true terminals cut the chain,
        # truncation bootstraps V(final obs), batch-end continuation bootstraps
        # V(current input) — no reset injection is performed for bootstrap.
        with torch.no_grad():
            if pending_reset:
                v_boot = 0.0  # last step was terminal (masked) — unused
            else:
                x_boot = build_input(cur_state, t_idx, q_ref, dq_ref, phase_sc, T)
                v_boot = float(value(rms.normalize(x_boot)))
        gamma, lam = 0.99, 0.95
        adv = compute_gae(br, bv, b_nonterm, b_next_v, v_boot, gamma, lam)
        ret = adv + bv
        # bias monitor: advantages by reset-phase bucket
        for b in range(N_PHASE_BUCKETS):
            m = b_step_bucket == b
            if bool(m.any()):
                av = adv[m]
                buckets[b]["adv_sum"] += float(av.sum())
                buckets[b]["adv_sq_sum"] += float((av ** 2).sum())
                buckets[b]["adv_n"] += int(m.sum())

        # PPO update (identical to baseline)
        rms.update(bx)
        xn = rms.normalize(bx)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        ploss = vloss = 0.0
        for _ in range(args.ppo_epochs):
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, args.minibatch):
                m = perm[s:s + args.minibatch]
                mu, std = policy(xn[m])
                logp_new = GaussianTanhPolicy.logp_squashed(bu[m], mu, std)
                ratio = torch.exp(logp_new - blp[m])
                pl = -torch.min(ratio * adv_n[m],
                                ratio.clamp(1 - 0.2, 1 + 0.2) * adv_n[m]).mean()
                vl = (value(xn[m]) - ret[m]).pow(2).mean()
                loss = pl + 0.5 * vl
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value.parameters()), 1.0)
                opt.step()
                ploss += float(pl.detach())
                vloss += float(vl.detach())
        n_up = args.ppo_epochs * math.ceil(N / args.minibatch)
        global_steps += N
        rec = {
            "batch": batch, "global_steps": global_steps,
            "mean_track_err": float(b_err.mean()),
            "track_err_p95": float(torch.quantile(b_err, 0.95)),
            "termination_reasons": reason_counts,
            "outcome_counts": {k: b_outcome.count(k) for k in OUTCOMES},
            "wrap_count": wrap_count,
            "clamp_frac_by_joint": {n: float(b_clamp_frac_joint[j] / N)
                                    for j, n in enumerate(joint_names)},
            "clamp_frac_overall": float(b_clamp_frac_joint.sum() / (N * ACT_DIM)),
            "max_preclamp_violation_rad": b_max_viol,
            "qdes_out_of_band_steps": clamp_stats["qdes_out_of_band_steps"],
            "track_err_last32": float(b_err[-32:].mean()),
            "clip_frac_presquash": float(b_clip.mean()),
            "mean_std": float(torch.exp(policy.log_std.detach().clamp(-2, 0.5)).mean()),
            "policy_loss": ploss / n_up, "value_loss": vloss / n_up,
            "episodes_started": len(ep_returns),
            "completion_rate": (ep_compls / len(ep_returns)) if ep_returns else None,
            "mean_episode_return": float(np.mean(ep_returns)) if ep_returns else None,
            "mean_episode_track_err": float(np.mean(ep_err_means)) if ep_err_means else None,
            "collect_seconds": collect_s, "update_seconds": time.time() - t0 - collect_s,
            "steps_per_s_total": global_steps / (time.time() - t_start),
        }
        batch_log.append(rec)
        logjl.write(json.dumps(rec) + "\n")
        logjl.flush()
        print(f"[cf] batch {batch}/{n_batches} err={rec['mean_track_err']:.4f} "
              f"succ={rec['outcome_counts']['success']} cat={rec['outcome_counts']['catastrophic']} "
              f"trunc={rec['outcome_counts']['truncate']} wraps={rec['wrap_count']} "
              f"clamp={rec['clamp_frac_overall']:.3f} "
              f"{rec['steps_per_s_total']:.0f} st/s", flush=True)
        if batch % args.ckpt_every == 0:
            save_ckpt(f"ckpt_batch{batch}.pth", batch, global_steps,
                      {"wrap_count": wrap_count, "qdes_margin": QDES_MARGIN,
                       "pkl": str(pkl_path)})

    if train_mode:
        save_ckpt("ckpt_final.pth", n_batches, global_steps,
                  {"wrap_count": wrap_count, "qdes_margin": QDES_MARGIN,
                   "pkl": str(pkl_path)})
    total_train_s = time.time() - t_start

    # ---- deterministic evaluation (SAME pkl reference, SAME clamp pipeline) ----
    policy.eval()
    inner.is_evaluating = True  # pushes OFF (repo eval convention)
    eval_seeds = []
    agg_curve = np.full((args.eval_seeds, T), np.nan)
    for k in range(args.eval_seeds):
        s = args.seed + 1000 + k
        seed_all(s)
        torch.manual_seed(s)
        cur = inject_episode(wrapped, inner, 0)  # phase 0 == canonical sequence start
        prev_a_env = torch.zeros(ACT_DIM, device=dev)
        per_err, per_worst, per_clip, per_ratio, per_viol, per_tsat = [], [], [], [], [], []
        per_joint_rows = []
        eval_clamp_frac = 0.0
        first_fail, fail_reasons_eval = None, []
        success_at_end = False
        frames = 0
        err_exceed_first = None
        for t in range(T):
            x_raw = build_input(cur, t, q_ref, dq_ref, phase_sc, T)
            with torch.no_grad():
                mu, _ = policy(rms.normalize(x_raw))
            u = mu
            a = torch.tanh(u)
            default_total, gain = live_action_map()
            q_des, a_env, d_raw, d_safe, clamped, viol = qdes_residual_clamp(
                a, q_ref[t], default_total, gain, lo, hi)
            obs, _, term, _, _ = wrapped.step(a_env[None], to_numpy=False)
            frames = t + 1
            q_true = inner.simulator.dof_pos[0].float()
            err = (q_true - q_ref[min(t + 1, T - 1)]).abs()
            tau = torch.from_numpy(inner.simulator.data.ctrl[-inner.num_dof:].copy()).to(dev).float()
            obs_state = obs["state"][0].to(dev).float()
            per_err.append(float(err.mean()))
            per_worst.append(float(err.max()))
            per_joint_rows.append(err.cpu())
            per_clip.append(float((u.abs() > 1).any()))
            eval_clamp_frac += float(clamped.float().mean())
            per_ratio.append(float((tau.abs() / torque_limits).max()))
            per_tsat.append(float((tau.abs() >= 0.98 * torque_limits).float().mean()))
            per_viol.append(int(((q_true < lo) | (q_true > hi)).any()))
            prev_a_env = a_env
            cur = obs_state
            if err_exceed_first is None and bool((err > ERR_TERM).any()):
                err_exceed_first = t + 1
            reasons = []
            if bool(term.any()):
                reasons.append("env_termination")
            reasons += fail_reasons(inner, q_true, err, obs_state, forbidden_idx,
                                    lo, hi, FALL_MIN_H, FALL_GXY)
            if reasons:
                if first_fail is None:
                    first_fail = t + 1
                    fail_reasons_eval = reasons
                if not (args.eval_no_stop and "env_termination" not in reasons):
                    break  # diagnostic no-stop mode continues through rule failures
            elif t == T - 1:
                success_at_end = True
        # success-at-end is a NECESSARY condition for completion
        completed = bool(success_at_end and frames == T)
        pj = torch.stack(per_joint_rows)
        f1 = pj[0]
        f1_arg = int(f1.argmax())
        eval_seeds.append({
            "seed": s, "completed": completed, "frames_survived": frames,
            "first_rule_fail_frame": first_fail, "fail_reasons": fail_reasons_eval,
            "success_at_end": success_at_end,
            "env_terminated_at": (first_fail if "env_termination" in fail_reasons_eval else None),
            "err_exceed_first_frame": err_exceed_first,
            "frame1_worst_joint": joint_names[f1_arg],
            "frame1_worst_joint_err": float(f1[f1_arg]),
            "mean_track_err": float(np.mean(per_err)),
            "track_err_p95": float(np.quantile(per_err, 0.95)),
            "worst_joint_err_mean": float(np.mean(per_worst)),
            "worst_joint_err_max": float(np.max(per_worst)),
            "err_exceed_0p8_frames": int(sum(e > ERR_TERM for e in per_err)),
            "presquash_clip_fraction": float(np.mean(per_clip)),
            "qdes_clamp_fraction": eval_clamp_frac / max(1, frames),
            "torque_ratio_max": float(np.max(per_ratio)),
            "torque_ratio_mean_of_frame_max": float(np.mean(per_ratio)),
            "torque_saturation_fraction_098": float(np.mean(per_tsat)),
            "joint_limit_violation_frames": int(sum(per_viol)),
            "per_joint_mean_abs_err": [float(x) for x in pj.mean(0)],
            "per_joint_max_abs_err": [float(x) for x in pj.max(0).values],
            "per_frame_mean_err": per_err,
        })
        agg_curve[k] = np.array(per_err + [np.nan] * (T - len(per_err)))
        print(f"[cf] eval seed {s}: completed={completed} frames={frames} "
              f"success_at_end={success_at_end} err={eval_seeds[-1]['mean_track_err']:.4f}",
              flush=True)
    inner.is_evaluating = False
    policy.train()

    n_comp = sum(e["completed"] for e in eval_seeds)
    tally: dict[str, int] = {}
    for e in eval_seeds:
        for rs in e["fail_reasons"]:
            tally[rs] = tally.get(rs, 0) + 1
    with np.errstate(invalid="ignore"):
        curve = np.nanmean(agg_curve, axis=0)
    n_per_frame = int(np.min(np.sum(np.isfinite(agg_curve), axis=0)))

    bias_table = {}
    for b in range(N_PHASE_BUCKETS):
        d = buckets[b]
        lo_f = b * T // N_PHASE_BUCKETS
        hi_f = (b + 1) * T // N_PHASE_BUCKETS - 1
        n = d["episodes"]
        an = d["adv_n"]
        adv_mean = d["adv_sum"] / an if an else None
        adv_var = (d["adv_sq_sum"] / an - adv_mean ** 2) if an and adv_mean is not None else None
        bias_table[f"bucket{b}_phases[{lo_f},{hi_f}]"] = {
            "episodes": n,
            "success_rate": (d["success"] / n) if n else None,
            "mean_episode_length": (d["len_sum"] / n) if n else None,
            "mean_episode_track_err": (d["err_sum"] / n) if n else None,
            "advantage_mean": adv_mean,
            "advantage_std": (float(math.sqrt(max(adv_var, 0.0))) if adv_var is not None else None),
            "advantage_n": an,
        }

    eval_agg = {
        "n_seeds": args.eval_seeds,
        "completion_rule": (f"success-at-end NECESSARY: completed <=> survived to frame T "
                            "AND success fired at the last frame AND none of the catastrophic "
                            "rules fired at any frame (env termination; any-joint |q-q_ref| "
                            "> 0.8; base-height fall < 0.2 m; orientation fall |pg_x|/|pg_y| "
                            "> 0.8; non-foot contact > 1 N; NaN/Inf; joint hard limit)"),
        "eval_mode": ("eval-only, no-stop diagnostic" if (args.eval_no_stop and args.eval_ckpt)
                      else "standard"),
        "completion_rate": n_comp / args.eval_seeds,
        "completion_wilson95": wilson_ci(n_comp, args.eval_seeds),
        "mean_track_err_over_seeds": float(np.mean([e["mean_track_err"] for e in eval_seeds])),
        "track_err_p95_over_seeds": float(np.mean([e["track_err_p95"] for e in eval_seeds])),
        "worst_joint_err_mean_over_seeds": float(np.mean([e["worst_joint_err_mean"] for e in eval_seeds])),
        "worst_joint_err_max_over_seeds": float(np.max([e["worst_joint_err_max"] for e in eval_seeds])),
        "per_joint_mean_abs_err_table": {
            n: float(np.mean([e["per_joint_mean_abs_err"][j] for e in eval_seeds]))
            for j, n in enumerate(joint_names)},
        "per_joint_max_abs_err_table": {
            n: float(np.max([e["per_joint_max_abs_err"][j] for e in eval_seeds]))
            for j, n in enumerate(joint_names)},
        "per_joint_note": "mean over eval seeds of time-mean |q-q_ref| per joint (max: time-max, worst seed)",
        "mean_frames_survived": float(np.mean([e["frames_survived"] for e in eval_seeds])),
        "mean_qdes_clamp_fraction": float(np.mean([e["qdes_clamp_fraction"] for e in eval_seeds])),
        "first_fail_frame_by_seed": [e["first_rule_fail_frame"] for e in eval_seeds],
        "frame1_worst_joint_counts": {
            n: sum(1 for e in eval_seeds if e["frame1_worst_joint"] == n)
            for n in set(e["frame1_worst_joint"] for e in eval_seeds)},
        "termination_reason_counts": tally,
        "err_exceed_first_frame_by_seed": [e["err_exceed_first_frame"] for e in eval_seeds],
        "per_frame_track_err_curve_nanmean": [None if not np.isfinite(x) else float(x) for x in curve],
        "per_frame_curve_min_seed_count": n_per_frame,
    }
    headline = {
        "arm": args.arm + "_cf",
        "completion_rate": eval_agg["completion_rate"],
        "completion_wilson95": eval_agg["completion_wilson95"],
        "mean_track_err": eval_agg["mean_track_err_over_seeds"],
        "track_err_p95": eval_agg["track_err_p95_over_seeds"],
        "worst_joint_err_mean": eval_agg["worst_joint_err_mean_over_seeds"],
        "worst_joint_err_max": eval_agg["worst_joint_err_max_over_seeds"],
        "n_seeds": args.eval_seeds,
        "train_steps": global_steps,
        "wrap_count": wrap_count,
        "reference": f"{clip_name} ({T} frames @ {clip_fps} fps, CANONICAL sequence)",
        "baseline_mean_track_err": args.openloop_baseline_mean,
        "baseline_worst_joint_err_mean": args.openloop_baseline_worst,
        "baseline_source": args.openloop_baseline_source,
        "baseline_note": ("null = no open-loop baseline supplied for THIS clip; "
                          "the 0.1120/0.9896 bounce_sway_13 Step-0 numbers previously "
                          "hardcoded here were WRONG provenance for ss4 runs"),
    }

    wall_clock_s = total_train_s
    throughput = {
        "num_envs": 1,
        "rollout_length": episode_cap,
        "environment_steps": global_steps if train_mode else int(ck.get("global_steps", 0)),
        "aggregate_transitions": global_steps if train_mode else int(ck.get("global_steps", 0)),
        "wall_clock_seconds": wall_clock_s if train_mode else None,
        "env_step_calls_per_second": ((global_steps / collect_time_total)
                                      if (train_mode and collect_time_total > 0) else None),
        "overall_steps_per_second_including_updates": (
            global_steps / wall_clock_s if (train_mode and wall_clock_s > 0) else None),
        "note": ("single-env sequential rollouts; None in eval-only mode (no fake "
                 "throughput numbers); profiling deferred until after the 100k gate"),
    }
    gates = {}
    for gate in (100_000, 500_000, 1_000_000, 2_000_000):
        if train_mode and global_steps >= gate and batch_log:
            hit = [r for r in batch_log if r["global_steps"] <= gate]
            r = hit[-1]
            gates[f"{gate//1000}k"] = {
                "global_steps": r["global_steps"],
                "mean_track_err": r["mean_track_err"],
                "completion_rate": r["completion_rate"],
            }
        else:
            gates[f"{gate//1000}k"] = None
    throughput["convergence_gates"] = gates

    report = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "script": "humanoidverse/scripts/fds_train_dance_core_cf.py",
            "project": "fixed_dance_reference_specialist — CONTRACT-FIX Stage A (cf line)",
            "baseline": "results/fds/dance_core_ss4_F_report.json (marker: ss4_01M_baseline_marker.json)",
            "quarantined_objective_v2": "results/fds/dance_core_ss4_Fv2_report.json (NOT comparable; reward changed)",
            "phase_scan": "results/fds/ss4_phase_scan_report.json",
            "arm": args.arm, "seed": args.seed,
            "reference_clip": f"{clip_name} ({T} frames @ {clip_fps} fps, {pkl_path.name})",
            "mode": ("eval-only from checkpoint " + str(args.eval_ckpt)
                     if args.eval_ckpt else "train + deterministic eval"),
            "train_seconds": total_train_s if train_mode else None,
            "steps_per_s_total": (global_steps / total_train_s
                                  if (train_mode and total_train_s > 0) else None),
        },
        "contract": {
            "reference_indexing": ("NON-CYCLIC: current index advances +1 per env step, "
                                   "CLAMPED at T-1, NEVER wraps; lookahead indices clamp at "
                                   "T-1 (CONSISTENT); reaching T-1 => success termination"),
            "wrap_count": wrap_count,
            "termination_split": {
                "success": "idx reaches T-1 with no catastrophic rule on that step; reward "
                           f"= step reward + {SUCCESS_BONUS} (NO death penalty)",
                "catastrophic": "v1 evening set (env term; any-joint err>0.8 divergence line; "
                                "base-height fall; orientation fall; non-foot contact; NaN; "
                                "joint hard limit); reward -5",
                "truncate": f"episode cap ({episode_cap}) without success; no penalty; GAE "
                            "bootstraps V(final observation)",
                "gae": "success/catastrophic are true terminals (mask 0); truncation "
                       "bootstraps V(final obs) never the new episode obs",
                "counts_over_training": outcome_counts,
                "catastrophic_reason_tally": cat_reason_tally if cat_reason_tally else reason_counts,
            },
            "qdes_clamp": {
                "form": ("residual: q_des = q_ref[t] + clip(q_des_raw - q_ref[t], "
                         f"lo+{QDES_MARGIN}-q_ref[t], hi-{QDES_MARGIN}-q_ref[t])"),
                "margin": QDES_MARGIN,
                "margin_policy": "FIXED; never auto-shrunk on reference margin",
                "startup_reference_band_check": {
                    "violations": band_viol, "n_violations": len(band_viol),
                    "tightest_margin_to_band_rad": ref_margins},
                "stats_over_training": {
                    "steps": clamp_stats["steps"],
                    "clamp_fraction_by_joint": {
                        n: float(clamp_stats["clamped_by_joint"][j] / max(1, clamp_stats["steps"]))
                        for j, n in enumerate(joint_names)},
                    "max_preclamp_violation_rad": clamp_stats["max_preclamp_viol"],
                    "mean_abs_residual_raw": (clamp_stats["resid_raw_abs_sum"]
                                              / max(1, clamp_stats["steps"])),
                    "mean_abs_residual_safe": (clamp_stats["resid_safe_abs_sum"]
                                               / max(1, clamp_stats["steps"])),
                    "qdes_out_of_band_steps": clamp_stats["qdes_out_of_band_steps"],
                    "actual_q_limit_note": "actual-q limit violations remain a catastrophic "
                                           "termination (evening set) and are tallied there",
                },
                "hiproll_note": ("if hip_roll clamps persistently: diagnose reference proximity "
                                 "/ residual direction / PD compensation / retarget — do NOT "
                                 "shrink the margin"),
            },
            "reward": (f"UNCHANGED baseline v1 objective: -{W_TRACK}*mean|q-qref| "
                       f"-{W_RATE}*mean|a_env diff| -{W_TERM}*1[catastrophic] "
                       f"+{W_UPRIGHT}*upright; success bonus {SUCCESS_BONUS}; "
                       "no worst-joint term, no logsumexp, no dense changes"),
            "episode": f"random-phase inject (uniform [0,{T-1}]), {episode_cap}-step cap, prev action zeroed",
        },
        "config": {
            "steps_total_requested": args.steps_total, "batches": n_batches,
            "batch_steps": args.batch_steps, "global_steps": global_steps,
            "policy": f"MLP 3x256 tanh-squashed diagonal Gaussian, log_std init {LOG_STD_INIT}",
            "value": "MLP 3x256", "normalizer": "per-dim running mean/std on full 106-d raw input",
            "obs_layout": ("state34 + e0(14) + de(14) + e5(14) + e10(14) + e20(14) + sin/cos "
                           f"phase; lookaheads clamped at frame {T-1}; NON-cyclic current index"),
            "ppo": {"gamma": 0.99, "lam": 0.95, "clip": 0.2, "lr": 3e-4,
                    "epochs": args.ppo_epochs, "minibatch": args.minibatch,
                    "vf_coef": 0.5, "ent_coef": 0.0, "grad_clip": 1.0,
                    "adv_normalized_per_batch": True},
            "env": "build_cpu_env (mujoco, obs noise ON, episodic DR ON, pushes ON in train, "
                   "OFF in eval via is_evaluating)",
        },
        "non_cyclic_bias_monitor": {
            "note": ("observation only: later reset phases are naturally shorter episodes "
                     "(fewer steps to T-1); fixed-length windows decision deferred"),
            "by_reset_phase_bucket": bias_table,
        },
        "batch_log": batch_log,
        "headline": headline,
        "throughput": throughput,
        "eval_aggregate": eval_agg,
        "eval_per_seed": eval_seeds,
    }
    assert not report_path.exists()
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[cf] saved {report_path}", flush=True)
    if logjl is not None:
        logjl.close()


if __name__ == "__main__":
    main()
