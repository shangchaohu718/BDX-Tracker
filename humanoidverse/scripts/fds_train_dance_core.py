"""fixed_dance_reference_specialist — Stage A dance-core trainer (arms F / P).

Independent PPO trainer (does NOT use humanoidverse/train.py) for tracking the
single reference clip re_dancegen_bounce_sway_13 (248 frames @ 50 fps,
pkl: humanoidverse/data/fds_bounce_sway_13.pkl, dof already in robot order).

Arms (preregistration results/fds/armFP_preregistration.json):
  F: fresh MLP 3x256 policy+value trained from scratch.
  P: same architecture/seed; before training, attempt v1-A actor-parameter
     transfer (results/v1-armA-300k/checkpoint) loading ONLY exact-shape
     matching actor tensors; everything else excluded and reported.

Observation vector (106-d, RAW units, per-dim running mean/std normalizer):
  [ obs["state"] 34 = dof_pos(14) dof_vel(14) proj_gravity(3) base_ang_vel(3),
    q_ref[t]-q (14), dq_ref[t]-dq (14),
    q_ref[min(t+5,247)]-q (14), q_ref[min(t+10,247)]-q (14),
    q_ref[min(t+20,247)]-q (14),
    sin(2*pi*t/248), cos(2*pi*t/248) ]
Reference index loops modulo 248 inside an episode; lookahead channels clamp.

Policy: Gaussian (diagonal, state-independent learnable log_std, clamped
[-2, 0.5], INIT -0.7 => std 0.50) -> a = tanh(u); env receives a directly
(its chain multiplies by 5 and clips +-5, i.e. the env-side clip corresponds
to |u| saturating tanh).
pre_normalization_action_clip_fraction is monitored as frac(|u| > 1)
pre-squash (sampled u during training, mean mu during deterministic eval).
[RETUNE NOTE, single documented retune] log_std init 0.0 (std 1.0) was
aborted after smoke + 7 flat batches: episodes lasted ~1.5 steps (any-joint
err > 0.8 one step after phase injection under noise-dominated actions), so
PPO saw almost only termination transitions; std init 0.50 restores usable
trajectory signal. Aborted partial runs: results/fds/dance_core_F*/,
dance_core_P*/ (batch_log.jsonl only, no report).

Reward per step (weights fixed, documented):
  v1: r = -1.0*mean_i|q_i - q_ref_i[t+1]|          (tracking, dominates)
          -0.05*mean_i|a_t - a_{t-1}|              (action smoothness, tanh space)
          -5.0*1[terminated]
          +0.01*clamp(-pg_z, 0, 1)                 (root upright bonus)
  v2 (authorized single objective change, 2026-08-25 ruling "objective_v2"):
      additionally -0.5*max_i|q_i - q_ref_i[t+1]|  (worst-joint penalty; aligns the
      objective with the any-joint completion gate; all other terms unchanged).
  v1 rationale for the change: frame-1 right-ankle veto (worst-joint ~1.0-1.2 rad
  while mean err ~0.11-0.13) replicated across bounce_sway_13 and side_step_4;
  mean-curve improvement does not fix completion.
q/dq/pg for the reward are read from simulator truth (inner.simulator /
inner.projected_gravity), NOT from the noisy obs. Termination (evening ruling
prereg v2, results/fds/armFP_preregistration_v2.json) — fail on ANY of:
env termination signal; any-joint |q_i - q_ref_i| > 0.8 rad; base-height fall
(root_z < termination_min_base_height=0.2); orientation fall (|pg_x| or |pg_y|
> termination_gravity_x/y=0.8); non-foot forbidden contact (any non-feet body
contact-force norm > 1 N, repo threshold); NaN/Inf in state; joint hard-limit
violation (q outside dof_pos limits). Termination reasons are logged per
batch and per eval seed. Episodes: random-phase reset (uniform frame in
[0,247], inject_read_exact_lean), 248-step cap, prev action zeroed. Eval
headline reports together: completion rate (+Wilson 95% CI), mean tracking
err, p95 tracking err, worst-joint err (never completion alone).

PPO: gamma 0.99, lambda 0.95, clip 0.2, lr 3e-4, 10 epochs, minibatch 256,
vf coef 0.5, adv normalized per batch, grad clip 1.0, ent coef 0.0.
Tanh-squashed importance sampling is exact in u-space (a = tanh(u) fixed).

Deterministic eval (10 fixed seeds, inject frame 0, 248 frames, mean action):
per-frame |q-q_ref|, termination, pre-squash clip fraction, torque ratio
(data.ctrl[-14:] vs inner.torque_limits), joint-limit violations; aggregate
mean over seeds + Wilson 95% CI on completion. Pushes disabled during eval
(inner.is_evaluating = True, repo convention); obs noise and episodic DR stay
ON (training obs path), seeded per eval episode so F and P see identical draws.

Outputs (never overwrite; suffix on collision):
  results/fds/{tag}/ckpt_batch{N}.pth every 10 batches + ckpt_final.pth
  results/fds/{tag}/batch_log.jsonl (incremental)
  results/fds/{tag}_report.json  (final, assert-not-exists)
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

PKL = ROOT / "humanoidverse/data/fds_bounce_sway_13.pkl"
ENV_CFG = ROOT / "results/bfm-dynamics-baseline/config.json"
V1A_CKPT = ROOT / "results/v1-armA-300k/checkpoint/model/model.safetensors"
OUT_BASE = ROOT / "results/fds"

T_FRAMES = 248
FPS = 50.0
OBS_STATE_DIM = 34       # dof_pos 14 + dof_vel 14 + proj_gravity 3 + base_ang_vel 3
INPUT_DIM = OBS_STATE_DIM + 5 * 14 + 2   # 106
ACT_DIM = 14

# reward weights (fixed, documented in module docstring)
W_TRACK, W_RATE, W_TERM, W_UPRIGHT = 1.0, 0.05, 5.0, 0.01
W_WORST = 0.5  # objective_v2: worst-joint penalty 0.5*max_j|q_j-q_ref_j|
ERR_TERM = 0.8           # any-joint |q - q_ref| termination threshold [rad]
ANKLE_IDX = {"left_ankle_joint": 4, "right_ankle_joint": 9}  # veto-watch channels

# ---- explicit obs schema (contract file; asserted at startup) ----
OBS_SCHEMA = {
    "_provenance": {
        "doc": "fixed_dance_reference_specialist Stage-A dance-core obs contract",
        "script": "humanoidverse/scripts/fds_train_dance_core.py",
        "preregistration": "results/fds/armFP_preregistration.json",
    },
    "total_dim": INPUT_DIM,
    "channels": {
        "current_state": {"offset": [0, 34],
                          "content": "env obs['state']: dof_pos(14), dof_vel(14), projected_gravity(3), base_ang_vel(3)"},
        "q_err_t": {"offset": [34, 48], "content": "q_ref[t] - q (q from noisy env obs)"},
        "dq_err_t": {"offset": [48, 62], "content": "dq_ref[t] - dq (dq from noisy env obs; dq_ref = central diff @50fps)"},
        "q_err_t_plus_5": {"offset": [62, 76], "content": "q_ref[min(t+5,247)] - q"},
        "q_err_t_plus_10": {"offset": [76, 90], "content": "q_ref[min(t+10,247)] - q"},
        "q_err_t_plus_20": {"offset": [90, 104], "content": "q_ref[min(t+20,247)] - q"},
        "sin_phase": {"offset": [104, 105], "content": "sin(2*pi*t/248)"},
        "cos_phase": {"offset": [105, 106], "content": "cos(2*pi*t/248)"},
    },
    "units": "RAW pre-normalizer; per-dim running mean/std normalizer applied downstream (nothing hand-tuned)",
    "reference_indexing": "t = injected reset frame at episode start, advances +1 per env step mod 248 (loops); lookahead indices CLAMPED to 247 (asserted, never OOB)",
    "phase_alignment": "phase channels must equal (sin,cos)(2*pi*frame/248) of the injected reset frame at episode start (asserted every reset)",
    "stage_B_reset_mix": {
        "active": False,
        "status": "NOT-ACTIVE (Stage B placeholder; Stage-A settings must not silently become final config)",
        "mix": {"dance_random_phase": 0.5, "intro_outro": 0.2, "home_stand_phi0": 0.2, "perturbation": 0.1},
        "stage_A_reset": "100% dance random-phase reset (uniform frame in [0,247])",
    },
    "stage_plan": {
        "current": "Stage A bounce_sway_13 F/P — selects trainer+init ONLY",
        "next": "side_step_4 Stage A (pure dance-core, random-phase, no intro/outro); Stage B untouched",
    },
}
SCHEMA_PATH = OUT_BASE / "dance_core_obs_schema.json"


def write_obs_schema():
    """Write the shared obs-schema contract; if it exists it must match exactly."""
    if SCHEMA_PATH.exists():
        existing = json.loads(SCHEMA_PATH.read_text())
        assert existing == OBS_SCHEMA, f"{SCHEMA_PATH} exists with DIFFERENT content"
    else:
        SCHEMA_PATH.write_text(json.dumps(OBS_SCHEMA, indent=2) + "\n")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------
class RunningMeanStd:
    """Per-dim running mean/var normalizer on the FULL raw input vector."""

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


LOG_STD_INIT = -0.7  # std 0.50; see RETUNE NOTE in module docstring


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
        """log pi(a=tanh(u)) = log N(u|mu,std) - sum_i log(1 - tanh(u_i)^2)."""
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
# arm P: v1-A actor transfer (exact shape match only)
# --------------------------------------------------------------------------
def try_v1a_transfer(policy: GaussianTanhPolicy) -> dict:
    report = {
        "source": str(V1A_CKPT),
        "attempted": False, "loaded": [], "loaded_param_count": 0,
        "policy_total_params": sum(p.numel() for p in policy.parameters()),
        "loaded_fraction": 0.0, "excluded_policy_layers": {}, "excluded_ckpt_actor_tensors": {},
        "below_10pct_threshold": None, "note": "",
    }
    try:
        from safetensors.torch import load_file
        sd = load_file(str(V1A_CKPT))
        report["attempted"] = True
    except Exception as e:  # noqa: BLE001
        report["note"] = f"checkpoint load failed ({type(e).__name__}: {e}); proceeding fresh"
        report["below_10pct_threshold"] = True
        return report
    actor = {k[len("_actor."):]: v for k, v in sd.items() if k.startswith("_actor.")}
    # semantic-axis restriction (prereg): only the state-obs encoder (embed_s,
    # consumes the same obs contract) and the action head (policy) are
    # candidates; embed_z (latent encoder, different input contract) excluded.
    pool = {k: v for k, v in actor.items()
            if k.startswith("embed_s.") or k.startswith("policy.")}
    excluded_pools = sorted(set(k.split(".")[0] for k in actor) - {"embed_s", "policy"})

    used: set[str] = set()
    with torch.no_grad():
        for name, p in policy.named_parameters():
            role = name.split(".")[-1]  # "weight" / "bias" / "log_std"
            cand = None
            for cn, t in pool.items():
                if cn in used or tuple(t.shape) != tuple(p.shape):
                    continue
                # same role class: bias-like names must not feed log_std etc.
                if role == "log_std":
                    continue
                if (role == "weight") != cn.endswith(".weight"):
                    continue
                if (role == "bias") != cn.endswith(".bias"):
                    continue
                cand = cn
                break
            if cand is not None:
                p.copy_(actor[cand].to(p.device, p.dtype))
                used.add(cand)
                report["loaded"].append({"policy_param": name, "ckpt_tensor": cand,
                                         "shape": list(p.shape), "numel": p.numel()})
            else:
                report["excluded_policy_layers"][name] = {
                    "shape": list(p.shape),
                    "reason": "no exact-shape role-matched v1-A actor tensor",
                }
    for cn, t in pool.items():
        if cn not in used:
            report["excluded_ckpt_actor_tensors"][cn] = list(t.shape)
    report["excluded_ckpt_submodules_by_semantics"] = {
        m: "excluded pool (not the state-obs encoder / action head; prereg semantic-axis rule)"
        for m in excluded_pools}
    report["loaded_param_count"] = sum(d["numel"] for d in report["loaded"])
    report["loaded_fraction"] = report["loaded_param_count"] / max(1, report["policy_total_params"])
    report["below_10pct_threshold"] = report["loaded_fraction"] < 0.10
    loaded_names = [d["policy_param"] for d in report["loaded"]]
    report["first_layer_loaded"] = "trunk.0.weight" in loaded_names
    report["action_head_loaded"] = {
        "mean.weight": "mean.weight" in loaded_names,
        "mean.bias": "mean.bias" in loaded_names,
    }
    report["interpretation"] = (
        "near-zero shape match => arm P is a trainer CONTROL (different init), "
        "NOT a pretraining comparison; the F-vs-P result cannot speak to BFM "
        "pretraining quality"
    )
    report["conclusion_wording_required"] = (
        "if P does not win beyond CI: 'Fresh chosen for structural simplicity' "
        "(never 'BFM pretraining is worse')"
    )
    report["note"] = (
        "v1-A actor is 512-wide with a 240-d state-encoder input; this policy is "
        "256-wide with a 106-d input, so shape-compatible transfer is expected to be "
        "near zero. Proceeding per instruction (report honestly, prereg tie rule "
        "already prefers FRESH within CI)."
    )
    return report


# --------------------------------------------------------------------------
# rollout helpers
# --------------------------------------------------------------------------
def build_reference(device, pkl_path: Path):
    """Load the single-clip reference pkl; returns (q_ref, dq_ref, T, fps, name)."""
    data = joblib.load(pkl_path)
    assert len(data) == 1, f"expected single-motion pkl, got keys {list(data.keys())}"
    name, entry = next(iter(data.items()))
    q_ref = torch.as_tensor(entry["dof"], dtype=torch.float32, device=device)
    T, n_dof = q_ref.shape
    assert n_dof == ACT_DIM, n_dof
    fps = float(entry["fps"])
    dof_np = np.asarray(entry["dof"])
    dq = np.zeros_like(dof_np)
    dq[1:-1] = (dof_np[2:] - dof_np[:-2]) * (fps / 2.0)
    dq[0], dq[-1] = dq[1], dq[-2]
    dq_ref = torch.as_tensor(dq, dtype=torch.float32, device=device)
    return q_ref, dq_ref, int(T), fps, name


def build_input(state34: torch.Tensor, t_idx: int, q_ref, dq_ref, phase_sc, T: int) -> torch.Tensor:
    """[106] raw input; state34 is the (noisy) env obs, refs are exact."""
    assert 0 <= t_idx < T, f"reference index OOB: {t_idx}"
    q, dq = state34[:14], state34[14:28]
    e = [q_ref[t_idx] - q, dq_ref[t_idx] - dq]
    for k in (5, 10, 20):
        idx = min(t_idx + k, T - 1)  # clamped lookahead, never OOB
        assert 0 <= idx < T
        e.append(q_ref[idx] - q)
    return torch.cat([state34] + e + [phase_sc[t_idx]])


class PhaseTable:
    def __init__(self, device, T: int):
        ang = 2 * math.pi * torch.arange(T, device=device) / T
        self.tab = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)  # [T,2]

    def __getitem__(self, t):
        return self.tab[t]


def inject_episode(wrapped, inner, frame: int):
    """Random/chosen-phase reset via the lean exact read; returns obs state."""
    from humanoidverse.scripts.generate_expert_sim_pool import inject_read_exact_lean
    ids = torch.zeros(1, dtype=torch.long, device=inner.device)
    times = torch.full((1,), frame / FPS, device=inner.device)
    obs = inject_read_exact_lean(wrapped, ids, times, settle_reps=2)
    return obs["state"][0].to(inner.device).float()


def fail_reasons(inner, q_true, err, obs_state34, forbidden_idx, lo, hi,
                 min_h: float, gxy: float) -> list:
    """Evening-ruling termination rule (any of these fails the step)."""
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


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["F", "P"], default="F")
    ap.add_argument("--tag", default=None, help="output tag (default: dance_core_{arm})")
    ap.add_argument("--steps-total", type=int, default=102400)
    ap.add_argument("--batch-steps", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=4728)
    ap.add_argument("--eval-seeds", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--ppo-epochs", type=int, default=10)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--objective", choices=["v1", "v2"], default="v2",
                    help="v2 = + worst-joint penalty 0.5 (authorized single change); "
                         "v1 = original mean-tracking objective (legacy runs)")
    ap.add_argument("--pkl", default=str(PKL), help="single-clip reference pkl")
    ap.add_argument("--eval-ckpt", default=None,
                    help="eval-only mode: load this checkpoint (no training)")
    ap.add_argument("--eval-no-stop", action="store_true",
                    help="diagnostic: keep stepping through extended-rule failures "
                         "(stop only on env termination); for per-joint tables")
    return ap.parse_args()


def unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    i = 1
    while True:
        q = p.with_name(f"{p.name}_{i}")
        if not q.exists():
            return q
        i += 1


def main():
    args = parse_args()
    pkl_path = Path(args.pkl).resolve()
    is_default_clip = pkl_path == PKL.resolve()
    if args.eval_ckpt:
        out_dir = Path(args.eval_ckpt).resolve().parent  # provenance only, never written
        report_path = OUT_BASE / f"{args.tag or 'evalonly'}_evalonly_report.json"
    else:
        tag = args.tag or f"dance_core_{args.arm}"
        out_dir = unique_path(OUT_BASE / tag)
        out_dir.mkdir(parents=True)
        report_path = OUT_BASE / f"{out_dir.name}_report.json"
    assert not report_path.exists()
    # obs schema contract: exact dim, offsets, clamped lookaheads (asserted
    # continuously in build_input), phase alignment (asserted every reset).
    # The on-disk contract file is written for the preregistered default clip
    # (bounce_sway_13, 248 frames); other clips share identical offsets and
    # only differ in the phase denominator (t/T_clip).
    assert OBS_SCHEMA["total_dim"] == INPUT_DIM == 106
    schema_written = False
    if is_default_clip and not args.eval_ckpt:
        write_obs_schema()
        schema_written = True
    print(f"[fds] arm={args.arm} out_dir={out_dir} report={report_path} "
          f"pkl={pkl_path.name} eval_only={bool(args.eval_ckpt)} "
          f"no_stop={args.eval_no_stop}", flush=True)

    os.environ["BFM_POOL_DATA"] = str(pkl_path)
    from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env

    cfg = json.loads(ENV_CFG.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapped = build_cpu_env(cfg, torch.device("cuda") if device.type == "cuda" else device)
    inner = wrapped._env
    dev = inner.device

    q_ref, dq_ref, T, clip_fps, clip_name = build_reference(dev, pkl_path)
    phase_sc = PhaseTable(dev, T)
    print(f"[fds] clip={clip_name} frames={T} fps={clip_fps}", flush=True)
    rcfg = yaml.safe_load(open(ROOT / "humanoidverse/config/robot/bdx/bdx_14dof.yaml"))["robot"]
    lo = torch.tensor(rcfg["dof_pos_lower_limit_list"], device=dev)
    hi = torch.tensor(rcfg["dof_pos_upper_limit_list"], device=dev)
    torque_limits = inner.torque_limits.flatten()
    # evening-ruling termination params (repo conventions, read from live env)
    ts = inner.config.termination_scales
    FALL_MIN_H = float(ts.termination_min_base_height)   # 0.2
    FALL_GXY = float(ts.termination_gravity_x)           # 0.8
    feet_set = set(inner.feet_indices.tolist())
    forbidden_idx = torch.tensor([i for i in range(len(inner.body_names))
                                  if i not in feet_set], dtype=torch.long, device=dev)
    print(f"[fds] termination: min_h={FALL_MIN_H} gxy={FALL_GXY} "
          f"forbidden_bodies={len(forbidden_idx)}/{len(inner.body_names)}", flush=True)

    # ---- nets (identical seeding across arms) ----
    seed_all(args.seed)
    torch.manual_seed(args.seed)  # re-seed after env build consumed RNG
    policy = GaussianTanhPolicy(INPUT_DIM).to(dev)
    value = ValueNet(INPUT_DIM).to(dev)
    transfer_report = None
    if args.arm == "P":
        transfer_report = try_v1a_transfer(policy)
        print(f"[fds] arm P transfer: loaded {transfer_report['loaded_param_count']}"
              f"/{transfer_report['policy_total_params']} params "
              f"({transfer_report['loaded_fraction']:.5f})", flush=True)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-4)
    rms = RunningMeanStd(INPUT_DIM, dev)
    phase_rng = np.random.default_rng(args.seed)
    joint_names = list(rcfg["dof_names"])
    train_mode = args.eval_ckpt is None

    if not train_mode:
        ck = torch.load(args.eval_ckpt, map_location=dev)
        policy.load_state_dict(ck["policy"])
        value.load_state_dict(ck["value"])
        rms.mean = ck["rms"]["mean"].to(dev)
        rms.var = ck["rms"]["var"].to(dev)
        rms.count = float(ck["rms"]["count"])
        print(f"[fds] eval-only: loaded {args.eval_ckpt} "
              f"(arm {ck.get('arm')}, steps {ck.get('global_steps')})", flush=True)

    def save_ckpt(name: str, batch: int, gsteps: int):
        torch.save({
            "batch": batch, "global_steps": gsteps, "arm": args.arm,
            "policy": policy.state_dict(), "value": value.state_dict(),
            "opt": opt.state_dict(), "rms": rms.state(),
        }, out_dir / name)

    # ---- rollout state ----
    pending_reset = True
    t_idx, steps_in_ep, prev_a = 0, 0, torch.zeros(ACT_DIM, device=dev)
    cur_state = None
    ep_ret, ep_steps, ep_errs, ep_completions, ep_starts = 0.0, 0, [], 0, 0

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
        bd = torch.zeros(N, device=dev)
        bv = torch.zeros(N, device=dev)
        b_err = torch.zeros(N, device=dev)
        b_clip = torch.zeros(N, device=dev)
        b_worst = torch.zeros(N, device=dev)
        b_ankle_a = [0.0, 0.0]  # sum |a| over steps: [left_ankle, right_ankle]
        reason_counts: dict[str, int] = {}
        ep_returns, ep_compls, ep_err_means = [], 0, []

        t0 = time.time()
        i = 0
        while i < N:
            if pending_reset:
                frame = int(phase_rng.integers(0, T))
                cur_state = inject_episode(wrapped, inner, frame)
                t_idx, steps_in_ep = frame, 0
                prev_a = torch.zeros(ACT_DIM, device=dev)
                ep_ret, ep_steps, ep_err = 0.0, 0, []
                ep_starts += 1
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
            obs, _, term, trunc, _ = wrapped.step(a[None], to_numpy=False)
            t_next = (t_idx + 1) % T
            q_true = inner.simulator.dof_pos[0].float()
            obs_state = obs["state"][0].to(dev).float()
            err = (q_true - q_ref[min(t_idx + 1, T - 1)]).abs()
            track_err = err.mean()
            reasons = []
            if bool(term.any()):
                reasons.append("env_termination")
            reasons += fail_reasons(inner, q_true, err, obs_state, forbidden_idx,
                                    lo, hi, FALL_MIN_H, FALL_GXY)
            term_flag = len(reasons) > 0
            upright = torch.clamp(-inner.projected_gravity[0, 2], 0.0, 1.0)
            r = (-W_TRACK * track_err
                 - W_RATE * (a - prev_a).abs().mean()
                 - W_TERM * float(term_flag)
                 + W_UPRIGHT * upright)
            if args.objective == "v2":
                r = r - W_WORST * err.max()  # worst-joint penalty
            bx[i], bu[i], blp[i], br[i], bd[i], bv[i] = x_raw, u, logp, r, float(term_flag), v
            b_err[i], b_clip[i], b_worst[i] = track_err, float((u.abs() > 1).any()), err.max()
            b_ankle_a[0] += float(a[ANKLE_IDX["left_ankle_joint"]].abs())
            b_ankle_a[1] += float(a[ANKLE_IDX["right_ankle_joint"]].abs())
            for rs in reasons:
                reason_counts[rs] = reason_counts.get(rs, 0) + 1
            ep_ret += float(r)
            ep_steps += 1
            ep_err.append(float(track_err))
            prev_a, cur_state, t_idx = a, obs_state, t_next
            i += 1
            if term_flag or ep_steps >= T:
                if not term_flag:
                    ep_completions += 1
                    ep_compls += 1
                ep_returns.append(ep_ret)
                ep_err_means.append(float(np.mean(ep_err)))
                pending_reset = True
        collect_s = time.time() - t0
        collect_time_total += collect_s
        if batch == 1:
            assert torch.isfinite(bx).all(), "non-finite obs input channels on first batch"
            assert torch.isfinite(br).all() and torch.isfinite(bv).all(), \
                "non-finite rewards/values on first batch"

        # GAE(0.95) with continuing-process bootstrap
        with torch.no_grad():
            if pending_reset:
                frame = int(phase_rng.integers(0, T))
                cur_state = inject_episode(wrapped, inner, frame)
                pending_reset = False
                ep_starts += 1
                ep_ret, ep_steps, ep_err = 0.0, 0, []
                t_idx, steps_in_ep = frame, 0
                prev_a = torch.zeros(ACT_DIM, device=dev)
            # NOTE: the injection above starts a NEW episode (counted next batch)
            x_boot = build_input(cur_state, t_idx, q_ref, dq_ref, phase_sc, T)
            v_boot = float(value(rms.normalize(x_boot)))
        gamma, lam = 0.99, 0.95
        adv = torch.zeros(N, device=dev)
        gae = 0.0
        for j in reversed(range(N)):
            nonterm = 1.0 - bd[j]
            next_v = bv[j + 1] if j + 1 < N else v_boot
            delta = br[j] + gamma * next_v * nonterm - bv[j]
            gae = delta + gamma * lam * nonterm * gae
            adv[j] = gae
        ret = adv + bv

        # PPO update
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
            "worst_joint_err_mean": float(b_worst.mean()),
            "worst_joint_err_max": float(b_worst.max()),
            "termination_reasons": reason_counts,
            "mean_abs_action_left_ankle": b_ankle_a[0] / N,
            "mean_abs_action_right_ankle": b_ankle_a[1] / N,
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
        print(f"[fds] batch {batch}/{n_batches} err={rec['mean_track_err']:.4f} "
              f"compl={rec['completion_rate']} ret={rec['mean_episode_return']:.2f} "
              f"clip={rec['clip_frac_presquash']:.3f} std={rec['mean_std']:.3f} "
              f"{rec['steps_per_s_total']:.0f} st/s", flush=True)
        if batch % args.ckpt_every == 0:
            save_ckpt(f"ckpt_batch{batch}.pth", batch, global_steps)

    if train_mode:
        save_ckpt("ckpt_final.pth", n_batches, global_steps)
    total_train_s = time.time() - t_start

    # ---- deterministic evaluation ----
    policy.eval()
    inner.is_evaluating = True  # disable pushes (repo eval convention)
    eval_seeds = []
    agg_curve = np.full((args.eval_seeds, T), np.nan)
    for k in range(args.eval_seeds):
        s = args.seed + 1000 + k
        seed_all(s)
        torch.manual_seed(s)
        cur = inject_episode(wrapped, inner, 0)
        prev_a = torch.zeros(ACT_DIM, device=dev)
        per_err, per_clip, per_ratio, per_viol, per_tsat, per_worst = [], [], [], [], [], []
        per_joint_rows = []
        terminated_at, frames = None, 0
        err_exceed_first = None  # first frame where any-joint |q-q_ref| > 0.8
        fail_reasons_eval: list = []
        for t in range(T):
            x_raw = build_input(cur, t, q_ref, dq_ref, phase_sc, T)
            with torch.no_grad():
                mu, _ = policy(rms.normalize(x_raw))
            u = mu
            a = torch.tanh(u)
            obs, _, term, _, _ = wrapped.step(a[None], to_numpy=False)
            frames = t + 1
            q_true = inner.simulator.dof_pos[0].float()
            err = (q_true - q_ref[min(t + 1, T - 1)]).abs()
            tau = torch.from_numpy(inner.simulator.data.ctrl[-inner.num_dof:].copy()).to(dev).float()
            obs_state = obs["state"][0].to(dev).float()
            per_err.append(float(err.mean()))
            per_worst.append(float(err.max()))
            per_joint_rows.append(err.cpu())
            per_clip.append(float((u.abs() > 1).any()))
            per_ratio.append(float((tau.abs() / torque_limits).max()))
            per_tsat.append(float((tau.abs() >= 0.98 * torque_limits).float().mean()))
            per_viol.append(int(((q_true < lo) | (q_true > hi)).any()))
            prev_a = a
            cur = obs_state
            if err_exceed_first is None and bool((err > ERR_TERM).any()):
                err_exceed_first = t + 1
            reasons = []
            if bool(term.any()):
                reasons.append("env_termination")
            reasons += fail_reasons(inner, q_true, err, obs_state, forbidden_idx,
                                    lo, hi, FALL_MIN_H, FALL_GXY)
            if reasons:
                terminated_at = t + 1
                fail_reasons_eval = reasons
                if not (args.eval_no_stop and "env_termination" not in reasons):
                    break  # diagnostic no-stop mode continues through rule failures
        # completion must discriminate trained vs untrained: full T frames
        # with none of the evening-ruling failure modes firing
        completed = terminated_at is None and frames == T
        pj = torch.stack(per_joint_rows)  # [frames, 14]
        f1 = pj[0]
        f1_arg = int(f1.argmax())
        eval_seeds.append({
            "seed": s, "completed": completed, "frames_survived": frames,
            "terminated_at": terminated_at, "termination_reasons": fail_reasons_eval,
            "env_terminated_at": terminated_at if "env_termination" in fail_reasons_eval else None,
            "err_exceed_first_frame": err_exceed_first,
            "frame1_worst_joint": joint_names[f1_arg],
            "frame1_worst_joint_err": float(f1[f1_arg]),
            "mean_track_err": float(np.mean(per_err)),
            "track_err_p95": float(np.quantile(per_err, 0.95)),
            "worst_joint_err_mean": float(np.mean(per_worst)),
            "worst_joint_err_max": float(np.max(per_worst)),
            "err_exceed_0p8_frames": int(sum(e > ERR_TERM for e in per_err)),
            "presquash_clip_fraction": float(np.mean(per_clip)),
            "torque_ratio_max": float(np.max(per_ratio)),
            "torque_ratio_mean_of_frame_max": float(np.mean(per_ratio)),
            "torque_saturation_fraction_098": float(np.mean(per_tsat)),
            "joint_limit_violation_frames": int(sum(per_viol)),
            "per_joint_mean_abs_err": [float(x) for x in pj.mean(0)],
            "per_joint_max_abs_err": [float(x) for x in pj.max(0).values],
            "per_frame_mean_err": per_err,
        })
        agg_curve[k] = np.array(per_err + [np.nan] * (T - len(per_err)))
        print(f"[fds] eval seed {s}: completed={completed} frames={frames} "
              f"err={eval_seeds[-1]['mean_track_err']:.4f}", flush=True)
    inner.is_evaluating = False
    policy.train()

    n_comp = sum(e["completed"] for e in eval_seeds)
    tally: dict[str, int] = {}
    for e in eval_seeds:
        for rs in e["termination_reasons"]:
            tally[rs] = tally.get(rs, 0) + 1
    curve_mat = agg_curve
    with np.errstate(invalid="ignore"):
        curve = np.nanmean(curve_mat, axis=0)
    n_per_frame = int(np.min(np.sum(np.isfinite(curve_mat), axis=0)))
    eval_agg = {
        "n_seeds": args.eval_seeds,
        "completion_rule": (f"survived {T} frames with NONE of the evening-ruling failure "
                            "modes firing: env termination; any-joint |q-q_ref| > 0.8; "
                            "base-height fall (< 0.2 m); orientation fall (|pg_x| or |pg_y| "
                            "> 0.8); non-foot contact > 1 N; NaN/Inf; joint hard limit"),
        "eval_mode": ("eval-only, no-stop diagnostic (rule failures recorded but rollout "
                      "continues; completion still judged by the strict rule)"
                      if args.eval_no_stop else "standard"),
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
        "frame1_worst_joint_counts": {
            n: sum(1 for e in eval_seeds if e["frame1_worst_joint"] == n)
            for n in set(e["frame1_worst_joint"] for e in eval_seeds)},
        "frame1_worst_joint_err_mean": float(np.mean(
            [e["frame1_worst_joint_err"] for e in eval_seeds])),
        "termination_reason_counts": tally,
        "err_exceed_first_frame_by_seed": [e["err_exceed_first_frame"] for e in eval_seeds],
        "err_exceed_fired_count": sum(e["err_exceed_first_frame"] is not None for e in eval_seeds),
        "mean_err_exceed_first_frame_when_fired": (
            float(np.mean([e["err_exceed_first_frame"] for e in eval_seeds
                           if e["err_exceed_first_frame"] is not None]))
            if any(e["err_exceed_first_frame"] is not None for e in eval_seeds) else None),
        "err_exceed_first_frame_note": "transient difficulty signature; None = never exceeded",
        "mean_presquash_clip_fraction": float(np.mean([e["presquash_clip_fraction"] for e in eval_seeds])),
        "mean_torque_ratio_max": float(np.mean([e["torque_ratio_max"] for e in eval_seeds])),
        "mean_joint_limit_violation_frames": float(np.mean([e["joint_limit_violation_frames"] for e in eval_seeds])),
        "per_frame_track_err_curve_nanmean": [None if not np.isfinite(x) else float(x) for x in curve],
        "per_frame_curve_min_seed_count": n_per_frame,
        "note": ("per-frame curve is nan-mean over seeds (frames past a seed's "
                 "termination are NaN); per-seed per_frame_mean_err has exact alignment"),
    }
    # headline: completion NEVER alone (evening ruling)
    headline = {
        "arm": args.arm,
        "completion_rate": eval_agg["completion_rate"],
        "completion_wilson95": eval_agg["completion_wilson95"],
        "mean_track_err": eval_agg["mean_track_err_over_seeds"],
        "track_err_p95": eval_agg["track_err_p95_over_seeds"],
        "worst_joint_err_mean": eval_agg["worst_joint_err_mean_over_seeds"],
        "worst_joint_err_max": eval_agg["worst_joint_err_max_over_seeds"],
        "n_seeds": args.eval_seeds,
        "train_steps": global_steps,
        "openloop_pd_baseline_mean_err": 0.117,
    }

    # throughput accounting (evening ruling #2) — no trainer rewrite
    wall_clock_s = total_train_s
    env_step_calls = global_steps
    throughput = {
        "num_envs": 1,
        "rollout_length": T_FRAMES,
        "environment_steps": env_step_calls,
        "aggregate_transitions": env_step_calls * 1,
        "wall_clock_seconds": wall_clock_s,
        "env_step_calls_per_second": (env_step_calls / collect_time_total
                                      if collect_time_total > 0 else None),
        "aggregate_transitions_per_second": (env_step_calls / collect_time_total
                                             if collect_time_total > 0 else None),
        "overall_steps_per_second_including_updates": global_steps / wall_clock_s,
        "note": ("the ~37-38 steps/s figure was per single-env step call; "
                 "num_envs=1 so aggregate_transitions == environment_steps "
                 "(sequential single-env rollouts; each loop iteration is one "
                 "wrapped.step call)"),
    }
    # convergence gates read from curves (no pre-commitment to 5-20M)
    gates = {}
    for gate in (100_000, 500_000, 1_000_000, 2_000_000):
        if global_steps >= gate and batch_log:
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
    throughput["gate_note"] = (
        f"budget {global_steps} steps; gates beyond budget are null — extend the "
        "budget only if curves are still improving at the current budget")

    next_stage_a = {
        "clip": "side_step_4",
        "stage": "A (pure dance-core, random-phase reset, NO intro/outro; Stage B untouched)",
        "pkl_planned": "humanoidverse/data/fds_side_step_4.pkl (to be produced via the "
                       "step0 link-validation/conversion path before launch)",
        "seed": 4728,
        "trainer": "humanoidverse/scripts/fds_train_dance_core.py (same trainer, same obs schema)",
        "steps_budget_per_convergence_gate": {
            "0.1M": "minimum useful budget (first gate); launch if F/P curves still improving at 51.2k",
            "0.5M": "extend if completion < 100% at 0.1M",
            "1M": "extend if still improving at 0.5M",
            "2M": "cap absent an explicit ruling",
        },
        "selection_note": "bounce_sway_13 F/P selects trainer+init ONLY; side_step_4 then "
                          "reuses the winner with the same protocol",
    }

    report = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "script": "humanoidverse/scripts/fds_train_dance_core.py",
            "project": "fixed_dance_reference_specialist — Stage A dance-core (armFP prereg)",
            "preregistration": "results/fds/armFP_preregistration.json",
            "preregistration_v2": "results/fds/armFP_preregistration_v2.json",
            "arm": args.arm, "seed": args.seed, "objective": args.objective,
            "reference_clip": f"{clip_name} ({T} frames @ {clip_fps} fps, {pkl_path.name})",
            "mode": ("eval-only from checkpoint " + str(args.eval_ckpt)
                     if args.eval_ckpt else "train + deterministic eval"),
            "backbone_selection": {
                "decision": "Fresh (tie rule)",
                "evidence": ("F 0.1287 vs P 0.1320 mean track err at 51.2k steps, both "
                             "0/10 completion — differences within CI"),
                "wording": "Fresh chosen for structural simplicity (P was a trainer "
                           "control: 0.166% params loaded, not a pretraining comparison)",
            } if is_default_clip else {
                "decision": "Fresh (selected on bounce_sway_13 F/P; see dance_core_F_1/P_1 reports)",
            },
            "schema_written_this_run": schema_written,
            "train_seconds": total_train_s,
            "steps_per_s_total": (global_steps / total_train_s) if total_train_s > 0 else None,
        },
        "config": {
            "steps_total_requested": args.steps_total, "batches": n_batches,
            "batch_steps": args.batch_steps, "global_steps": global_steps,
            "policy": ("MLP 3x256 tanh-squashed diagonal Gaussian, learnable log_std clamp "
                       f"[-2,0.5], init {LOG_STD_INIT} (std 0.50; single documented retune)"),
            "value": "MLP 3x256", "normalizer": "per-dim running mean/std on full 106-d raw input",
            "obs_layout": ("state34 + e0(14) + de(14) + e5(14) + e10(14) + e20(14) + sin/cos phase; "
                           "lookaheads clamped at frame 247; reference loops mod 248"),
            "ppo": {"gamma": 0.99, "lam": 0.95, "clip": 0.2, "lr": 3e-4,
                    "epochs": args.ppo_epochs, "minibatch": args.minibatch,
                    "vf_coef": 0.5, "ent_coef": 0.0, "grad_clip": 1.0,
                    "adv_normalized_per_batch": True},
            "reward": {"objective": args.objective,
                       "objective_v2": ("+ worst-joint penalty 0.5*max_j|q-q_ref| "
                                        "(authorized single objective change, 2026-08-25)")
                                      if args.objective == "v2" else None,
                       "w_track": W_TRACK, "w_action_rate": W_RATE, "w_termination": W_TERM,
                       "w_upright": W_UPRIGHT,
                       "w_worst_joint": W_WORST if args.objective == "v2" else 0.0,
                       "upright_def": "clamp(-projected_gravity_z, 0, 1) from sim truth",
                       "termination": ("FAIL on any of: env termination; any-joint |q-q_ref| > "
                                       f"{ERR_TERM} rad; base-height fall (< {FALL_MIN_H} m); "
                                       f"orientation fall (|pg_x|/|pg_y| > {FALL_GXY}); non-foot "
                                       "contact > 1 N; NaN/Inf; joint hard limit "
                                       "(evening ruling, prereg v2)")},
            "episode": "random-phase inject (uniform [0,247]), 248-step cap, prev action zeroed",
            "env": "build_cpu_env (mujoco, obs noise ON, episodic DR ON, pushes ON in train, "
                   "OFF in eval via is_evaluating; max_episode_length_s=10000)",
        },
        "transfer": transfer_report,
        "batch_log": batch_log,
        "headline": headline,
        "throughput": throughput,
        "next_stage_a_launch_config": next_stage_a,
        "eval_aggregate": eval_agg,
        "eval_per_seed": eval_seeds,
    }
    assert not report_path.exists()
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[fds] saved {report_path}", flush=True)
    if logjl is not None:
        logjl.close()


if __name__ == "__main__":
    main()
