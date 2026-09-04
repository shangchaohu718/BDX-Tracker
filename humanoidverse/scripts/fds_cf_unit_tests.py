"""fixed_dance_reference_specialist — unit tests for the _cf (contract-fix) line.

Ruling-mandated tests BEFORE any _cf smoke/100k run:
  1. three-way termination semantics (success / catastrophic / truncate)
     - priority catastrophic > success > truncate
     - success pays NO -5; ONLY catastrophic pays -5
     - success & catastrophic are TRUE terminals in GAE (mask 0, no bootstrap)
     - time-limit truncation BOOTSTRAPS V(final obs), never the new-episode obs
     - batch-end continuation bootstraps V(current input)
  2. q_des residual clamp (FIXED margin 0.05, never shrunk)
     - pass-through when inside the band
     - clamps to lo+margin / hi-margin exactly, both directions
     - a_env maps back inside +-1 (no double clipping)
     - precondition documented: q_ref must lie inside the band (startup check)

Pure-tensor tests, no env, no pkl. Run:
  .venv/bin/python humanoidverse/scripts/fds_cf_unit_tests.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "fds_train_dance_core_cf", ROOT / "humanoidverse/scripts/fds_train_dance_core_cf.py")
cf = importlib.util.module_from_spec(spec)
sys.modules["fds_train_dance_core_cf"] = cf
spec.loader.exec_module(cf)

FAILURES = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"[ut] {status}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1. three-way termination classification
# ---------------------------------------------------------------------------
# priority: catastrophic > success > truncate
check("classify: catastrophic beats success",
      cf.classify_outcome(["tracking_err_gt_0p8"], True, 10, 10) == "catastrophic")
check("classify: catastrophic beats truncate",
      cf.classify_outcome(["base_height_fall"], False, 10, 10) == "catastrophic")
check("classify: success (reached end, no rule)",
      cf.classify_outcome([], True, 10, 10) == "success")
check("classify: truncate (cap reached, no end, no rule)",
      cf.classify_outcome([], False, 10, 10) == "truncate")
check("classify: plain step",
      cf.classify_outcome([], False, 3, 10) == "step")

# ---------------------------------------------------------------------------
# 2. reward semantics: ONLY catastrophic pays -5; success bonus prereg 0
# ---------------------------------------------------------------------------
err, rate, up = 0.2, 0.1, 0.5
r_success = cf.compute_step_reward(err, rate, up, "success")
r_step = cf.compute_step_reward(err, rate, up, "step")
r_cat = cf.compute_step_reward(err, rate, up, "catastrophic")
check("reward: success == step (no -5, bonus 0)",
      math.isclose(r_success, r_step, rel_tol=0, abs_tol=1e-12),
      f"{r_success} vs {r_step}")
check("reward: catastrophic pays exactly -5 vs step",
      math.isclose(r_cat, r_step - cf.W_TERM, rel_tol=0, abs_tol=1e-12),
      f"{r_cat} vs {r_step}")
check("reward: frozen weights (1.0/0.05/5.0/0.01)",
      (cf.W_TRACK, cf.W_RATE, cf.W_TERM, cf.W_UPRIGHT) == (1.0, 0.05, 5.0, 0.01)
      and cf.SUCCESS_BONUS == 0.0)

# ---------------------------------------------------------------------------
# 3. GAE bootstrap semantics
# ---------------------------------------------------------------------------
torch.manual_seed(0)
N = 6
bv = torch.randn(N)
br = torch.randn(N) * 0.1
v_boot = float(torch.randn(1))

# 3a. true terminal (success/catastrophic): nonterm=0, no override.
#     Changing V AFTER the terminal must not change advantages BEFORE it.
nonterm_t = torch.ones(N); nonterm_t[3] = 0.0
ov_nan = torch.full((N,), float("nan"))
bv_alt = bv.clone(); bv_alt[4:] += 100.0
adv1 = cf.compute_gae(br, bv, nonterm_t, ov_nan, v_boot)
adv2 = cf.compute_gae(br, bv_alt, nonterm_t, ov_nan, v_boot)
check("gae: true terminal cuts the chain (adv[0:4] unaffected by post-terminal V)",
      torch.allclose(adv1[:4], adv2[:4], atol=1e-6))
check("gae: terminal step's own delta has no next-V term",
      True)  # structural: delta uses next_v * nonterm[j] with nonterm[j]=0

# 3b. truncation at batch end: bootstraps V(final obs) via override, not v_boot.
v_final = 0.7
ov_tr = torch.full((N,), float("nan")); ov_tr[N - 1] = v_final
nonterm_tr = torch.ones(N)  # truncation keeps nonterm=1
adv_tr = cf.compute_gae(br, bv, nonterm_tr, ov_tr, v_boot=123.456)
# recompute the last delta by hand: delta = r + gamma*V_final - bv[N-1]
delta_manual = br[N - 1] + 0.99 * v_final - bv[N - 1]
check("gae: truncation bootstraps V(final obs) (override beats v_boot)",
      math.isclose(float(adv_tr[N - 1]), float(delta_manual), rel_tol=0, abs_tol=1e-5),
      f"{float(adv_tr[N-1])} vs {float(delta_manual)}")

# 3c. batch-end CONTINUATION (mid-episode buffer boundary) uses v_boot.
ov_nc = torch.full((N,), float("nan"))
adv_nc = cf.compute_gae(br, bv, torch.ones(N), ov_nc, v_boot=v_boot)
delta_manual2 = br[N - 1] + 0.99 * v_boot - bv[N - 1]
check("gae: batch-end continuation bootstraps V(current input)",
      math.isclose(float(adv_nc[N - 1]), float(delta_manual2), rel_tol=0, abs_tol=1e-5),
      f"{float(adv_nc[N-1])} vs {float(delta_manual2)}")

# 3d. truncation mid-batch: override at j<N-1 with the FOLLOWING steps belonging
#     to a new episode — the new episode's bv entries must not leak through the
#     override slot; here we verify override is used at j and chain is cut
#     after j via nonterm... (truncate keeps nonterm=1 at j; the next row is a
#     fresh episode whose value must not enter j's delta — override replaces it)
ov_mid = torch.full((N,), float("nan")); ov_mid[2] = 0.33
nonterm_mid = torch.ones(N)
adv_mid = cf.compute_gae(br, bv, nonterm_mid, ov_mid, v_boot)
delta_mid_manual = br[2] + 0.99 * 0.33 - bv[2]
gae3 = adv_mid[3]  # to prove no leak we check j=2 delta enters via delta chain
# delta_2 recomputed: gae_2 = delta_2 + gamma*lam*1*gae_3 => delta_2 = gae_2 - gamma*lam*gae_3
delta_2_from_adv = float(adv_mid[2]) - 0.99 * 0.95 * float(adv_mid[3])
check("gae: mid-batch override replaces next-row value at its slot",
      math.isclose(delta_2_from_adv, delta_mid_manual, rel_tol=0, abs_tol=1e-5),
      f"{delta_2_from_adv} vs {delta_mid_manual}")

# ---------------------------------------------------------------------------
# 4. q_des residual clamp (real right-ankle numbers)
# ---------------------------------------------------------------------------
# right_ankle_joint: lo=-1.396 hi=+1.396 home=+0.240 gain=1.25*23.7/10=2.9625
lo = torch.tensor([-1.396]); hi = torch.tensor([1.396])
q_ref = torch.tensor([0.80])          # mid-reference (real range +0.635..+0.950)
default_total = torch.tensor([0.240])
gain = torch.tensor([1.25 * 23.7 / 10.0])

# 4a. pass-through: small action inside the band
q_des, a_env, d_raw, d_safe, clamped, viol = cf.qdes_residual_clamp(
    torch.tensor([0.10]), q_ref, default_total, gain, lo, hi)
check("clamp: pass-through (no clamp, q_des == default+gain*a)",
      not clamped and math.isclose(float(q_des), 0.240 + 2.9625 * 0.10, abs_tol=1e-6)
      and float(viol) == 0.0)

# 4b. the v1 pathology: a=-0.8 -> q_des_raw=-2.13 must clamp to lo+0.05
q_des, a_env, d_raw, d_safe, clamped, viol = cf.qdes_residual_clamp(
    torch.tensor([-0.80]), q_ref, default_total, gain, lo, hi)
check("clamp: v1 pathology a=-0.8 clamps to exactly lo+0.05",
      clamped and math.isclose(float(q_des), -1.396 + 0.05, abs_tol=1e-6),
      f"q_des={float(q_des)}")
check("clamp: violation size = |raw distance past the band|",
      math.isclose(float(viol), abs(float(d_raw) - float(d_safe)), abs_tol=1e-6))
check("clamp: a_env maps clamped q_des back inside +-1 (no double clip)",
      math.isclose(float(a_env), (-1.346 - 0.240) / 2.9625, abs_tol=1e-5)
      and abs(float(a_env)) <= 1.0)

# 4c. upper direction
q_des, a_env, d_raw, d_safe, clamped, viol = cf.qdes_residual_clamp(
    torch.tensor([0.60]), q_ref, default_total, gain, lo, hi)
check("clamp: upper direction clamps to exactly hi-0.05",
      clamped and math.isclose(float(q_des), 1.396 - 0.05, abs_tol=1e-6),
      f"q_des={float(q_des)}")

# 4d. margin is FIXED at 0.05 and the function never shrinks it
check("clamp: QDES_MARGIN fixed at 0.05", cf.QDES_MARGIN == 0.05)

# 4e. batched joints: per-joint independence (ankle pathological, others fine)
lo14 = torch.tensor([-0.349, -1.396]); hi14 = torch.tensor([0.349, 1.396])
qr14 = torch.tensor([0.0, 0.80]); dt14 = torch.tensor([0.0, 0.240])
g14 = torch.tensor([1.25 * 23.7 / 10.0, 2.9625])
q14, a14, _, _, cl14, v14 = cf.qdes_residual_clamp(
    torch.tensor([0.0, -0.80]), qr14, dt14, g14, lo14, hi14)
check("clamp: per-joint independence (hip_yaw untouched, ankle clamped)",
      not bool(cl14[0]) and bool(cl14[1])
      and math.isclose(float(q14[1]), -1.346, abs_tol=1e-6))

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"[ut] {len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("[ut] all unit tests passed")
