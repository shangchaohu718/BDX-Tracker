#!/usr/bin/env python3
# scale_report.py — magnitude audit of BFM-Zero training: reward terms, activations, gradients, losses.
#
# Purpose: hot grad norms (fb_bwd 60-70k) are only meaningful relative to the scales of the things
# that produce them. This prints, for the last N log rows:
#   - reward components (raw, pre-reward_scale) + which DOMINATES the signed scaled sum
#   - activation / output magnitudes (z, B, F1, Q values) -> are they exploding?
#   - all grad norms side by side -> which map is the outlier?
#   - losses -> relative scales
# Same data source as wandb (train_log.txt). No action taken; alarm-only.
import csv, sys, os, math

WORK_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/bfmzero-isaac-diag-fresh"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20   # last N rows to summarize
LOG = os.path.join(WORK_DIR, "train_log.txt")
rows = list(csv.DictReader(open(LOG)))
if not rows:
    print("no rows"); sys.exit(0)
rows = rows[-N:]

# reward_scale weights (from config/rewards/reward_bfm_zero.yaml) to compute the SCALED contribution
REWARD_SCALE = {
    "penalty_torques": -0.000001,
    "penalty_action_rate": -0.5,
    "penalty_ankle_roll": -0.5,
    "penalty_feet_ori": -0.1,
    "penalty_slippage": -1.0,
    "penalty_undesired_contact": -1.0,
    "limits_dof_pos": -10.0,
    "limits_torque": -5.0,
}

def f(r, k):
    v = r.get(k)
    if v in (None, "", "nan", "inf", "-inf"):
        return None
    try:
        return float(v)
    except ValueError:
        return None

def stat(name):
    vals = [f(r, name) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (vals[0], sum(vals)/len(vals), max(min(vals), max(vals), key=abs), vals[-1])

def fmt(t):
    if t is None:
        return f"{'—':>12s} {'—':>12s} {'—':>12s} {'—':>12s}"
    a, m, mx, z = t
    return f"{a:>12.4g} {m:>12.4g} {mx:>12.4g} {z:>12.4g}"

print(f"[scale] {WORK_DIR}  (last {len(rows)} rows, timestep {f(rows[-1],'timestep'):,.0f} -> {f(rows[0],'timestep'):,.0f} of earliest)")
print()
print("=== REWARD COMPONENTS (raw aux_rew, pre-scale) and their SCALED contribution ===")
print(f"{'component':>32s} {'weight':>10s} | {'first':>12s} {'mean':>12s} {'max|abs|':>12s} {'last':>12s} | {'scaled_mean':>12s} {'%of|scaled|':>11s}")
# compute scaled contributions to find the dominator
scaled = {}
for comp, w in REWARD_SCALE.items():
    key = f"aux_rew/{comp}"
    s = stat(key)
    if s is None:
        continue
    mean_raw = s[1]
    sc = w * mean_raw
    scaled[comp] = sc
tot = sum(abs(v) for v in scaled.values()) or 1.0
for comp, w in REWARD_SCALE.items():
    key = f"aux_rew/{comp}"
    s = stat(key)
    if s is None:
        continue
    sc = scaled.get(comp, 0)
    pct = 100.0 * abs(sc) / tot
    flag = "  <-- DOMINATES" if pct > 40 else ""
    print(f"{comp:>32s} {w:>10g} | {fmt(s)} | {sc:>12.4g} {pct:>10.1f}%{flag}")
print(f"{'(sum of scaled)':>32s} {'':>10s} | {'':>12s} {'':>12s} {'':>12s} {'':>12s} | {sum(scaled.values()):>12.4g}")

print()
print("=== GRADIENT NORMS (all maps) ===")
print(f"{'metric':>32s} {'first':>12s} {'mean':>12s} {'max|abs|':>12s} {'last':>12s}")
for k in ["fb_bwd_grad_norm", "fb_fwd_grad_norm", "actor_grad_norm", "critic_grad_norm",
          "aux_critic_grad_norm", "disc_grad_norm"]:
    print(f"{k:>32s} {fmt(stat(k))}")

print()
print("=== ACTIVATION / OUTPUT MAGNITUDES (exploding?) ===")
print(f"{'metric':>32s} {'first':>12s} {'mean':>12s} {'max|abs|':>12s} {'last':>12s}")
for k in ["z_norm", "B", "B_norm", "F1", "F_norm", "Q1", "Q_fb", "Q_fb_abs", "Q_aux",
          "Q_discriminator", "M1", "target_Q", "target_M", "scale_reg_weight"]:
    print(f"{k:>32s} {fmt(stat(k))}")

print()
print("=== LOSSES (relative scale) ===")
print(f"{'metric':>32s} {'first':>12s} {'mean':>12s} {'max|abs|':>12s} {'last':>12s}")
for k in ["fb_loss", "actor_loss", "critic_loss", "aux_critic_loss", "q_loss",
          "disc_loss", "disc_train_loss", "disc_expert_loss", "disc_wgan_gp_loss",
          "orth_loss", "orth_loss_diag", "orth_loss_offdiag", "fb_diag", "fb_offdiag",
          "mean_aux_reward", "mean_disc_reward"]:
    print(f"{k:>32s} {fmt(stat(k))}")

print()
print("=== HEALTH FLAGS (per-row ok checks) ===")
for k in ["actor_ok", "critic_ok", "aux_critic_ok", "fb_ok", "disc_ok"]:
    vals = [f(r, k) for r in rows]
    vals = [v for v in vals if v is not None]
    if vals:
        nbad = sum(1 for v in vals if v < 0.5)
        print(f"  {k}: {len(vals)-nbad}/{len(vals)} ok  ({nbad} flagged bad)")