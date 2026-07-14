#!/usr/bin/env python3
# group_trend.py — split train_log metrics into FB vs aux/penalty/disc groups and report
# first/mid/last + a windowed (last 100 rows) trend with arrow.
# Usage: python group_trend.py [work_dir]
import csv, sys, os

WORK_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/bfmzero-fresh-qloss"
LOG = os.path.join(WORK_DIR, "train_log.txt")
rows = list(csv.DictReader(open(LOG)))
N = len(rows)

def col(name):
    out = []
    for r in rows:
        v = r.get(name)
        if v not in (None, "", "nan", "inf", "-inf"):
            try: out.append(float(v))
            except ValueError: pass
    return out

def trend(name, W=100):
    v = col(name)
    if len(v) < 4: return None
    if len(v) < 2*W: W = max(1, len(v)//4)
    prev = v[-2*W:-W]; last = v[-W:]
    pm = sum(prev)/len(prev); lm = sum(last)/len(last)
    pct = float("inf") if pm == 0 else 100*(lm-pm)/abs(pm)
    arrow = "▲" if lm > pm*1.05 else ("▼" if lm < pm*0.95 else "→")
    return dict(first=v[0], mid=v[len(v)//2], last=v[-1],
                prev_mean=pm, last_mean=lm, pct=pct, arrow=arrow)

FB = ["B","B_norm","F1","F_norm","M1","target_M","Q1","Q_fb","Q_fb_abs",
      "mean_next_Q","target_Q","unc_Q","z_norm","scale_reg_weight",
      "fb_bwd_grad_norm","fb_fwd_grad_norm","fb_loss","fb_offdiag","fb_diag",
      "orth_loss","orth_loss_diag","orth_loss_offdiag","q_loss"]
AUX = ["actor_grad_norm","actor_loss","critic_grad_norm","critic_loss",
       "auxQ1","aux_critic_grad_norm","aux_critic_loss","mean_next_auxQ",
       "target_auxQ","unc_auxQ","Q_aux","Q_discriminator",
       "disc_loss","disc_expert_loss","disc_train_loss","disc_wgan_gp_loss","disc_grad_norm",
       "aux_rew/limits_dof_pos","aux_rew/limits_torque","aux_rew/penalty_action_rate",
       "aux_rew/penalty_ankle_roll","aux_rew/penalty_feet_ori","aux_rew/penalty_slippage",
       "aux_rew/penalty_torques","aux_rew/penalty_undesired_contact",
       "mean_aux_reward","mean_disc_reward"]

ts = col("timestep")
print(f"rows={N}  step {int(ts[0]):,} -> {int(ts[-1]):,}  (window = last 100 rows vs prev 100)")

def report(title, keys):
    print(f"\n{'='*104}\n{title}\n{'='*104}")
    print(f"{'variable':>32s} {'first':>12s} {'mid':>12s} {'last':>12s} | {'prev100':>12s} {'last100':>12s} {'d%':>9s} trend")
    for k in keys:
        t = trend(k)
        if t is None: print(f"{k:>32s}   (no data)"); continue
        pct = "inf" if t['pct']==float("inf") else f"{t['pct']:.1f}%"
        print(f"{k:>32s} {t['first']:>12.4g} {t['mid']:>12.4g} {t['last']:>12.4g} | "
              f"{t['prev_mean']:>12.4g} {t['last_mean']:>12.4g} {pct:>9s} {t['arrow']}")

report("GROUP A — FB (Forward-Backward representation)", FB)
report("GROUP B — Auxiliary / penalty / discrimination", AUX)