#!/usr/bin/env python3
# watch_fresh_metrics.py — periodic health snapshot of the BFM-Zero fresh-init run.
#
# Reads the SAME metrics dict that wandb.log() receives (train.py logs to both wandb and this CSV),
# so this is the wandb data without the offline-sync / entity-benasano / API round-trip. Reports the
# FB-critic divergence signature (fb_bwd_grad_norm, Q_fb_abs) that diagnosed the 2026-07-07 resume
# NaN (poisoned checkpoint hit fb_bwd_grad_norm=18500, Q_fb_abs=260 at divergence), the reward
# dominators, and a trend (first/mid/last) so divergence is visible as a climbing tail.
#
# Usage:
#   python3 watch_fresh_metrics.py                       # default: results/bfmzero-isaac-diag-fresh
#   python3 watch_fresh_metrics.py /path/to/work_dir
import csv, sys, os, json

WORK_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/bfmzero-isaac-diag-fresh"
LOG = os.path.join(WORK_DIR, "train_log.txt")
TS = os.path.join(WORK_DIR, "checkpoint", "train_status.json")

if not os.path.exists(LOG):
    print("[watch] no train_log.txt yet at", LOG, "(metrics flush on first log_every_updates boundary)")
    sys.exit(0)

rows = list(csv.DictReader(open(LOG)))
if not rows:
    print("[watch] train_log.txt empty"); sys.exit(0)

# progress
t0 = int(float(rows[0].get("timestep", 0)))
tN = int(float(rows[-1].get("timestep", 0)))
target = 384_000_000
pct = 100.0 * tN / target
try:
    saved = json.load(open(TS))["time"]
except Exception:
    saved = tN

KEYS = [
    "fb_bwd_grad_norm", "Q_fb_abs", "fb_fwd_grad_norm", "fb_loss",   # FB critic (the divergence signal)
    "actor_grad_norm", "actor_loss", "critic_grad_norm", "critic_loss",
    "aux_rew/penalty_action_rate", "aux_rew/penalty_ankle_roll",     # reward dominators
    "aux_rew/limits_dof_pos", "aux_rew/penalty_torques",
    "mean_aux_reward", "mean_disc_reward", "FPS",
]

def col(name):
    out = []
    for r in rows:
        v = r.get(name)
        if v not in (None, "", "nan", "inf", "-inf"):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out

# health verdict from the FB-critic signature (the known NaN precursor)
bwd = col("fb_bwd_grad_norm")
qfb = col("Q_fb_abs")
last_bwd = bwd[-1] if bwd else 0.0
last_qfb = qfb[-1] if qfb else 0.0
peak_bwd = max(bwd) if bwd else 0.0
peak_qfb = max(qfb) if qfb else 0.0
saw_nan = any(v != v for v in bwd + qfb)

# Persistent high-water mark: record the worst FB-critic state ever seen, so a transient divergence
# that the supervisor auto-recovered from between checks is NOT silently forgotten. Alarm-only by
# design (per user 2026-07-07) -- this file records the event for the next check to surface; it takes
# no action. Kept in the run dir so it survives supervisor restarts.
HW = os.path.join(WORK_DIR, "fb_critic_highwater.json")
prev = {"peak_bwd": 0.0, "peak_qfb": 0.0, "saw_nan": False, "step_at_peak": 0}
if os.path.exists(HW):
    try:
        prev = json.load(open(HW))
    except Exception:
        pass
if peak_bwd > prev.get("peak_bwd", 0) or peak_qfb > prev.get("peak_qfb", 0) or saw_nan:
    # find the step at which the peak occurred
    step_at_peak = tN
    for r, v in zip(rows, bwd):
        if v == peak_bwd:
            step_at_peak = int(float(r.get("timestep", tN)))
            break
    prev = {
        "peak_bwd": max(peak_bwd, prev.get("peak_bwd", 0)),
        "peak_qfb": max(peak_qfb, prev.get("peak_qfb", 0)),
        "saw_nan": bool(prev.get("saw_nan") or saw_nan),
        "step_at_peak": step_at_peak,
    }
    json.dump(prev, open(HW, "w"), indent=2)

# Divergence signature = a MONOTONIC climb in fb_bwd over the last window, not a single hot reading.
# Early training is genuinely noisy (untrained critic + warmup normalizers), so a hot-but-oscillating
# fb_bwd is NOT by itself divergence. The durable signal is: last window strictly climbing AND past
# the poisoned-resume scale (fb_bwd>50k or Q_fb>1000). NaN in the column is always alarming.
WINDOW = min(8, len(bwd))
tail = bwd[-WINDOW:] if WINDOW >= 2 else [last_bwd]
climbing = all(tail[i] <= tail[i + 1] for i in range(len(tail) - 1)) if len(tail) >= 2 else False
monotonic_hot = climbing and (last_bwd > 50000 or last_qfb > 1000)

verdict = "OK"
if saw_nan or monotonic_hot:
    verdict = ("*** DIVERGING — fb_bwd climbing monotonically past poisoned-resume scale "
               "(poisoned 384M NaN was fb_bwd~18500/Q_fb~260) ***")
elif last_bwd > 50000 or last_qfb > 1000:
    verdict = "ELEVATED (hot, not yet monotonic) — could be warmup noise; watch next check for climb"
# surface a transient past event the live row has recovered from
if verdict == "OK" and (prev["peak_bwd"] > 50000 or prev["peak_qfb"] > 1000 or prev["saw_nan"]):
    verdict = (f"RECOVERED but PAST SPIKE on disk: peak fb_bwd={prev['peak_bwd']:.0f} "
               f"Q_fb={prev['peak_qfb']:.0f} at step {prev['step_at_peak']}"
               + (" (saw NaN)" if prev["saw_nan"] else "")
               + " — investigate iter_logs/ before all-clear")

# NaN-reset rate: a SEPARATE signal from the grad norms. Low rate (<1%) = sim healthy even if grads
# are hot; high rate = sim producing non-finite state (the trim-off contact-pile signature). These
# two can disagree — report both so they aren't conflated.
import glob, re
ilog_dir = os.path.join(WORK_DIR, "iter_logs")
ilogs = sorted(glob.glob(os.path.join(ilog_dir, "*.log")))
reset_rate_str = ""
if ilogs:
    last_reset = None
    for lf in ilogs:
        for ln in open(lf, errors="ignore"):
            m = re.search(r"NaN-reset safety net: (\d+) env-reset\(s\) so far \(step_count=(\d+), profile_envs=(\d+)\)", ln)
            if m:
                last_reset = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if last_reset:
        r, s, e = last_reset
        rate = 100.0 * r / (s * e) if s and e else 0.0
        reset_rate_str = f"  NaN-reset: {r} resets by step {s} = {rate:.3f}% of env-substeps (baseline ~1%)"

print(f"[watch] {WORK_DIR}")
print(f"        step={tN:,} / {target:,} ({pct:.2f}%)  checkpoint_saved={saved:,}  rows={len(rows)}  verdict={verdict}")
print(f"        peak-so-far: fb_bwd={peak_bwd:.0f}  Q_fb={peak_qfb:.0f}  tail(window={WINDOW})_climbing={climbing}  (highwater: {HW})")
if reset_rate_str:
    print(f"       {reset_rate_str}")
print(f"{'metric':>28s} {'first':>12s} {'mid':>12s} {'last':>12s}")
for k in KEYS:
    v = col(k)
    if not v:
        continue
    n = len(v)
    print(f"{k:>28s} {v[0]:>12.4g} {v[n//2]:>12.4g} {v[-1]:>12.4g}")