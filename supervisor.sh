#!/bin/bash
# [BFM-OPS] BFM-Zero auto-resume supervisor (run-mode driven).
#
# ONE supervisor for all run intents. The mode (fresh | resume | snapshot) is selected by $1 or
# BFM_ZERO_RUN_MODE, default "fresh". It mirrors _resolve_run_mode() in train.py -- same mapping
# run-mode -> (work_dir, target) -- so train.py and the supervisor agree on where checkpoints live
# and what step count counts as "done".
#
#   fresh    -> results/bfmzero-isaac-diag-fresh,   target 384M (paper scale), eval ON,  random weights
#   resume   -> results/bfmzero-isaac-diag-resume,  target 750M (extended),     eval OFF, load checkpoint
#   snapshot -> (exits during seeding; not driven by this supervisor)
#
# Relaunches training across process deaths -- including the observed ~4h death of GPU jobs on
# this host -- until results/.../checkpoint/train_status.json reaches TARGET. Launched detached
# (systemd-run, or setsid+nohup) so it survives SSH disconnects. The supervisor itself is CPU-only,
# so it is not a GPU-job-kill target; only the child training run is. A crash guard (short run +
# no step progress) stops the loop on a REAL failure instead of infinite-looping.
#
# Usage:
#   ./supervisor.sh              # fresh (default)
#   ./supervisor.sh resume       # resume from latest checkpoint
#   BFM_ZERO_RUN_MODE=resume ./supervisor.sh
#   BFM_ZERO_NUM_ENV_STEPS=500000000 ./supervisor.sh resume   # override target
set -u

PROJECT=/root/epfs/tcl/bdx_BFMzero
cd "$PROJECT" || { echo "no project dir"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
export BFM_ZERO_PROFILE=h20

# --- run-mode selection (arg > env > default) --------------------------------- #
MODE="${1:-${BFM_ZERO_RUN_MODE:-fresh}}"
case "$MODE" in
  fresh)
    export BFM_ZERO_RUN_MODE=fresh
    DEFAULT_WORK_DIR="results/bfmzero-isaac-diag-fresh"
    DEFAULT_TARGET=384000000       # paper scale (~7d on H20); extend only after inference eval
    ;;
  resume)
    export BFM_ZERO_RUN_MODE=resume
    DEFAULT_WORK_DIR="results/bfmzero-isaac-diag-resume"
    DEFAULT_TARGET=750000000       # extended target past paper scale
    ;;
  *)  # snapshot is a one-shot debug mode (exits during seeding); not driven by this supervisor
    echo "unknown/unsupported mode '$MODE' for the supervisor (use: fresh | resume)"; exit 1 ;;
esac
export BFM_ZERO_WORK_DIR="${BFM_ZERO_WORK_DIR:-$DEFAULT_WORK_DIR}"
export BFM_ZERO_NAN_RESET=1     # per-substep isfinite NaN-reset safety net (the validated fix)
export PYTHONUNBUFFERED=1       # child prints stream to the per-iter log instead of block-buffering

RUN_DIR="$BFM_ZERO_WORK_DIR"
TS="$RUN_DIR/checkpoint/train_status.json"
TARGET="${BFM_ZERO_NUM_ENV_STEPS:-$DEFAULT_TARGET}"
LOG="$RUN_DIR/supervisor.log"              # log lives INSIDE the run dir, one per mode (no clobber)
ITER_LOG_DIR="$RUN_DIR/iter_logs"
mkdir -p "$ITER_LOG_DIR"

ts_time() { .venv/bin/python -c "import json;print(json.load(open('$TS'))['time'])" 2>/dev/null || echo 0; }
now()     { date -u +%FT%TZ; }

echo "[$(now)] supervisor START mode=$MODE target=$TARGET run_dir=$RUN_DIR" | tee -a "$LOG"

iter=0
MAX_ITER=80
MIN_RUN_SEC=600   # a legit run lasts ~4h; shorter than this with no progress = real crash
while [ $iter -lt $MAX_ITER ]; do
  iter=$((iter + 1))
  before=$(ts_time)
  if [ "${before:-0}" -ge "$TARGET" ]; then
    echo "[$(now)] DONE train_status=$before >= $TARGET" | tee -a "$LOG"
    exit 0
  fi
  ilog="$ITER_LOG_DIR/train_iter_${iter}_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "[$(now)] iter=$iter before=$before launching .venv/bin/python -m humanoidverse.train -> $ilog" | tee -a "$LOG"
  start=$(date +%s)
  # Direct module entry (NOT run_train.sh -- that script's embedded USD-patch python is syntactically
  # broken, and the USD patch is unneeded for the mujoco_warp backend which loads XML terrain, not USD).
  .venv/bin/python -m humanoidverse.train > "$ilog" 2>&1
  rc=$?
  end=$(date +%s); dur=$((end - start))
  after=$(ts_time); advanced=$((after - before))
  echo "[$(now)] iter=$iter rc=$rc dur=${dur}s before=$before after=$after (+$advanced)" | tee -a "$LOG"
  # crash guard: short run with no step advance = real failure. Stop, do not loop.
  if [ "$dur" -lt "$MIN_RUN_SEC" ] && [ "$advanced" -le 0 ]; then
    echo "[$(now)] CRASH GUARD tripped: run lasted ${dur}s, no progress. Tail of $ilog:" | tee -a "$LOG"
    tail -25 "$ilog" | tee -a "$LOG"
    echo "[$(now)] stopping to avoid an infinite relaunch loop." | tee -a "$LOG"
    exit 1
  fi
  sleep 20
done
echo "[$(now)] reached MAX_ITER=$MAX_ITER, stopping." | tee -a "$LOG"
exit 0