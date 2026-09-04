#!/bin/bash
# Watcher for the 300k checkpoint-harvest run: copies the (overwriting)
# checkpoint dir to numbered snapshots when train_log crosses thresholds.
# Zero-risk to training (read-only polling + cp).
set -u
RUN_DIR="results/bfm-dynamics-rnorm-harvest300k"
LOG="$RUN_DIR/train_log.txt"
THRESHOLDS=(5000 10000 25000 50000 75000 100000 150000 200000 250000)
declare -A DONE
while true; do
  if [ -f "$LOG" ]; then
    ROWS=$(grep -c "^" "$LOG" 2>/dev/null || echo 0)
    STEP=$(( ROWS * 8192 ))
    for T in "${THRESHOLDS[@]}"; do
      if [ -z "${DONE[$T]:-}" ] && [ "$STEP" -ge "$T" ]; then
        # wait for a stable checkpoint (train_status stops advancing briefly
        # after a save; just sleep a little and copy)
        sleep 3
        cp -r "$RUN_DIR/checkpoint" "$RUN_DIR/ckpt_$T"
        DONE[$T]=1
        echo "$(date +%H:%M:%S) snapshotted ckpt_$T (log step ~$STEP)"
      fi
    done
    if [ "${#DONE[@]}" -eq "${#THRESHOLDS[@]}" ]; then
      echo "all thresholds snapshotted"
      break
    fi
  fi
  if ! pgrep -f "bfm-dynamics-rnorm-harvest300k" > /dev/null; then
    sleep 10
    if ! pgrep -f "bfm-dynamics-rnorm-harvest300k" > /dev/null; then
      echo "training process gone; exiting watcher"
      break
    fi
  fi
  sleep 5
done
