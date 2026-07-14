#!/usr/bin/env bash
# watch_nan_reset.sh — monitor the BFM-Zero NaN-reset safety-net firing rate on the H20 run.
#
# Deployed to the remote box (not a git checkout). Tail the training log, parse the latest
# "[BFM-DIAG-NAN] NaN-reset safety net: N env-reset(s) so far (step_count=S, profile_envs=E)"
# line, and report the per-env-per-substep reset rate against the known-good baseline (~1%).
#
# Why this matters NOW: collision trim (trim_collision) was disabled 2026-07-07 for both train
# + infer. At 8192 training worlds a deep collapse can pile up contacts and approach njmax=1024
# -> dropped constraint rows -> penetration cascade -> NaN. BFM_ZERO_NAN_RESET still heals them,
# but if it fires MUCH more often than ~1%/substep, the net is masking constant collapses
# (the same artifact that hid real falls at inference). Recovery: re-enable trim, or raise
# nconmax/njmax (mind the nworld-product OOM).
#
# Usage on remote:
#   bash watch_nan_reset.sh                # one-shot snapshot of the latest log
#   bash watch_nan_reset.sh -f             # follow: re-check every 30s until Ctrl-C
#   bash watch_nan_reset.sh /path/to.log   # point at a specific log (default: supervisor.log)
set -u
PROJ=/root/epfs/tcl/bdx_BFMzero
LOG="$PROJ/supervisor.log"
FOLLOW=0
for arg in "$@"; do
    case "$arg" in
        -f|--follow) FOLLOW=1 ;;
        -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
        *) LOG="$arg" ;;
    esac
done

BASELINE_PCT=1.0   # validated-good NaN-reset rate with trim ON (% of env-substeps), per bfmzero-nan-root-cause

snapshot() {
    if [ ! -f "$LOG" ]; then
        echo "[watch] log not found: $LOG (training not started yet?)"; return 1
    fi
    # all NaN-reset lines (cumulative), newest last
    local lines; lines=$(grep -a 'NaN-reset safety net' "$LOG")
    local line; line=$(echo "$lines" | tail -1)
    if [ -z "$line" ]; then
        local last; last=$(tail -1 "$LOG" 2>/dev/null | cut -c1-80)
        echo "[watch] no NaN-reset lines yet in $LOG (net has not fired, or log just started). last: $last"
        return 0
    fi
    local resets steps envs
    resets=$(echo "$line" | sed -n 's/.*safety net: \([0-9]*\) env-reset.*/\1/p')
    steps=$(echo "$line"  | sed -n 's/.*step_count=\([0-9]*\).*/\1/p')
    envs=$(echo "$line"  | sed -n 's/.*profile_envs=\([0-9]*\).*/\1/p')
    if [ -z "$resets" ] || [ -z "$steps" ] || [ -z "$envs" ] || [ "$steps" -eq 0 ] || [ "$envs" -eq 0 ]; then
        echo "[watch] parse failed: $line"; return 1
    fi
    # rate = cumulative NaN-resets / (cumulative substeps * envs) = fraction of env-substeps healed.
    local rate; rate=$(awk -v r="$resets" -v s="$steps" -v e="$envs" 'BEGIN{printf "%.3f", 100.0*r/(s*e)}')

    # trend: compare against the previous cumulative line (detect a RISING rate, the buffer-pressure signature)
    local trend=""
    local prev; prev=$(echo "$lines" | tail -2 | head -1)
    if [ -n "$prev" ] && [ "$prev" != "$line" ]; then
        local pr ps
        pr=$(echo "$prev" | sed -n 's/.*safety net: \([0-9]*\) env-reset.*/\1/p')
        ps=$(echo "$prev" | sed -n 's/.*step_count=\([0-9]*\).*/\1/p')
        if [ -n "$pr" ] && [ -n "$ps" ] && [ "$ps" -gt 0 ]; then
            local prate; prate=$(awk -v r="$pr" -v s="$ps" -v e="$envs" 'BEGIN{printf "%.3f", 100.0*r/(s*e)}')
            trend=$(awk -v n="$rate" -v p="$prate" 'BEGIN{ d=n-p; if(d>0.1) printf "  RISING (%.3f%% -> %.3f%%)", p, n; else if(d< -0.1) printf "  falling (%.3f%% -> %.3f%%)", p, n; else printf "  steady (~%.3f%%)", n}')
        fi
    fi

    local verdict
    verdict=$(awk -v r="$rate" -v b="$BASELINE_PCT" 'BEGIN{
        if (r+0 < 0.01) print "ZERO — net idle, sim healthy";
        else if (r+0 <= 2*b) print "OK — near baseline";
        else if (r+0 <= 5*b) print "ELEVATED — net firing above norm; trim-off may be causing buffer-pressure NaNs";
        else print "HIGH — net masking constant collapses; likely njmax overflow. Recovery: re-enable trim_collision or raise nconmax/njmax";
    }')
    printf '[watch] resets=%s  step_count(substeps)=%s  envs=%s  ->  %s%% of env-substeps  [baseline ~%s%%]%s\n' \
        "$resets" "$steps" "$envs" "$rate" "$BASELINE_PCT" "$trend"
    printf '         verdict: %s\n' "$verdict"
}

echo "[watch] target log: $LOG"
if [ "$FOLLOW" -eq 1 ]; then
    echo "[watch] following every 30s (Ctrl-C to stop)..."
    while true; do snapshot; echo "---"; sleep 30; done
else
    snapshot
fi