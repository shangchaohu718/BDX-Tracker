#!/bin/bash
# Memory watchdog for BFM-Zero training.
# Usage: mem_watchdog.sh <PGID> <logfile>
# Kills the whole process group if MemAvailable drops below 3 GB or GPU memory
# above 15.4 GB / 16 GB. Samples every 20 s, logs to csv.

PGID="$1"; LOG="$2"
echo "time,rss_mb,avail_mb,gpu_mib" > "$LOG"
while kill -0 -"$PGID" 2>/dev/null; do
    rss=0
    for p in $(pgrep -g "$PGID" 2>/dev/null); do
        r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null)
        rss=$((rss + ${r:-0}))
    done
    rss_mb=$((rss / 1024))
    avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    gpu_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "$(date +%H:%M:%S),$rss_mb,$avail_mb,${gpu_mib:-na}" >> "$LOG"
    if [ "${avail_mb:-99999}" -lt 3000 ]; then
        echo "$(date +%H:%M:%S) WATCHDOG: MemAvailable=${avail_mb}MB < 3000MB -> killing group $PGID" >> "$LOG"
        kill -TERM -"$PGID" 2>/dev/null; sleep 5; kill -KILL -"$PGID" 2>/dev/null
        exit 42
    fi
    if [ "${gpu_mib:-0}" -gt 15850 ]; then
        echo "$(date +%H:%M:%S) WATCHDOG: GPU=${gpu_mib}MiB > 15850MiB -> killing group $PGID" >> "$LOG"
        kill -TERM -"$PGID" 2>/dev/null; sleep 5; kill -KILL -"$PGID" 2>/dev/null
        exit 43
    fi
    sleep 20
done
echo "$(date +%H:%M:%S) target exited normally" >> "$LOG"
