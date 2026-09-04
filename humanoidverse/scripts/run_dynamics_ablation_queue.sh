#!/usr/bin/env bash
set -euo pipefail

# Wait for an already-running diagnostic baseline, verify its 2M checkpoint,
# then run the two single-factor ablations sequentially on the same GPU.
baseline_pid="${1:?usage: $0 BASELINE_PID}"
project_dir="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$project_dir"

while kill -0 "$baseline_pid" 2>/dev/null; do
    sleep 30
done

baseline_status="results/bfm-dynamics-baseline/checkpoint/train_status.json"
baseline_time="$($project_dir/.venv/bin/python -c "import json; print(json.load(open('$baseline_status'))['time'])")"
if [[ "$baseline_time" -lt 2000000 ]]; then
    echo "baseline did not reach 2M (time=$baseline_time); refusing to run incomparable ablations" >&2
    exit 1
fi

run_one() {
    local work_dir="$1"
    local scale_reg="$2"
    local clip="$3"
    mkdir -p "$work_dir"
    env \
        BDX_MODE=full \
        BDX_STEPS=2000000 \
        BDX_DATA="$project_dir/humanoidverse/data/bdx_14dof_train.pkl" \
        BDX_WORK_DIR="$work_dir" \
        BDX_SCALE_REG="$scale_reg" \
        BDX_CLIP_GRAD_NORM="$clip" \
        BDX_RELABEL_RATIO=0.8 \
        BDX_DIAG_DYNAMICS=1 \
        BDX_DISABLE_EVAL=1 \
        "$project_dir/.venv/bin/python" -m humanoidverse.scripts.train_bfm_zero_bdx \
        > "$work_dir/console.log" 2>&1
}

run_one results/bfm-dynamics-scale-off false 0
run_one results/bfm-dynamics-clip1 true 1

