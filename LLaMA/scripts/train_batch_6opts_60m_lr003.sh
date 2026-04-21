#!/bin/bash
# Batch launcher for 60M experiments:
# shampoo, muon, soap, mano
# Runs two experiments at once, each on two GPUs.
# Auto-detaches with nohup so closing the terminal will not stop the batch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default behavior: detach the whole batch from the current terminal.
# Set RUN_IN_FOREGROUND=1 to keep it in the current shell.
if [[ "${BATCH_DETACHED:-0}" != "1" && "${RUN_IN_FOREGROUND:-0}" != "1" ]]; then
    DETACH_TIMESTAMP="$(date +%Y-%m-%d-%H-%M-%S)"
    DETACH_LOG="$PROJECT_DIR/checkpoints/batch-launch-${DETACH_TIMESTAMP}.log"
    mkdir -p "$PROJECT_DIR/checkpoints"

    BATCH_DETACHED=1 nohup bash "$0" "$@" >"$DETACH_LOG" 2>&1 < /dev/null &
    DETACHED_PID=$!

    echo "Batch launched in background."
    echo "Launcher PID: ${DETACHED_PID}"
    echo "Launcher log: ${DETACH_LOG}"
    echo "Use: tail -f ${DETACH_LOG}"
    exit 0
fi

LR_MATRIX="0.03"
NUM_GPUS_PER_RUN="2"
GPU_PAIR_A="0,1"
GPU_PAIR_B="2,3"

RUN_SCRIPTS=(
    "train_mano_60m.sh"
)

TIMESTAMP="$(date +%Y-%m-%d-%H-%M-%S)"
LOG_DIR="$PROJECT_DIR/checkpoints/batch-${TIMESTAMP}-lr_matrix_${LR_MATRIX}"
mkdir -p "$LOG_DIR"

declare -a ACTIVE_PIDS=()

cleanup_children() {
    local pid
    for pid in "${ACTIVE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}

trap cleanup_children EXIT

launch_one() {
    local script_name="$1"
    local gpu_ids="$2"
    local exp_name
    local log_file

    exp_name="${script_name#train_}"
    exp_name="${exp_name%_60m.sh}"
    log_file="$LOG_DIR/${exp_name}.log"

    echo "[$(date '+%F %T')] Launching ${exp_name} on GPUs ${gpu_ids}" >&2
    echo "  log: ${log_file}" >&2

    CUDA_VISIBLE_DEVICES="$gpu_ids" bash "$SCRIPT_DIR/$script_name" \
        --num_gpus "$NUM_GPUS_PER_RUN" \
        --lr_matrix "$LR_MATRIX" \
        >"$log_file" 2>&1 &

    LAST_PID=$!
}

echo "Batch start: ${TIMESTAMP}"
echo "LR_MATRIX override: ${LR_MATRIX}"
echo "Per-run GPUs: ${NUM_GPUS_PER_RUN}"
echo "Logs directory: ${LOG_DIR}"

total_runs="${#RUN_SCRIPTS[@]}"

for ((i = 0; i < total_runs; i += 2)); do
    script_a="${RUN_SCRIPTS[$i]}"
    script_b="${RUN_SCRIPTS[$((i + 1))]}"

    echo ""
    echo "========== Round $((i / 2 + 1)) =========="

    launch_one "$script_a" "$GPU_PAIR_A"
    pid_a="$LAST_PID"

    launch_one "$script_b" "$GPU_PAIR_B"
    pid_b="$LAST_PID"

    ACTIVE_PIDS=("$pid_a" "$pid_b")

    status_a=0
    status_b=0

    if wait "$pid_a"; then
        status_a=0
    else
        status_a=$?
    fi
    if wait "$pid_b"; then
        status_b=0
    else
        status_b=$?
    fi

    if [[ "$status_a" -ne 0 || "$status_b" -ne 0 ]]; then
        echo ""
        echo "A run failed in round $((i / 2 + 1))."
        echo "  ${script_a} exit code: ${status_a}"
        echo "  ${script_b} exit code: ${status_b}"
        echo "Check logs in: ${LOG_DIR}"
        exit 1
    fi

    ACTIVE_PIDS=()

    echo "Round $((i / 2 + 1)) completed successfully."
done

echo ""
echo "All experiments finished successfully."
echo "Logs directory: ${LOG_DIR}"
