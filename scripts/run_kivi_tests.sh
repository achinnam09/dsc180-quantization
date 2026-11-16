#!/usr/bin/env bash
# Comprehensive KIVI test on a DSMLP GPU pod: unit tests, the Task 3 smoke test,
# and the Task 4 LongBench subset with both prefill settings.
#
# Run INSIDE the pod, from the repo root, in the background (pods don't keep stdout):
#     source ~/private/.kivi_env && cd ~/private/dsc180-quantization
#     mkdir -p logs && nohup bash scripts/run_kivi_tests.sh > logs/kivi_tests_$(date +%m%d_%H%M).log 2>&1 &
# Watch it:
#     tail -f logs/kivi_tests_*.log                      # whole run
#     tail -f results/longbench/subset_n30_quantprefill/run.log   # per-prompt progress
# Stop it:
#     pkill -f run_kivi_tests.sh; pkill -f run_longbench.py
#
# Options (environment variables):
#     N=30                              prompts per task (KIVI's 8 tasks -> 8*N prompts)
#     CONFIGS="fp16 kivi4 kivi2 token2" cache configs to compare
#     STAGES="pytest smoke quant exact" which stages to run, in order:
#         pytest  unit tests (should take < 1 min)
#         smoke   Task 3 again: 1 Qasper prompt, 5 configs (a few minutes)
#         quant   subset, mentor's setting (prefill attention on the quantized cache)
#         exact   subset, KIVI paper's setting (exact prefill attention)
#   e.g.  STAGES="quant" nohup bash scripts/run_kivi_tests.sh > logs/quant.log 2>&1 &
#
# Resuming: the harness writes one line per finished prompt and skips finished
# prompts on re-run, so after a pod timeout just run the same command again.
#
# Runtime estimate (N=30, 4 configs): per run (one prefill setting) the 240
# prompts average ~3,590 input tokens (5 of 8 tasks are mostly truncated to
# 4096) and at most 45K generated tokens per config (QMSum/MultiNews allow 512
# each). Rough guess on an Ampere-class GPU: 15-40 min per config, so ~1-2.5 h
# per prefill setting and ~2-5 h for both. Pods stop after 6 h: if `quant` is
# slower than ~2.5 h, run `exact` separately (STAGES="exact") in a later
# invocation or a second pod. Pre-Ampere GPUs (no bf16) will be much slower.

set -uo pipefail  # no -e: a failed stage is recorded and the next stage still runs

N="${N:-30}"
CONFIGS="${CONFIGS:-fp16 kivi4 kivi2 token2}"
STAGES="${STAGES:-pytest smoke quant exact}"
KIVI_TASKS="qasper qmsum multi_news trec triviaqa samsum lcc repobench-p"

cd "$(dirname "$0")/.." || exit 1
for env_file in "$HOME/private/.kivi_env" "$HOME/.kivi_env"; do
    # shellcheck disable=SC1090
    [ -f "$env_file" ] && source "$env_file" && break
done
command -v uv >/dev/null || { echo "ERROR: uv not on PATH; run scripts/dsmlp_setup.sh first"; exit 1; }
mkdir -p logs

FAILED=()
START=$(date +%s)
stage() {
    local name="$1"; shift
    echo
    echo "=================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STAGE: $name"
    echo "  \$ $*"
    echo "=================================================================="
    local t0=$(date +%s)
    if "$@"; then
        echo "[$(date '+%H:%M:%S')] STAGE $name: OK ($(( ($(date +%s) - t0) / 60 )) min)"
    else
        echo "[$(date '+%H:%M:%S')] STAGE $name: FAILED (exit $?) after $(( ($(date +%s) - t0) / 60 )) min"
        FAILED+=("$name")
    fi
}
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

echo "KIVI tests  N=$N  CONFIGS='$CONFIGS'  STAGES='$STAGES'"
echo "commit: $(git log --oneline -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: no nvidia-smi"
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# shellcheck disable=SC2086  # $CONFIGS / $KIVI_TASKS are intentionally word-split
if has_stage pytest; then
    stage pytest uv run pytest -q
fi
if has_stage smoke; then
    # Mac (MPS) results for this prompt, quantized prefill: fp16 'yes', kivi4 'yes',
    # kivi2 'unanswerable', token2 gibberish, channel2 rambling. CUDA bf16 numerics
    # differ slightly from MPS, so borderline answers may change; token2 should
    # still be gibberish and fp16/kivi4 should still agree.
    stage smoke uv run python src/evaluation/run_longbench.py --tasks qasper --n 1 \
        --configs fp16 kivi4 kivi2 token2 channel2 --run-name gpu_task3_quantprefill --show
fi
if has_stage quant; then
    stage subset_quantprefill uv run python src/evaluation/run_longbench.py --tasks $KIVI_TASKS \
        --n "$N" --configs $CONFIGS --run-name "subset_n${N}_quantprefill"
fi
if has_stage exact; then
    stage subset_exactprefill uv run python src/evaluation/run_longbench.py --tasks $KIVI_TASKS \
        --n "$N" --configs $CONFIGS --run-name "subset_n${N}_exactprefill" --exact-prefill
fi

echo
echo "=================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINAL SCORES (LongBench, 0-100)"
echo "=================================================================="
for run in gpu_task3_quantprefill "subset_n${N}_quantprefill" "subset_n${N}_exactprefill"; do
    [ -d "results/longbench/$run" ] || continue
    echo "--- $run"
    if [ "$run" = gpu_task3_quantprefill ]; then
        uv run python src/evaluation/run_longbench.py --tasks qasper --configs fp16 kivi4 kivi2 token2 channel2 \
            --run-name "$run" --score-only 2>&1 | tail -n +2
    else
        uv run python src/evaluation/run_longbench.py --tasks $KIVI_TASKS --configs $CONFIGS \
            --run-name "$run" --score-only 2>&1 | tail -n +2
    fi
done

echo
echo "Total time: $(( ($(date +%s) - START) / 60 )) min"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED STAGES: ${FAILED[*]}"
    exit 1
fi
echo "All stages OK."
