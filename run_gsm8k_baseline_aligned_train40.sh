#!/usr/bin/env bash
# Fixed source-order train40 cohort shared with the GSM8K pruning baselines.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=4,5 ./run_gsm8k_baseline_aligned_train40.sh
  ./run_gsm8k_baseline_aligned_train40.sh --dry-run
  STAGE=prepare ./run_gsm8k_baseline_aligned_train40.sh
  STAGE=eval CUDA_VISIBLE_DEVICES=4,5 ./run_gsm8k_baseline_aligned_train40.sh

The independent run uses the baselines' exact first 40 source-order GSM8K train
questions, fixed worker order and unchanged correctness-aware/all attribution.
Stages: prepare -> collect -> attribute -> aggregate -> policies -> evaluate
global static pruning 80%, 70%, 60% -> summarize. Each stage/rate runs in a
separate process. Evaluation uses batch 4 and all 1,319 test questions.
An empty Round 1 (both spatial and temporal) is skipped; reviewer reads Round 0.
Resume is always enabled. Existing seed0/seed1/seed2 experiments are preserved.

Optional stage argument: prepare|collect|attribute|aggregate|policies|eval|summarize|all.
Overrides: STAGE, RUN_DIR, REFERENCE_RUN, BASELINE_DATA, COHORT_AUDIT_DIR,
           CUDA_VISIBLE_DEVICES (default 4,5), PYTHON_BIN, LOG_ROOT, DRY_RUN.
Dry-run only prints commands, without creating files or loading models.
EOF
}

STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
stage_argument=0
for argument in "$@"; do
  case "$argument" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    prepare|collect|attribute|aggregate|policies|eval|summarize|all)
      if (( stage_argument )); then echo "Choose only one stage." >&2; exit 2; fi
      STAGE="$argument"; stage_argument=1 ;;
    *) echo "Unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done
case "$STAGE" in prepare|collect|attribute|aggregate|policies|eval|summarize|all) ;;
  *) echo "Unknown STAGE: $STAGE" >&2; exit 2 ;;
esac
is_true() { case "${1,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac; }

PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python}"
REFERENCE_RUN="${REFERENCE_RUN:-$REPO_ROOT/results/AgentAttrPrune/gsm8k/qwen3_8b/train40_correctness_aware}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/AgentAttrPrune/gsm8k/qwen3_8b/train40_baseline_aligned_correctness_aware}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/gsm8k_baseline_aligned_train40}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export PYTHONUNBUFFERED=1 HF_CHAT_ENABLE_THINKING=0 HF_CHAT_CAPTURE_SCORES=0 TOKENIZERS_PARALLELISM=false

if ! is_true "$DRY_RUN"; then
  mkdir -p "$LOG_ROOT"
  exec {launcher_lock}> "$LOG_ROOT/.launcher.lock"
  if ! flock -n "$launcher_lock"; then
    echo "The baseline-aligned GSM8K launcher is already active." >&2
    exit 2
  fi
  log_file="$LOG_ROOT/${STAGE}_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}.log"
  exec > >(tee -a "$log_file") 2>&1
  started=$SECONDS
  trap 'status=$?; printf "[%s] cohort=baseline_source_order_train40 exit_status=%d elapsed_seconds=%d\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" "$((SECONDS - started))"' EXIT
  printf 'Log: %s\nRun: %s\n' "$log_file" "$RUN_DIR"
  printf '[%s] host=%s GPUs=%s cohort=baseline_source_order_train40 generation_seed=0 stage=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)" "$CUDA_VISIBLE_DEVICES" "$STAGE"
fi

run_stage() {
  local stage="$1"
  shift
  local -a command=("$PYTHON_BIN" -u -m experiments_attrprune.baseline_aligned_gsm8k
    --reference-run "$REFERENCE_RUN" --run-dir "$RUN_DIR" --stage "$stage"
    --eval-batch-size 4 --limit 1319 --resume)
  [[ -z "${BASELINE_DATA:-}" ]] || command+=(--baseline-data "$BASELINE_DATA")
  [[ -z "${COHORT_AUDIT_DIR:-}" ]] || command+=(--cohort-audit-dir "$COHORT_AUDIT_DIR")
  command+=("$@")
  if is_true "$DRY_RUN"; then
    printf '%q ' "${command[@]}" --dry-run
    printf '\n'
    return
  fi
  local stage_started=$SECONDS
  printf '[%s] stage=%s %s started\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage" "$*"
  "${command[@]}"
  printf '[%s] stage=%s completed elapsed_seconds=%d\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage" "$((SECONDS-stage_started))"
}

run_evaluations() {
  local percentage
  for percentage in 80 70 60; do run_stage eval --pruning-percent "$percentage"; done
}

if [[ "$STAGE" == all ]]; then
  for stage in prepare collect attribute aggregate policies; do run_stage "$stage"; done
  run_evaluations
  run_stage summarize
elif [[ "$STAGE" == eval ]]; then
  run_evaluations
else
  run_stage "$STAGE"
fi
