#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  ./run_gsm8k_train40_uniform_pruning.sh [all|prepare|eval] [PERCENTAGE ...]
  ./run_gsm8k_train40_uniform_pruning.sh summarize

Default: prepare, evaluate 60% then 80%, and summarize.
Reuses the existing GSM8K train40 correctness-aware attribution.
Pruning is allocated independently per round and edge type.
Policies and evaluations use directories without the _global suffix.

Defaults: GPUs 4,5; evaluation batch size 4; all 1319 test questions;
seed 0; resume enabled; generation traces saved.

Parallel example (prepare once, then eval in separate terminals):
  ./run_gsm8k_train40_uniform_pruning.sh prepare 50 60 80
  CUDA_VISIBLE_DEVICES=4,5 ./run_gsm8k_train40_uniform_pruning.sh eval 50
  CUDA_VISIBLE_DEVICES=6,7 ./run_gsm8k_train40_uniform_pruning.sh eval 60 80
  ./run_gsm8k_train40_uniform_pruning.sh summarize  # after all three finish

prepare replaces the sweep manifest with the requested percentages.
Parallel jobs should use eval after a single shared prepare.

Overrides: CUDA_VISIBLE_DEVICES, EVAL_BATCH_SIZE, PYTHON_BIN,
MODEL_PATH, GSM8K_DATA_DIR, RUN_DIR, DRY_RUN=1.
Keep the same batch size when resuming an existing evaluation.
EOF
}

STAGE="${1:-all}"
if (( $# > 0 )); then shift; fi
case "${STAGE}" in
  -h|--help|help) usage; exit 0 ;;
  all|prepare|eval|summarize) ;;
  *) usage >&2; exit 2 ;;
esac
if [[ "${STAGE}" == "summarize" ]] && (( $# > 0 )); then
  echo "summarize uses all percentages in the prepared manifest; it takes no percentages." >&2
  exit 2
fi
if (( $# > 0 )); then
  PERCENTAGES=("$@")
else
  PERCENTAGES=(60 80)
fi
declare -A SEEN_PERCENTAGES=()
for index in "${!PERCENTAGES[@]}"; do
  percentage="${PERCENTAGES[index]}"
  if [[ ! "${percentage}" =~ ^[0-9]{1,3}$ ]] || (( 10#${percentage} > 100 )); then
    echo "Pruning percentages must be integers from 0 to 100: ${percentage}" >&2
    exit 2
  fi
  percentage=$((10#${percentage}))
  if [[ -n "${SEEN_PERCENTAGES[${percentage}]:-}" ]]; then
    echo "Duplicate pruning percentage: ${percentage}" >&2
    exit 2
  fi
  SEEN_PERCENTAGES[${percentage}]=1
  PERCENTAGES[index]="${percentage}"
done

PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python}"
MODEL_PATH="${MODEL_PATH:-/egr/research-optml/ruoyu.chen/Models/Qwen3-8B}"
GSM8K_DATA_DIR="${GSM8K_DATA_DIR:-${REPO_ROOT}/data/GSM8K}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/AgentAttrPrune/gsm8k/qwen3_8b/train40_correctness_aware}"
SWEEP_ROOT="${RUN_DIR}/attribution/static_pruning/token_ratio_conditional_mean"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export HF_CHAT_ENABLE_THINKING=0 HF_CHAT_CAPTURE_SCORES=0
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

if [[ "${STAGE}" == "all" || "${STAGE}" == "prepare" ]]; then
  run_cmd "${PYTHON_BIN}" -m experiments_attrprune.static_pruning prepare \
    --run-dir "${RUN_DIR}" \
    --metric token_ratio --statistic conditional_mean \
    --pruning-scope per_round_edge_type --percentages "${PERCENTAGES[@]}"
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "eval" ]]; then
  for percentage in "${PERCENTAGES[@]}"; do
    printf -v label 'prune_%02dpct' "${percentage}"
    run_cmd "${PYTHON_BIN}" -m experiments_attrprune.evaluate_benchmark \
      --benchmark gsm8k --model-name "${MODEL_PATH}" \
      --data-dir "${GSM8K_DATA_DIR}" --run-dir "${RUN_DIR}" \
      --mode token_ratio \
      --policy-json "${SWEEP_ROOT}/policies/${label}.json" \
      --evaluation-name "static_token_ratio_conditional_mean_${label}" \
      --split test --start-index 0 --limit 1319 \
      --eval-batch-size "${EVAL_BATCH_SIZE}" --seed 0 \
      --resume --save-traces --no-download
  done
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "summarize" ]]; then
  run_cmd "${PYTHON_BIN}" -m experiments_attrprune.static_pruning summarize \
    --manifest "${SWEEP_ROOT}/manifest.json"
fi
