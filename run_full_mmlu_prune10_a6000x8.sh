#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

CONDA_BIN="/egr/research-optml/ruoyu.chen/anaconda3/bin/conda"
CONDA_ENV="agentprune"
MODEL_PATH="/egr/research-optml/ruoyu.chen/Models/Qwen3-8B"
RUN_DIR="${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/full_dev_a6000x8_tok2048"
POLICY_JSON="${RUN_DIR}/attribution/static_pruning/token_ratio_conditional_mean/policies/prune_10pct.json"
SUMMARY_JSON="${RUN_DIR}/evaluation/static_token_ratio_conditional_mean_prune_10pct_val/evaluation_summary.json"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable not found: ${CONDA_BIN}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/run_config.json" ]]; then
  echo "Run config not found: ${RUN_DIR}/run_config.json" >&2
  exit 1
fi
if [[ ! -f "${POLICY_JSON}" ]]; then
  echo "Pruning policy not found: ${POLICY_JSON}" >&2
  exit 1
fi
if [[ -f "${SUMMARY_JSON}" ]]; then
  echo "10% full-set MMLU result already exists; not starting a duplicate run."
  echo "Summary: ${SUMMARY_JSON}"
  exit 0
fi

mkdir -p "${REPO_ROOT}/logs"
LOG_FILE="${REPO_ROOT}/logs/full_mmlu_prune10_$(hostname -s)_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_CHAT_ENABLE_THINKING=0
export HF_CHAT_CAPTURE_SCORES=0
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export PYTHONUNBUFFERED=1

DEFER_ARGS=()
for sample_index in ${DEFER_INDICES:-}; do
  if [[ ! "${sample_index}" =~ ^[0-9]+$ ]]; then
    echo "Invalid DEFER_INDICES value: ${sample_index}" >&2
    exit 1
  fi
  DEFER_ARGS+=(--defer-index "${sample_index}")
done

echo "Starting 10% full-set MMLU evaluation in the foreground."
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Log: ${LOG_FILE}"
if (( ${#DEFER_ARGS[@]} )); then
  echo "Deferred sample indices: ${DEFER_INDICES}"
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -m experiments_attrprune.evaluate_mmlu \
  --benchmark mmlu \
  --model-name "${MODEL_PATH}" \
  --run-dir "${RUN_DIR}" \
  --mode token_ratio \
  --policy-json "${POLICY_JSON}" \
  --evaluation-name static_token_ratio_conditional_mean_prune_10pct \
  --split val \
  --start-index 0 \
  --limit 1531 \
  --seed 0 \
  --resume \
  "${DEFER_ARGS[@]}" \
  --no-download \
  2>&1 | tee "${LOG_FILE}"

echo "Completed 10% full-set MMLU evaluation."
echo "Summary: ${SUMMARY_JSON}"
