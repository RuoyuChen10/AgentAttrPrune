#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_CHAT_ENABLE_THINKING=0
export HF_CHAT_CAPTURE_SCORES=0
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

CONDA_BIN="${CONDA_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-agentprune}"
MODEL_PATH="${MODEL_PATH:-/egr/research-optml/ruoyu.chen/Models/Qwen3-8B}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/full_dev_a6000x8_tok2048}"
VAL_START_INDEX="${VAL_START_INDEX:-0}"
VAL_LIMIT="${VAL_LIMIT:-1531}"
PRUNING_PERCENTAGES="${PRUNING_PERCENTAGES:-10 20 40 50}"
read -r -a PERCENTAGES <<< "${PRUNING_PERCENTAGES}"

POLICY_DIR="${RUN_DIR}/attribution/static_pruning/token_ratio_conditional_mean/policies"
DENSE_SUMMARY="${RUN_DIR}/evaluation/dense_val/evaluation_summary.json"

RUN_TAG="$(hostname -s)_$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/full_mmlu_sweep_a6000x8_${RUN_TAG}}"
DENSE_LOG="${LOG_DIR}/dense_0pct.log"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable not found: ${CONDA_BIN}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/run_config.json" ]]; then
  echo "Run config not found: ${RUN_DIR}/run_config.json" >&2
  exit 1
fi
for percentage in "${PERCENTAGES[@]}"; do
  if [[ ! "${percentage}" =~ ^[0-9]+$ ]] || (( percentage < 1 || percentage > 100 )); then
    echo "Invalid pruning percentage: ${percentage}" >&2
    exit 1
  fi
  printf -v policy_label 'prune_%02dpct' "${percentage}"
  policy_json="${POLICY_DIR}/${policy_label}.json"
  if [[ ! -f "${policy_json}" ]]; then
    echo "Pruning policy not found: ${policy_json}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}"

echo "Repository:      ${REPO_ROOT}"
echo "Run directory:   ${RUN_DIR}"
echo "Validation data: val[${VAL_START_INDEX}:$((VAL_START_INDEX + VAL_LIMIT))]"
echo "Visible GPUs:    ${CUDA_VISIBLE_DEVICES}"
echo "Pruning rates:   0 ${PERCENTAGES[*]}"
echo "Logs:            ${LOG_DIR}"

dense_status=0
if [[ -f "${DENSE_SUMMARY}" ]]; then
  echo "0% dense result already exists; skipping: ${DENSE_SUMMARY}"
else
  echo "===== Starting 0% dense evaluation ====="
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python -m experiments_attrprune.evaluate_mmlu \
    --benchmark mmlu \
    --model-name "${MODEL_PATH}" \
    --run-dir "${RUN_DIR}" \
    --mode dense \
    --split val \
    --start-index "${VAL_START_INDEX}" \
    --limit "${VAL_LIMIT}" \
    --seed 0 \
    --resume \
    --no-download \
    2>&1 | tee "${DENSE_LOG}"
  dense_status=${PIPESTATUS[0]}
  echo "0% dense exit status: ${dense_status}"
fi

overall_status=${dense_status}
for percentage in "${PERCENTAGES[@]}"; do
  printf -v policy_label 'prune_%02dpct' "${percentage}"
  policy_json="${POLICY_DIR}/${policy_label}.json"
  evaluation_name="static_token_ratio_conditional_mean_${policy_label}"
  summary_json="${RUN_DIR}/evaluation/${evaluation_name}_val/evaluation_summary.json"
  pruning_log="${LOG_DIR}/prune_${percentage}pct.log"

  if [[ -f "${summary_json}" ]]; then
    echo "${percentage}% pruning result already exists; skipping: ${summary_json}"
    continue
  fi

  echo "===== Starting ${percentage}% pruning evaluation ====="
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python -m experiments_attrprune.evaluate_mmlu \
    --benchmark mmlu \
    --model-name "${MODEL_PATH}" \
    --run-dir "${RUN_DIR}" \
    --mode token_ratio \
    --policy-json "${policy_json}" \
    --evaluation-name "${evaluation_name}" \
    --split val \
    --start-index "${VAL_START_INDEX}" \
    --limit "${VAL_LIMIT}" \
    --seed 0 \
    --resume \
    --no-download \
    2>&1 | tee "${pruning_log}"
  pruning_status=${PIPESTATUS[0]}
  echo "${percentage}% pruning exit status: ${pruning_status}"
  if (( pruning_status != 0 )); then
    overall_status=1
  fi
done

echo "===== Final status ====="
echo "0% summary: ${DENSE_SUMMARY}"
for percentage in "${PERCENTAGES[@]}"; do
  printf -v policy_label 'prune_%02dpct' "${percentage}"
  evaluation_name="static_token_ratio_conditional_mean_${policy_label}"
  echo "${percentage}% summary: ${RUN_DIR}/evaluation/${evaluation_name}_val/evaluation_summary.json"
done
echo "Logs: ${LOG_DIR}"

if (( overall_status != 0 )); then
  exit 1
fi
