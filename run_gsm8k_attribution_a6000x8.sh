#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

export CONDA_ENV="${CONDA_ENV:-agentprune}"
export MODEL_PATH="${MODEL_PATH:-/egr/research-optml/ruoyu.chen/Models/Qwen3-8B}"
export GSM8K_DATA_DIR="${GSM8K_DATA_DIR:-${REPO_ROOT}/data/GSM8K}"
export RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/AgentAttrPrune}"
export RUN_NAME="${RUN_NAME:-gsm8k/qwen3_8b/train285_a6000x8_tok2048}"

export CALIBRATION_START_INDEX="${CALIBRATION_START_INDEX:-0}"
export CALIBRATION_LIMIT="${CALIBRATION_LIMIT:-285}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export NUM_ROUNDS="${NUM_ROUNDS:-2}"
export TEMPERATURE="${TEMPERATURE:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

export ATTR_BATCH_SIZE="${ATTR_BATCH_SIZE:-8}"
export ATTR_SAMPLE_SCOPE="${ATTR_SAMPLE_SCOPE:-all}"
export ATTR_NODE_SCOPE="${ATTR_NODE_SCOPE:-all}"
export TARGET_RECOVERY="${TARGET_RECOVERY:-0.95}"
export SALIENCY_FLOOR="${SALIENCY_FLOOR:-0.4}"
export POLICY_PRUNING_RATE="${POLICY_PRUNING_RATE:-0.25}"
export RESUME="${RESUME:-1}"

RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}"
RUN_TAG="$(hostname -s)_$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/gsm8k_attribution_a6000x8_${RUN_TAG}.log}"

if [[ ! -x "${REPO_ROOT}/run_full_gsm8k.sh" ]]; then
  echo "Missing runner: ${REPO_ROOT}/run_full_gsm8k.sh" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Missing model config: ${MODEL_PATH}/config.json" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "Benchmark:           GSM8K"
echo "Output directory:    ${RUN_DIR}"
echo "Calibration records: train[${CALIBRATION_START_INDEX}:$((CALIBRATION_START_INDEX + CALIBRATION_LIMIT))]"
echo "Workers/rounds:      ${NUM_WORKERS}/${NUM_ROUNDS}"
echo "Max new tokens:      ${MAX_NEW_TOKENS}"
echo "Visible GPUs:        ${CUDA_VISIBLE_DEVICES}"
echo "Log:                 ${LOG_PATH}"

./run_full_gsm8k.sh calibrate 2>&1 | tee "${LOG_PATH}"

echo "Completed GSM8K trace collection and attribution."
echo "Results: ${RUN_DIR}"
