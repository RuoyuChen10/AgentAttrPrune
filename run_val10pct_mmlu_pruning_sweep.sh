#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_CHAT_ENABLE_THINKING=0
export HF_CHAT_CAPTURE_SCORES=0
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SOURCE_RUN="${SOURCE_RUN:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/full_dev_a6000x8_tok2048}"
MODEL_PATH="${MODEL_PATH:-/egr/research-optml/ruoyu.chen/Models/Qwen3-8B}"
CONDA_BIN="${CONDA_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-agentprune}"

VAL_START_INDEX="${VAL_START_INDEX:-0}"
VAL_LIMIT="${VAL_LIMIT:-153}"
PRUNING_PERCENTAGES="${PRUNING_PERCENTAGES:-5 10 20 40 50}"
METRIC="${METRIC:-token_ratio}"
STATISTIC="${STATISTIC:-conditional_mean}"

RUN_TAG="$(hostname -s)_$(date +%Y%m%d_%H%M%S)_$$"
OUTPUT_RUN="${OUTPUT_RUN:-${SOURCE_RUN}_val10pct_${RUN_TAG}}"

if [[ ! -f "${SOURCE_RUN}/run_config.json" ]]; then
  echo "Missing source run configuration: ${SOURCE_RUN}/run_config.json" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_RUN}/attribution/aggregate_edge_matrices.json" ]]; then
  echo "Missing aggregate edge matrix under: ${SOURCE_RUN}/attribution" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Missing model configuration: ${MODEL_PATH}/config.json" >&2
  exit 1
fi
if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable is not available: ${CONDA_BIN}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_RUN}/attribution"
cp "${SOURCE_RUN}/run_config.json" "${OUTPUT_RUN}/run_config.json"
cp "${SOURCE_RUN}/attribution/aggregate_edge_matrices.json" \
  "${OUTPUT_RUN}/attribution/aggregate_edge_matrices.json"

PYTHON_CMD=(
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python
)

echo "Source run:          ${SOURCE_RUN}"
echo "Output run:          ${OUTPUT_RUN}"
echo "Validation records:  val[${VAL_START_INDEX}:$((VAL_START_INDEX + VAL_LIMIT))]"
echo "Pruning percentages: ${PRUNING_PERCENTAGES}"
echo "Importance matrix:   ${METRIC}/${STATISTIC}"
echo "Visible GPUs:        ${CUDA_VISIBLE_DEVICES}"

{
  # True unpruned baseline using the dense graph.
  "${PYTHON_CMD[@]}" -m experiments_attrprune.evaluate_mmlu \
    --benchmark mmlu \
    --model-name "${MODEL_PATH}" \
    --run-dir "${OUTPUT_RUN}" \
    --mode dense \
    --split val \
    --start-index "${VAL_START_INDEX}" \
    --limit "${VAL_LIMIT}" \
    --seed 0 \
    --no-download

  # Build frozen policies, evaluate every requested rate, and write JSON/CSV.
  RUN_DIR="${OUTPUT_RUN}" \
  MODEL_PATH="${MODEL_PATH}" \
  CONDA_BIN="${CONDA_BIN}" \
  CONDA_ENV="${CONDA_ENV}" \
  VAL_START_INDEX="${VAL_START_INDEX}" \
  VAL_LIMIT="${VAL_LIMIT}" \
  PRUNING_PERCENTAGES="${PRUNING_PERCENTAGES}" \
  METRIC="${METRIC}" \
  STATISTIC="${STATISTIC}" \
  SAVE_EVAL_TRACES=0 \
  SKIP_COMPLETED=1 \
  ./run_static_mmlu_pruning_sweep.sh all
} 2>&1 | tee "${OUTPUT_RUN}/val10pct_sweep.log"

echo "Dense summary: ${OUTPUT_RUN}/evaluation/dense_val/evaluation_summary.json"
echo "Sweep CSV:     ${OUTPUT_RUN}/evaluation/static_${METRIC}_${STATISTIC}_sweep_val/summary.csv"
echo "Sweep JSON:    ${OUTPUT_RUN}/evaluation/static_${METRIC}_${STATISTIC}_sweep_val/summary.json"
echo "Run log:       ${OUTPUT_RUN}/val10pct_sweep.log"
