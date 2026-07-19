#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-/egr/research-optml/ruoyu.chen/Models/Qwen3-8B}"
MMLU_DATA_DIR="${MMLU_DATA_DIR:-${REPO_ROOT}/data/MMLU}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/AgentAttrPrune}"
RUN_NAME="${RUN_NAME:-}"

DEV_LIMIT="${DEV_LIMIT:-285}"
DEV_START_INDEX="${DEV_START_INDEX:-0}"
VAL_LIMIT="${VAL_LIMIT:-1531}"
NUM_WORKERS="${NUM_WORKERS:-5}"
NUM_ROUNDS="${NUM_ROUNDS:-2}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SEED="${SEED:-0}"

ATTR_BATCH_SIZE="${ATTR_BATCH_SIZE:-4}"
ATTR_SAMPLE_LIMIT="${ATTR_SAMPLE_LIMIT:-}"
ATTR_EVENT_LIMIT="${ATTR_EVENT_LIMIT:-}"
ATTR_SAMPLE_SCOPE="${ATTR_SAMPLE_SCOPE:-all}"
ATTR_NODE_SCOPE="${ATTR_NODE_SCOPE:-all}"
TARGET_TOKEN_LIMIT="${TARGET_TOKEN_LIMIT:-}"
TARGET_RECOVERY="${TARGET_RECOVERY:-0.95}"
SALIENCY_FLOOR="${SALIENCY_FLOOR:-0.4}"
POLICY_PRUNING_RATE="${POLICY_PRUNING_RATE:-0.25}"

RESUME="${RESUME:-0}"
OVERWRITE_ATTRIBUTION="${OVERWRITE_ATTRIBUTION:-0}"
KEEP_GOING="${KEEP_GOING:-0}"
SAVE_EVAL_TRACES="${SAVE_EVAL_TRACES:-0}"
DRY_RUN="${DRY_RUN:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_BIN="${CONDA_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-}"

export HF_CHAT_ENABLE_THINKING=0
export HF_CHAT_CAPTURE_SCORES=0
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

usage() {
  cat <<'EOF'
Usage:
  ./experiments_attrprune/run_mmlu.sh STAGE

Stages:
  traces             Dense dev calibration; save a full trace for every sample.
  attribution        Attribute all saved dev traces; aggregate correct-only policies.
  render             Rebuild per-node sentence/insertion SVGs from cached event JSON.
  aggregate          Rebuild aggregate matrices/policies from cached events.
  one-sample         Run one dev sample and save every node result, without aggregate.
  calibrate          Run all selected dev traces, attribution, and correct-only aggregate.
  eval-dense         Evaluate the dense graph on the complete val selection.
  eval-binary        Evaluate the binary attribution policy.
  eval-token-ratio   Evaluate the primary token-ratio attribution policy.
  eval-word-ratio    Evaluate the word-ratio attribution policy.
  eval               Run dense, binary, and token-ratio evaluations.
  compare            Build comparison_summary.json and .csv.
  all                Run traces, attribution, all evaluations, and compare.

Environment overrides:
  MODEL_PATH, MMLU_DATA_DIR, RESULTS_ROOT, RUN_NAME, DEV_START_INDEX,
  DEV_LIMIT, VAL_LIMIT,
  NUM_WORKERS, NUM_ROUNDS, TEMPERATURE, MAX_NEW_TOKENS, SEED,
  TARGET_RECOVERY, SALIENCY_FLOOR, POLICY_PRUNING_RATE, ATTR_BATCH_SIZE,
  ATTR_SAMPLE_LIMIT, ATTR_EVENT_LIMIT, ATTR_SAMPLE_SCOPE, ATTR_NODE_SCOPE,
  TARGET_TOKEN_LIMIT,
  RESUME=1, OVERWRITE_ATTRIBUTION=1, KEEP_GOING=1,
  SAVE_EVAL_TRACES=1, DRY_RUN=1.

Python environment:
  Use the active environment by default. Set CONDA_ENV=agentprune (and
  optionally CONDA_BIN=...) to execute every command through conda run.
EOF
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=("${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python)
else
  PYTHON_CMD=("${PYTHON_BIN}")
fi

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if ! is_true "${DRY_RUN}"; then
    "$@"
  fi
}

common_args() {
  COMMON_ARGS=(
    --model-name "${MODEL_PATH}"
    --data-dir "${MMLU_DATA_DIR}"
    --results-root "${RESULTS_ROOT}"
    --seed "${SEED}"
  )
  if [[ -n "${RUN_NAME}" ]]; then
    COMMON_ARGS+=(--run-name "${RUN_NAME}")
  fi
}

run_traces() {
  common_args
  local args=(
    -m experiments_attrprune.collect_correct
    "${COMMON_ARGS[@]}"
    --split dev
    --start-index "${DEV_START_INDEX}"
    --limit "${DEV_LIMIT}"
    --num-workers "${NUM_WORKERS}"
    --num-rounds "${NUM_ROUNDS}"
    --temperature "${TEMPERATURE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
  )
  if is_true "${RESUME}"; then
    args+=(--resume)
  fi
  run_cmd "${PYTHON_CMD[@]}" "${args[@]}"
}

run_attribution() {
  local attribute_only="${1:-0}"
  common_args
  local args=(
    -m experiments_attrprune.attribute_edges
    "${COMMON_ARGS[@]}"
    --target-mode all
    --score-type logit
    --score-reduction sum
    --target-recovery "${TARGET_RECOVERY}"
    --saliency-floor "${SALIENCY_FLOOR}"
    --batch-size "${ATTR_BATCH_SIZE}"
    --sample-scope "${ATTR_SAMPLE_SCOPE}"
    --node-scope "${ATTR_NODE_SCOPE}"
    --min-incoming 2
    --min-selected-sentences 2
    --min-sentence-words 4
    --policy-pruning-rate "${POLICY_PRUNING_RATE}"
  )
  if [[ -n "${ATTR_SAMPLE_LIMIT}" ]]; then
    args+=(--sample-limit "${ATTR_SAMPLE_LIMIT}")
  fi
  if [[ -n "${ATTR_EVENT_LIMIT}" ]]; then
    args+=(--event-limit "${ATTR_EVENT_LIMIT}")
  fi
  if [[ -n "${TARGET_TOKEN_LIMIT}" ]]; then
    args+=(--target-token-limit "${TARGET_TOKEN_LIMIT}")
  fi
  if is_true "${OVERWRITE_ATTRIBUTION}"; then
    args+=(--overwrite)
  fi
  if is_true "${KEEP_GOING}"; then
    args+=(--keep-going)
  fi
  if is_true "${attribute_only}"; then
    args+=(--attribute-only)
  fi
  run_cmd "${PYTHON_CMD[@]}" "${args[@]}"
}

run_aggregate() {
  common_args
  run_cmd "${PYTHON_CMD[@]}" -m experiments_attrprune.attribute_edges \
    "${COMMON_ARGS[@]}" \
    --policy-pruning-rate "${POLICY_PRUNING_RATE}" \
    --aggregate-only
}

run_render() {
  common_args
  run_cmd "${PYTHON_CMD[@]}" -m experiments_attrprune.attribute_edges \
    "${COMMON_ARGS[@]}" \
    --saliency-floor "${SALIENCY_FLOOR}" \
    --render-only
}

run_evaluation() {
  local mode="$1"
  common_args
  local args=(
    -m experiments_attrprune.evaluate_mmlu
    "${COMMON_ARGS[@]}"
    --mode "${mode}"
    --split val
    --limit "${VAL_LIMIT}"
  )
  if is_true "${SAVE_EVAL_TRACES}"; then
    args+=(--save-traces)
  fi
  run_cmd "${PYTHON_CMD[@]}" "${args[@]}"
}

run_compare() {
  local args=(
    -m experiments_attrprune.compare
    --model-name "${MODEL_PATH}"
    --results-root "${RESULTS_ROOT}"
    --seed "${SEED}"
  )
  if [[ -n "${RUN_NAME}" ]]; then
    args+=(--run-name "${RUN_NAME}")
  fi
  run_cmd "${PYTHON_CMD[@]}" "${args[@]}"
}

STAGE="${1:-all}"
case "${STAGE}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  traces|attribution|render|aggregate|one-sample|calibrate|eval-dense|eval-binary|eval-token-ratio|eval-word-ratio|eval|compare|all)
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "${STAGE}" == "one-sample" ]]; then
  DEV_LIMIT=1
  ATTR_SAMPLE_LIMIT=1
fi

echo "Stage:        ${STAGE}"
echo "Model:        ${MODEL_PATH}"
echo "MMLU data:    ${MMLU_DATA_DIR}"
echo "Results root: ${RESULTS_ROOT}"
echo "Run name:     ${RUN_NAME:-mmlu/<model-slug>/seed_${SEED}}"
echo "Workers:      ${NUM_WORKERS}; rounds: ${NUM_ROUNDS}"
echo "Dev start:    ${DEV_START_INDEX}; dev/val: ${DEV_LIMIT}/${VAL_LIMIT}"
echo "Dry run:      ${DRY_RUN}"

case "${STAGE}" in
  traces) run_traces ;;
  attribution) run_attribution ;;
  render) run_render ;;
  aggregate) run_aggregate ;;
  one-sample)
    run_traces
    run_attribution 1
    ;;
  calibrate)
    run_traces
    run_attribution
    ;;
  eval-dense) run_evaluation dense ;;
  eval-binary) run_evaluation binary ;;
  eval-token-ratio) run_evaluation token_ratio ;;
  eval-word-ratio) run_evaluation word_ratio ;;
  eval)
    run_evaluation dense
    run_evaluation binary
    run_evaluation token_ratio
    ;;
  compare) run_compare ;;
  all)
    run_traces
    run_attribution
    run_evaluation dense
    run_evaluation binary
    run_evaluation token_ratio
    run_compare
    ;;
esac
