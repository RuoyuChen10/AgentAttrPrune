#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_CHAT_ENABLE_THINKING=0
export HF_CHAT_CAPTURE_SCORES=0
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/full_dev_a6000x8_tok2048}"
MODEL_PATH="${MODEL_PATH:-/egr/research-optml/ruoyu.chen/Models/Qwen3-8B}"
MMLU_DATA_DIR="${MMLU_DATA_DIR:-}"
VAL_SPLIT="${VAL_SPLIT:-val}"
VAL_START_INDEX="${VAL_START_INDEX:-0}"
VAL_LIMIT="${VAL_LIMIT:-1531}"
SEED="${SEED:-0}"

METRIC="${METRIC:-token_ratio}"
STATISTIC="${STATISTIC:-conditional_mean}"
PRUNING_SCOPE="${PRUNING_SCOPE:-per_round_edge_type}"
PRUNING_PERCENTAGES="${PRUNING_PERCENTAGES:-5 10 20 30 40 50 60 70 80}"

STAGE="${1:-all}"
if (( $# > 0 )); then
  shift
fi
if (( $# > 0 )); then
  PERCENTAGES=("$@")
else
  read -r -a PERCENTAGES <<< "${PRUNING_PERCENTAGES}"
fi

CONDA_BIN="${CONDA_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-agentprune}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_EVAL_TRACES="${SAVE_EVAL_TRACES:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"

SCOPE_SUFFIX=""
if [[ "${PRUNING_SCOPE}" == "global" ]]; then
  SCOPE_SUFFIX="_global"
fi
SWEEP_ROOT="${RUN_DIR}/attribution/static_pruning/${METRIC}_${STATISTIC}${SCOPE_SUFFIX}"
MANIFEST_JSON="${SWEEP_ROOT}/manifest.json"

usage() {
  cat <<'EOF'
Usage:
  ./run_static_mmlu_pruning_sweep.sh STAGE [PERCENTAGE ...]

Stages:
  prepare     Build frozen policies for the selected percentages.
  eval        Evaluate every prepared policy on the MMLU validation selection.
  summarize   Validate all results and write one summary JSON and CSV.
  all         Prepare, evaluate and summarize (default).

Defaults:
  - source run: results/AgentAttrPrune/mmlu/qwen3_8b/full_dev_a6000x8_tok2048
  - importance: token_ratio conditional_mean
  - evaluation: all 1531 MMLU val records
  - percentages: 5 10 20 30 40 50 60 70 80

Useful overrides:
  CUDA_VISIBLE_DEVICES, RUN_DIR, MODEL_PATH, MMLU_DATA_DIR,
  VAL_START_INDEX, VAL_LIMIT, SEED, PRUNING_PERCENTAGES,
  PRUNING_SCOPE=per_round_edge_type|global,
  SAVE_EVAL_TRACES=1, SKIP_COMPLETED=0, DRY_RUN=1.

Examples:
  DRY_RUN=1 ./run_static_mmlu_pruning_sweep.sh all
  PRUNING_SCOPE=global ./run_static_mmlu_pruning_sweep.sh all 10 20 30 40 50
  ./run_static_mmlu_pruning_sweep.sh all 60 70 80
  ./run_static_mmlu_pruning_sweep.sh eval 70
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    ./run_static_mmlu_pruning_sweep.sh all 60 70 80
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

percentage_label() {
  local percentage="$1"
  printf 'prune_%02dpct' "${percentage}"
}

evaluation_name() {
  local percentage="$1"
  printf 'static_%s_%s%s_%s' \
    "${METRIC}" "${STATISTIC}" "${SCOPE_SUFFIX}" \
    "$(percentage_label "${percentage}")"
}

preflight() {
  if [[ "${PRUNING_SCOPE}" != "per_round_edge_type" \
    && "${PRUNING_SCOPE}" != "global" ]]; then
    echo "PRUNING_SCOPE must be per_round_edge_type or global." >&2
    exit 2
  fi
  if [[ ${#PERCENTAGES[@]} -eq 0 ]]; then
    echo "PRUNING_PERCENTAGES must contain at least one integer percentage." >&2
    exit 2
  fi
  local percentage
  for percentage in "${PERCENTAGES[@]}"; do
    if [[ ! "${percentage}" =~ ^[0-9]+$ ]] \
      || (( percentage < 0 || percentage > 100 )); then
      echo "Invalid pruning percentage: ${percentage}" >&2
      exit 2
    fi
  done
  if is_true "${DRY_RUN}"; then
    return
  fi
  if [[ -n "${CONDA_ENV}" && ! -x "${CONDA_BIN}" ]]; then
    echo "Conda executable not found: ${CONDA_BIN}" >&2
    exit 1
  fi
  if [[ ! -f "${RUN_DIR}/run_config.json" ]]; then
    echo "Run configuration not found: ${RUN_DIR}/run_config.json" >&2
    exit 1
  fi
  if [[ ! -f "${RUN_DIR}/attribution/aggregate_edge_matrices.json" ]]; then
    echo "Aggregate matrix not found under ${RUN_DIR}/attribution" >&2
    exit 1
  fi
  if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Model config not found: ${MODEL_PATH}/config.json" >&2
    exit 1
  fi
}

prepare_policies() {
  run_cmd "${PYTHON_CMD[@]}" -m experiments_attrprune.static_pruning \
    prepare \
    --run-dir "${RUN_DIR}" \
    --metric "${METRIC}" \
    --statistic "${STATISTIC}" \
    --pruning-scope "${PRUNING_SCOPE}" \
    --percentages "${PERCENTAGES[@]}"
}

evaluate_policies() {
  local percentage label policy_json eval_name summary_json
  for percentage in "${PERCENTAGES[@]}"; do
    label="$(percentage_label "${percentage}")"
    policy_json="${SWEEP_ROOT}/policies/${label}.json"
    eval_name="$(evaluation_name "${percentage}")"
    summary_json="${RUN_DIR}/evaluation/${eval_name}_${VAL_SPLIT}/evaluation_summary.json"

    if ! is_true "${DRY_RUN}" && [[ ! -f "${policy_json}" ]]; then
      echo "Policy not found: ${policy_json}" >&2
      echo "Run the prepare stage first." >&2
      exit 1
    fi
    if ! is_true "${DRY_RUN}" && is_true "${SKIP_COMPLETED}" \
      && [[ -f "${summary_json}" ]]; then
      echo "Skipping completed evaluation: ${summary_json}"
      continue
    fi

    local args=(
      -m experiments_attrprune.evaluate_mmlu
      --benchmark mmlu
      --model-name "${MODEL_PATH}"
      --run-dir "${RUN_DIR}"
      --mode "${METRIC}"
      --policy-json "${policy_json}"
      --evaluation-name "${eval_name}"
      --split "${VAL_SPLIT}"
      --start-index "${VAL_START_INDEX}"
      --limit "${VAL_LIMIT}"
      --seed "${SEED}"
      --resume
      --no-download
    )
    if [[ -n "${MMLU_DATA_DIR}" ]]; then
      args+=(--data-dir "${MMLU_DATA_DIR}")
    fi
    if is_true "${SAVE_EVAL_TRACES}"; then
      args+=(--save-traces)
    fi
    run_cmd "${PYTHON_CMD[@]}" "${args[@]}"
  done
}

summarize_results() {
  if ! is_true "${DRY_RUN}" && [[ ! -f "${MANIFEST_JSON}" ]]; then
    echo "Sweep manifest not found: ${MANIFEST_JSON}" >&2
    exit 1
  fi
  run_cmd "${PYTHON_CMD[@]}" -m experiments_attrprune.static_pruning \
    summarize --manifest "${MANIFEST_JSON}"
}

case "${STAGE}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  prepare|eval|summarize|all)
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    usage >&2
    exit 2
    ;;
esac

echo "Stage:              ${STAGE}"
echo "Source run:         ${RUN_DIR}"
echo "Model:              ${MODEL_PATH}"
echo "Importance:         ${METRIC}/${STATISTIC}"
echo "Pruning scope:      ${PRUNING_SCOPE}"
echo "Pruning percent:    ${PERCENTAGES[*]}"
echo "Evaluation:         ${VAL_SPLIT}[${VAL_START_INDEX}:$((VAL_START_INDEX + VAL_LIMIT))]"
echo "Visible GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo "Skip completed:     ${SKIP_COMPLETED}"
echo "Dry run:            ${DRY_RUN}"

preflight

case "${STAGE}" in
  prepare) prepare_policies ;;
  eval) evaluate_policies ;;
  summarize) summarize_results ;;
  all)
    prepare_policies
    evaluate_policies
    summarize_results
    ;;
esac
