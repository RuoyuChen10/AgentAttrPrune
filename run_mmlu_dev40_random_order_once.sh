#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  ./run_mmlu_dev40_random_order_once.sh prepare
  CUDA_VISIBLE_DEVICES=2,3 ./run_mmlu_dev40_random_order_once.sh [calibrate]
  CUDA_VISIBLE_DEVICES=2,3 ./run_mmlu_dev40_random_order_once.sh attribute
  ./run_mmlu_dev40_random_order_once.sh aggregate
  ./run_mmlu_dev40_random_order_once.sh visualize

Use the same 40 questions and exactly repeat_00 from the frozen 400-rollout run.
prepare validates and copies existing traces and completed attribution on CPU.
calibrate prepares, resumes missing attribution, aggregates, and visualizes.
There is no new generation, pruning, or validation evaluation.
Stop the old 400-rollout attribution in its tmux pane before a GPU stage.

Defaults:
  SOURCE_RUN=results/AgentAttrPrune/mmlu/qwen3_8b/dev40_random_order10_correctness_aware
  RUN_NAME=mmlu/qwen3_8b/dev40_random_order1_correctness_aware
  ATTR_BATCH_SIZE=4, ORDER_SEED=0; exactly one repeat per question.
  CONDA_ENV=agentprune; set CONDA_ENV='' and PYTHON_BIN to use another interpreter.
  DRY_RUN=1 prints commands without changing files or loading models.
  Logs include stage start/end times and elapsed seconds, under
  logs/mmlu_dev40_random_order1/ (override LOG_DIR).
EOF
}

is_true() {
  case "${1,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac
}

STAGE="${1:-calibrate}"
if (( $# > 0 )); then shift; fi
case "${STAGE}" in
  -h|--help|help) usage; exit 0 ;;
  prepare|calibrate|attribute|aggregate|visualize) ;;
  *) echo "Unknown stage: ${STAGE}" >&2; usage >&2; exit 2 ;;
esac
if (( $# > 0 )); then
  echo "${STAGE} does not accept positional arguments." >&2
  exit 2
fi

DRY_RUN="${DRY_RUN:-0}"
REFERENCE_OVERRIDE="${REFERENCE_RUN:-}"
ORDER_SEED_OVERRIDE="${ORDER_SEED:-}"
ATTR_BATCH_OVERRIDE="${ATTR_BATCH_SIZE:-}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/AgentAttrPrune}"
SOURCE_RUN="${SOURCE_RUN:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/dev40_random_order10_correctness_aware}"
REFERENCE_RUN="${REFERENCE_RUN:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware}"
RUN_NAME="${RUN_NAME:-mmlu/qwen3_8b/dev40_random_order1_correctness_aware}"
EXPECTED_RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}"
if [[ -n "${RUN_DIR:-}" && "${RUN_DIR}" != "${EXPECTED_RUN_DIR}" ]]; then
  echo "Set RUN_NAME and RESULTS_ROOT; RUN_DIR must equal ${EXPECTED_RUN_DIR}." >&2
  exit 2
fi
if [[ "${STAGE}" == "calibrate" || "${STAGE}" == "attribute" ]]; then
  if ! is_true "${DRY_RUN}" && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Select available GPUs explicitly with CUDA_VISIBLE_DEVICES." >&2
    exit 2
  fi
  if ! is_true "${DRY_RUN}" && [[ -f "${SOURCE_RUN}/.random_order.lock" ]]; then
    # A shared read lock refuses overlap with the old writer and prevents it
    # from restarting while the smaller experiment is using the same GPUs.
    exec {source_lock_fd}< "${SOURCE_RUN}/.random_order.lock"
    if ! flock -n -s "${source_lock_fd}"; then
      echo "The old 400-rollout run is still active. Stop it with Ctrl-C in its tmux pane, then rerun this command." >&2
      exit 2
    fi
  fi
fi

CONDA_ENV="${CONDA_ENV-agentprune}"
if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=("${CONDA_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/bin/conda}" run --no-capture-output -n "${CONDA_ENV}" python)
else
  PYTHON_CMD=("${PYTHON_BIN:-python}")
fi
export PYTHONUNBUFFERED=1
if [[ -f "${SOURCE_RUN}/run_config.json" ]]; then
  # Read frozen settings as data; never evaluate paths as shell code. Capturing
  # the command first preserves a Python failure instead of hiding it in < <().
  frozen_output="$("${PYTHON_CMD[@]}" -c '
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())["random_order_experiment"]
values = [config["reference"]["run_dir"], config["order_seed"], config["attribution_batch_size"]]
if any("\n" in str(value) or "\r" in str(value) for value in values):
    raise SystemExit("Frozen settings cannot contain line breaks")
print("\n".join(map(str, values)))
' "${SOURCE_RUN}/run_config.json")"
  mapfile -t frozen_settings <<< "${frozen_output}"
  if (( ${#frozen_settings[@]} != 3 )); then
    echo "Could not read the source run's frozen attribution settings." >&2
    exit 2
  fi
  if [[ -n "${REFERENCE_OVERRIDE}" && "${REFERENCE_OVERRIDE}" != "${frozen_settings[0]}" ]] \
    || [[ -n "${ORDER_SEED_OVERRIDE}" && "${ORDER_SEED_OVERRIDE}" != "${frozen_settings[1]}" ]] \
    || [[ -n "${ATTR_BATCH_OVERRIDE}" && "${ATTR_BATCH_OVERRIDE}" != "${frozen_settings[2]}" ]]; then
    echo "The once-only experiment must preserve the source reference, order seed, and attribution batch size; remove conflicting overrides." >&2
    exit 2
  fi
  REFERENCE_RUN="${frozen_settings[0]}"
  ORDER_SEED="${frozen_settings[1]}"
  ATTR_BATCH_SIZE="${frozen_settings[2]}"
fi
PREPARE_CMD=("${PYTHON_CMD[@]}" -m experiments_attrprune.random_order_subset
  --source-run "${SOURCE_RUN}" --run-dir "${EXPECTED_RUN_DIR}" --resume)
COMMON_ARGS=(--reference-run "${REFERENCE_RUN}" --run-dir "${EXPECTED_RUN_DIR}"
  --repeats 1 --order-seed "${ORDER_SEED:-0}" --attr-batch-size "${ATTR_BATCH_SIZE:-4}" --resume)

run_stage() {
  local step="$1"
  local started=${SECONDS}
  local -a command
  if [[ "${step}" == "prepare" ]]; then
    command=("${PREPARE_CMD[@]}")
  else
    command=("${PYTHON_CMD[@]}" -m experiments_attrprune.random_order_mmlu --stage "${step}" "${COMMON_ARGS[@]}")
  fi
  if is_true "${DRY_RUN}"; then
    printf '%q ' "${command[@]}" --dry-run
    printf '\n'
    return
  fi
  printf '[%s] stage=%s started\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${step}"
  "${command[@]}"
  printf '[%s] stage=%s completed elapsed_seconds=%d\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${step}" "$((SECONDS - started))"
}

if ! is_true "${DRY_RUN}"; then
  LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/mmlu_dev40_random_order1}"
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/${STAGE}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  printf 'Log: %s\nRun: %s\nSource: %s\n' "${LOG_FILE}" "${EXPECTED_RUN_DIR}" "${SOURCE_RUN}"
  printf '[%s] host=%s GPUs=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)" "${CUDA_VISIBLE_DEVICES:-CPU}"
  launch_started=${SECONDS}
  trap 'launch_status=$?; printf "[%s] exit_status=%d elapsed_seconds=%d\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${launch_status}" "$((SECONDS - launch_started))"' EXIT
fi

if [[ "${STAGE}" == "calibrate" ]]; then
  for step in prepare attribute aggregate visualize; do run_stage "${step}"; done
else
  run_stage "${STAGE}"
fi
