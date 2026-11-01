#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=4,5 ./run_mmlu_reordered_pruning.sh direct [all] [80 60 50]
  CUDA_VISIBLE_DEVICES=6,7 ./run_mmlu_reordered_pruning.sh recalibrate [all] [80 60 50]
  ./run_mmlu_reordered_pruning.sh direct prepare
  DRY_RUN=1 ./run_mmlu_reordered_pruning.sh recalibrate

Stages: all (default), prepare, collect, attribute, aggregate, visualize, eval, summarize.
Global static pruning of 45 fixed-order candidates, with no Bernoulli sampling.
Default evaluation: 1531 MMLU val questions, batch=4, percentages 80 then 60 then 50.
Calibration: same dev40, one dense rollout per question, generation serial, attribution batch=4.
Each GPU stage/rate uses a separate process. Resume is always enabled.
Overrides: SOURCE_RUN, RUN_DIR, VAL_LIMIT, EVAL_BATCH_SIZE, ATTR_BATCH_SIZE,
           PYTHON_BIN, LOG_ROOT, DRY_RUN.
DRY_RUN prints commands without running Python, creating files, or loading a model.
EOF
}

is_true() { case "${1,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac; }
BRANCH="${1:-}"
case "${BRANCH}" in
  -h|--help|help) usage; exit 0 ;;
  direct|recalibrate) shift ;;
  *) usage >&2; exit 2 ;;
esac
STAGE=all
if (( $# > 0 )) && [[ ! "$1" =~ ^[0-9]+$ ]]; then STAGE="$1"; shift; fi
case "${STAGE}" in
  all|prepare|collect|attribute|aggregate|visualize|eval|summarize) ;;
  *) echo "Unknown stage: ${STAGE}" >&2; exit 2 ;;
esac
if [[ "${BRANCH}" == direct && "${STAGE}" =~ ^(collect|attribute|aggregate)$ ]]; then
  echo "The direct branch uses prepare; it does not generate or re-attribute calibration traces." >&2
  exit 2
fi
if (( $# == 0 )); then set -- 80 60 50; fi
PERCENTAGES=("$@")
declare -A seen=()
for percentage in "${PERCENTAGES[@]}"; do
  case "${percentage}" in 80|60|50) ;; *) echo "Supported rates: 80 60 50." >&2; exit 2 ;; esac
  if [[ -n "${seen[${percentage}]:-}" ]]; then echo "Duplicate percentage: ${percentage}" >&2; exit 2; fi
  seen[${percentage}]=1
done
DRY_RUN="${DRY_RUN:-0}"
if ! is_true "${DRY_RUN}" && [[ "${STAGE}" =~ ^(all|collect|attribute|eval)$ ]] \
  && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Select CUDA_VISIBLE_DEVICES explicitly for ${STAGE}." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python}"
SOURCE_RUN="${SOURCE_RUN:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/dev40_random_order1_correctness_aware}"
if [[ "${BRANCH}" == direct ]]; then SUFFIX=direct; else SUFFIX=recalibrated; fi
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/dev40_rowminuscol_${SUFFIX}_correctness_aware}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/mmlu_reordered}"
export PYTHONUNBUFFERED=1
export HF_CHAT_ENABLE_THINKING=0
export HF_CHAT_CAPTURE_SCORES=0
export TOKENIZERS_PARALLELISM=false
COMMON=(--branch "${BRANCH}" --source-run "${SOURCE_RUN}" --run-dir "${RUN_DIR}"
  --percentages "${PERCENTAGES[@]}" --limit "${VAL_LIMIT:-1531}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-4}" --attr-batch-size "${ATTR_BATCH_SIZE:-4}" --resume)

run_stage() {
  local stage="$1"
  shift
  local -a command=("${PYTHON_BIN}" -u -m experiments_attrprune.reordered_mmlu
    "${COMMON[@]}" --stage "${stage}" "$@")
  if is_true "${DRY_RUN}"; then
    printf '%q ' "${command[@]}" --dry-run
    printf '\n'
    return
  fi
  local started=${SECONDS}
  printf '[%s] stage=%s %s started\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${stage}" "$*"
  "${command[@]}"
  printf '[%s] stage=%s completed elapsed_seconds=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${stage}" "$((SECONDS - started))"
}
run_evaluations() {
  local percentage
  for percentage in "${PERCENTAGES[@]}"; do
    run_stage eval --pruning-percent "${percentage}"
  done
}

if ! is_true "${DRY_RUN}"; then
  mkdir -p "${LOG_ROOT}/${BRANCH}"
  exec {launcher_lock}> "${LOG_ROOT}/${BRANCH}/.launcher.lock"
  if ! flock -n "${launcher_lock}"; then
    echo "A ${BRANCH} launcher is already active; use its existing tmux/log." >&2
    exit 2
  fi
  LOG_FILE="${LOG_ROOT}/${BRANCH}/${STAGE}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  started=${SECONDS}
  trap 'status=$?; printf "[%s] branch=%s exit_status=%d elapsed_seconds=%d\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${BRANCH}" "${status}" "$((SECONDS - started))"' EXIT
  printf 'Log: %s\nRun: %s\n' "${LOG_FILE}" "${RUN_DIR}"
  printf '[%s] host=%s GPUs=%s branch=%s stage=%s rates=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)" "${CUDA_VISIBLE_DEVICES:-CPU}" "${BRANCH}" "${STAGE}" "${PERCENTAGES[*]}"
fi

if [[ "${STAGE}" == all ]]; then
  run_stage prepare
  if [[ "${BRANCH}" == recalibrate ]]; then
    for stage in collect attribute aggregate; do run_stage "${stage}"; done
  fi
  run_stage visualize
  run_evaluations
  run_stage summarize
elif [[ "${STAGE}" == eval ]]; then
  run_evaluations
else
  run_stage "${STAGE}"
fi
