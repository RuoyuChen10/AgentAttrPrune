#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=2,3 ./run_mmlu_reordered_skip_empty80.sh both
  CUDA_VISIBLE_DEVICES=2,3 ./run_mmlu_reordered_skip_empty80.sh direct
  ./run_mmlu_reordered_skip_empty80.sh both prepare
  DRY_RUN=1 ./run_mmlu_reordered_skip_empty80.sh both

Branches: direct, recalibrate, both (default; sequential separate processes).
Stages: eval (default), prepare (CPU only).
Only re-evaluates 80% with the empty Round 1 skip rule. No new calibration.
Uses independent *_val_skip_empty_round1 checkpoints; original results stay intact.
Defaults: 1531 val questions, batch=4, resume enabled, static edges.
Overrides: PYTHON_BIN, SOURCE_RUN, DIRECT_RUN_DIR, RECALIBRATE_RUN_DIR,
           VAL_LIMIT, EVAL_BATCH_SIZE, LOG_ROOT, DRY_RUN.
DRY_RUN only prints commands; it does not create files or execute Python.
EOF
}
is_true() { case "${1,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac; }
BRANCH="${1:-both}"
case "${BRANCH}" in
  -h|--help|help) usage; exit 0 ;;
  direct|recalibrate|both) ;;
  *) usage >&2; exit 2 ;;
esac
STAGE="${2:-eval}"
if (( $# > 2 )) || [[ "${STAGE}" != eval && "${STAGE}" != prepare ]]; then
  usage >&2; exit 2
fi
DRY_RUN="${DRY_RUN:-0}"
if ! is_true "${DRY_RUN}" && [[ "${STAGE}" == eval && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Select idle CUDA_VISIBLE_DEVICES explicitly for evaluation." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/mmlu_reordered_skip_empty80}"
export PYTHONUNBUFFERED=1
export HF_CHAT_ENABLE_THINKING=0
export HF_CHAT_CAPTURE_SCORES=0
export TOKENIZERS_PARALLELISM=false
if [[ "${BRANCH}" == both ]]; then BRANCHES=(direct recalibrate); else BRANCHES=("${BRANCH}"); fi

run_branch() (
  local branch="$1"
  local -a command=("${PYTHON_BIN}" -u -m experiments_attrprune.reordered_skip_empty_evaluation
    --branch "${branch}" --stage "${STAGE}" --limit "${VAL_LIMIT:-1531}"
    --eval-batch-size "${EVAL_BATCH_SIZE:-4}" --resume)
  if [[ -n "${SOURCE_RUN:-}" ]]; then command+=(--source-run "${SOURCE_RUN}"); fi
  if [[ "${branch}" == direct && -n "${DIRECT_RUN_DIR:-}" ]]; then
    command+=(--run-dir "${DIRECT_RUN_DIR}")
  elif [[ "${branch}" == recalibrate && -n "${RECALIBRATE_RUN_DIR:-}" ]]; then
    command+=(--run-dir "${RECALIBRATE_RUN_DIR}")
  fi
  if is_true "${DRY_RUN}"; then
    printf '%q ' "${command[@]}" --dry-run
    printf '\n'
    exit 0
  fi
  mkdir -p "${LOG_ROOT}/${branch}"
  exec {branch_lock}> "${LOG_ROOT}/${branch}/.launcher.lock"
  if ! flock -n "${branch_lock}"; then
    echo "A skip-empty ${branch} launcher is already active." >&2
    exit 2
  fi
  local log_file="${LOG_ROOT}/${branch}/${STAGE}_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}.log"
  exec > >(tee -a "${log_file}") 2>&1
  local started=${SECONDS}
  trap 'status=$?; printf "[%s] branch=%s skip_empty_round1=1 exit_status=%d elapsed_seconds=%d\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${branch}" "${status}" "$((SECONDS - started))"' EXIT
  printf 'Log: %s\n' "${log_file}"
  printf '[%s] host=%s GPUs=%s branch=%s stage=%s pruning=80%% skip_empty_round1=1\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)" "${CUDA_VISIBLE_DEVICES:-CPU}" "${branch}" "${STAGE}"
  "${command[@]}"
)
for branch in "${BRANCHES[@]}"; do run_branch "${branch}"; done
