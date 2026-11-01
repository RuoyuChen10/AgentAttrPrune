#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_random_order_sampling.sh global 50 60
  CUDA_VISIBLE_DEVICES=2,3 ./run_mmlu_random_order_sampling.sh uniform 50 60
  PREPARE_ONLY=1 ./run_mmlu_random_order_sampling.sh global 50 60
  DRY_RUN=1 ./run_mmlu_random_order_sampling.sh uniform 50 60

Use the completed random-order dev40 calibration, retaining per-round matrices.
Importance is used as the logit; each inference samples sigmoid(logit) edges.
Global pruning ranks all 65 round/type candidates together. Uniform pruning
ranks separately within round-0 spatial, round-1 spatial, and round-1 temporal.
Each percentage starts from the original matrix, not the previous percentage.
The two percentages run sequentially on the selected pair of GPUs.
No new calibration, attribution, or AgentPrune training is started.

Defaults:
  RUN_DIR=results/AgentAttrPrune/mmlu/qwen3_8b/dev40_random_order1_correctness_aware
  VAL_LIMIT=1531, SEED=0, one two-round inference per val question, resume enabled.
  PYTHON_BIN=/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python
  OUTPUT_ROOT may override evaluation output (separate subdirs by scope/rate).
  LOG_ROOT defaults to logs/mmlu_random_order_sampling; timestamps are UTC.
GPU execution requires CUDA_VISIBLE_DEVICES. PREPARE_ONLY and DRY_RUN use CPU.
EOF
}

is_true() {
  case "${1,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac
}

SCOPE="${1:-}"
case "${SCOPE}" in
  -h|--help|help) usage; exit 0 ;;
  global|uniform) shift ;;
  *) echo "Select global or uniform." >&2; usage >&2; exit 2 ;;
esac
if (( $# == 0 )); then set -- 50 60; fi
PERCENTAGES=("$@")
declare -A seen=()
for percentage in "${PERCENTAGES[@]}"; do
  case "${percentage}" in
    50|60) ;;
    *) echo "This experiment supports independent 50% and 60% settings." >&2; exit 2 ;;
  esac
  if [[ -n "${seen[${percentage}]:-}" ]]; then
    echo "Duplicate pruning percentage: ${percentage}" >&2
    exit 2
  fi
  seen[${percentage}]=1
done

DRY_RUN="${DRY_RUN:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
if ! is_true "${DRY_RUN}" && ! is_true "${PREPARE_ONLY}" && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Select available GPUs explicitly using CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/dev40_random_order1_correctness_aware}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/mmlu_random_order_sampling}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

run_one() (
  local percentage="$1"
  local -a command=("${PYTHON_BIN}" -u -m experiments_attrprune.evaluate_sampled_mmlu
    --run-dir "${RUN_DIR}" --pruning-scope "${SCOPE}" --pruning-percent "${percentage}"
    --round-policy per_round --split val --limit "${VAL_LIMIT:-1531}" --seed "${SEED:-0}" --resume)
  if [[ -n "${OUTPUT_ROOT:-}" ]]; then
    command+=(--output-dir "${OUTPUT_ROOT}/${SCOPE}/prune_${percentage}pct")
  fi
  if is_true "${PREPARE_ONLY}"; then command+=(--prepare-only); fi
  if is_true "${DRY_RUN}"; then
    printf '%q ' "${command[@]}" --dry-run
    printf '\n'
    exit 0
  fi
  mkdir -p "${LOG_ROOT}/${SCOPE}"
  local log_file="${LOG_ROOT}/${SCOPE}/prune_${percentage}pct_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}.log"
  exec > >(tee -a "${log_file}") 2>&1
  local started=${SECONDS}
  trap 'status=$?; printf "[%s] scope=%s pruning=%s%% exit_status=%d elapsed_seconds=%d\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SCOPE}" "${percentage}" "${status}" "$((SECONDS - started))"' EXIT
  printf 'Log: %s\n' "${log_file}"
  printf '[%s] host=%s GPUs=%s scope=%s pruning=%s%% prepare_only=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)" "${CUDA_VISIBLE_DEVICES:-CPU}" "${SCOPE}" "${percentage}" "${PREPARE_ONLY}"
  "${command[@]}"
)

for percentage in "${PERCENTAGES[@]}"; do run_one "${percentage}"; done
