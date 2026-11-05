#!/usr/bin/env bash
# Independent calibration cohorts; generation/evaluation seed stays frozen.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=6,7 ./run_gsm8k_train40_resampling.sh 1 2
  CUDA_VISIBLE_DEVICES=6,7 ./run_gsm8k_train40_resampling.sh 2
  STAGE=prepare ./run_gsm8k_train40_resampling.sh 1 2
  DRY_RUN=1 ./run_gsm8k_train40_resampling.sh 1 2

Each calibration seed independently selects 40 training questions without replacement.
Different cohorts may overlap. The original seed0 run is read-only.
For each new cohort: dense collection -> unchanged correctness-aware attribution
-> aggregation -> static global policies -> test25/50/60/70/80/90 -> summary.
Every GPU stage and pruning rate uses a separate process, sequentially.
If Round1 spatial AND temporal are empty, reviewer reads Round0 directly.
Defaults preserve Qwen3-8B, two rounds, canonical order, temperature0,2048tokens,
thinking off, generation seed0, serial calibration, attributionbatch4, evalbatch4.
All 1319 test questions are evaluated and traces/checkpoints saved; resume enabled.

Overrides: STAGE=all|prepare|collect|attribute|aggregate|policies|eval|summarize,
           PYTHON_BIN, REFERENCE_RUN, RUNS_ROOT, LOG_ROOT, DRY_RUN.
DRY_RUN only prints commands and never creates files or loads models.
Results are saved independently for each cohort and rate; no curves are drawn.
EOF
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if (( $# == 0 )); then set -- 1 2; fi
SEEDS=("$@")
declare -A seen=()
for seed in "${SEEDS[@]}"; do
  case "${seed}" in 1|2) ;; *) echo "Calibration seeds must be 1 or 2." >&2; exit 2 ;; esac
  if [[ -n "${seen[${seed}]:-}" ]]; then echo "Duplicate calibration seed: ${seed}" >&2; exit 2; fi
  seen[${seed}]=1
done
STAGE="${STAGE:-all}"
case "${STAGE}" in all|prepare|collect|attribute|aggregate|policies|eval|summarize) ;;
  *) echo "Unknown STAGE: ${STAGE}" >&2; exit 2 ;;
esac
is_true() { case "${1,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac; }
DRY_RUN="${DRY_RUN:-0}"
if ! is_true "${DRY_RUN}" && [[ "${STAGE}" =~ ^(all|collect|attribute|eval)$ ]] \
    && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Select idle CUDA_VISIBLE_DEVICES explicitly for GPU stages." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python}"
REFERENCE_RUN="${REFERENCE_RUN:-${REPO_ROOT}/results/AgentAttrPrune/gsm8k/qwen3_8b/train40_correctness_aware}"
RUNS_ROOT="${RUNS_ROOT:-$(dirname -- "${REFERENCE_RUN}")}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/gsm8k_train40_resampling}"
export PYTHONUNBUFFERED=1 HF_CHAT_ENABLE_THINKING=0 HF_CHAT_CAPTURE_SCORES=0 TOKENIZERS_PARALLELISM=false

run_seed() (
  local seed="$1"
  local run_dir="${RUNS_ROOT}/train40_resample_seed${seed}_correctness_aware"
  if ! is_true "${DRY_RUN}"; then
    mkdir -p "${LOG_ROOT}/seed_${seed}"
    exec {seed_lock}> "${LOG_ROOT}/seed_${seed}/.launcher.lock"
    if ! flock -n "${seed_lock}"; then echo "Calibration seed ${seed} launcher is already active." >&2; exit 2; fi
    local log_file="${LOG_ROOT}/seed_${seed}/${STAGE}_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}.log"
    exec > >(tee -a "${log_file}") 2>&1
    local started=${SECONDS}
    trap 'status=$?; printf "[%s] calibration_seed=%s exit_status=%d elapsed_seconds=%d\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${seed}" "${status}" "$((SECONDS - started))"' EXIT
    printf 'Log: %s\nRun: %s\n' "${log_file}" "${run_dir}"
    printf '[%s] host=%s GPUs=%s calibration_seed=%s generation_seed=0 stage=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)" "${CUDA_VISIBLE_DEVICES:-CPU}" "${seed}" "${STAGE}"
  fi
  run_stage() {
    local stage="$1"
    shift
    local -a command=("${PYTHON_BIN}" -u -m experiments_attrprune.resampled_gsm8k
      --calibration-seed "${seed}" --reference-run "${REFERENCE_RUN}" --run-dir "${run_dir}"
      --stage "${stage}" --resume "$@")
    if is_true "${DRY_RUN}"; then printf '%q ' "${command[@]}" --dry-run; printf '\n'; return; fi
    local stage_started=${SECONDS}
    printf '[%s] stage=%s %s started\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${stage}" "$*"
    "${command[@]}"
    printf '[%s] stage=%s completed elapsed_seconds=%d\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${stage}" "$((SECONDS-stage_started))"
  }
  run_evaluations() {
    local percentage
    for percentage in 25 50 60 70 80 90; do run_stage eval --pruning-percent "${percentage}"; done
  }
  if [[ "${STAGE}" == all ]]; then
    for stage in prepare collect attribute aggregate policies; do run_stage "${stage}"; done
    run_evaluations
    run_stage summarize
  elif [[ "${STAGE}" == eval ]]; then
    run_evaluations
  else
    run_stage "${STAGE}"
  fi
)
for seed in "${SEEDS[@]}"; do run_seed "${seed}"; done
