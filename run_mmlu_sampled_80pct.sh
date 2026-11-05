#!/usr/bin/env bash
# Run the two 80% sampled-policy ablations sequentially on one selected GPU pair.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=4,5 ./run_mmlu_sampled_80pct.sh
  CUDA_VISIBLE_DEVICES=4,5 ./run_mmlu_sampled_80pct.sh global uniform
  CUDA_VISIBLE_DEVICES=4,5 ./run_mmlu_sampled_80pct.sh uniform
  PREPARE_ONLY=1 ./run_mmlu_sampled_80pct.sh
  DRY_RUN=1 ./run_mmlu_sampled_80pct.sh

Default: global then uniform, each at 80% candidate pruning and eval batch 4.
Use the completed random-order dev40 attribution without new calibration.
Each question samples edges with sigmoid(importance) after pruning and spatial
cycle rejection; the sampled DAG determines execution order. Preserve all two
worker rounds, including when a sampled round has no edges.
Qwen3-8B, temperature 0, 2048 tokens, thinking disabled, 1531 val questions,
per-round policies, sampling seed 0, traces saved, resume enabled.
Global and uniform each retain 13 of 65 candidates before edge sampling.
Existing 50%/60% runs are not modified or replayed.

Overrides: RUN_DIR, OUTPUT_ROOT, LOG_ROOT, PYTHON_BIN, PREPARE_ONLY, DRY_RUN.
OUTPUT_ROOT defaults to <RUN_DIR>/evaluation/agentprune_sampling/per_round_sigmoid.
Each scope writes <OUTPUT_ROOT>/<scope>/prune_80pct.
DRY_RUN only prints commands, creates no files, and loads no model.
EOF
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if (( $# == 0 )); then set -- global uniform; fi
SCOPES=("$@")
declare -A seen=()
for scope in "${SCOPES[@]}"; do
  case "${scope}" in global|uniform) ;; *) echo "Select global and/or uniform; this launcher runs only 80%." >&2; exit 2 ;; esac
  if [[ -n "${seen[${scope}]:-}" ]]; then echo "Duplicate scope: ${scope}" >&2; exit 2; fi
  seen[${scope}]=1
done
is_true() { case "${1,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac; }
DRY_RUN="${DRY_RUN:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
if ! is_true "${DRY_RUN}" && ! is_true "${PREPARE_ONLY}" && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Select available CUDA_VISIBLE_DEVICES explicitly." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/dev40_random_order1_correctness_aware}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_DIR}/evaluation/agentprune_sampling/per_round_sigmoid}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/mmlu_sampled_80pct}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_CHAT_ENABLE_THINKING=0 HF_CHAT_CAPTURE_SCORES=0

run_scope() (
  local scope="$1"
  local -a command=("${PYTHON_BIN}" -u -m experiments_attrprune.evaluate_sampled_mmlu_batched
    --run-dir "${RUN_DIR}" --output-dir "${OUTPUT_ROOT}/${scope}/prune_80pct"
    --pruning-scope "${scope}" --pruning-percent 80 --eval-batch-size 4
    --round-policy per_round --split val --limit 1531 --seed 0 --resume)
  if is_true "${PREPARE_ONLY}"; then command+=(--prepare-only); fi
  if is_true "${DRY_RUN}"; then
    printf '%q ' "${command[@]}" --dry-run
    printf '\n'
    exit 0
  fi
  mkdir -p "${LOG_ROOT}/${scope}"
  exec {launcher_lock}> "${LOG_ROOT}/${scope}/.launcher.lock"
  if ! flock -n "${launcher_lock}"; then echo "An 80% ${scope} launcher is already active." >&2; exit 2; fi
  local log_file="${LOG_ROOT}/${scope}/prune_80pct_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}.log"
  exec > >(tee -a "${log_file}") 2>&1
  local started=${SECONDS}
  trap 'status=$?; printf "[%s] scope=%s pruning=80%% eval_batch_size=4 exit_status=%d elapsed_seconds=%d\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${scope}" "${status}" "$((SECONDS-started))"' EXIT
  printf 'Log: %s\nOutput: %s\n' "${log_file}" "${OUTPUT_ROOT}/${scope}/prune_80pct"
  printf '[%s] host=%s GPUs=%s scope=%s pruning=80%% eval_batch_size=4 prepare_only=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)" "${CUDA_VISIBLE_DEVICES:-CPU}" "${scope}" "${PREPARE_ONLY}"
  "${command[@]}"
)
for scope in "${SCOPES[@]}"; do run_scope "${scope}"; done
