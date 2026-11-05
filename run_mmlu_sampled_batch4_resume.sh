#!/usr/bin/env bash
# Preserve completed serial answers; batch only unfinished sampled-graph questions.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_sampled_batch4_resume.sh global
  CUDA_VISIBLE_DEVICES=2,3 ./run_mmlu_sampled_batch4_resume.sh uniform
  PREPARE_ONLY=1 ./run_mmlu_sampled_batch4_resume.sh global
  DRY_RUN=1 ./run_mmlu_sampled_batch4_resume.sh uniform

Continue only the unfinished 60% sampled-policy evaluation with batch size 4.
Stop the corresponding old serial process before running this launcher.
The evaluator acquires its source lock and refuses an active source writer.
Completed batch-1 answers/traces remain unchanged and are imported by reference;
only missing questions run. The final result records mixed batch-1/batch-4 use.
The completed 50% evaluations are not replayed or modified.

Defaults:
  SOURCE_EVALUATION=<random-order-dev40>/evaluation/agentprune_sampling/
                    per_round_sigmoid/<scope>/prune_60pct
  OUTPUT_DIR=<same scope>/prune_60pct_batch4_continuation
  resume enabled; original question identities, sampling seeds and masks retained.

Overrides: SOURCE_EVALUATION, OUTPUT_DIR, PYTHON_BIN, LOG_ROOT,
           PREPARE_ONLY=1, DRY_RUN=1.
DRY_RUN only prints commands; it creates no files and loads no model.
EOF
}

SCOPE="${1:-}"
case "${SCOPE}" in
  -h|--help|help) usage; exit 0 ;;
  global|uniform) ;;
  *) echo "Select global or uniform." >&2; usage >&2; exit 2 ;;
esac
if (( $# != 1 )); then
  echo "This launcher resumes only 60%; pass one scope and no percentages." >&2
  exit 2
fi
is_true() { case "${1,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac; }
DRY_RUN="${DRY_RUN:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
if ! is_true "${DRY_RUN}" && ! is_true "${PREPARE_ONLY}" && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Select available CUDA_VISIBLE_DEVICES explicitly." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python}"
SOURCE_EVALUATION="${SOURCE_EVALUATION:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/dev40_random_order1_correctness_aware/evaluation/agentprune_sampling/per_round_sigmoid/${SCOPE}/prune_60pct}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname -- "${SOURCE_EVALUATION}")/prune_60pct_batch4_continuation}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/mmlu_sampled_batch4_continuation}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_CHAT_ENABLE_THINKING=0 HF_CHAT_CAPTURE_SCORES=0
command=("${PYTHON_BIN}" -u -m experiments_attrprune.continue_sampled_mmlu
  --source-evaluation "${SOURCE_EVALUATION}" --output-dir "${OUTPUT_DIR}"
  --eval-batch-size 4 --resume)
if is_true "${DRY_RUN}"; then
  printf '%q ' "${command[@]}" --dry-run
  printf '\n'
  exit 0
fi
if is_true "${PREPARE_ONLY}"; then command+=(--prepare-only); fi
mkdir -p "${LOG_ROOT}/${SCOPE}"
exec {launcher_lock}> "${LOG_ROOT}/${SCOPE}/.launcher.lock"
if ! flock -n "${launcher_lock}"; then
  echo "A batch-4 ${SCOPE} continuation launcher is already active." >&2
  exit 2
fi
log_file="${LOG_ROOT}/${SCOPE}/prune_60pct_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}.log"
exec > >(tee -a "${log_file}") 2>&1
started=${SECONDS}
trap 'status=$?; printf "[%s] scope=%s batch4_continuation=1 exit_status=%d elapsed_seconds=%d\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SCOPE}" "${status}" "$((SECONDS - started))"' EXIT
printf 'Log: %s\nSource: %s\nOutput: %s\n' "${log_file}" "${SOURCE_EVALUATION}" "${OUTPUT_DIR}"
printf '[%s] host=%s GPUs=%s scope=%s pruning=60%% continuation_batch=4 prepare_only=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)" "${CUDA_VISIBLE_DEVICES:-CPU}" "${SCOPE}" "${PREPARE_ONLY}"
"${command[@]}"
