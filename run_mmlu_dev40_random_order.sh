#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  DRY_RUN=1 ./run_mmlu_dev40_random_order.sh [calibrate]
  CUDA_VISIBLE_DEVICES=<devices> ./run_mmlu_dev40_random_order.sh [calibrate|collect|attribute]
  ./run_mmlu_dev40_random_order.sh aggregate
  ./run_mmlu_dev40_random_order.sh visualize

calibrate (default) collects 40 questions x 10 independently ordered dense
rollouts, attributes all nodes, aggregates observed edges, and renders matrices.
The reference run fixes the questions, model, prompts, workers, and decoding.
No pruning policies or validation inference are produced by this launcher.

Defaults:
  REFERENCE_RUN=<repo>/results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware
  RUN_NAME=mmlu/qwen3_8b/dev40_random_order10_correctness_aware
  ORDER_SEED=0, ATTR_BATCH_SIZE=4, RESUME=1, DRY_RUN=0.

Overrides:
  REFERENCE_RUN, RESULTS_ROOT, RUN_NAME, ORDER_SEED, ATTR_BATCH_SIZE,
  RESUME, DRY_RUN, CONDA_ENV=agentprune, CONDA_BIN.
  Set CONDA_ENV='' and PYTHON_BIN=/path/to/python to bypass conda run.
  GPU stages require an explicitly selected, nonempty CUDA_VISIBLE_DEVICES.
  DRY_RUN=1 prints the resolved command without running Python or writing files.
EOF
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

STAGE="${1:-calibrate}"
if (( $# > 0 )); then shift; fi
case "${STAGE}" in
  -h|--help|help) usage; exit 0 ;;
  collect|attribute|aggregate|visualize|calibrate) ;;
  *) echo "Unknown stage: ${STAGE}" >&2; usage >&2; exit 2 ;;
esac
if (( $# > 0 )); then
  echo "${STAGE} does not accept positional arguments; use the documented environment overrides." >&2
  exit 2
fi

DRY_RUN="${DRY_RUN:-0}"
if [[ "${STAGE}" == "collect" || "${STAGE}" == "attribute" || "${STAGE}" == "calibrate" ]] \
  && ! is_true "${DRY_RUN}" && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Select available GPUs explicitly with CUDA_VISIBLE_DEVICES before ${STAGE}." >&2
  exit 2
fi

RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/AgentAttrPrune}"
RUN_NAME="${RUN_NAME:-mmlu/qwen3_8b/dev40_random_order10_correctness_aware}"
EXPECTED_RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}"
if [[ -n "${RUN_DIR:-}" && "${RUN_DIR}" != "${EXPECTED_RUN_DIR}" ]]; then
  echo "Set RUN_NAME and RESULTS_ROOT; RUN_DIR must equal ${EXPECTED_RUN_DIR}." >&2
  exit 2
fi
REFERENCE_RUN="${REFERENCE_RUN:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware}"
CONDA_ENV="${CONDA_ENV-agentprune}"
if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=("${CONDA_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/bin/conda}" run --no-capture-output -n "${CONDA_ENV}" python)
else
  PYTHON_CMD=("${PYTHON_BIN:-python}")
fi

COMMAND=(
  "${PYTHON_CMD[@]}" -m experiments_attrprune.random_order_mmlu
  --stage "${STAGE}"
  --reference-run "${REFERENCE_RUN}"
  --run-dir "${EXPECTED_RUN_DIR}"
  --order-seed "${ORDER_SEED:-0}"
  --attr-batch-size "${ATTR_BATCH_SIZE:-4}"
)
if is_true "${RESUME:-1}"; then COMMAND+=(--resume); fi
if is_true "${DRY_RUN}"; then COMMAND+=(--dry-run); fi

printf '%q ' "${COMMAND[@]}"
printf '\n'
if is_true "${DRY_RUN}"; then exit 0; fi
exec "${COMMAND[@]}"
