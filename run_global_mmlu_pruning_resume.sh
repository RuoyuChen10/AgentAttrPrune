#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# CUDA_VISIBLE_DEVICES is deliberately required from the caller so GPU
# allocation remains under external control.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set by the caller." >&2
  echo "Example: CUDA_VISIBLE_DEVICES=4,5,6,7 $0 10 20 40 50 60 70 80" >&2
  exit 2
fi

CONDA_BIN="${CONDA_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-agentprune}"
MODEL_PATH="${MODEL_PATH:-/egr/research-optml/ruoyu.chen/Models/Qwen3-8B}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/full_dev_a6000x8_tok2048}"
VAL_START_INDEX="${VAL_START_INDEX:-0}"
VAL_LIMIT="${VAL_LIMIT:-1531}"
SEED="${SEED:-0}"
if [[ -z "${VALIDATION_PYTHON:-}" ]]; then
  VALIDATION_PYTHON="$(command -v python3 || true)"
fi

# Use DEFER_INDICES=269 for the long pass on limited GPUs. Explicitly set
# DEFER_INDICES='' in a later invocation to fill deferred samples and write
# final summaries. Commas and whitespace are both accepted.
DEFER_INDICES="${DEFER_INDICES-269}"

if (( $# > 0 )); then
  PERCENTAGES=("$@")
else
  PERCENTAGES=(10 20 40 50 60 70 80)
fi

export HF_CHAT_ENABLE_THINKING="${HF_CHAT_ENABLE_THINKING:-0}"
export HF_CHAT_CAPTURE_SCORES="${HF_CHAT_CAPTURE_SCORES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

fail() {
  echo "Error: $*" >&2
  exit 1
}

[[ -x "${CONDA_BIN}" ]] || fail "Conda executable not found: ${CONDA_BIN}"
[[ -f "${MODEL_PATH}/config.json" ]] || fail "Model config not found: ${MODEL_PATH}"
[[ -f "${RUN_DIR}/run_config.json" ]] || fail "Run config not found: ${RUN_DIR}/run_config.json"
[[ -n "${VALIDATION_PYTHON}" && -x "${VALIDATION_PYTHON}" ]] \
  || fail "python3 is required to validate resumable checkpoints"

[[ "${VAL_START_INDEX}" =~ ^[0-9]+$ ]] \
  || fail "VAL_START_INDEX must be a non-negative integer"
[[ "${VAL_LIMIT}" =~ ^[1-9][0-9]*$ ]] \
  || fail "VAL_LIMIT must be a positive integer"
[[ "${SEED}" =~ ^[0-9]+$ ]] || fail "SEED must be a non-negative integer"

for percentage in "${PERCENTAGES[@]}"; do
  [[ "${percentage}" =~ ^[0-9]+$ ]] \
    || fail "Invalid pruning percentage: ${percentage}"
  (( percentage >= 0 && percentage <= 100 )) \
    || fail "Pruning percentage must be between 0 and 100: ${percentage}"
done

NORMALIZED_DEFER_INDICES="${DEFER_INDICES//,/ }"
DEFERRED_INDICES=()
DEFER_ARGS=()
SELECTION_STOP=$((VAL_START_INDEX + VAL_LIMIT))
for sample_index in ${NORMALIZED_DEFER_INDICES}; do
  [[ "${sample_index}" =~ ^[0-9]+$ ]] \
    || fail "Invalid DEFER_INDICES value: ${sample_index}"
  (( sample_index >= VAL_START_INDEX && sample_index < SELECTION_STOP )) \
    || fail "Deferred index ${sample_index} is outside the evaluation selection"
  if [[ " ${DEFERRED_INDICES[*]} " != *" ${sample_index} "* ]]; then
    DEFERRED_INDICES+=("${sample_index}")
    DEFER_ARGS+=(--defer-index "${sample_index}")
  fi
done

echo "Global MMLU pruning resume"
echo "GPUs:               ${CUDA_VISIBLE_DEVICES}"
echo "Pruning percent:    ${PERCENTAGES[*]}"
echo "Evaluation:         val[${VAL_START_INDEX}:${SELECTION_STOP}]"
echo "Deferred indices:   ${DEFERRED_INDICES[*]:-(none)}"
echo "Run directory:      ${RUN_DIR}"

for percentage in "${PERCENTAGES[@]}"; do
  printf -v label 'prune_%02dpct' "${percentage}"
  evaluation_name="static_token_ratio_conditional_mean_global_${label}"
  policy_json="${RUN_DIR}/attribution/static_pruning/token_ratio_conditional_mean_global/policies/${label}.json"
  evaluation_dir="${RUN_DIR}/evaluation/${evaluation_name}_val"
  predictions_jsonl="${evaluation_dir}/predictions.jsonl"
  summary_json="${evaluation_dir}/evaluation_summary.json"

  [[ -f "${policy_json}" ]] || fail "Global policy not found: ${policy_json}"

  if [[ -f "${summary_json}" ]]; then
    echo "[skip] ${percentage}% already complete: ${summary_json}"
    continue
  fi

  echo "[run] global pruning ${percentage}%"
  command_status=0
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python -m experiments_attrprune.evaluate_mmlu \
    --benchmark mmlu \
    --model-name "${MODEL_PATH}" \
    --run-dir "${RUN_DIR}" \
    --mode token_ratio \
    --policy-json "${policy_json}" \
    --evaluation-name "${evaluation_name}" \
    --split val \
    --start-index "${VAL_START_INDEX}" \
    --limit "${VAL_LIMIT}" \
    --seed "${SEED}" \
    --resume \
    "${DEFER_ARGS[@]}" \
    --no-download || command_status=$?

  if [[ -f "${summary_json}" ]]; then
    echo "[done] ${percentage}% complete: ${summary_json}"
    continue
  fi

  [[ -s "${predictions_jsonl}" ]] \
    || fail "${percentage}% stopped without a predictions checkpoint (exit ${command_status})"

  validation_status=0
  validation_output="$(
    "${VALIDATION_PYTHON}" - \
      "${predictions_jsonl}" \
      "${VAL_START_INDEX}" \
      "${SELECTION_STOP}" \
      "${DEFERRED_INDICES[@]}" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

predictions_path = Path(sys.argv[1])
selection_start = int(sys.argv[2])
selection_stop = int(sys.argv[3])
expected_missing = sorted({int(value) for value in sys.argv[4:]})

indices = []
with predictions_path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "sample_index" not in record:
            raise ValueError(f"line {line_number} has no sample_index")
        indices.append(int(record["sample_index"]))

unique_indices = set(indices)
selected_indices = set(range(selection_start, selection_stop))
missing = sorted(selected_indices - unique_indices)
outside = sorted(unique_indices - selected_indices)
details = {
    "lines": len(indices),
    "unique": len(unique_indices),
    "missing": missing,
    "outside_selection": outside,
}
print(json.dumps(details, separators=(",", ":")))

if len(indices) != len(unique_indices):
    raise SystemExit(1)
if outside:
    raise SystemExit(1)
if missing != expected_missing:
    raise SystemExit(1)
PY
  )" || validation_status=$?

  if (( validation_status == 0 && ${#DEFERRED_INDICES[@]} > 0 )); then
    echo "[deferred] ${percentage}% completed all non-deferred samples: ${validation_output}"
    continue
  fi

  fail "${percentage}% stopped unexpectedly (exit ${command_status}); ${validation_output}"
done

MANIFEST_JSON="${RUN_DIR}/attribution/static_pruning/token_ratio_conditional_mean_global/manifest.json"
[[ -f "${MANIFEST_JSON}" ]] || fail "Global sweep manifest not found: ${MANIFEST_JSON}"

missing_summaries=0
while IFS= read -r evaluation_directory; do
  if [[ ! -f "${evaluation_directory}/evaluation_summary.json" ]]; then
    missing_summaries=$((missing_summaries + 1))
  fi
done < <(
  "${VALIDATION_PYTHON}" - "${MANIFEST_JSON}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
for policy in manifest["policies"]:
    print(policy["evaluation_directory"])
PY
)

if (( missing_summaries == 0 )); then
  echo "[summarize] all global evaluations are complete"
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python -m experiments_attrprune.static_pruning \
    summarize --manifest "${MANIFEST_JSON}"
else
  echo "[checkpointed] ${missing_summaries} global evaluation(s) still lack a final summary."
  if (( ${#DEFERRED_INDICES[@]} > 0 )); then
    echo "Rerun later with DEFER_INDICES='' and enough GPUs to fill deferred samples."
  fi
fi
