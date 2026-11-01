#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=6,7 ./run_gsm8k_train40_correctness_aware.sh [calibrate]
  ./run_gsm8k_train40_correctness_aware.sh prepare [PERCENTAGE ...]
  CUDA_VISIBLE_DEVICES=6,7 ./run_gsm8k_train40_correctness_aware.sh eval [PERCENTAGE ...]
  ./run_gsm8k_train40_correctness_aware.sh summarize

calibrate (default) records train indices 0..39 after the seed-888 shuffle,
attributes every node, and aggregates all 40 samples. It never starts test inference.
prepare and summarize do not load a model. eval runs requested policies sequentially.

Defaults:
  RUN_NAME=gsm8k/qwen3_8b/train40_correctness_aware
  PRUNING_PERCENTAGES="50 60 80"; global token_ratio / conditional_mean.
  Four workers, two rounds, temperature 0, seed 0, MAX_NEW_TOKENS=2048.
  ATTR_BATCH_SIZE=4, EVAL_BATCH_SIZE=4, TEST_LIMIT=1319, RESUME=1.
  0% is available explicitly and retains all 28 worker edges.
  The four final-reviewer edges are always retained and excluded from the ratio.

Overrides:
  RUN_NAME, RESULTS_ROOT, MODEL_PATH, GSM8K_DATA_DIR, CONDA_ENV, PYTHON_BIN,
  ATTR_BATCH_SIZE, EVAL_BATCH_SIZE, MAX_NEW_TOKENS, TEST_LIMIT, SEED,
  SAVE_EVAL_TRACES=1 (default), DRY_RUN=1.
  GPU stages require an explicit nonempty CUDA_VISIBLE_DEVICES.
  Use a new RUN_NAME when changing the experiment or generation configuration.
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
  calibrate|prepare|eval|summarize) ;;
  *) echo "Unknown stage: ${STAGE}" >&2; usage >&2; exit 2 ;;
esac
if [[ "${STAGE}" == "calibrate" || "${STAGE}" == "summarize" ]] && (( $# > 0 )); then
  echo "${STAGE} does not accept pruning percentages." >&2
  exit 2
fi

export DRY_RUN="${DRY_RUN:-0}"
if [[ "${STAGE}" == "calibrate" || "${STAGE}" == "eval" ]] \
  && ! is_true "${DRY_RUN}" && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Select available GPUs explicitly, for example CUDA_VISIBLE_DEVICES=6,7, before ${STAGE}." >&2
  exit 2
fi

export CONDA_ENV="${CONDA_ENV-agentprune}"
export MODEL_PATH="${MODEL_PATH:-/egr/research-optml/ruoyu.chen/Models/Qwen3-8B}"
export GSM8K_DATA_DIR="${GSM8K_DATA_DIR:-${REPO_ROOT}/data/GSM8K}"
export RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/AgentAttrPrune}"
export RUN_NAME="${RUN_NAME:-gsm8k/qwen3_8b/train40_correctness_aware}"
EXPECTED_RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}"
if [[ -n "${RUN_DIR:-}" && "${RUN_DIR}" != "${EXPECTED_RUN_DIR}" ]]; then
  echo "Set RUN_NAME and RESULTS_ROOT; RUN_DIR must equal ${EXPECTED_RUN_DIR}." >&2
  exit 2
fi
export RUN_DIR="${EXPECTED_RUN_DIR}"

# Fix the dataset and selection even when another experiment exported settings.
export BENCHMARK=gsm8k DATA_DIR="${GSM8K_DATA_DIR}"
export CALIBRATION_SPLIT=train CALIBRATION_START_INDEX=0 CALIBRATION_LIMIT=40
export EVALUATION_SPLIT=test EVALUATION_LIMIT="${TEST_LIMIT:-1319}"
export ATTR_EXPECTED_START_INDEX=0 ATTR_EXPECTED_SAMPLE_COUNT=40
export ATTRIBUTION_MODE=correctness-aware AGGREGATION_SCOPE=all
export ATTR_SAMPLE_SCOPE=all ATTR_NODE_SCOPE=all
export ATTR_SAMPLE_LIMIT="" ATTR_EVENT_LIMIT="" TARGET_TOKEN_LIMIT=""
export NUM_WORKERS=4 NUM_ROUNDS=2 TEMPERATURE=0 TARGET_RECOVERY=0.95
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}" SEED="${SEED:-0}"
export ATTR_BATCH_SIZE="${ATTR_BATCH_SIZE:-4}" EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export RESUME="${RESUME:-1}" SAVE_EVAL_TRACES="${SAVE_EVAL_TRACES:-1}"
export HF_CHAT_ENABLE_THINKING=0 HF_CHAT_CAPTURE_SCORES=0
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PRUNING_SCOPE=global METRIC=token_ratio STATISTIC=conditional_mean
export PRUNING_PERCENTAGES="${PRUNING_PERCENTAGES:-50 60 80}"

if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=("${CONDA_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/bin/conda}" run --no-capture-output -n "${CONDA_ENV}" python)
else
  PYTHON_CMD=("${PYTHON_BIN:-python}")
fi

if (( $# > 0 )); then
  PERCENTAGES=("$@")
else
  read -r -a PERCENTAGES <<< "${PRUNING_PERCENTAGES}"
fi
if [[ "${STAGE}" == "prepare" || "${STAGE}" == "eval" ]]; then
  if (( ${#PERCENTAGES[@]} == 0 )); then
    echo "At least one pruning percentage is required." >&2
    exit 2
  fi
  for index in "${!PERCENTAGES[@]}"; do
    percentage="${PERCENTAGES[index]}"
    if [[ ! "${percentage}" =~ ^[0-9]{1,3}$ ]] || (( 10#${percentage} > 100 )); then
      echo "Pruning percentages must be integers from 0 to 100: ${percentage}" >&2
      exit 2
    fi
    PERCENTAGES[index]=$((10#${percentage}))
  done
fi

if [[ "${STAGE}" == "calibrate" ]]; then
  exec ./experiments_attrprune/run_gsm8k.sh calibrate
fi

SWEEP_ROOT="${RUN_DIR}/attribution/static_pruning/token_ratio_conditional_mean_global"
MANIFEST_JSON="${SWEEP_ROOT}/manifest.json"

# Validate completed publication, frozen train selection, and current policy source
# before a post-calibration stage can create artifacts or load a model.
if ! is_true "${DRY_RUN}"; then
  "${PYTHON_CMD[@]}" - "${RUN_DIR}" "${STAGE}" "${PERCENTAGES[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from experiments_attrprune.common import config_fingerprint, validate_prompt_contract

run_dir = Path(sys.argv[1]).resolve()
stage = sys.argv[2]

def read(path):
    if not path.is_file():
        raise SystemExit(f"Missing {path}; finish calibrate first.")
    return json.loads(path.read_text(encoding="utf-8"))

config = read(run_dir / "run_config.json")
workers = [item.get("agent_id") for item in config.get("workers", [])]
if (config.get("benchmark") != "gsm8k"
        or config.get("splits") != {"calibration": "train", "evaluation": "test"}
        or workers != ["math_solver", "mathematical_analyst", "programming_expert", "inspector"]
        or config.get("num_rounds") != 2
        or config.get("decoding", {}).get("temperature") != 0
        or config.get("dataset", {}).get("shuffle_seed") != 888):
    raise SystemExit("This entrypoint requires the GSM8K train/test, four-worker, two-round, seed-888 dataset configuration.")
fingerprint = config.get("fingerprint")
if not fingerprint or config_fingerprint(config) != fingerprint:
    raise SystemExit("Run configuration fingerprint is invalid.")
validate_prompt_contract(config, benchmark="gsm8k")

attribution_dir = run_dir / "attribution"
manifest = read(attribution_dir / "manifest.json")
expected = {f"sample_{index:05d}" for index in range(40)}
sample_ids = [item.get("sample_id") for item in manifest.get("samples", [])]
if (manifest.get("complete") is not True or manifest.get("failures")
        or len(sample_ids) != 40 or set(sample_ids) != expected):
    raise SystemExit("Calibration manifest is incomplete or failed; finish all 40 samples first.")
aggregate_path = attribution_dir / "aggregate_edge_matrices.json"
aggregate = read(aggregate_path)
digest = hashlib.sha256(aggregate_path.read_bytes()).hexdigest()
if (manifest.get("aggregation_complete") is not True
        or manifest.get("aggregate_sha256") != digest):
    raise SystemExit("Calibration aggregate publication is incomplete or changed; finish calibrate first.")
if (aggregate.get("num_aggregated_samples") != 40
        or aggregate.get("aggregation_scope") != "all"
        or aggregate.get("run_fingerprint") != fingerprint
        or aggregate.get("attribution_configuration", {}).get("attribution_mode") != "correctness-aware"):
    raise SystemExit("This entrypoint requires the current correctness-aware, all-40-sample aggregate.")

selection = read(run_dir / "calibration" / "selection_manifest.json")
records = selection.get("samples", [])
trace_root = run_dir / "calibration" / "traces"
trace_paths = {path.parent.name: path for path in trace_root.glob("sample_*/trace.json")}
if (selection.get("expected_sample_count") != 40
        or selection.get("expected_start_index") != 0
        or len(records) != 40
        or {item.get("sample_id") for item in records} != expected
        or set(trace_paths) != expected):
    raise SystemExit("Frozen calibration selection must contain exactly train indices 0..39.")
for item in records:
    path = trace_paths[item["sample_id"]]
    raw = path.read_bytes()
    trace_config = json.loads(raw).get("config", {})
    expected_index = int(item["sample_id"].split("_")[1])
    if (Path(item["trace_json"]).resolve() != path.resolve()
            or item.get("sha256") != hashlib.sha256(raw).hexdigest()
            or item.get("sample_index") != expected_index
            or trace_config.get("sample_index") != expected_index
            or trace_config.get("benchmark") != "gsm8k"
            or trace_config.get("fingerprint") != fingerprint):
        raise SystemExit(f"Frozen calibration trace changed or has incompatible identity: {path}")

if stage in {"eval", "summarize"}:
    sweep_root = attribution_dir / "static_pruning" / "token_ratio_conditional_mean_global"
    sweep_path = sweep_root / "manifest.json"
    if not sweep_path.is_file():
        raise SystemExit("Missing prepared sweep manifest; run prepare first.")
    sweep = read(sweep_path)
    if sweep.get("aggregate_sha256") != digest:
        raise SystemExit("Prepared sweep uses a stale aggregate; rerun prepare.")
    if (sweep.get("run_fingerprint") != fingerprint
            or sweep.get("benchmark") != "gsm8k"
            or sweep.get("evaluation_split") != "test"
            or sweep.get("pruning_allocation") != "global"
            or sweep.get("metric") != "token_ratio"
            or sweep.get("statistic") != "conditional_mean"):
        raise SystemExit("Prepared sweep does not match this GSM8K global experiment; rerun prepare.")
    entries = sweep.get("policies", [])
    prepared = {int(item["requested_pruning_percent"]): item for item in entries}
    percentages = [int(value) for value in sys.argv[3:]] if stage == "eval" else list(prepared)
    if not percentages or len(prepared) != len(entries):
        raise SystemExit("Prepared sweep has missing or duplicate policies; rerun prepare.")
    for percentage in percentages:
        path = sweep_root / "policies" / f"prune_{percentage:02d}pct.json"
        entry = prepared.get(percentage)
        if (entry is None or not path.is_file()
                or Path(entry["policy_json"]).resolve() != path.resolve()):
            raise SystemExit(f"Policy {percentage}% is not prepared for this sweep; rerun prepare.")
        policy = read(path)
        if policy.get("static_sweep", {}).get("aggregate_sha256") != digest:
            raise SystemExit(f"Policy {percentage}% uses a stale aggregate; rerun prepare.")
        if (policy.get("run_fingerprint") != fingerprint
                or policy.get("pruning_allocation") != "global"
                or policy.get("metric") != "token_ratio"
                or policy.get("statistic") != "conditional_mean"
                or policy.get("static_sweep", {}).get("requested_pruning_percent") != percentage):
            raise SystemExit(f"Policy {percentage}% has incompatible settings; rerun prepare.")
PY
fi

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if ! is_true "${DRY_RUN}"; then
    "$@"
  fi
}

case "${STAGE}" in
  prepare)
    run_cmd "${PYTHON_CMD[@]}" -m experiments_attrprune.static_pruning prepare \
      --run-dir "${RUN_DIR}" --metric token_ratio --statistic conditional_mean \
      --pruning-scope global --percentages "${PERCENTAGES[@]}"
    ;;
  eval)
    for percentage in "${PERCENTAGES[@]}"; do
      printf -v label 'prune_%02dpct' "${percentage}"
      args=(
        -m experiments_attrprune.evaluate_benchmark --benchmark gsm8k
        --model-name "${MODEL_PATH}" --data-dir "${GSM8K_DATA_DIR}"
        --run-dir "${RUN_DIR}" --mode token_ratio
        --policy-json "${SWEEP_ROOT}/policies/${label}.json"
        --evaluation-name "static_token_ratio_conditional_mean_global_${label}"
        --split test --start-index 0 --limit "${EVALUATION_LIMIT}"
        --eval-batch-size "${EVAL_BATCH_SIZE}" --seed "${SEED}" --resume --no-download
      )
      if is_true "${SAVE_EVAL_TRACES}"; then args+=(--save-traces); fi
      # Enter --resume even for a completed evaluation to validate its identity.
      run_cmd "${PYTHON_CMD[@]}" "${args[@]}"
    done
    ;;
  summarize)
    run_cmd "${PYTHON_CMD[@]}" -m experiments_attrprune.static_pruning summarize \
      --manifest "${MANIFEST_JSON}"
    ;;
esac
