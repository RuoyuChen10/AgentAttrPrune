#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=0 ./run_mmlu_dev40_correctness_aware.sh [calibrate]
  ./run_mmlu_dev40_correctness_aware.sh prepare [PERCENTAGE ...]
  CUDA_VISIBLE_DEVICES=0 ./run_mmlu_dev40_correctness_aware.sh eval [PERCENTAGE ...]
  ./run_mmlu_dev40_correctness_aware.sh summarize

calibrate (default) records exactly dev indices 0..39, attributes every node,
and aggregates all 40 samples. It never starts validation inference.
prepare and summarize run without loading a model. eval requires prepared
policies and starts validation inference only when explicitly requested.

Defaults:
  RUN_NAME=mmlu/qwen3_8b/dev40_correctness_aware
  ATTR_BATCH_SIZE=4, EVAL_BATCH_SIZE=4, MAX_NEW_TOKENS=2048
  PRUNING_PERCENTAGES="0 5 10 20 30 40 50 60 70 80"
  Global token_ratio / conditional_mean; 0% is the dense graph baseline.
  Five workers, two rounds, temperature 0, seed 0, all 1531 val questions.

Overrides:
  RUN_NAME, RESULTS_ROOT, MODEL_PATH, MMLU_DATA_DIR, CONDA_ENV, PYTHON_BIN,
  ATTR_BATCH_SIZE, EVAL_BATCH_SIZE, MAX_NEW_TOKENS, VAL_LIMIT, SEED,
  RESUME=1 (default), SAVE_EVAL_TRACES=1 (default), DRY_RUN=1.
  GPU stages require an explicitly selected, nonempty CUDA_VISIBLE_DEVICES.
  Change RUN_NAME for a different model/generation configuration or experiment.
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
  echo "Select available GPUs explicitly, for example CUDA_VISIBLE_DEVICES=0, before ${STAGE}." >&2
  exit 2
fi

export CONDA_ENV="${CONDA_ENV-agentprune}"
export MODEL_PATH="${MODEL_PATH:-/egr/research-optml/ruoyu.chen/Models/Qwen3-8B}"
export MMLU_DATA_DIR="${MMLU_DATA_DIR:-${REPO_ROOT}/data/MMLU}"
export RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/AgentAttrPrune}"
export RUN_NAME="${RUN_NAME:-mmlu/qwen3_8b/dev40_correctness_aware}"
EXPECTED_RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}"
if [[ -n "${RUN_DIR:-}" && "${RUN_DIR}" != "${EXPECTED_RUN_DIR}" ]]; then
  echo "Set RUN_NAME and RESULTS_ROOT; RUN_DIR must equal ${EXPECTED_RUN_DIR}." >&2
  exit 2
fi
export RUN_DIR="${EXPECTED_RUN_DIR}"

# This entrypoint defines a fixed 40-question experiment. Partial attribution
# controls inherited from other shell sessions must not silently shrink it.
export DEV_START_INDEX=0 DEV_LIMIT=40
export ATTR_EXPECTED_START_INDEX=0 ATTR_EXPECTED_SAMPLE_COUNT=40
export ATTRIBUTION_MODE=correctness-aware AGGREGATION_SCOPE=all
export ATTR_SAMPLE_SCOPE=all ATTR_NODE_SCOPE=all
export ATTR_SAMPLE_LIMIT="" ATTR_EVENT_LIMIT="" TARGET_TOKEN_LIMIT=""
export NUM_WORKERS=5 NUM_ROUNDS=2 TEMPERATURE=0 TARGET_RECOVERY=0.95
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export SEED="${SEED:-0}"
export ATTR_BATCH_SIZE="${ATTR_BATCH_SIZE:-4}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export VAL_LIMIT="${VAL_LIMIT:-1531}"
export VAL_SPLIT=val VAL_START_INDEX=0
export RESUME="${RESUME:-1}"
export SAVE_EVAL_TRACES="${SAVE_EVAL_TRACES:-1}"
# Always enter evaluate_mmlu --resume, including completed evaluations, so its
# full fingerprint, policy, dataset selection and batch checks run before reuse.
export SKIP_COMPLETED=0
export PRUNING_SCOPE=global METRIC=token_ratio STATISTIC=conditional_mean
export PRUNING_PERCENTAGES="${PRUNING_PERCENTAGES:-0 5 10 20 30 40 50 60 70 80}"

if [[ "${STAGE}" == "calibrate" ]]; then
  exec ./experiments_attrprune/run_mmlu.sh calibrate
fi

# Refuse a legacy or partial aggregate before preparing/evaluating this profile.
if ! is_true "${DRY_RUN}"; then
  if [[ -n "${CONDA_ENV}" ]]; then
    PYTHON_CMD=("${CONDA_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/bin/conda}" run --no-capture-output -n "${CONDA_ENV}" python)
  else
    PYTHON_CMD=("${PYTHON_BIN:-python}")
  fi
  "${PYTHON_CMD[@]}" -c '
import hashlib
import json
import os
from pathlib import Path
import sys

attribution_dir = Path(sys.argv[1]) / "attribution"
manifest_path = attribution_dir / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"Missing {manifest_path}; finish calibrate first.")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
sample_ids = [item.get("sample_id") for item in manifest.get("samples", [])]
if (manifest.get("complete") is not True or manifest.get("failures")
        or len(sample_ids) != 40
        or set(sample_ids) != {f"sample_{index:05d}" for index in range(40)}):
    raise SystemExit("Calibration manifest is incomplete or failed; finish all 40 samples first.")
path = attribution_dir / "aggregate_edge_matrices.json"
if not path.is_file():
    raise SystemExit(f"Missing {path}; finish calibrate first.")
aggregate_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
if (manifest.get("aggregation_complete") is not True
        or manifest.get("aggregate_sha256") != aggregate_sha256):
    raise SystemExit("Calibration aggregate publication is incomplete or changed; finish calibrate first.")
aggregate = json.loads(path.read_text(encoding="utf-8"))
if (aggregate.get("num_aggregated_samples") != 40
        or aggregate.get("aggregation_scope") != "all"
        or aggregate.get("attribution_configuration", {}).get("attribution_mode") != "correctness-aware"):
    raise SystemExit("This entrypoint requires the complete correctness-aware, all-40-sample aggregate.")
if sys.argv[2] in {"eval", "summarize"}:
    sweep_root = attribution_dir / "static_pruning" / "token_ratio_conditional_mean_global"
    sweep_manifest_path = sweep_root / "manifest.json"
    if not sweep_manifest_path.is_file():
        raise SystemExit("Missing prepared sweep manifest; run prepare first.")
    sweep_manifest = json.loads(sweep_manifest_path.read_text(encoding="utf-8"))
    if sweep_manifest.get("aggregate_sha256") != aggregate_sha256:
        raise SystemExit("Prepared sweep uses a stale aggregate; rerun prepare.")
    entries = sweep_manifest.get("policies", [])
    prepared = {int(item["requested_pruning_percent"]): item for item in entries}
    if sys.argv[2] == "eval":
        requested = sys.argv[3:] or os.environ["PRUNING_PERCENTAGES"].split()
        if any(not value.isdigit() or not 0 <= int(value) <= 100 for value in requested):
            raise SystemExit("Pruning percentages must be integers from 0 to 100.")
        percentages = [int(value) for value in requested]
    else:
        percentages = list(prepared)
    if not percentages:
        raise SystemExit("Prepared sweep contains no policies; rerun prepare.")
    for percentage in percentages:
        policy_path = sweep_root / "policies" / f"prune_{percentage:02d}pct.json"
        entry = prepared.get(percentage)
        if (entry is None or not policy_path.is_file()
                or Path(entry["policy_json"]).resolve() != policy_path.resolve()):
            raise SystemExit(f"Policy {percentage}% is not prepared for this sweep; rerun prepare.")
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if policy.get("static_sweep", {}).get("aggregate_sha256") != aggregate_sha256:
            raise SystemExit(f"Policy {percentage}% uses a stale aggregate; rerun prepare.")
' "${RUN_DIR}" "${STAGE}" "$@"
fi

exec ./run_static_mmlu_pruning_sweep.sh "${STAGE}" "$@"
