#!/usr/bin/env bash
# Complete the frozen seed2 calibration and attribution without an evaluation sweep.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=2,3 ./run_mmlu_seed2_attribution_only.sh
  DRY_RUN=1 ./run_mmlu_seed2_attribution_only.sh

Resume the already frozen MMLU seed2 dev40 selection: collect -> attribute ->
aggregate matrices only. Keep Qwen3-8B, canonical order, two rounds, greedy
2048-token generation, thinking off, serial collection and attribution batch 4.
Stop after validating 40 traces and 440 node attribution records. Do not create
pruning policies or start any val evaluations. Hold the normal seed2 launcher
lock throughout so the existing sweep queue cannot run the same seed concurrently.

Overrides: PYTHON_BIN, REFERENCE_RUN, RUN_DIR, LOG_ROOT, DRY_RUN=1.
DRY_RUN prints the stages without writing files or allocating a model.
EOF
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if (( $# != 0 )); then echo "This launcher only runs the frozen MMLU seed2 attribution; pass no arguments." >&2; exit 2; fi
is_true() { case "${1,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac; }
DRY_RUN="${DRY_RUN:-0}"
if ! is_true "${DRY_RUN}" && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Select idle CUDA_VISIBLE_DEVICES explicitly." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python}"
REFERENCE_RUN="${REFERENCE_RUN:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/dev40_resample_seed2_correctness_aware}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/mmlu_dev40_resampling}"
export PYTHONUNBUFFERED=1 HF_CHAT_ENABLE_THINKING=0 HF_CHAT_CAPTURE_SCORES=0 TOKENIZERS_PARALLELISM=false

run_command() {
  if is_true "${DRY_RUN}"; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
if ! is_true "${DRY_RUN}"; then
  mkdir -p "${LOG_ROOT}/seed_2"
  exec {seed_lock}> "${LOG_ROOT}/seed_2/.launcher.lock"
  if ! flock -n "${seed_lock}"; then echo "Calibration seed 2 launcher is already active." >&2; exit 2; fi
  log_file="${LOG_ROOT}/seed_2/attribution_only_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}.log"
  exec > >(tee -a "${log_file}") 2>&1
  started=${SECONDS}
  trap 'status=$?; printf "[%s] calibration_seed=2 attribution_only exit_status=%d elapsed_seconds=%d\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" "$((SECONDS-started))"' EXIT
  printf 'Log: %s\nRun: %s\n' "${log_file}" "${RUN_DIR}"
  printf '[%s] host=%s GPUs=%s calibration_seed=2 generation_seed=0 attribution_only=1\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)" "${CUDA_VISIBLE_DEVICES}"
fi

check_or_aggregate() {
  if is_true "${DRY_RUN}"; then
    run_command "${PYTHON_BIN}" - "${REFERENCE_RUN}" "${RUN_DIR}" "$1"
    return
  fi
  "${PYTHON_BIN}" - "${REFERENCE_RUN}" "${RUN_DIR}" "$1" <<'PY'
from pathlib import Path
import sys

from experiments_attrprune import resampled_mmlu as runner
from experiments_attrprune.common import load_json, write_json

reference, root, phase = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
if not (root / "run_config.json").is_file():
    raise FileNotFoundError("The seed2 run must already have its frozen configuration")
args = runner.parse_args([
    "--calibration-seed", "2", "--reference-run", str(reference),
    "--run-dir", str(root), "--stage", "aggregate", "--resume",
])
ctx = runner.prepare_context(args)
if ctx.config["fingerprint"] != "6d926a2eb1ff68b4":
    raise ValueError("The original frozen seed2 configuration changed")
runner._validate_frozen(ctx)
runner._check_hashes(ctx.source_hashes)
if phase == "check":
    print("Verified frozen seed2 dev40; serial collection, attribution batch=4.", flush=True)
elif phase == "aggregate_without_policies":
    with runner.run_lock(ctx.paths.run_dir):
        runner._validate_frozen(ctx)
        runner._check_hashes(ctx.source_hashes)
        manifest_path = ctx.paths.attribution / "manifest.json"
        aggregate_path = ctx.paths.attribution / "aggregate_edge_matrices.json"
        manifest = load_json(manifest_path)
        if manifest.get("aggregation_complete") is True:
            aggregate = runner.validate_ready_for_evaluation(ctx)
        else:
            # This helper validates all 40 traces, 440 nodes and every edge count.
            # The regular aggregate stage also creates policies; this path stops
            # after publishing the same verified attribution matrix.
            aggregate = runner._aggregate_from_artifacts(ctx, manifest)
            runner.freeze_json(aggregate_path, aggregate)
            manifest.update(aggregation_complete=True,
                            aggregate_sha256=runner.file_sha(aggregate_path))
            write_json(manifest_path, manifest)
            runner.validate_ready_for_evaluation(ctx)
        if (aggregate.get("num_rollouts") != 40
                or sum(aggregate.get("node_status_counts", {}).values()) != 440):
            raise ValueError("Attribution requires exactly 40 traces and 440 node records")
        print("Complete: 40 traces, 440 node records; attribution matrix verified.", flush=True)
        print(f"Matrix: {aggregate_path}", flush=True)
else:
    raise ValueError(f"Unknown attribution-only phase: {phase}")
PY
}

check_or_aggregate check
for stage in collect attribute; do
  if ! is_true "${DRY_RUN}"; then
    printf '[%s] stage=%s started\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${stage}"
  fi
  run_command "${PYTHON_BIN}" -u -m experiments_attrprune.resampled_mmlu \
    --calibration-seed 2 --reference-run "${REFERENCE_RUN}" --run-dir "${RUN_DIR}" \
    --stage "${stage}" --resume
done
check_or_aggregate aggregate_without_policies
