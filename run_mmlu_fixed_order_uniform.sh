#!/usr/bin/env bash
# Uniform static pruning of the original fixed-order, seed-0 dev40 matrix.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=6,7 ./run_mmlu_fixed_order_uniform.sh
  PREPARE_ONLY=1 ./run_mmlu_fixed_order_uniform.sh
  DRY_RUN=1 ./run_mmlu_fixed_order_uniform.sh

Reuse the original seed-0 fixed-order dev40 attribution matrix. Run uniform
static pruning at 50%, 60%, then 80%, with eval batch 4 and resume enabled.
Uniform means per_round_edge_type: prune R0 spatial, R1 spatial and R1 temporal
independently. All retained edges execute; there is no Bernoulli sampling.
Preserve both worker rounds, original agent/reviewer order, Qwen3-8B,
temperature 0, 2048 tokens, thinking off, seed 0 and all 1531 val questions.
Existing global policies/results are read-only. Save traces and separate logs.

Overrides: RUN_DIR, LOG_ROOT, PYTHON_BIN, PREPARE_ONLY=1, DRY_RUN=1.
DRY_RUN prints commands only, without creating files or loading a model.
EOF
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if (( $# != 0 )); then echo "This launcher evaluates exactly 50%, 60%, 80%; pass no arguments." >&2; exit 2; fi
is_true() { case "${1,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac; }
DRY_RUN="${DRY_RUN:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
if ! is_true "${DRY_RUN}" && ! is_true "${PREPARE_ONLY}" && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Select idle CUDA_VISIBLE_DEVICES explicitly." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware}"
SWEEP_ROOT="${RUN_DIR}/attribution/static_pruning/token_ratio_conditional_mean"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/mmlu_fixed_order_uniform}"
export PYTHONUNBUFFERED=1 HF_CHAT_ENABLE_THINKING=0 HF_CHAT_CAPTURE_SCORES=0 TOKENIZERS_PARALLELISM=false

run_command() {
  if is_true "${DRY_RUN}"; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
if ! is_true "${DRY_RUN}"; then
  mkdir -p "${LOG_ROOT}"
  exec {launcher_lock}> "${LOG_ROOT}/.launcher.lock"
  if ! flock -n "${launcher_lock}"; then echo "A fixed-order uniform queue is already active." >&2; exit 2; fi
  log_file="${LOG_ROOT}/uniform_50_60_80_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}.log"
  exec > >(tee -a "${log_file}") 2>&1
  started=${SECONDS}
  trap 'status=$?; printf "[%s] fixed_order_uniform exit_status=%d elapsed_seconds=%d\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" "$((SECONDS-started))"' EXIT
  printf 'Log: %s\nRun: %s\n' "${log_file}" "${RUN_DIR}"
  printf '[%s] host=%s GPUs=%s rates=50,60,80 scope=per_round_edge_type eval_batch_size=4 prepare_only=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)" "${CUDA_VISIBLE_DEVICES:-CPU}" "${PREPARE_ONLY}"

  # Validate the exact completed baseline and never replace a different sweep.
  "${PYTHON_BIN}" - "${RUN_DIR}" <<'PY'
from pathlib import Path
import hashlib,json,sys
from AgentAttrPrune.attribution.edge import build_pruned_masks
from AgentAttrPrune.pruning import RoundPolicy
from experiments_attrprune.static_pruning import _validate_sources
root=Path(sys.argv[1]).resolve()
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
config=load(root/'run_config.json')
ap=root/'attribution/aggregate_edge_matrices.json'
aggregate=load(ap)
calibration=load(root/'attribution/manifest.json')
baseline=root/'evaluation/static_token_ratio_conditional_mean_global_prune_80pct_val'
summary=load(baseline/'evaluation_summary.json')
assert config['fingerprint']=='e94a0ae0428ddc10', 'Expected original fixed-order seed0 run'
assert config['seed']==0 and config['num_rounds']==2 and len(config['workers'])==5
assert config['decoding']=={'temperature':0.0,'max_new_tokens':2048,'enable_thinking':False}
assert summary['run_fingerprint']==config['fingerprint']
assert summary['total']==1531 and summary['correct']==1184 and summary['eval_batch_size']==4
assert summary['llm_calls']==1531*11 and summary['candidate_edges']==45
assert aggregate['num_aggregated_samples']==40 and aggregate['aggregation_scope']=='all'
assert aggregate['attribution_configuration']['attribution_mode']=='correctness-aware'
assert calibration['complete'] and calibration['aggregation_complete'] and not calibration.get('failures')
assert calibration['aggregate_sha256']==sha(ap)
assert load(Path(summary['source_policy_json']))['static_sweep']['aggregate_sha256']==sha(ap)
_validate_sources(config,aggregate,metric='token_ratio',statistic='conditional_mean')
sweep=root/'attribution/static_pruning/token_ratio_conditional_mean'
manifest=sweep/'manifest.json'
if manifest.exists():
    m=load(manifest)
    assert m['run_fingerprint']==config['fingerprint'] and m['aggregate_sha256']==sha(ap)
    assert m['percentages']==[50,60,80] and m['pruning_allocation']=='per_round_edge_type'
    assert m['metric']=='token_ratio' and m['statistic']=='conditional_mean' and m['unobserved']=='keep'
for rate,expected in ((50,[5,5,13]),(60,[4,4,10]),(80,[2,2,5])):
    payload=build_pruned_masks(aggregate,metric='token_ratio',statistic='conditional_mean',
        threshold=None,pruning_rate=rate/100,unobserved='keep',pruning_scope='per_round_edge_type')
    payload['static_sweep']={'label':f'prune_{rate:02d}pct','requested_pruning_percent':rate,
        'aggregate_json':str(ap),'aggregate_sha256':sha(ap)}
    policy=RoundPolicy.from_dict(payload).to_dict()
    counts=[sum(map(sum,policy['masks'][edge][r])) for r,edge in ((0,'spatial'),(1,'spatial'),(1,'temporal'))]
    assert counts==expected,(rate,counts)
    path=sweep/'policies'/f'prune_{rate:02d}pct.json'
    if path.exists():assert load(path)==policy,f'Existing uniform policy changed: {path}'
    if manifest.exists():assert path.exists(),f'Frozen uniform sweep has a missing policy: {path}'
print('Verified original baseline and uniform retained-edge counts: 23,18,9.',flush=True)
PY
fi

if is_true "${DRY_RUN}" || [[ ! -f "${SWEEP_ROOT}/manifest.json" ]]; then
  run_command "${PYTHON_BIN}" -u -m experiments_attrprune.static_pruning prepare \
    --run-dir "${RUN_DIR}" --percentages 50 60 80 --metric token_ratio \
    --statistic conditional_mean --unobserved keep --pruning-scope per_round_edge_type
fi
if is_true "${PREPARE_ONLY}"; then exit 0; fi
for percentage in 50 60 80; do
  run_command "${PYTHON_BIN}" -u -m experiments_attrprune.evaluate_mmlu \
    --benchmark mmlu --model-name /egr/research-optml/ruoyu.chen/Models/Qwen3-8B \
    --run-dir "${RUN_DIR}" --mode token_ratio \
    --policy-json "${SWEEP_ROOT}/policies/prune_${percentage}pct.json" \
    --evaluation-name "static_token_ratio_conditional_mean_prune_${percentage}pct" \
    --data-dir /egr/research-optml/ruoyu.chen/Project/AgentPrune/dataset/MMLU \
    --split val --start-index 0 --limit 1531 --eval-batch-size 4 --seed 0 \
    --resume --save-traces --no-download
done
run_command "${PYTHON_BIN}" -u -m experiments_attrprune.static_pruning summarize \
  --manifest "${SWEEP_ROOT}/manifest.json"
