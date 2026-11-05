#!/usr/bin/env bash
# Add one seed-0 evaluation without replacing the completed global sweep.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=4,5 ./run_gsm8k_seed0_90pct.sh
  PREPARE_ONLY=1 ./run_gsm8k_seed0_90pct.sh
  DRY_RUN=1 ./run_gsm8k_seed0_90pct.sh

Reuse the original train40_correctness_aware seed-0 matrix; add only global
static 90% pruning, batch 4, all 1319 test questions, resume and saved traces.
Preserve Qwen3-8B, four workers, two rounds, original fixed order/reviewer IDs,
temperature 0, 2048 tokens, thinking disabled, generation seed 0 and shuffle 888.
Retain 3/28 worker edges; no edge sampling or new calibration/attribution.
Round 1 has no spatial edge but retains inspector(R0) -> inspector(R1), so it runs.
Keep the existing 50/60/70/80 sweep and concurrent 25% evaluation unchanged.

Overrides: RUN_DIR, LOG_ROOT, PYTHON_BIN, PREPARE_ONLY=1, DRY_RUN=1.
DRY_RUN prints commands only and creates no files or model allocations.
EOF
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if (( $# != 0 )); then echo "This launcher adds only seed0 global 90%; pass no arguments." >&2; exit 2; fi
is_true() { case "${1,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac; }
DRY_RUN="${DRY_RUN:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
if ! is_true "${DRY_RUN}" && ! is_true "${PREPARE_ONLY}" && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Select idle CUDA_VISIBLE_DEVICES explicitly." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/AgentAttrPrune/gsm8k/qwen3_8b/train40_correctness_aware}"
SWEEP_ROOT="${RUN_DIR}/attribution/static_pruning/token_ratio_conditional_mean_global"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/gsm8k_seed0_90pct}"
export PYTHONUNBUFFERED=1 HF_CHAT_ENABLE_THINKING=0 HF_CHAT_CAPTURE_SCORES=0 TOKENIZERS_PARALLELISM=false

run_command() {
  if is_true "${DRY_RUN}"; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
if ! is_true "${DRY_RUN}"; then
  mkdir -p "${LOG_ROOT}"
  exec {launcher_lock}> "${LOG_ROOT}/.launcher.lock"
  if ! flock -n "${launcher_lock}"; then echo "A seed0 90% launcher is already active." >&2; exit 2; fi
  log_file="${LOG_ROOT}/global_90pct_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}.log"
  exec > >(tee -a "${log_file}") 2>&1
  started=${SECONDS}
  trap 'status=$?; printf "[%s] gsm8k_seed0_90pct exit_status=%d elapsed_seconds=%d\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" "$((SECONDS-started))"' EXIT
  printf 'Log: %s\nRun: %s\n' "${log_file}" "${RUN_DIR}"
  printf '[%s] host=%s GPUs=%s seed=0 global_pruning=90%% eval_batch_size=4 prepare_only=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)" "${CUDA_VISIBLE_DEVICES:-CPU}" "${PREPARE_ONLY}"
fi

prepare_or_verify() {
  if is_true "${DRY_RUN}"; then run_command "${PYTHON_BIN}" - "${RUN_DIR}" "$1"; return; fi
  "${PYTHON_BIN}" - "${RUN_DIR}" "$1" <<'PY'
from pathlib import Path
import hashlib,json,sys
from AgentAttrPrune.attribution.edge import build_pruned_masks
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.graph.fixed_order_skip_empty import execution_contract_for_policy
from experiments_attrprune.common import config_fingerprint,validate_prompt_contract,write_json
from experiments_attrprune.static_pruning import _validate_sources
root=Path(sys.argv[1]).resolve()
verify_only=sys.argv[2]=='verify'
sweep=root/'attribution/static_pruning/token_ratio_conditional_mean_global'
out=sweep/'supplement_90pct'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
config=load(root/'run_config.json')
ap=root/'attribution/aggregate_edge_matrices.json'
aggregate=load(ap)
calibration=load(root/'attribution/manifest.json')
master=load(sweep/'manifest.json')
assert config['fingerprint']==config_fingerprint(config)=='dd1386dc95e08ff0'
assert config['benchmark']=='gsm8k' and config['seed']==0 and config['num_rounds']==2
assert [w['agent_id'] for w in config['workers']]==['math_solver','mathematical_analyst','programming_expert','inspector']
assert config['dataset']['shuffle_seed']==888 and config['dataset']['splits']['test']['record_count']==1319
assert config['decoding']=={'temperature':0.0,'max_new_tokens':2048,'enable_thinking':False}
validate_prompt_contract(config,benchmark='gsm8k')
_validate_sources(config,aggregate,metric='token_ratio',statistic='conditional_mean')
assert sha(ap)=='7ec711ae340d1036701a8fa3c6b6d6a82facd7aa1276dd7ee1847613c76d0877'
assert aggregate['num_aggregated_samples']==40 and aggregate['aggregation_scope']=='all'
assert aggregate['attribution_configuration']['attribution_mode']=='correctness-aware'
assert calibration['complete'] and calibration['aggregation_complete'] and not calibration.get('failures')
assert calibration['aggregate_sha256']==sha(ap)==master['aggregate_sha256']
assert master['percentages']==[50,60,70,80] and master['run_fingerprint']==config['fingerprint']
assert master['pruning_allocation']=='global'

protected=[root/'run_config.json',ap,root/'attribution/manifest.json',sweep/'manifest.json']
protected.extend(sweep/'policies'/f'prune_{rate}pct.json' for rate in (50,60,70,80))
# The 25% supplement is concurrently evaluating on another host. Only freeze
# the original completed global sweeps, never mutable checkpoints from other runs.
for rate in (50,60,70,80):
    directory=root/'evaluation'/f'static_token_ratio_conditional_mean_global_prune_{rate}pct_test'
    if directory.is_dir():
        protected.extend(p for p in directory.iterdir() if p.is_file() and p.suffix in {'.json','.jsonl','.csv'})
before={str(p):sha(p) for p in protected}
snapshot=out/'preserved_original_files.json'
if snapshot.exists():
    assert all(sha(Path(p))==value for p,value in load(snapshot).items()),'Original source/result changed'
elif verify_only:raise FileNotFoundError(snapshot)

policy=build_pruned_masks(aggregate,metric='token_ratio',statistic='conditional_mean',
    threshold=None,pruning_rate=0.90,unobserved='keep',pruning_scope='global')
policy['static_sweep']={'label':'prune_90pct','requested_pruning_percent':90,
    'aggregate_json':str(ap),'aggregate_sha256':sha(ap)}
policy=RoundPolicy.from_dict(policy).to_dict()
counts=[sum(map(sum,policy['masks'][edge][r])) for r,edge in ((0,'spatial'),(1,'spatial'),(1,'temporal'))]
assert counts==[2,0,1],counts
contract=execution_contract_for_policy(policy)
assert contract['skipped_worker_rounds']==[] and contract['reviewer_source_round']==1, (
    'This source needs the empty-Round-1 evaluator; refusing full-round evaluation',contract)
assert policy['masks']['temporal'][1][3][3], 'Expected inspector(R0) to inspector(R1)'

policy_path=sweep/'policies/prune_90pct.json'
name='static_token_ratio_conditional_mean_global_prune_90pct'
manifest={'schema_version':1,'kind':'additional_fixed_order_evaluation','run_dir':str(root),
    'run_fingerprint':config['fingerprint'],'aggregate_json':str(ap),'aggregate_sha256':sha(ap),
    'pruning_percent':90,'pruning_scope':'global','metric':'token_ratio','statistic':'conditional_mean',
    'policy_json':str(policy_path),'policy_fingerprint':config_fingerprint(policy),
    'evaluation_name':name,'evaluation_directory':str(root/'evaluation'/f'{name}_test'),
    'retained_edges':3,'candidate_edges':28,'retained_r0_spatial_r1_spatial_r1_temporal':counts,
    'test_start_index':0,'test_limit':1319,'eval_batch_size':4,'save_traces':True,
    'execution_contract':contract,
    'execution':'static retained edges; Round 1 temporal inspector self-edge requires both worker rounds'}
mp=out/'manifest.json'
for path,payload in ((policy_path,policy),(mp,manifest)):
    if path.exists():assert load(path)==payload,f'Existing supplement changed: {path}'
    elif verify_only:raise FileNotFoundError(path)
if not snapshot.exists():write_json(snapshot,before)
if not policy_path.exists():write_json(policy_path,policy)
if not mp.exists():write_json(mp,manifest)
assert all(sha(Path(p))==value for p,value in before.items()),'An original source/result was changed'
print(f'Verified seed0 Global90: retained {counts}=3/28; original sweep and results preserved.',flush=True)
print(f'Round 1: spatial={counts[1]}, temporal={counts[2]}; skip=False; reviewer_source_round=1.',flush=True)
print(f'Supplement: {mp}',flush=True)
PY
}

prepare_or_verify prepare
if is_true "${PREPARE_ONLY}"; then exit 0; fi
run_command "${PYTHON_BIN}" -u -m experiments_attrprune.evaluate_mmlu \
  --benchmark gsm8k --model-name /egr/research-optml/ruoyu.chen/Models/Qwen3-8B \
  --run-dir "${RUN_DIR}" --mode token_ratio \
  --policy-json "${SWEEP_ROOT}/policies/prune_90pct.json" \
  --evaluation-name static_token_ratio_conditional_mean_global_prune_90pct \
  --data-dir "${REPO_ROOT}/data/GSM8K" \
  --split test --start-index 0 --limit 1319 --eval-batch-size 4 --seed 0 \
  --resume --save-traces --no-download
prepare_or_verify verify
