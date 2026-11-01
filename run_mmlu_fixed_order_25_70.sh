#!/usr/bin/env bash
# Add two evaluations to the original dev40 fixed-order experiment.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
STAGE="${1:-all}"
case "${STAGE}" in prepare|all) ;; *) echo "Usage: $0 [prepare|all]" >&2; exit 2 ;; esac
if (( $# > 1 )); then echo "Usage: $0 [prepare|all]" >&2; exit 2; fi
if [[ "${STAGE}" == all && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Select idle CUDA_VISIBLE_DEVICES explicitly." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python}"
RUN_DIR="${REPO_ROOT}/results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware"
SWEEP_ROOT="${RUN_DIR}/attribution/static_pruning/token_ratio_conditional_mean_global"
LOG_ROOT="${REPO_ROOT}/logs/mmlu_fixed_order_25_70"
mkdir -p "${LOG_ROOT}"
exec {launcher_lock}> "${LOG_ROOT}/.launcher.lock"
if ! flock -n "${launcher_lock}"; then echo "This 25%/70% queue is already active." >&2; exit 2; fi
LOG_FILE="${LOG_ROOT}/${STAGE}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
started=${SECONDS}
trap 'status=$?; printf "[%s] fixed_order_25_70 exit_status=%d elapsed_seconds=%d\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" "$((SECONDS - started))"' EXIT
printf 'Log: %s\n' "${LOG_FILE}"
printf '[%s] host=%s GPUs=%s stage=%s rates=25,70 batch=4\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)" "${CUDA_VISIBLE_DEVICES:-CPU}" "${STAGE}"
export PYTHONUNBUFFERED=1 HF_CHAT_ENABLE_THINKING=0 HF_CHAT_CAPTURE_SCORES=0 TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" - "${RUN_DIR}" <<'PY'
from pathlib import Path
import hashlib,json,sys
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.attribution.edge import build_pruned_masks
from experiments_attrprune.static_pruning import _validate_sources
from experiments_attrprune.common import write_json

root=Path(sys.argv[1]).resolve()
sweep=root/'attribution/static_pruning/token_ratio_conditional_mean_global'
out=sweep/'supplement_25_70'
load=lambda path:json.loads(path.read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
config=load(root/'run_config.json')
ap=root/'attribution/aggregate_edge_matrices.json'
aggregate=load(ap)
calibration=load(root/'attribution/manifest.json')
_validate_sources(config,aggregate,metric='token_ratio',statistic='conditional_mean')
assert config['fingerprint']=='e94a0ae0428ddc10'
assert aggregate['num_aggregated_samples']==40 and aggregate['aggregation_scope']=='all'
assert aggregate['attribution_configuration']['attribution_mode']=='correctness-aware'
assert calibration['complete'] and calibration['aggregation_complete'] and not calibration.get('failures')
assert calibration['aggregate_sha256']==sha(ap)
assert config['decoding']=={'temperature':0.0,'max_new_tokens':2048,'enable_thinking':False}
protected=[root/'run_config.json',ap,root/'attribution/manifest.json',sweep/'manifest.json']
protected.extend(sorted((sweep/'policies').glob('*.json')))
for rate in (50,60,80,90):
    directory=root/'evaluation'/f'static_token_ratio_conditional_mean_global_prune_{rate}pct_val'
    protected.extend(p for p in directory.iterdir() if p.is_file() and p.suffix in {'.json','.jsonl'})
prior_summary=root/'evaluation/static_token_ratio_conditional_mean_global_sweep_val'
protected.extend(sorted(prior_summary.glob('summary.*')))
before={str(p):sha(p) for p in protected}
snapshot=out/'preserved_original_files.json'
if snapshot.exists():
    saved=load(snapshot)
    assert all(sha(Path(p))==value for p,value in saved.items()), 'Original source/result changed'
else:
    write_json(snapshot,before)
entries=[]
for rate,expected in ((25,[10,7,17]),(70,[7,1,6])):
    policy=build_pruned_masks(aggregate,metric='token_ratio',statistic='conditional_mean',
        threshold=None,pruning_rate=rate/100,unobserved='keep',pruning_scope='global')
    policy['static_sweep']={'label':f'prune_{rate:02d}pct','requested_pruning_percent':rate,
        'aggregate_json':str(ap),'aggregate_sha256':sha(ap)}
    policy=RoundPolicy.from_dict(policy).to_dict()
    counts=[sum(map(sum,policy['masks'][edge][r])) for r,edge in ((0,'spatial'),(1,'spatial'),(1,'temporal'))]
    assert counts==expected,(rate,counts)
    path=sweep/'policies'/f'prune_{rate:02d}pct.json'
    if path.exists():
        assert load(path)==policy, f'Existing policy changed: {path}'
    else:
        write_json(path,policy)
    name=f'static_token_ratio_conditional_mean_global_prune_{rate:02d}pct'
    entries.append({'percentage':rate,'policy_json':str(path),'policy_sha256':sha(path),
                    'evaluation_name':name,'evaluation_directory':str(root/'evaluation'/f'{name}_val'),
                    'retained_edges':sum(counts),'retained_r0_spatial_r1_spatial_r1_temporal':counts})
manifest={'kind':'additional_fixed_order_evaluations','run_dir':str(root),
    'run_fingerprint':config['fingerprint'],'aggregate_json':str(ap),'aggregate_sha256':sha(ap),
    'percentages':[25,70],'mode':'static global; execute all retained edges',
    'val_start_index':0,'val_limit':1531,'eval_batch_size':4,'save_traces':True,'policies':entries}
mp=out/'manifest.json'
if mp.exists():assert load(mp)==manifest,'Supplement configuration changed'
else:write_json(mp,manifest)
assert all(sha(Path(p))==value for p,value in before.items()),'An original file was changed'
print(json.dumps(manifest,indent=2),flush=True)
PY

if [[ "${STAGE}" == prepare ]]; then exit 0; fi
for percentage in 25 70; do
  printf '[%s] pruning=%s%% started\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${percentage}"
  "${PYTHON_BIN}" -u -m experiments_attrprune.evaluate_mmlu \
    --benchmark mmlu --model-name /egr/research-optml/ruoyu.chen/Models/Qwen3-8B \
    --run-dir "${RUN_DIR}" --mode token_ratio \
    --policy-json "${SWEEP_ROOT}/policies/prune_${percentage}pct.json" \
    --evaluation-name "static_token_ratio_conditional_mean_global_prune_${percentage}pct" \
    --data-dir /egr/research-optml/ruoyu.chen/Project/AgentPrune/dataset/MMLU \
    --split val --start-index 0 --limit 1531 --eval-batch-size 4 --seed 0 \
    --resume --save-traces --no-download
  printf '[%s] pruning=%s%% completed\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${percentage}"
done

# Keep the original four-rate sweep summary unchanged; publish this supplement separately.
"${PYTHON_BIN}" - "${RUN_DIR}" <<'PY'
from pathlib import Path
import csv,hashlib,json,sys
from experiments_attrprune.common import write_json
root=Path(sys.argv[1]).resolve()
out=root/'attribution/static_pruning/token_ratio_conditional_mean_global/supplement_25_70'
load=lambda path:json.loads(path.read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
manifest=load(out/'manifest.json')
original=load(out/'preserved_original_files.json')
assert all(sha(Path(p))==value for p,value in original.items()),'An original file changed'
rows=[]
for entry in manifest['policies']:
    d=Path(entry['evaluation_directory'])
    summary=load(d/'evaluation_summary.json')
    predictions=[json.loads(line) for line in (d/'predictions.jsonl').read_text().splitlines() if line.strip()]
    assert len(predictions)==1531 and {r['sample_index'] for r in predictions}==set(range(1531))
    assert summary['total']==1531 and summary['correct']==sum(bool(r['correct']) for r in predictions)
    assert summary['accuracy']==summary['correct']/1531
    assert summary['run_fingerprint']==manifest['run_fingerprint'] and summary['eval_batch_size']==4
    assert summary['retained_edges']==entry['retained_edges'] and summary['candidate_edges']==45
    assert load(d/'resolved_policy.json')==load(Path(entry['policy_json']))
    assert sha(Path(entry['policy_json']))==entry['policy_sha256']
    rows.append({'pruning_percent':entry['percentage'],'total':summary['total'],
        'correct':summary['correct'],'accuracy':summary['accuracy'],
        'retained_edges':summary['retained_edges'],'prompt_tokens':summary['prompt_tokens'],
        'generated_tokens':summary['generated_tokens'],'elapsed_seconds':summary['elapsed_seconds'],
        'summary_json':str(d/'evaluation_summary.json')})
destination=root/'evaluation/static_token_ratio_conditional_mean_global_supplement_25_70_val'
write_json(destination/'summary.json',{'complete':True,'source_manifest':str(out/'manifest.json'),'rows':rows})
with (destination/'summary.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({'complete':True,'summary_dir':str(destination),'results':rows},indent=2),flush=True)
PY
