#!/usr/bin/env bash
# Run the three prepared MMLU baseline experiments serially on one GPU pair.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname -- "$repo_root")"
python_bin="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python}"
: "${CUDA_VISIBLE_DEVICES:?Select the physical GPU pair explicitly}"
allocation="${GPU_ALLOCATION:-idle}"
case "$allocation" in idle|shared) ;; *) echo 'GPU_ALLOCATION must be idle or shared' >&2; exit 2 ;; esac
queue_root="${QUEUE_ROOT:-$repo_root/logs/mmlu_baseline_alignment_queue/$(hostname -s)_gpu_${CUDA_VISIBLE_DEVICES//,/_}}"
mkdir -p "$queue_root"
exec 9>"$queue_root/.queue.lock"
flock -n 9 || { echo 'This MMLU baseline queue is already active.' >&2; exit 2; }
queue_log="$queue_root/queue_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
exec > >(tee -a "$queue_log") 2>&1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export QUEUE_ROOT="$queue_root" GPU_ALLOCATION="$allocation"
printf 'Queue log: %s\nHost: %s; GPUs: %s; allocation: %s\n' "$queue_log" "$(hostname -s)" "$CUDA_VISIBLE_DEVICES" "$allocation"

wait_for_gpus() {
  "$python_bin" - <<'PY'
import csv, json, os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

indices = [int(value) for value in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
if len(indices) != 2 or len(set(indices)) != 2:
    raise ValueError('Select two distinct physical GPU indices')
mode = os.environ['GPU_ALLOCATION']
last_message, ready_count = None, 0
while ready_count < 2:
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    rows = [[field.strip() for field in row] for row in csv.reader(raw.splitlines())]
    selected = [row for row in rows if int(row[0]) in indices]
    if len(selected) != 2:
        raise RuntimeError('Requested GPUs are not both visible to nvidia-smi')
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    apps = [[field.strip() for field in row] for row in csv.reader(raw.splitlines()) if row]
    selected_apps = [row for row in apps if row[0] in {r[1] for r in selected}]
    if mode == 'idle':
        ready = not selected_apps and all(int(r[2]) < 1024 and int(r[4]) <= 10 for r in selected)
    else:
        ready = all(int(r[3]) >= 30 * 1024 for r in selected)
    state = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'allocation': mode,
             'method': os.environ['QUEUE_METHOD'], 'status': 'gpu_check' if ready else 'waiting_for_gpus',
             'gpus': [{'index': int(r[0]), 'used_mib': int(r[2]), 'free_mib': int(r[3]), 'utilization': int(r[4])} for r in selected],
             'compute_pids': selected_apps}
    path = Path(os.environ['QUEUE_ROOT']) / 'gpu_wait.json'
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(state, indent=2) + '\n'); temporary.replace(path)
    message = (ready, tuple(tuple(row) for row in selected_apps))
    if message != last_message:
        print(json.dumps(state), flush=True); last_message = message
    ready_count = ready_count + 1 if ready else 0
    if ready_count < 2:
        time.sleep(3 if ready else 30)
PY
}

exit_status=0
for method in AgentPrune AgentDropout SafeSieve; do
  export QUEUE_METHOD="$method"
  printf '[%s] %s waiting for GPUs\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$method"
  wait_for_gpus
  printf '[%s] %s started\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$method"
  status=0
  case "$method" in
    AgentPrune) "$project_root/AgentPrune/run_mmlu_attrprune_aligned.sh" || status=$? ;;
    AgentDropout) "$project_root/AgentDropout/run_mmlu_attrprune_dev40.sh" all --resume || status=$? ;;
    SafeSieve) "$project_root/SafeSieve/run_qwen3_mmlu_attrprune_aligned.sh" || status=$? ;;
  esac
  printf '[%s] %s exit_status=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$method" "$status"
  printf '%s\n' "$status" > "$queue_root/${method}.exit_status"
  if (( status != 0 )); then exit_status=1; fi
done
printf '[%s] All three baseline jobs attempted; queue exit_status=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$exit_status"
exit "$exit_status"
