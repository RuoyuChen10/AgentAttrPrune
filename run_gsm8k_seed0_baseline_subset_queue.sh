#!/usr/bin/env bash
# Original AgentAttrPrune GSM8K seed0 train40, shared prompts and greedy decoding.
# Run each baseline's unchanged training/pruning budget, then its full test set.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname -- "$repo_root")"
cd "$repo_root"
python_bin="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/agentprune/bin/python}"
if (( $# == 0 )); then set -- AgentPrune AgentDropout SafeSieve; fi
methods=("$@")
declare -A selected_methods=()
for method in "${methods[@]}"; do
  case "$method" in AgentPrune|AgentDropout|SafeSieve) ;;
    *) echo "Unknown baseline: $method" >&2; exit 2 ;;
  esac
  if [[ -n "${selected_methods[$method]:-}" ]]; then
    echo "Duplicate baseline: $method" >&2; exit 2
  fi
  selected_methods[$method]=1
done
gpu_pair="${CUDA_VISIBLE_DEVICES:-4,5}"
host="$(hostname -s)"
expected_host="${EXPECTED_HOST:-otml-seit}"
if [[ "$host" != "$expected_host" ]]; then
  echo "Expected host $expected_host; current host is $host." >&2
  exit 2
fi
if [[ ! "$gpu_pair" =~ ^[0-9]+,[0-9]+$ ]] || [[ "${gpu_pair%,*}" == "${gpu_pair#*,}" ]]; then
  echo 'Select exactly two distinct physical GPU indices.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$gpu_pair" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 STAGE=all
queue_root="$repo_root/logs/gsm8k_seed0_baseline_queue/${host}_gpu_${gpu_pair//,/_}"
mkdir -p "$queue_root"
exec 9>"$queue_root/.queue.lock"
flock -n 9 || { echo 'This GSM8K seed0 baseline queue is already active.' >&2; exit 2; }
launch_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
queue_log="$queue_root/queue_${launch_id}.log"
exec > >(tee -a "$queue_log") 2>&1
printf 'Queue log: %s\nHost: %s; GPUs: %s\n' "$queue_log" "$host" "$gpu_pair"
printf 'Cohort: original AgentAttrPrune GSM8K seed0 train40; test=1319; temperature=0; LLM/eval batch=4\n'
printf 'Methods: %s\n' "${methods[*]}"

exit_status=0
for method in "${methods[@]}"; do
  case "$method" in
    AgentPrune) command=("$project_root/AgentPrune/run_gsm8k_attrprune_seed0.sh") ;;
    AgentDropout) command=("$project_root/AgentDropout/run_gsm8k_attrprune_seed0.sh" all --resume) ;;
    SafeSieve) command=("$project_root/SafeSieve/run_gsm8k_attrprune_seed0.sh") ;;
  esac
  printf '[%s] %s waiting for idle GPUs %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$method" "$gpu_pair"
  status=0
  "$python_bin" "$repo_root/run_when_gpus_idle.py" \
    --gpus "$gpu_pair" --cwd "$repo_root" --poll-seconds 30 --idle-seconds 30 \
    --log-dir "$queue_root/${launch_id}_${method}" -- "${command[@]}" || status=$?
  printf '%s\n' "$status" > "$queue_root/${launch_id}_${method}.exit_status"
  printf '[%s] %s exit_status=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$method" "$status"
  if (( status != 0 )); then exit_status=1; fi
done
printf '[%s] Queue complete; exit_status=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$exit_status"
exit "$exit_status"
