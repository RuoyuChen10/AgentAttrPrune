#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,7}"
export PRUNING_SCOPE="global"
export VAL_START_INDEX="0"
export VAL_LIMIT="1531"
export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

exec ./run_static_mmlu_pruning_sweep.sh \
  all \
  10 20 40 50 60 70 80
