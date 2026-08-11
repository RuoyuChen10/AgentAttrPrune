#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BENCHMARK=gsm8k
exec "${SCRIPT_DIR}/run_benchmark.sh" "$@"
