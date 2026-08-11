#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export CONDA_ENV="${CONDA_ENV:-agentprune}"
export MODEL_PATH="${MODEL_PATH:-/egr/research-optml/ruoyu.chen/Models/Qwen3-8B}"
export GSM8K_DATA_DIR="${GSM8K_DATA_DIR:-${REPO_ROOT}/data/GSM8K}"
export RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/AgentAttrPrune}"
export RUN_NAME="${RUN_NAME:-gsm8k/qwen3_8b/full_train_complete_2048}"

export CALIBRATION_START_INDEX="${CALIBRATION_START_INDEX:-0}"
export CALIBRATION_LIMIT="${CALIBRATION_LIMIT:-7473}"
export EVALUATION_LIMIT="${EVALUATION_LIMIT:-1319}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export NUM_ROUNDS="${NUM_ROUNDS:-2}"
export TEMPERATURE="${TEMPERATURE:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

export ATTR_BATCH_SIZE="${ATTR_BATCH_SIZE:-8}"
export ATTR_SAMPLE_SCOPE="${ATTR_SAMPLE_SCOPE:-all}"
export ATTR_NODE_SCOPE="${ATTR_NODE_SCOPE:-all}"
export TARGET_RECOVERY="${TARGET_RECOVERY:-0.95}"
export SALIENCY_FLOOR="${SALIENCY_FLOOR:-0.4}"
export POLICY_PRUNING_RATE="${POLICY_PRUNING_RATE:-0.25}"

STAGE="${1:-all}"
exec ./experiments_attrprune/run_gsm8k.sh "${STAGE}"
