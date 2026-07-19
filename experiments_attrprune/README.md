# MMLU attribution-pruning experiments

This directory contains thin experiment entrypoints. Reusable graph, tracing,
attribution, visualization, matrix, and policy logic lives in the
`AgentAttrPrune` package.

The formal protocol:

1. records a complete dense trace for every selected MMLU dev sample;
2. writes an independent JSON/HTML/SVG attribution bundle for every
   worker-round node and the final reviewer;
3. writes round-specific spatial/temporal matrices for every sample;
4. aggregates only final-correct dev samples into pruning policies;
5. evaluates frozen policies on the complete selected MMLU val split.

Print the resolved pipeline without running a model:

```bash
DRY_RUN=1 ./experiments_attrprune/run_mmlu.sh all
```

Run one dev sample:

```bash
DEV_START_INDEX=0 \
MAX_NEW_TOKENS=512 \
./experiments_attrprune/run_mmlu.sh one-sample
```

Run complete dev calibration without val/test evaluation:

```bash
DEV_LIMIT=285 \
MAX_NEW_TOKENS=2048 \
./experiments_attrprune/run_mmlu.sh calibrate
```

Run the full dev-to-val protocol:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./experiments_attrprune/run_mmlu.sh all
```

Use an existing Conda environment:

```bash
CONDA_ENV=agentprune \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./experiments_attrprune/run_mmlu.sh all
```

All generated output is written under `results/AgentAttrPrune/`. See the
top-level `README.md`, the detailed algorithm document, and:

```bash
./experiments_attrprune/run_mmlu.sh --help
```
