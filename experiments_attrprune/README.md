# Attribution-pruning benchmark experiments

This directory contains thin experiment entrypoints. Reusable graph, tracing,
attribution, visualization, matrix, and policy logic lives in the
`AgentAttrPrune` package.

The formal protocol:

1. records a complete dense trace for every selected calibration sample;
2. writes an independent JSON/HTML/SVG attribution bundle for every
   worker-round node and the final reviewer;
3. writes round-specific spatial/temporal matrices for every sample;
4. aggregates only final-correct dev samples into pruning policies;
5. evaluates frozen policies on a disjoint benchmark evaluation split.

Supported benchmark protocols:

| Benchmark | Calibration | Evaluation | Workers | Role order |
|---|---|---|---:|---|
| MMLU | `dev` (285) | `val` (1531) | 5 | Knowlegable Expert, Critic, Mathematician, Psychologist, Historian |
| GSM8K | `train` (7473) | `test` (1319) | 4 | Math Solver, Mathematical Analyst, Programming Expert, Inspector |
| HumanEval | fixed `dev` (80) | fixed `test` (81) | 5 | Project Manager, Algorithm Designer, Programming Expert, Test Analyst, Bug Fixer |

The worker configurations are not identical. Their default counts and roles
follow AgentPrune's benchmark-specific defaults. They do use the same
two-round graph executor and pruning machinery, and every run finishes with a
task-aware dense reviewer outside the worker matrices.

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

Run the equivalent GSM8K or HumanEval pipeline:

```bash
./experiments_attrprune/run_gsm8k.sh one-sample
./experiments_attrprune/run_humaneval.sh one-sample
```

Full eight-GPU launchers are available at the repository root. Their default
generation ceiling is 2048 tokens per node:

```bash
# Calibration split only: traces, every-node attribution, matrices, heatmaps, policies
./run_full_gsm8k.sh calibrate
./run_full_humaneval.sh calibrate

# Complete calibration plus frozen-policy evaluation
./run_full_gsm8k.sh all
./run_full_humaneval.sh all
```

The GSM8K and HumanEval wrappers use `run_benchmark.sh`; the MMLU wrapper keeps
its original compatible entrypoint. The shared runner can also be called
directly:

```bash
BENCHMARK=gsm8k DRY_RUN=1 \
  ./experiments_attrprune/run_benchmark.sh all
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

## Static MMLU pruning-rate sweep

When attribution and `aggregate_edge_matrices.json` already exist, the static
sweep does not rerun calibration or attribution. It ranks the frozen
`token_ratio/conditional_mean` matrices and evaluates policies requested at
5%, 10%, 20%, 30%, 40%, and 50% pruning:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./run_static_mmlu_pruning_sweep.sh all
```

The default source is
`results/AgentAttrPrune/mmlu/qwen3_8b/full_dev_a6000x8_tok2048`. The accuracy
protocol matches the AgentPrune attribution-pruning evaluation: policies are
calibrated only from verified-correct MMLU `dev` samples, then frozen and
evaluated by exact answer match on every selected MMLU `val` sample (1531 by
default).

The established policy builder applies each requested rate independently per
round and per edge type (`spatial`/`temporal`). Integer rounding and the rule
that a positive rate removes at least one observed edge can therefore make the
effective global rate differ slightly from the requested rate. Both rates and
the exact retained/pruned edge counts are recorded in the manifest and final
summary.

To allocate one pruning budget across every round and both edge types, select
the global scope. All observed candidate edges are flattened, sorted by
importance, and the globally lowest edges are removed. The legacy per-matrix
scope remains the default, and global policies/results use `_global`-suffixed
directories and evaluation names so the two sweeps can coexist:

```bash
PRUNING_SCOPE=global \
  ./run_static_mmlu_pruning_sweep.sh all 10 20 30 40 50
```

To build only the global policy JSON files before launching evaluation:

```bash
PRUNING_SCOPE=global \
  ./run_static_mmlu_pruning_sweep.sh prepare 10 20 30 40 50
```

Useful stages and checks:

```bash
# Print all six evaluation commands without loading the model.
DRY_RUN=1 ./run_static_mmlu_pruning_sweep.sh all

# Build only the six small policy JSON files from the existing matrix.
./run_static_mmlu_pruning_sweep.sh prepare

# Evaluate; completed rates are skipped on a rerun by default.
./run_static_mmlu_pruning_sweep.sh eval

# Rebuild the combined JSON/CSV after all evaluations finish.
./run_static_mmlu_pruning_sweep.sh summarize
```

The combined results are written under
`evaluation/static_token_ratio_conditional_mean_sweep_val/summary.{json,csv}`
inside the source run directory. Evaluation traces are disabled by default;
set `SAVE_EVAL_TRACES=1` only when the additional disk use is intentional.

Evaluations checkpoint `predictions.jsonl` atomically after every completed
sample. Pass `--resume` (or set `RESUME=1` in the pipeline runners) to validate
`evaluation_state.json` and skip every saved sample, whether its prediction was
correct or incorrect. The state locks the run fingerprint, policy hash, dataset
selection, seed, and trace mode, so incompatible checkpoints are never mixed.
`evaluation_summary.json` is written only after the complete requested range is
present and all totals and accuracy have been recomputed from the persisted
records. The static-sweep and full-set MMLU launchers enable this evaluation
resume mode automatically.

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

HumanEval runs execute model-generated code in a limited subprocess. See
`data/README.md` for the trust boundary.
