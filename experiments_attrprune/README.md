# Attribution-pruning benchmark experiments

This directory contains thin experiment entrypoints. Reusable graph, tracing,
attribution, visualization, matrix, and policy logic lives in the
`AgentAttrPrune` package.

For the current experiment profiles and verified commands, see the
[main README](../README.md), [MMLU ablation guide](../docs/mmlu_ablations.md),
and [GSM8K guide](../docs/gsm8k_experiments.md). The sections below also retain
the general-purpose and legacy benchmark interfaces; their defaults are not
identical to the dedicated dev40/train40 profiles.

The legacy default protocol:

1. records a complete dense trace for every selected calibration sample;
2. writes an independent JSON/HTML/SVG attribution bundle for every
   worker-round node and the final reviewer;
3. writes round-specific spatial/temporal matrices for every sample;
4. aggregates only final-correct dev samples into pruning policies;
5. evaluates frozen policies on a disjoint benchmark evaluation split.

The opt-in MMLU dev40 protocol uses
`--attribution-mode correctness-aware --aggregation-scope all` instead. It
records the ordinary dense graph for all 40 selected questions, retains every
upstream output including wrong answers, and independently routes each receiving
worker by its own explicit `Decision`. Correct workers use the unchanged
full-output attribution. Wrong workers use complete greedy gold-prefix
insertion/deletion with the earliest raw-logit maximum, then one no-gold greedy
generation under the original reasoning-first prompt. Only successful
verification contributes the selected-token ratios; failure contributes observed
zeros that remain in the average denominator. It does not replace the original
worker output for subsequent agents. Reviewer reports are preserved and reviewer
edges remain dense and outside the matrices.

Use the dedicated root launcher from the repository root:

```bash
# Print the resolved default calibration command without loading a model.
DRY_RUN=1 ./run_mmlu_dev40_correctness_aware.sh

# Replace 0 with GPUs available for this job; no evaluation starts here.
CUDA_VISIBLE_DEVICES=0 ./run_mmlu_dev40_correctness_aware.sh calibrate

# CPU policy preparation, then manually scheduled GPU inference.
./run_mmlu_dev40_correctness_aware.sh prepare 0 5 10 20 30 40 50 60 70 80
CUDA_VISIBLE_DEVICES=0 ./run_mmlu_dev40_correctness_aware.sh eval 0 5 10 20 30 40 50 60 70 80
./run_mmlu_dev40_correctness_aware.sh summarize
```

The profile fixes dev indices 0..39 after dataset shuffle seed 888, five workers,
two rounds, temperature 0, global token-ratio conditional-mean ranking, and all
sample/all node attribution. The default run is
`mmlu/qwen3_8b/dev40_correctness_aware`; keep any `RUN_NAME` override identical
across stages. Attribution and evaluation batches default to 4 independently.
The 0% policy provides the dense control with all 45 worker edges enabled;
every evaluation uses the same batch size and the default full 1531-question
val selection. `MAX_NEW_TOKENS=2048`, `RESUME=1`, and `SAVE_EVAL_TRACES=1` are
defaults. Actual GPU stages require explicit `CUDA_VISIBLE_DEVICES`.

The selection manifest freezes trace paths and SHA256 hashes before attribution.
All node reports, full insertion/deletion rankings and candidate scores,
probability diagnostics, full-input answer-first diagnostic generation, local
verification input/output/token IDs, sample matrices, and heatmaps are saved.
Probability is diagnostic; raw-logit sum remains the optimization target.
Malformed/missing decisions use the wrong-node branch, while truncation and
runtime/alignment errors stop publication instead of counting as failed rescue.
The peak subset is not a global minimum, and one failed verification is not
proof that every subset fails. Forty questions is a fixed small-data budget,
not an exact match to the current local baseline scripts.
After recalibration, run `prepare` again: the launcher checks successful
aggregate publication and matches the current aggregate hash against the sweep
and requested policies before evaluation or summary. Completed evaluations are
reused only after the evaluator validates the full saved configuration.

The equivalent GSM8K profile is `run_gsm8k_train40_correctness_aware.sh`:

```bash
CUDA_VISIBLE_DEVICES=6,7 ./run_gsm8k_train40_correctness_aware.sh calibrate
# Run the following stages separately after calibration and when GPUs are free.
./run_gsm8k_train40_correctness_aware.sh prepare 50 60 80
CUDA_VISIBLE_DEVICES=6,7 EVAL_BATCH_SIZE=4 ./run_gsm8k_train40_correctness_aware.sh eval 50 60 80
./run_gsm8k_train40_correctness_aware.sh summarize
```

It fixes train indices 0..39 after shuffle seed 888, four workers and two rounds,
all-sample correctness-aware attribution, and a new run
`gsm8k/qwen3_8b/train40_correctness_aware`. The original GSM8K numeric answer
format is preserved: strict explicit `The answer is <number>` parsing routes
workers, and the wrong-node scoring target is `The answer is <gold number>`.
The full insertion/peak/no-gold verification workflow otherwise matches MMLU.
The reviewer uses four run-fixed opaque source IDs, with bodies unchanged;
collection, resume, and evaluation reuse the saved mapping. Old GSM8K runs with
semantic reviewer labels need a different run directory for the new experiment.

GSM8K globally ranks all 28 worker edges (12 spatial + 16 temporal), excludes
the four reviewer edges, and evaluates the three ratios sequentially on test1319
with batch4. Defaults are `calibrate` only and pruning percentages `50 60 80`.
Explicit 0% supplies a dense control. `TEST_LIMIT` overrides the test selection;
`ATTR_BATCH_SIZE` and `EVAL_BATCH_SIZE` control the respective batches. All
publication, frozen-selection, policy-source, and batch consistency checks apply.
Final summaries are in
`evaluation/static_token_ratio_conditional_mean_global_sweep_test/summary.{json,csv}`.

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

## Batched evaluation

Evaluation defaults to `--eval-batch-size 1`, which retains the original serial
inference path. Set `EVAL_BATCH_SIZE=4` in the MMLU, shared benchmark, or static
pruning sweep runner to enable batching for a new evaluation. The setting is
independent of `ATTR_BATCH_SIZE`, which controls attribution scoring.

Following AgentDropout's evaluation pattern, several independent questions run
concurrently and their pending Hugging Face generation requests are combined
into model batches. Each question keeps its worker order, round dependencies,
reviewer inputs, prompts, and separate trace. Sizes greater than one require
`temperature=0`; sampling is rejected to avoid changes in random-number usage.
Larger batches require more GPU memory, and throughput depends on prompt and
output lengths.

Batching preserves the graph computation, but different GPU matrix shapes can
introduce floating-point differences, even with greedy decoding. Compare a
small selection against serial execution before using batching for formal
results. Use distinct evaluation names so the comparison cannot overwrite or
resume the other run, for example, with an existing calibrated run:

```bash
python -m experiments_attrprune.evaluate_mmlu \
  --run-dir /path/to/calibrated/run --mode dense --limit 8 \
  --evaluation-name serial_check --eval-batch-size 1 --save-traces
python -m experiments_attrprune.evaluate_mmlu \
  --run-dir /path/to/calibrated/run --mode dense --limit 8 \
  --evaluation-name batch_check --eval-batch-size 4 --save-traces
```

Use the run's original `--model-name` and `--seed` if they differ from the
defaults. Compare final predictions and per-node trace outputs, not only
aggregate accuracy. Keep size 1 when exact reproduction of serial outputs is
required. Resume validation records the batch size; legacy checkpoints without
this field are treated as size 1. A different size requires a different
evaluation name. Dynamic batch membership may still differ after resuming.
Even with the same maximum batch size, resume cannot guarantee bit-for-bit
reproduction of an uninterrupted batched run.

CUDA out-of-memory failures during batched generation release the failed
generation's tensor references and split that physical batch into smaller
sequential batches, down to one request. A singleton that still runs out of
memory stops the evaluation. The configured maximum sample concurrency and
checkpoint identity stay unchanged. Each new batched trace event records
`requested_batch_size`, `effective_batch_size`, `oom_split_depth`, and
`oom_fallback` in `llm_call`; actual batch shapes can change floating-point
results. Per-sample metrics and the evaluation's `generation_batching` summary
record affected calls and samples. Older checkpoints are counted as lacking
batch metadata, not assumed to have used a particular physical batch size.
The sweep CSV includes this coverage and fallback counts. Token totals count
successful calls; wall time also includes failed attempts and recovery.

Timing distinguishes summed per-question latency
(`sample_elapsed_seconds_sum`) from elapsed processing time
(`elapsed_seconds`, accumulated from nonoverlapping intervals between completed
checkpoints). Time spent on failed work after the last successful checkpoint is
excluded from that accumulated value.
`invocation_elapsed_seconds` measures only the current evaluation invocation;
the accumulated fields include resumed records. For serial versus batched
throughput, compare this invocation field on fresh runs over the same selection;
it excludes model loading. Serial `elapsed_seconds` retains its historical
sample-latency definition. Policy comparison requires the same batch size.

Validation at implementation time: the full suite passed 119 tests, including
batch padding/EOS handling, observer isolation, cancellation, checkpoints, and
resume compatibility. A CPU integration check using Transformers 4.51.3, a
small randomly initialized Qwen3, and the real Qwen3 tokenizer ran two questions
through two workers, one round, and a reviewer. Serial and batched outputs,
trace token IDs, and usage matched exactly; generation calls decreased from
six single requests to three batches of two. This CPU check does not establish
equivalence for the full model, as the GPU smoke below demonstrates.

### Real Qwen3-8B GPU smoke (2026-10-28)

The full local Qwen3-8B weights were tested in BF16 on one shared RTX A6000
(physical GPU 4), using MMLU val index 0 with shuffle seed 888 (`moral_disputes`,
gold C). The dense graph used five workers, two rounds, the saved anonymous
reviewer IDs from `full_dev_tok2048_random_reviewer_v2_legacy_policy_transfer`,
temperature 0, and a 2048-token ceiling. Both batch sizes were warmed up before
timing; model loading was excluded and CUDA was synchronized at each boundary.

Only one unique question was selected. It ran once serially, then two copies
ran concurrently so real generation batching could be exercised.

| Mode | Completed copies | Wall time | Time per copy | Generated tokens per copy | Final answer |
|---|---:|---:|---:|---:|---|
| Serial, batch size 1 | 1 | 29.32 s | 29.32 s | 788 | B |
| Batched, batch size 2 | 2 | 27.17 s | 13.58 s | 683 | C, C |

**The serial/batched final-answer equivalence check failed.** All eleven events
per copy reached EOS, but the first worker already diverged at its twelfth
generated token despite identical prompt token IDs, attention masks, and
decoding parameters. The two batched copies matched each other exactly.
Direct Transformers generation from the same prompt reproduced the divergence
without the batching wrapper: at token 12, serial scores for ` the` and ` his`
were both 45.75 (two repeat runs), while batch size 2 produced 45.50 and 45.75,
respectively. Both modes generated the same preceding token prefix. This
localizes the change to model numerics under different batch shapes rather
than prompt construction or padding; details are in `first_divergence.json`.

The apparent sample-throughput ratio was 2.16x, but batched outputs were shorter
and the GPU was shared; this is neither an equal-work speed measurement nor an
estimate for the whole evaluation set. Batch size 1 remains the default for
preserving the original execution path.

The benchmark script, environment metadata, per-node traces, and summary are
saved locally under
`results/AgentAttrPrune/diagnostics/qwen3_8b_batch_smoke_20261028/` (generated
artifacts are ignored by Git).

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
calibrated from the aggregate's recorded MMLU `dev` selection (verified-correct
by default, or all 40 in the correctness-aware protocol), then frozen and
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
Completed evaluations are skipped only when their saved batch size matches
`EVAL_BATCH_SIZE`; legacy summaries are treated as size 1.

Evaluations checkpoint `predictions.jsonl` atomically after every completed
sample. Pass `--resume` (or set `RESUME=1` in the pipeline runners) to validate
`evaluation_state.json` and skip every saved sample, whether its prediction was
correct or incorrect. The state locks the run fingerprint, policy hash, dataset
selection, seed, trace mode, and evaluation batch size, so incompatible
checkpoints are never mixed.
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
