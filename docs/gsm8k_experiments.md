# GSM8K calibration, pruning, and visualization

The correctness-aware GSM8K experiment calibrates on **40 training questions**
and evaluates frozen communication policies on **all 1,319 test questions**.
The training subset is indices `0..39` after the dataset shuffle with seed 888.
It is separate from the MMLU dev40 experiments and from older full-training-set
or correct-only GSM8K calibrations.

Run all commands from the repository root. Install the package and select a
Python environment as described in the [main README](../README.md).

## Data and environment

The loader expects the original `train.jsonl` and `test.jsonl` files, with
`question` and `answer` fields and the `####` final-answer delimiter. Download
missing files with:

```bash
python -m AgentAttrPrune.datasets.download gsm8k --data-root data
```

On a machine other than the original experiment server, override the local
model and interpreter defaults. These exports apply to the examples below:

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export GSM8K_DATA_DIR="$PWD/data/GSM8K"
export CONDA_ENV=""
export PYTHON_BIN="$(command -v python)"
```

The launchers otherwise default to the original server's Qwen3-8B and Conda
paths. An explicit empty `CONDA_ENV` makes the correctness-aware launcher use
`PYTHON_BIN` directly. Choose available GPUs explicitly; `6,7` below is only an
example device pair.

## Protocol

| Setting | GSM8K train40 |
|---|---|
| Worker order | Math Solver → Mathematical Analyst → Programming Expert → Inspector |
| Worker rounds | 2 |
| Calls per Dense sample | 9: eight worker calls and one reviewer |
| Spatial candidates | 6 per round, 12 total; no same-round self-connections |
| Temporal candidates | Round 0: none; Round 1: all 16 previous-to-current edges |
| Reviewer inputs | All four final-round workers; excluded from pruning |
| Generation | Qwen3-8B, temperature 0, at most 2,048 new tokens, thinking disabled |
| Seeds | Generation seed 0; dataset shuffle seed 888 |
| Attribution / evaluation batch | 4 / 4 |
| Attribution / aggregation | `correctness-aware` / `all` |
| Edge importance | `token_ratio` with `conditional_mean` |

All 40 traces contribute regardless of the reviewer's final correctness. The
calibration produces 360 node records, including the reviewer. Correct workers
use the original full-output attribution and its recovery-based early stopping.
Incorrect workers use the explicit `The answer is <gold number>` prefix for
local scoring, complete greedy insertion, and the earliest maximum-scoring
subset. A separate greedy verification restores the original reasoning-first
prompt and omits the gold answer. Numeric targets may span multiple tokens.

Correctness routing requires an explicit numeric answer field. Missing,
malformed, or conflicting fields take the incorrect-worker branch; numbers
mentioned only in reasoning do not establish a correct answer. This strict
attribution routing is separate from the benchmark's ordinary final-answer
normalization. Reviewer attribution uses the original attribution procedure
and does not enter worker-edge matrices.

A failed local rescue contributes zero on each existing edge and remains in
the denominator. Runtime, truncation, and alignment failures block publication
of a complete calibration. Local verification never changes the original trace
or downstream inputs. See the [algorithm specification](attribution_edge_pruning.md)
for the complete objective and validation rules.

Reviewer source labels are run-fixed anonymous IDs, stored in
`prompt.reviewer_agent_ids`. They are reused for collection, resume, and every
pruning rate. Old runs with a different reviewer prompt contract require a new
run directory.

## Calibrate and evaluate global pruning

The default run is
`results/AgentAttrPrune/gsm8k/qwen3_8b/train40_correctness_aware`.

```bash
CUDA_VISIBLE_DEVICES=6,7 ./run_gsm8k_train40_correctness_aware.sh calibrate
```

This command collects the 40 Dense traces, attributes them, and aggregates the
matrices. It stops before test evaluation. Then prepare the policies and run
the test sweep:

```bash
./run_gsm8k_train40_correctness_aware.sh prepare 50 60 80
CUDA_VISIBLE_DEVICES=6,7 ./run_gsm8k_train40_correctness_aware.sh eval 50 60 80
./run_gsm8k_train40_correctness_aware.sh summarize
```

Run each stage after the preceding stage succeeds. Global pruning ranks all
28 worker candidates together across rounds and edge types. Each percentage
starts from the unpruned aggregate. The 50%, 60%, and 80% policies retain 14, 11,
and 6 worker edges, respectively, because pruning counts use the existing
rounding rule. Add `0` to both `prepare` and `eval` for a Dense control using the
same run configuration and reviewer IDs.

The masks are static: each inference executes every retained edge. These
launchers do not apply sigmoid probabilities or sample Bernoulli edge masks.
They use the standard two-round evaluator; the optional MMLU experiment that
skips an entirely empty final round is a separate execution protocol.

Resume is enabled by default and evaluation traces are saved. Reuse the same
command to resume, with the same batch size and selection. Change `RUN_NAME`
when changing calibration or generation settings. The global launcher accepts
`ATTR_BATCH_SIZE`, `EVAL_BATCH_SIZE`, `TEST_LIMIT`, `RESULTS_ROOT`, and `RUN_NAME`
overrides. For command inspection without inference:

```bash
DRY_RUN=1 ./run_gsm8k_train40_correctness_aware.sh calibrate
```

## Uniform pruning allocation

The uniform-allocation experiment reuses the same completed train40 aggregate,
but applies the pruning percentage independently within each round and edge
type: six Round 0 spatial edges, six Round 1 spatial edges, and sixteen Round 1
temporal edges. It also uses fixed masks, without edge sampling.

```bash
./run_gsm8k_train40_uniform_pruning.sh prepare 50 60 80
CUDA_VISIBLE_DEVICES=6,7 ./run_gsm8k_train40_uniform_pruning.sh eval 50 60 80
./run_gsm8k_train40_uniform_pruning.sh summarize
```

This wrapper defaults to a `60 80` sweep when percentages are omitted. Its GPU
default is `4,5`, so pass `CUDA_VISIBLE_DEVICES` explicitly. If calibration used
a custom output directory, set `RUN_DIR` to that directory when using this
wrapper. Uniform 50%, 60%, and 80% retain 14, 10, and 5 worker edges respectively;
rounding within partitions can differ from the global edge budget.

Both sweep preparers replace their respective sweep manifest with the requested
percentages. Prepare the complete set once before launching separate `eval`
jobs. The global sweep uses a directory ending in `_global`; the uniform sweep
uses a separate directory without that suffix.

## Matrix and insertion-curve visualization

After calibration and policy preparation, the shared matrix renderer supports
both GSM8K pruning allocations. It uses short worker names (`Solver`, `Anal.`,
`Prog.`, `Insp.`), rows for sources, columns for targets, and one panel per round
and edge type. Round 0 temporal has no candidate edges. Pruned cells are marked
separately from non-candidates and from retained edges with zero importance.

```bash
GSM8K_RUN=results/AgentAttrPrune/gsm8k/qwen3_8b/train40_correctness_aware
GSM8K_SWEEP="$GSM8K_RUN/attribution/static_pruning/token_ratio_conditional_mean_global"

python -m experiments_attrprune.visualize_pruning_matrices \
  "$GSM8K_SWEEP/policies/prune_50pct.json" \
  "$GSM8K_SWEEP/policies/prune_60pct.json" \
  "$GSM8K_SWEEP/policies/prune_80pct.json" \
  --output-dir "$GSM8K_SWEEP/figures" --color-max 1.0
```

This produces PNG and SVG files. Supply the same `--color-max` for comparisons;
the renderer rejects a scale that would clip an importance value. For uniform
policies, use `token_ratio_conditional_mean` instead of the `_global` directory.
To show the original importance matrix without a pruning overlay, pass any
prepared policy from that aggregate with `--intensity-only`, and choose a
separate output directory:

```bash
python -m experiments_attrprune.visualize_pruning_matrices \
  "$GSM8K_SWEEP/policies/prune_50pct.json" --intensity-only \
  --output-dir "$GSM8K_RUN/attribution/intensity" --color-max 1.0
```

Per-node reports in `attribution/sample_*/` include JSON data, an HTML report,
a sentence-attribution SVG, and `event_*.insertion_curve.svg`. The original
attribution curve may stop early; the incorrect-worker gold-prefix curve
contains the complete insertion search and the selected earliest peak. A raw
logit score is not a probability, and a failure's zero matrix contribution does
not imply that its insertion curve was flat.

Cached per-node reports can be regenerated on CPU without rerunning inference:

```bash
python -m experiments_attrprune.attribute_edges \
  --benchmark gsm8k --model-name "$MODEL_PATH" --run-dir "$GSM8K_RUN" \
  --attribution-mode correctness-aware --aggregation-scope all --render-only
```

## Saved outputs

| Output | Location relative to the run directory |
|---|---|
| Frozen configuration and reviewer IDs | `run_config.json` |
| Dense traces and frozen selection | `calibration/traces/`, `calibration/selection_manifest.json` |
| Node reports and aggregate | `attribution/sample_*/`, `attribution/aggregate_edge_matrices.json` |
| Global policies and manifest | `attribution/static_pruning/token_ratio_conditional_mean_global/` |
| Uniform policies and manifest | `attribution/static_pruning/token_ratio_conditional_mean/` |
| Global test sweep summary | `evaluation/static_token_ratio_conditional_mean_global_sweep_test/summary.json` and `.csv` |
| Uniform test sweep summary | `evaluation/static_token_ratio_conditional_mean_sweep_test/summary.json` and `.csv` |

Per-rate evaluation directories contain predictions, metrics, and saved traces.
Summaries include accuracy, token use, runtime, and retained-edge counts. Use
the saved configurations and observed edge counts when comparing rounded
pruning budgets or separate calibration protocols.
