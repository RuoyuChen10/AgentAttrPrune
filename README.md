# AgentAttrPrune

Sentence-attribution-based communication pruning for local LLM multi-agent systems.
AgentAttrPrune records complete reasoning traces, measures which incoming messages
support each receiving agent, and converts those measurements into round-specific
communication policies. It supports MMLU, GSM8K, and HumanEval with a local
Hugging Face backend.

The repository includes fixed-order pruning, correctness-aware attribution,
random-order calibration, per-inference edge sampling, attribution-based agent
ordering, and independent calibration-set resampling. Experiment outputs are
kept separate from source code and are not distributed with the repository.

## Documentation

| Guide | Contents |
|---|---|
| [MMLU ablations](docs/mmlu_ablations.md) | Random order, sampled policies, reordered graphs, empty-round skipping, and calibration resampling |
| [GSM8K experiments](docs/gsm8k_experiments.md) | Train40 calibration, global/uniform static pruning, test evaluation, and visualization |
| [GSM8K resampling](docs/gsm8k_resampling.md) | Independent seed1/seed2 train40 cohorts and six fixed-order static pruning rates |
| [Aligned baselines](baselines/README.md) | Minimal AgentPrune, AgentDropout, and SafeSieve runners with shared prompts and inference batch size 4 |
| [Attribution specification](docs/attribution_edge_pruning.md) | Scoring, sentence search, correctness routing, denominators, and artifact schemas |
| [Experiment reference](experiments_attrprune/README.md) | General benchmark stages, batching, and legacy workflows |
| [Dataset setup](data/README.md) | Local layouts, downloads, and calibration/evaluation splits |

## Installation

Use Python 3.10+, CUDA-enabled PyTorch, Transformers 4.51+, Accelerate, and
local Qwen3 model weights. From the repository root:

```bash
conda create -n agentattrprune python=3.10 -y
conda activate agentattrprune
pip install -e '.[test]'

# Use this environment instead of the launchers' lab-specific defaults.
export CONDA_ENV=''
export PYTHON_BIN="$(command -v python)"
export MODEL_PATH=/path/to/Qwen3-8B
export MMLU_DATA_DIR="$PWD/data/MMLU"
export GSM8K_DATA_DIR="$PWD/data/GSM8K"

python -m AgentAttrPrune.datasets.download all --data-root data
```

Run commands from the repository root. Select available GPUs explicitly with
`CUDA_VISIBLE_DEVICES`; the GPU IDs in the examples are illustrative. Shell
launchers support `--help` and, where documented, `DRY_RUN=1`. Some convenience
scripts retain the original lab's model/interpreter paths, so set the environment
above when using another machine. Experiments derived from an existing run inherit
its frozen model and data configuration instead of accepting a replacement model.

Model placement uses `device_map=auto`. Making two GPUs visible does not imply
data parallelism or equal utilization of both devices.

## Method and experiment contract

A dense round executes workers in order: each worker can receive spatial messages
from all earlier workers. Round 0 has no temporal inputs; Round 1 can receive all
previous-round worker outputs, including the same worker's earlier output. The
reviewer reads all final-round worker outputs and is excluded from pruning.

| Benchmark | Calibration / evaluation | Workers in canonical order | Two-round worker edges | Calls per dense sample |
|---|---|---|---:|---:|
| MMLU | dev (285) / val (1,531) | Knowledgeable Expert, Critic, Mathematician, Psychologist, Historian | 45 | 11 |
| GSM8K | train (7,473) / test (1,319) | Math Solver, Mathematical Analyst, Programming Expert, Inspector | 28 | 9 |
| HumanEval | fixed dev (80) / disjoint test (81) | Project Manager, Algorithm Designer, Programming Expert, Test Analyst, Bug Fixer | 45 | 11 |

The current fixed-order small-calibration profiles use Qwen3-8B, two rounds, temperature 0,
a 2,048-token generation ceiling, thinking disabled, serial calibration
collection, attribution batch 4, and evaluation batch 4. MMLU and GSM8K preserve
opaque reviewer source IDs in the run configuration; canonical agent identities
remain available in every trace and matrix.

### Correctness-aware attribution

The `dev40` and `train40` profiles use **correctness-aware attribution with all
selected samples**, including final-reviewer errors:

1. Record the ordinary dense graph, preserving every original upstream output.
2. Correct receiving workers use teacher-forced attribution of their full output,
   with logit-sum scoring and the original 0.95-recovery early stop.
3. Incorrect workers use full greedy gold-prefix insertion. Select the earliest
   maximum-scoring prefix, then verify once without gold under the original prompt.
4. Successful verification contributes the selected-source-token ratio. Failed
   verification contributes zero, while the observed edge remains in the denominator.

Local verification never rewrites the original trace or propagates a corrected
answer downstream. The reviewer retains ordinary attribution. Incorrect-node
insertion runs to completion; it does not use the correct-node recovery early stop.
The selected prefix is not guaranteed to be the globally smallest sufficient set.

MMLU routes workers using their explicit `Decision` field. GSM8K uses the explicit
`The answer is <number>` field and supports multi-token numeric answers. Missing
or invalid answer fields take the incorrect-worker branch. Truncation, runtime
failure, and token-alignment errors cannot be counted as successful zero attribution.

The general-purpose legacy runners still default to `original` attribution and
final-correct-only aggregation. Record the attribution mode and aggregation scope
when comparing experiments.

### Edge importance and averaging

The primary metric is `token_ratio`: selected source-output tokenizer tokens
divided by all visible tokens from that source. Task text, prompts, labels,
separators, and special tokens are excluded. `binary` and `word_ratio` are also
available. Matrices always use **`[round][source][target]`**.

For an edge that appears in only some traces:

\[
\operatorname{conditional\_mean}_e =
\frac{\sum_{t:\,e\text{ exists in }t} \operatorname{importance}_{e,t}}
     {\#\{t:e\text{ exists in }t\}}.
\]

An absent edge does not enter the denominator. An existing edge whose local rescue
fails contributes zero and does enter it. For example, scores `0.6, 0.3, 0` across
three observed occurrences give `0.3`, even if a fourth trace lacks the edge.
Spatial self-edges are forbidden; cross-round self-temporal edges are allowed.
Reviewer attribution is stored separately from worker matrices.

## MMLU: original fixed-order baseline

This profile selects the first 40 dev questions after dataset shuffle seed 888.
The run seed is 0. Calibration stops before validation; prepare and
evaluate pruning policies separately:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_dev40_correctness_aware.sh calibrate

./run_mmlu_dev40_correctness_aware.sh prepare 25 50 60 70 80 90
CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_dev40_correctness_aware.sh eval 25 50 60 70 80 90
./run_mmlu_dev40_correctness_aware.sh summarize
```

The default run is `results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware`.
Explicit rates override the launcher's default rate list. Add `0` to both
`prepare` and `eval` for a dense control. Preparing a sweep writes its rate
manifest: prepare the complete intended rate list before parallel evaluation.

Global static pruning ranks the 45 worker candidates together. The six requested
rates above retain **34, 23, 18, 14, 9, and 5** worker edges under the existing
rounding rule. Every retained edge is used on every inference; there is no
Bernoulli sampling. The five reviewer inputs are never pruned. Evaluation uses
all 1,531 val questions and saves per-question predictions and traces.

`run_mmlu_fixed_order_25_70.sh` is a source-preserving supplement for the original
local run: it adds 25% and 70% results without replacing the earlier sweep
manifest. Its source-preservation checks are tied to that run. Use the general
profile above for a new installation.

For a uniform-allocation comparison using the same seed-0 matrix and fixed
execution order, run `CUDA_VISIBLE_DEVICES=6,7 ./run_mmlu_fixed_order_uniform.sh`.
This evaluates 50%, 60%, and 80% sequentially with batch size 4 and resume enabled.
Uniform allocation prunes Round-0 spatial, Round-1 spatial, and Round-1 temporal
independently (`per_round_edge_type`), retaining 23, 18, and 9 worker edges in
total. Every retained edge is used, both rounds execute, and the original global
policies and results remain separate. No calibration or attribution is repeated.

## MMLU ablation workflows

| Experiment | Calibration or policy change | Evaluation behavior |
|---|---|---|
| Random-order Dense | Independently permute workers in each round | Calibration/attribution only |
| Random-order sampled policy | Prune random-order importance matrices globally or uniformly; use importance as a logit | Sample edges at every inference, rejecting spatial cycles |
| Row-sum minus column-sum, direct | Freeze an order per round; project existing importance onto forward edges | Static global pruning |
| Row-sum minus column-sum, recalibrate | Regenerate the same dev40 under the frozen new orders and re-attribute | Static global pruning |
| Skip empty Round 1 | If both Round 1 masks are empty, send Round 0 outputs directly to reviewer | Separate evaluation checkpoints |
| Independent dev40 resampling | Change only the 40 calibration questions, using selection seeds 1 and 2 | Six static global pruning rates |

### Random-order Dense calibration

Each round uses its own independently drawn uniform worker permutation. Agent
identities and matrix coordinates stay fixed. Each rollout still has 10 spatial
edges per round and 25 Round 1 temporal edges. The full profile freezes
**40 questions × 10 repeats = 400 rollouts**:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_dev40_random_order.sh calibrate
./run_mmlu_dev40_random_order.sh visualize
```

For the smaller experiment used subsequently, reuse exactly `repeat_00` from each
question in the frozen larger run:

```bash
./run_mmlu_dev40_random_order_once.sh prepare
CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_dev40_random_order_once.sh calibrate
```

This subset wrapper requires existing source traces; it does not generate 40 new
traces. It reuses valid completed attribution and computes missing attribution.
The larger attribution run must not write concurrently. A fresh 40-rollout CLI
recipe, source requirements, and all individual stages are in the
[MMLU guide](docs/mmlu_ablations.md#random-order-400-rollouts-and-the-40-rollout-subset).

Random ordering makes both directions of a worker pair observable across traces.
Unobserved positions remain distinguishable from observed zero importance. The
400-rollout budget reproduces a local AgentPrune GSM8K sampling budget; it is not
a claim of exact alignment with AgentPrune's MMLU setup.

### Global and uniform pruning followed by edge sampling

Use the completed random-order dev40 matrix, retaining separate round matrices:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_random_order_sampling.sh global 50 60
CUDA_VISIBLE_DEVICES=2,3 ./run_mmlu_random_order_sampling.sh uniform 50 60
```

Global pruning ranks **65** candidates together: 20 spatial directions per round
plus 25 temporal edges. Uniform pruning allocates the rate independently to
Round 0 spatial, Round 1 spatial, and Round 1 temporal. Each rate starts from
the original unpruned matrices.

After pruning, the remaining edge's raw importance is treated as a logit:
`p(edge) = sigmoid(importance)`. Each inference samples a new graph; spatial
cycle checks keep it executable. These probabilities are not learned AgentPrune
logits or estimates of causal importance. This is an AgentPrune-style graph
sampling ablation, not a reproduction of its trainable optimizer. See the
[MMLU guide](docs/mmlu_ablations.md) for exact replay and visualization details.

To accelerate an unfinished serial 60% evaluation, stop its serial process and
run `CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_sampled_batch4_resume.sh global`
(or `CUDA_VISIBLE_DEVICES=2,3 ./run_mmlu_sampled_batch4_resume.sh uniform`).
The separate continuation checkpoint retains completed answers and evaluates
only the remaining questions with batch size 4. Results explicitly record mixed
batch-1/batch-4 provenance; completed 50% results are unchanged. See the
[continuation instructions](docs/random_order_sampled_pruning.md#continuing-a-serial-60-evaluation-with-batch-size-4).

For additional 80% sampled-policy evaluations, run
`CUDA_VISIBLE_DEVICES=4,5 ./run_mmlu_sampled_80pct.sh`. This runs global then
uniform pruning sequentially, both with evaluation batch size 4 from the start.
It preserves the original two-round sampled execution rule, including empty
Round 1, and saves independent resumable checkpoints for each scope.

### Agent ordering: direct projection versus recalibration

For each round separately, compute `score_i = row_sum_i - column_sum_i` on the
raw spatial conditional means, then sort descending, breaking ties by canonical
identity order. Freeze those orders before either branch. Retain only forward
spatial edges; keep all 25 temporal candidates.

```bash
CUDA_VISIBLE_DEVICES=4,5 ./run_mmlu_reordered_pruning.sh direct all 80 60 50
CUDA_VISIBLE_DEVICES=6,7 ./run_mmlu_reordered_pruning.sh recalibrate all 80 60 50
```

The direct branch preserves the original importance sums and observation counts
on allowed edges. The recalibration branch reruns exactly the same 40 questions
and attributes the new traces; each allowed edge then has 40 observations.
Both branches prune globally over 45 candidates, retain 9/18/23 edges at
80/60/50%, and use fixed masks at inference. Recalibration does not change the
already frozen orders.

The original reordered evaluations execute both worker rounds. To reproduce
the separate 80% empty-round ablation while preserving those results:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_reordered_skip_empty80.sh both
```

Only when **both** Round 1 spatial and temporal masks are empty does this evaluator
skip its five calls and give the reviewer all five Round 0 outputs. The resulting
sample has six calls instead of eleven. Other ratios and existing checkpoints
are unchanged.

### Two additional independent dev40 cohorts

```bash
# CPU preparation, then the complete two-cohort queue.
STAGE=prepare ./run_mmlu_dev40_resampling.sh 1 2
CUDA_VISIBLE_DEVICES=6,7 ./run_mmlu_dev40_resampling.sh 1 2
```

Each selection seed independently draws 40 distinct dev questions; overlap
between cohorts is allowed. Only calibration membership changes: generation seed
0, canonical order, reviewer aliases, attribution settings, and all 1,531 val
questions are inherited from the original run. Cohorts run sequentially, each
with **25%, 50%, 60%, 70%, 80%, 90%** static pruning. These evaluators apply the
empty-Round-1 rule at every rate. The original cohort is reused, not regenerated.

The launcher saves results and per-cohort summaries; it does not draw an error-bar
curve. Repeating the same command resumes completed work. See
[resampling details](docs/resampled_mmlu.md).

## GSM8K: train40 and static pruning

The correctness-aware profile uses the first 40 shuffled training questions,
four workers, two rounds, and the full 1,319-question test split. Run global
pruning as follows:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_gsm8k_train40_correctness_aware.sh calibrate
./run_gsm8k_train40_correctness_aware.sh prepare 50 60 80
CUDA_VISIBLE_DEVICES=0,1 ./run_gsm8k_train40_correctness_aware.sh eval 50 60 80
./run_gsm8k_train40_correctness_aware.sh summarize
```

Global pruning ranks all 28 worker edges together: 12 spatial and 16 temporal.
At 50/60/80%, it retains 14/11/6 edges. The four reviewer inputs remain dense.

To supplement the existing seed-0 run with 25% global static pruning, use
`CUDA_VISIBLE_DEVICES=0,1 ./run_gsm8k_seed0_25pct.sh`. It reuses the existing
matrix, retains 21 of 28 worker edges, and evaluates all 1,319 test questions
with batch size 4 and resume enabled. Original sweep metadata and results are
preserved; no calibration or attribution is repeated.

To compare uniform allocation using the same completed attribution:

```bash
./run_gsm8k_train40_uniform_pruning.sh prepare 50 60 80
CUDA_VISIBLE_DEVICES=2,3 ./run_gsm8k_train40_uniform_pruning.sh eval 50 60 80
./run_gsm8k_train40_uniform_pruning.sh summarize
```

Here "uniform" means pruning independently per round and edge type. It is still
**static** pruning and has no sigmoid or per-inference edge sampling. Separate
policy/evaluation directories preserve global results. The detailed
[GSM8K guide](docs/gsm8k_experiments.md) covers numeric answer parsing, data,
rounding, output paths, and visualization commands.

To draw two additional independent 40-question training cohorts while preserving
the original fixed order, model, reviewer aliases, and generation seed 0:

```bash
STAGE=prepare ./run_gsm8k_train40_resampling.sh 1 2
CUDA_VISIBLE_DEVICES=0,1 ./run_gsm8k_train40_resampling.sh 1 2
```

Each cohort runs correctness-aware calibration followed by global static
**25%, 50%, 60%, 70%, 80%, 90%** pruning on the same full test split. The retained
worker budgets are **21, 14, 11, 8, 6, 3**. Sampling is without replacement within
each cohort; overlap between cohorts is allowed. These resampling evaluators
skip Round 1 only when both of its worker edge masks are empty, passing Round 0
outputs to the reviewer. Results are isolated by cohort and rate, resume is
enabled, and no accuracy curve is generated. See the
[GSM8K resampling guide](docs/gsm8k_resampling.md) for configuration and commands.

The general `run_full_gsm8k.sh`, `run_full_humaneval.sh`, and
`experiments_attrprune/run_*.sh` entrypoints remain available for full-split or
legacy experiments. Their defaults differ from the train40/dev40 profiles;
consult the [experiment reference](experiments_attrprune/README.md).

## Visualization

All plotting can use saved JSON without loading the language model. The display
uses short names such as **Knowl., Critic, Math., Psych., Hist.** while preserving
canonical IDs in saved data. PNG supports inspection, SVG supports scalable
figures, and JSON/CSV retain values and provenance where exported by the renderer.

### Frozen policies and unpruned importance

```bash
BASE=results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware
POLICIES="$BASE/attribution/static_pruning/token_ratio_conditional_mean_global/policies"

python -m experiments_attrprune.visualize_pruning_matrices \
  "$POLICIES/prune_50pct.json" "$POLICIES/prune_60pct.json" \
  "$POLICIES/prune_80pct.json" \
  --output-dir "$BASE/visualizations/pruning" --color-max 1.0

python -m experiments_attrprune.visualize_pruning_matrices \
  "$POLICIES/prune_50pct.json" --intensity-only \
  --output-dir "$BASE/visualizations/importance" --color-max 1.0
```

The first command writes a figure for each supplied policy. The second shows all
candidate attribution values without the pruning overlay. Choose a shared color
maximum large enough for all compared matrices; values above a supplied maximum
must not be silently clipped. The same renderer accepts GSM8K policies.

### Random-order and reordered matrices

```bash
./run_mmlu_dev40_random_order_once.sh visualize
./run_mmlu_reordered_pruning.sh direct visualize
./run_mmlu_reordered_pruning.sh recalibrate visualize
```

Random-order plots include per-round importance and observation-count heatmaps.
The cross-round spatial summary is **the two rounds' importance sum divided by
their total observed count**, not an unweighted average of means. Fixed-order
comparison panels are explicitly labeled with their own calibration budget.

For reordered plots, spatial axes follow that round's execution order. Round 1
temporal rows follow the Round 0 source order and columns follow the Round 1
target order. Round 0 temporal is marked absent. Spatial diagonals indicate
forbidden self-connections; unobserved entries and measured zero are distinct.
See [visualization recipes](docs/mmlu_ablations.md) for sigmoid probability
matrices and global/uniform comparisons.

For a completed or prepared sampled-policy experiment, export the sigmoid
probabilities and compare global versus uniform pruning on the same scale:

```bash
RANDOM_RUN=results/AgentAttrPrune/mmlu/qwen3_8b/dev40_random_order1_correctness_aware
python -m experiments_attrprune.visualize_sampled_mmlu \
  --evaluation-root "$RANDOM_RUN/evaluation/agentprune_sampling/per_round_sigmoid" \
  --output-dir "$RANDOM_RUN/visualizations/sampled_policies" \
  --scopes global uniform --percentages 50 60
```

These are Bernoulli sampling thresholds for cycle-eligible edges, not empirical
edge-inclusion frequencies after cycle rejection. Pruned candidates and absent
edges have separate visual markers.

### Sentence reports and insertion curves

Each attributed event stores JSON, HTML, a sentence-attribution SVG, and insertion
curve SVG. Correct-node curves show recovery and the selected early-stop prefix;
incorrect-node reports include full gold-prefix insertion/deletion diagnostics,
the earliest peak, and no-gold verification. Curves explain the attribution
search, not validation accuracy.

Re-render saved event reports without model inference:

```bash
ATTRIBUTION_MODE=correctness-aware AGGREGATION_SCOPE=all \
RUN_NAME=mmlu/qwen3_8b/dev40_correctness_aware \
  ./experiments_attrprune/run_mmlu.sh render
ATTRIBUTION_MODE=correctness-aware AGGREGATION_SCOPE=all \
RUN_NAME=gsm8k/qwen3_8b/train40_correctness_aware \
  ./experiments_attrprune/run_gsm8k.sh render
```

## Outputs, resume, and comparison

A typical run contains:

```text
results/AgentAttrPrune/<benchmark>/qwen3_8b/<run>/
├── run_config.json
├── calibration/                 # Selection, state, and per-question traces
├── attribution/
│   ├── manifest.json
│   ├── aggregate_edge_matrices.json
│   └── sample_*/                # Event reports, curves, node index, matrices
├── evaluation/                  # Separate directories per policy/experiment
│   └── <evaluation>/
│       ├── predictions.jsonl
│       └── evaluation_summary.json
└── ...                          # Frozen order plans, policies, summaries
```

Exact paths depend on the experiment; policies may live under
`attribution/static_pruning/` or a run-level `policies/` directory. Launchers print
resolved run and log paths. Independent evaluations preserve the original
calibration artifacts and use distinct checkpoints.

Keep model, dataset, prompt contract, source identities, aggregation scope,
decoding, and evaluation batch size consistent. Floating-point differences in
batched generation can change greedy outputs, so batch 1 and batch 4 should not
be mixed in an accuracy comparison. Changing frozen settings requires a new
run/evaluation identity. Missing, changed, duplicate, or incomplete artifacts
are rejected rather than silently reused as complete results.

Report accuracy together with the calibration budget, pruning allocation,
requested and actual retained edges, sampled edge counts when applicable,
prompt/generated tokens, runtime, and empty-round execution rule. Validation or
test questions never contribute to calibration attribution. Token-ratio
importance and its sigmoid transform are not calibrated causal probabilities.

## Repository layout and testing

```text
AgentAttrPrune/          # Graphs, prompts, local model backend, tracing, attribution
experiments_attrprune/   # Collection, attribution, evaluation, and visualization CLIs
ablation/               # Additional isolated benchmark controls
run_*.sh                # Experiment launchers
docs/                   # Method and experiment guides
tests/                  # Unit and CPU stub integration tests
```

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q
bash -n run_mmlu_dev40_resampling.sh
python -m experiments_attrprune.resampled_mmlu --help
python -m experiments_attrprune.visualize_pruning_matrices --help
```

Tests exercise provenance, graph execution, correctness routing, observed-edge
averaging, static/sampled policies, batching, resume, and visualization. They do
not launch a full benchmark. Local datasets, model weights, logs, and generated
experiment results are excluded from Git.

## Relationship to AgentPrune

AgentAttrPrune adopts the multi-agent communication-pruning setting from
[AgentPrune](https://openreview.net/forum?id=LkzuPorQ5L). Its main method derives
round-specific policies from sentence attribution. The optional sampled-DAG
ablation borrows graph-sampling mechanics but uses attribution-derived logits
without AgentPrune's training objective.

```bibtex
@inproceedings{zhang2025cut,
  title={Cut the Crap: An Economical Communication Pipeline for {LLM}-based Multi-Agent Systems},
  author={Guibin Zhang and Yanwei Yue and Zhixun Li and Sukwon Yun and Guancheng Wan and Kun Wang and Dawei Cheng and Jeffrey Xu Yu and Tianlong Chen},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=LkzuPorQ5L}
}
```
