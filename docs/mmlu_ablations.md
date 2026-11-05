# MMLU ablation protocols and visualization

These experiments separate calibration order, candidate pruning, inference-time
edge sampling, and calibration-question selection. They are different
interventions; a pruning percentage alone does not identify the experiment.

Run commands from the repository root after installing the package. Choose
available GPUs explicitly. To use the active Python environment instead of the
original server's interpreter defaults:

```bash
export CONDA_ENV=''
export PYTHON_BIN="$(command -v python)"
```

The ablations inherit model paths, dataset identity, prompts, reviewer aliases,
and decoding from an existing reference run. The examples use the default
Qwen3-8B run names. A new reference must be calibrated on the current machine;
the repository does not distribute model weights, dataset snapshots, or traces.

## Shared definitions

The reference uses 40 MMLU dev questions, five workers, two rounds, temperature
0, 2,048 generated tokens, and thinking disabled. The canonical worker identity
order is Knowledgeable Expert, Critic, Mathematician, Psychologist, Historian.
The validation split contains 1,531 questions in the frozen dataset snapshot.

For any executable dense worker order, each round has 10 forward spatial edges.
Round 0 has no temporal edges; Round 1 has all 25 previous-round-to-current-round
temporal edges, including self-temporal links. The reviewer's five inputs are
outside the worker pruning budget. A two-round dense trace contains 11 calls.

Correctness-aware attribution includes **all** calibration questions and nodes.
Correct workers use the existing output-recovery search and early stopping.
Incorrect workers use full gold-prefix insertion, the earliest global peak, and
verification without gold in the prompt. Local correction never changes the
recorded trace or the downstream inputs from that trace. The reviewer retains
ordinary attribution and stays outside the worker matrices.

The primary metric is `token_ratio`, aggregated independently for each round,
edge type, source, and target:

```text
conditional_mean = importance_sum / observed_edge_count
```

An absent edge does not enter the denominator. An observed edge whose local
correction fails contributes zero and remains in the denominator. Runtime
failures, incomplete attribution, and truncated calibration outputs do not
become zero-importance observations. Frozen configurations, trace hashes, and
node-completion records must validate before a complete matrix is published.

## Experiment map

| Protocol | Calibration | Inference-time behavior |
|---|---|---|
| Original fixed order | Same canonical order, 40 traces | Global static pruning of 45 candidates |
| Original fixed order, uniform allocation | Reuse the same seed-0 fixed-order matrix | Independently prune each round/edge-type matrix; static execution |
| Random-order dense | Independent order for each question/repeat/round | Attribution experiment; no validation or pruning |
| Sampled global/uniform | Completed random-order dev40 matrix | Hard prune 65 candidates, then sigmoid/Bernoulli sampling with spatial cycle rejection |
| Row-minus-column: direct | Project the existing random-order matrix into a frozen order | Global static pruning of 45 forward candidates |
| Row-minus-column: recalibrate | Re-run the same dev40 under that frozen order and re-attribute | Global static pruning of 45 forward candidates |
| Empty-final-round ablation | Reuse the two reordered 80% policies | Skip Round 1 only when both of its edge types are empty |
| Independent dev40 resamples | Two new 40-question cohorts, original canonical order | Six global static rates; empty-final-round rule enabled |
| Expert-only baseline | Inherit an existing run configuration | One first-worker Round 0 answer; no communication or reviewer |

Every requested pruning rate starts independently from the unpruned source
matrix. Static policies execute every retained edge on every question. Only the
explicit sampled protocol applies Bernoulli draws.

## Fixed-order global versus uniform allocation

```bash
CUDA_VISIBLE_DEVICES=6,7 ./run_mmlu_fixed_order_uniform.sh
```

This comparison reuses `dev40_correctness_aware`, the source of the original
77.34% global-80 result, without collecting or attributing new samples. It keeps
the same fixed worker order, two full rounds, model, prompts, reviewer aliases,
seed, 1,531 validation questions, and evaluation batch size 4. Only pruning
allocation changes from `global` to `per_round_edge_type` (uniform).

| Pruning percentage | Round-0 spatial | Round-1 spatial | Round-1 temporal | Total retained |
|---|---:|---:|---:|---:|
| 50% | 5 | 5 | 13 | 23 |
| 60% | 4 | 4 | 10 | 18 |
| 80% | 2 | 2 | 5 | 9 |

All retained edges are used for every question. No sigmoid/Bernoulli sampling,
reordering, or empty-final-round shortcut is introduced. The launcher runs the
three rates sequentially with resume enabled and keeps original global artifacts
unchanged. Uniform policies use
`attribution/static_pruning/token_ratio_conditional_mean/`; their evaluation
names are `static_token_ratio_conditional_mean_prune_<rate>pct_val`, following
the existing naming convention without the `_global` suffix.

## Random order: 400 rollouts and the 40-rollout subset

Each `(question, repeat, round)` independently draws a uniform worker permutation.
Repeated permutations are allowed. Spatial messages follow the current execution
prefix; temporal and reviewer inputs retain canonical identity order. Stored
matrix coordinates always refer to identities, not execution positions.

The original budget is 40 questions times 10 repeats. Its default destination is
`dev40_random_order10_correctness_aware`:

```bash
DRY_RUN=1 ./run_mmlu_dev40_random_order.sh
CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_dev40_random_order.sh calibrate
```

Stages are `collect`, `attribute`, `aggregate`, `visualize`, and `calibrate`.
The full pipeline ends with attribution plots; it does not evaluate validation
questions. The 40×10 budget was motivated by a local AgentPrune GSM8K run, not by
an identical MMLU training configuration or graph distribution.

The reduced protocol uses exactly `repeat_00` for each of the same 40 questions.
It selects repeats independently of correctness, copies immutable existing
traces, and reuses only compatible attribution caches:

```bash
# Requires the 40 repeat_00 source traces from the frozen 400-rollout run.
./run_mmlu_dev40_random_order_once.sh prepare
CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_dev40_random_order_once.sh calibrate
```

Its default destination is `dev40_random_order1_correctness_aware`. It does not
generate new traces. The source writer must be stopped before the subset's GPU
stages; the launcher checks the source lock. The original 400-rollout files remain
intact. For a new 40-rollout experiment without a 400-rollout source, use the
underlying runner and a separate output directory:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m experiments_attrprune.random_order_mmlu \
  --stage calibrate --repeats 1 --order-seed 0 --attr-batch-size 4 --resume \
  --reference-run results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware \
  --run-dir results/AgentAttrPrune/mmlu/qwen3_8b/dev40_random_order1_fresh
```

Across `N` complete rollouts, each unordered worker pair's two spatial directions
must have counts summing to `N` in each round. Every Round 1 temporal cell has
count `N`. Unobserved directions remain missing; an observed mean may be zero.

## Hard pruning followed by per-inference sampling

This protocol reads the completed `dev40_random_order1_correctness_aware` result.
It retains separate Round 0 spatial, Round 1 spatial, and Round 1 temporal
matrices: 20 + 20 + 25 = **65** candidate slots.

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_random_order_sampling.sh global 50 60
CUDA_VISIBLE_DEVICES=2,3 ./run_mmlu_random_order_sampling.sh uniform 50 60

# Inspect or prepare without GPU inference:
DRY_RUN=1 ./run_mmlu_random_order_sampling.sh global 50 60
PREPARE_ONLY=1 ./run_mmlu_random_order_sampling.sh uniform 50 60
```

`global` ranks all 65 slots together. `uniform` prunes the same requested
percentage separately from each of the three active round/type matrices; it
does not enforce equal degrees for individual workers. Candidate removal uses
the sampler's float32 `torch.round` convention.

The attribution value is used as a **logit**, so an eligible retained edge is
sampled with `p = sigmoid(importance)`, not `p = importance`. In particular,
importance zero maps to probability 0.5 before the hard mask. Spatial candidates
are visited in canonical source-major order, cycle eligibility is checked
before the Bernoulli draw, and the resulting DAG executes in FIFO topological
order. Temporal candidates do not need cycle rejection.

The sampling and cycle rules follow the local AgentPrune implementation, but
these are untrained attribution logits and separate per-round matrices.
AgentPrune's learned shared matrices and training procedure are not reproduced.
Actual executed edges can be fewer than the hard mask retains. Each trace saves
the sampled masks, execution orders, and seeds.

Results are under
`<source>/evaluation/agentprune_sampling/per_round_sigmoid/<scope>/prune_<rate>pct/`.
`RUN_DIR`, `OUTPUT_ROOT`, `VAL_LIMIT`, and `SEED` are documented launcher overrides.

## Freeze an order using row sum minus column sum

For each round independently, compute

```text
score[i] = sum_j W[i,j] - sum_j W[j,i]
```

using the raw spatial conditional-mean importance matrix. Sort in descending
score order, resolving exact ties by canonical identity order. The order is
frozen before either branch runs. For the current source matrix, Round 0 is
Hist. → Math. → Psych. → Knowl. → Critic; Round 1 is Math. → Hist. → Psych. →
Knowl. → Critic. Other calibrated source matrices may produce other orders.

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_reordered_pruning.sh direct
CUDA_VISIBLE_DEVICES=2,3 ./run_mmlu_reordered_pruning.sh recalibrate
```

The `direct` branch keeps only forward spatial edges under the new order and
preserves their existing sums, observation counts, and means. It is a projection
of an existing attribution matrix, not a new 40-trace calibration. Temporal
values are unchanged.

The `recalibrate` branch runs the exact same 40 questions once under the frozen
orders and redoes correctness-aware attribution. It requires 40 traces and 440
node records; every allowed edge has 40 observations. It never reorders a second
time after attribution.

Both branches evaluate **80%, 60%, 50%**, retaining **9, 18, 23** of the 45 worker
candidates, respectively. No sigmoid or edge sampling is used. Generation is
serial, attribution batch size is 4, and validation batch size is 4. Default
destinations are `dev40_rowminuscol_direct_correctness_aware` and
`dev40_rowminuscol_recalibrated_correctness_aware`. Stages can also be run
separately; `direct` has no collection or attribution stage.

The original reordered evaluation executes both rounds even if Round 1 has no
messages. The explicit 80% ablation changes this behavior:

```bash
./run_mmlu_reordered_skip_empty80.sh both prepare
CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_reordered_skip_empty80.sh both
```

Only when **both** Round 1 spatial and temporal masks are empty, its five worker
calls are skipped and the reviewer reads the five Round 0 outputs. A skipped
trace has six calls instead of eleven. New results use
`evaluation/global_prune_80pct_val_skip_empty_round1`; original results remain
under `evaluation/global_prune_80pct_val`. `both` can be replaced with `direct`
or `recalibrate`.

## Independent calibration resamples

```bash
STAGE=prepare ./run_mmlu_dev40_resampling.sh 1 2
CUDA_VISIBLE_DEVICES=0,1 ./run_mmlu_dev40_resampling.sh 1 2
```

Each seed independently selects 40 questions without replacement from the 285
dev questions; overlap between cohorts is allowed. Seeds 1 and 2 select the
calibration questions only. Generation/evaluation seed 0, original fixed order,
reviewer aliases, attribution settings, and all 1,531 validation questions remain
unchanged. The reference cohort is reused read-only.

Each new cohort completes calibration and **25%, 50%, 60%, 70%, 80%, 90%** static
global evaluations before the next cohort starts. The retained budgets are
**34, 23, 18, 14, 9, 5**, following the existing static-pruning rounding rule.
The empty-final-round rule applies to every rate. Outputs are isolated in
`dev40_resample_seed{1,2}_correctness_aware`, with final `sweep_summary.json` and
`sweep_summary.csv`. Repeating the command resumes verified checkpoints. This
launcher does not draw accuracy curves. See [resampling details](resampled_mmlu.md).

`run_mmlu_fixed_order_25_70.sh` is a historical supplement for the original
reference snapshot, with its fingerprint and edge counts pinned. It preserves
the older four-rate results. For a new reference, use the general correctness-
aware `prepare`/`eval` interface rather than that snapshot-specific script.

## Expert-only baseline

The standalone baseline uses the first configured MMLU worker's original Round 0
prompt, with no incoming messages or reviewer. It inherits model, dataset, and
decoding settings from the selected run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m ablation.evaluate_expert_only_mmlu \
  --run-dir results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware \
  --split val --resume
```

It writes separate `evaluation/expert_only_val` checkpoints and predictions.
This is a single-worker baseline, rather than a zero-edge multi-worker policy.

## Figures and exports

All matrix figures use **row = source, column = target**. Display labels are
shortened to Knowl., Critic, Math., Psych., and Hist.; JSON/CSV keep canonical
identities. Round 0 temporal is absent, spatial self-links are forbidden, and
self-temporal edges are valid. Unobserved cells, observed zeros, and hard-pruned
cells have different meanings.

Random-order importance, observation-count, pooled-spatial, and reference
comparison figures can be rebuilt on CPU:

```bash
./run_mmlu_dev40_random_order_once.sh visualize
```

Pooled spatial means are `sum across rounds / count across rounds`, not an
unweighted average of round means. The fixed-order comparison is an explicit
reference panel; it is not part of the random-order run. Titles report the
actual budgets, and importance panels share a 0–1 color scale. PNG, SVG, per-edge
CSV, pooled JSON, and a visualization manifest are saved under
`<run>/attribution/visualizations/`.

For frozen static policies, the compact renderer overlays pruning masks and uses
the same short names for MMLU and GSM8K. `--intensity-only` hides the mask overlay,
and `--color-max` fixes a common color scale across different source matrices:

```bash
python -m experiments_attrprune.visualize_pruning_matrices \
  results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware/attribution/static_pruning/token_ratio_conditional_mean_global/policies/prune_90pct.json \
  --output-dir figures/mmlu_fixed90 --formats png svg --color-max 1
```

The reordered renderer displays spatial axes in that round's order. Round 1
temporal rows use Round 0 order, while columns use Round 1 order. Canonical data
coordinates stay unchanged:

```bash
./run_mmlu_reordered_pruning.sh direct visualize
./run_mmlu_reordered_pruning.sh recalibrate visualize
```

To display sigmoid probabilities before pruning, both pruning scopes, and their
side-by-side differences using already prepared sampled policies:

```bash
python -m experiments_attrprune.visualize_sampled_mmlu \
  --evaluation-root results/AgentAttrPrune/mmlu/qwen3_8b/dev40_random_order1_correctness_aware/evaluation/agentprune_sampling/per_round_sigmoid \
  --output-dir figures/mmlu_sampled_sigmoid
```

This CPU command does not sample graphs, require completed validation, or load
a model. It verifies the frozen policy/config hashes and shared source matrices,
then writes PNG/SVG, per-edge CSV, numerical JSON, and a provenance manifest.
Orange outlines mark edges retained only by one scope. Colors show the sigmoid
threshold for an eligible retained edge; spatial cycle rejection means that
this is **not** its marginal observed inclusion frequency. The common probability
color range is 0.5–0.75, which covers sigmoid values for importance in [0,1].
Use `--scopes global` or `--percentages 50` to render a subset.

Per-node sentence maps and insertion curves come from cached `event_*.json`
attribution records. They are distinct from edge-matrix plots and can be rebuilt
without loading a model using `experiments_attrprune.attribute_edges --render-only`.
See the [attribution specification](attribution_edge_pruning.md) for the event
schema, early-stop curves, and correctness-rescue diagnostics.
