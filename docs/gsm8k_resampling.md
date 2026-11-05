# GSM8K: independent train40 calibration cohorts

This experiment adds two independently sampled calibration cohorts to the
original fixed-order GSM8K train40 run. It changes calibration question
selection while preserving the reference model, worker roles and order,
reviewer aliases, decoding, attribution settings, and test questions.

## Run

From the repository root, select an available GPU pair. `6,7` below is an
example; do not reuse devices occupied by another experiment.

```bash
CUDA_VISIBLE_DEVICES=6,7 ./run_gsm8k_train40_resampling.sh 1 2
```

The launcher runs seed 1, then seed 2. For each cohort it performs Dense
collection, correctness-aware attribution, aggregation, policy preparation,
and **25%, 50%, 60%, 70%, 80%, 90% global static pruning evaluation** in that
order, followed by a result summary. Each GPU stage and pruning rate runs in a
separate process. The launcher saves results and does not draw accuracy curves.

Set the interpreter explicitly when using a different installation:

```bash
export PYTHON_BIN="$(command -v python)"
CUDA_VISIBLE_DEVICES=6,7 ./run_gsm8k_train40_resampling.sh 1 2
```

The reference is
`results/AgentAttrPrune/gsm8k/qwen3_8b/train40_correctness_aware` by default.
Set `REFERENCE_RUN` for another compatible completed reference. Model and data
paths are inherited from that run's frozen configuration.

## Sampling and fixed settings

Each selection seed independently draws 40 distinct questions from all 7,473
training questions after the reference's seed-888 dataset shuffle. The sampling
rule is `sorted(random.Random(seed).sample(range(dataset_size), 40))`. Questions
cannot repeat within a cohort; overlap between cohorts is allowed. The selected
indices, complete question records, and content hashes are frozen before GPU
work starts and are reused on resume.

Selection seeds **1 and 2 are not generation seeds**. The original generation
seed remains 0. The original reference cohort is the first 40 shuffled training
records and is read only.

| Setting | Value |
|---|---|
| Worker order, both rounds | Math Solver → Mathematical Analyst → Programming Expert → Inspector |
| Worker rounds | 2 |
| Model and decoding | Reference Qwen3-8B; temperature 0; 2,048-token ceiling; thinking off |
| Calibration generation | Serial; one complete Dense inference per selected question |
| Attribution | Exact reference `correctness-aware` configuration; all samples and nodes; batch 4 |
| Evaluation | The same 1,319 test questions; batch 4; saved traces |
| Worker candidate edges | 6 spatial per round; 16 Round 1 temporal; 28 total |
| Reviewer inputs | All four outputs of the last executed worker round; outside the pruning budget |

Every complete calibration has 40 traces and 360 node attribution records.
Each allowed edge has 40 observations. Incorrect-node gold-prefix verification
does not modify the original trace or downstream messages. Failed verification
contributes observed zero importance; runtime failure, truncation, or incomplete
node attribution prevents a complete aggregate from being published.

## Static policies and empty Round 1

All 28 worker candidates are ranked together by `token_ratio / conditional_mean`.
Each percentage starts from the same unpruned aggregate. The policies retain
**21, 14, 11, 8, 6, and 3** worker edges at 25%, 50%, 60%, 70%, 80%, and 90%,
respectively, under the existing rounding convention. Retained edges execute on
every question; there is no sigmoid or Bernoulli sampling.

The resampling evaluator skips Round 1 only if **both** its spatial and temporal
masks are empty. In that case the reviewer receives all four Round 0 outputs,
and inference uses five calls instead of nine. Otherwise both worker rounds
execute. The original fixed-order GSM8K launcher remains unchanged.

For the current reference matrix, all six derived policies retain at least one
Round 1 edge. Thus the conditional skip rule would preserve its execution at
those rates. This matrix check is not a claim that all six reference evaluations
have completed; compare only complete saved evaluation results.

## Resume and outputs

Run the same command to resume. Completed trace samples and compatible node
attributions are reused after validation. To run only the second cohort:

```bash
CUDA_VISIBLE_DEVICES=6,7 ./run_gsm8k_train40_resampling.sh 2
```

The independent output directories are:

```text
results/AgentAttrPrune/gsm8k/qwen3_8b/
  train40_resample_seed1_correctness_aware/
  train40_resample_seed2_correctness_aware/
```

Each contains frozen configuration/question/order plans, calibration traces,
node attributions, policies, and six evaluation directories named
`evaluation/static_token_ratio_conditional_mean_global_prune_<rate>pct_test`.
After all six evaluations complete, `sweep_summary.json` and
`sweep_summary.csv` record accuracy, token use, runtime, retained edges, and the
reviewer source round. Logs are isolated under
`logs/gsm8k_train40_resampling/seed_1/` and `seed_2/`.

Changed source files, altered question selections, duplicate or missing samples,
incomplete attributions, and incompatible policies cannot be silently reused as
complete results. Original calibration and evaluation artifacts stay intact.

## CPU preparation and individual stages

```bash
STAGE=prepare ./run_gsm8k_train40_resampling.sh 1 2
DRY_RUN=1 ./run_gsm8k_train40_resampling.sh 1 2
```

`STAGE` accepts `all` (default), `prepare`, `collect`, `attribute`, `aggregate`,
`policies`, `eval`, and `summarize`. The `eval` stage runs all six requested rates
for each selected cohort. Command dry-run does not load models or create run
files. The launcher supports `PYTHON_BIN`, `REFERENCE_RUN`, `RUNS_ROOT`, and
`LOG_ROOT` overrides.

For the original GSM8K experiment and matrix/insertion-curve commands, see
[GSM8K experiments](gsm8k_experiments.md).
