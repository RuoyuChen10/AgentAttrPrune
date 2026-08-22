# AgentAttrPrune

AgentAttrPrune is a compact, standalone repository for attribution-based
communication pruning in multi-agent systems. It records dense multi-agent
traces, identifies the incoming agent-output sentences that are sufficient to
recover each receiving node's original output score, and aggregates those
sentence selections into round-specific edge-importance matrices and
deterministic pruning policies.

The current release focuses on local Hugging Face/Qwen3 inference,
sentence-level edge attribution, and three text benchmarks: MMLU, GSM8K, and
HumanEval. It does not include AgentPrune's trainable edge optimizer, hosted
model APIs, visual tasks, tool use, or AutoGen.

See [the attribution and edge-pruning specification](docs/attribution_edge_pruning.md)
for the full algorithm, matrix orientation, schemas, and interpretation
boundaries.

## Repository layout

```text
AgentAttrPrune/
├── AgentAttrPrune/             # Graph, HF backend, tracing, attribution, policy
├── experiments_attrprune/      # Benchmark calibration, evaluation, and shell entrypoints
├── docs/                       # Algorithm and output-schema documentation
├── baselines/                  # Contract for future external baselines
├── data/                       # Local datasets; contents are ignored by Git
├── results/AgentAttrPrune/     # Generated experiment artifacts; ignored by Git
├── tests/
├── run_full_*.sh              # Complete MMLU, GSM8K, and HumanEval launchers
└── pyproject.toml
```

Reusable implementation code belongs in the `AgentAttrPrune` package.
`experiments_attrprune` only handles argument parsing, dataset iteration, run
configuration, and artifact paths.

## Current implementation

### Dense graph and trace collection

The benchmark configurations are intentionally task-specific. Their default
worker counts and role order match the corresponding AgentPrune experiment and
prompt-set defaults:

| Benchmark | Calibration / evaluation | Workers | Roles in matrix order | Events at 2 rounds | Worker candidate edges |
|---|---|---:|---|---:|---:|
| MMLU | `dev` / `val` | 5 | Knowlegable Expert; Critic; Mathematician; Psychologist; Historian | 11 | 45 |
| GSM8K | `train` / `test` | 4 | Math Solver; Mathematical Analyst; Programming Expert; Inspector | 9 | 28 |
| HumanEval | fixed `dev` / disjoint `test` | 5 | Project Manager; Algorithm Designer; Programming Expert; Test Analyst; Bug Fixer | 11 | 45 |

They are therefore not the same multi-agent system in worker count or roles.
They do share one graph executor, two-round topology, trace schema, attribution
search, matrix definition, pruning logic, and evaluation path. The original
AgentPrune entrypoints name the worker implementations `AnalyzeAgent`,
`MathSolver`, and `CodeWriting`, respectively. AgentAttrPrune standardizes their
execution through `AnalyzeAgent` while preserving the benchmark-specific roles
and prompts, followed by one task-aware dense reviewer. A policy is bound to its
benchmark, worker order, roles, dataset snapshot, and run fingerprint; it cannot
be applied to another benchmark accidentally.

- Within each round, spatial communication is an executable ordered DAG:
  earlier workers send to later workers.
- Round 0 has no temporal edges.
- From round 1 onward, temporal communication is a complete previous-round to
  current-round `N x N` topology, including self-temporal edges.
- A final dense reviewer reads all workers from the last round.

For five workers and two rounds, every sample must contain:

```text
5 workers x 2 rounds + 1 reviewer = 11 generation events
```

`GenerationTraceRecorder` stores the exact input messages, rendered prompt,
prompt token IDs, output text, generated token IDs, incoming-edge provenance,
and generation termination metadata for every event. Outputs that reach
`max_new_tokens` without EOS are marked as truncated. The diagnostic trace is
retained, but truncated samples are rejected from formal attribution and policy
calibration.

### Attribution scope

Every worker-round node and the final reviewer receive a standalone attribution
record. Only sentences inside incoming agent outputs or temporal history are
removable. The following content always remains visible:

- system prompts;
- the benchmark task (including MMLU answer options when present);
- message headers and source labels;
- separators and chat-template tokens.

A node with no incoming agent/history content still receives complete JSON,
HTML, and SVG artifacts with status `no_attributable_input`.

### Teacher-forced target score

The default attribution target is the receiving node's complete recorded output.
The scorer teacher-forces the original generated token IDs under every
perturbed prompt and computes:

```text
score_type      = logit
score_reduction = sum
target_mode     = all
```

The baseline prompt contains no incoming sentences. The full prompt contains
all incoming sentences. Hidden sentence tokens are actually removed and the
remaining prompt is compacted; they are not replaced by pad or EOS tokens.

### Exact greedy sentence search and early stopping

At every search step, the implementation evaluates **all remaining candidate
sentences** and adds the sentence whose resulting score is closest to the full
input score. Candidate variants are split into inference batches only to control
memory usage.

The older efficient-attribution controls `search_scope`, `pending_samples`, and
`update_step` are intentionally **not implemented** in the current search.
`ATTR_BATCH_SIZE` is a compute batch size, not a search window.

Let:

```text
s0 = score with no incoming sentences
s* = score with all incoming sentences
sk = score after greedy step k
```

Recovery is:

```text
recovery_k = clip(1 - abs(sk - s*) / max(abs(s* - s0), 1e-12), 0, 1)
```

Search stops when recovery reaches `0.95`, or when the absolute score tolerance
is satisfied. `patience=1` is the current default. If a larger patience value is
used directly through the Python CLI, consecutive qualifying steps validate
the stop, but the returned subset remains the earliest qualifying prefix.

The current search hyperparameters are:

| Parameter | Default | Meaning |
|---|---:|---|
| `target_mode` | `all` | Explain all recorded output tokens |
| `target_token_limit` | unset | Do not truncate the attribution target |
| `score_type` | `logit` | Use the raw target-token logits |
| `score_reduction` | `sum` | Sum over target output tokens |
| `target_recovery` | `0.95` | Early-stop recovery threshold |
| `absolute_atol` | `1e-4` | Absolute full-score tolerance |
| `relative_rtol` | `1e-3` | Relative full-score tolerance |
| `patience` | `1` | Consecutive qualifying steps required |
| `batch_size` | `4` | Candidate variants per model batch |
| `max_selected_sentences` | unset | Search until recovery or exhaustion |
| `min_selected_sentences` | `2` | Substantive sentences required by binary metric |
| `min_sentence_words` | `4` | Minimum substantive sentence length |
| `saliency_floor` | `0.4` | Display-only saliency of the final selected sentence |

### Sentence saliency

The selected early-stop prefix receives a display saliency without additional
model forwards. For consecutive selected scores:

```text
d_k = abs(score_k - score_(k-1))
C_k = sum(d_i for i <= k)
```

The selected prefix is normalized so the first greedy sentence has saliency
`1.0` and the last selected sentence has saliency `0.4`. Unselected sentences
have `null` saliency and remain gray in the visualization. This saliency only
explains differences within the selected prefix; it does not replace edge
importance.

### Edge importance and matrices

The primary edge metric follows the method definition:

```text
token_ratio =
    selected source-output tokenizer tokens
    / all visible source-output tokenizer tokens
```

Token ownership is source-local and deduplicated across sentence boundaries.
System, task, header, template, and special tokens are excluded from both the
numerator and denominator. `binary` and `word_ratio` remain available as
ablation metrics.

Every sample stores separate spatial and temporal matrices with orientation:

```text
[round][source][target]
```

Reviewer attribution is stored separately and never inserted into the worker
`N x N` matrices. Incorrect samples retain all per-node artifacts and sample
matrices, but only `correct_verified=true` samples contribute to aggregate
matrices and pruning policies.

### Per-node and per-sample visualizations

Each event produces:

- `event_*.json`: complete input, output, incoming text, candidates, curve,
  edge scores, generation metadata, and attribution metadata;
- `event_*.html`: readable complete report;
- `event_*.svg`: sentence-level attribution map with orange saliency gradient;
- `event_*.insertion_curve.svg`: observed insertion score versus retained
  sentence percentage, with the early-stop sentence count, sentence ratio,
  token ratio, recovery, target line, and separate full-input endpoint.

Each sample also produces:

- `node_attribution_index.json`;
- `sample_edge_matrices.json`;
- `sample_token_ratio_matrices.png`.

Correct-only aggregation produces aggregate JSON, round-specific heatmaps, and
token-ratio/binary/word-ratio policies.

## Code map

| File | Responsibility |
|---|---|
| `AgentAttrPrune/graph/multi_agent.py` | Shared dense/pruned multi-agent execution loop |
| `AgentAttrPrune/llm/hf_chat.py` | Local Hugging Face generation and exact termination metadata |
| `AgentAttrPrune/tracing/recorder.py` | Schema-v1 traces and incoming provenance |
| `AgentAttrPrune/attribution/scoring.py` | Batched teacher-forced logit/log-probability/probability scoring |
| `AgentAttrPrune/attribution/edge.py` | Incoming-only sentence masks, exact greedy search, edge metrics, matrices |
| `AgentAttrPrune/attribution/visualization.py` | Sentence SVGs, insertion curves, HTML reports, matrix heatmaps |
| `AgentAttrPrune/attribution/calibrator.py` | Per-event attribution, caching, aggregation, policy publication |
| `AgentAttrPrune/pruning/policy.py` | Validation and replay of round-specific spatial/temporal masks |
| `experiments_attrprune/common.py` | Run paths, dataset identity, worker order, configuration fingerprint |
| `experiments_attrprune/*.py` | Thin experiment CLIs |

## Invariants

- Every matrix uses `[round][source][target]`.
- Spatial and temporal matrices are never merged.
- Dense and pruned runs share the same workers, prompts, reviewer, decoding
  parameters, and execution loop.
- Reviewer edges remain dense and do not enter worker matrices.
- Policies are calibrated only from baseline-correct calibration traces.
- Val/test samples never contribute to attribution aggregation.
- Formal traces require exact prompt and generated token IDs.
- Sample matrices, aggregate matrices, policies, and evaluations are bound to
  the same `run_config.json` fingerprint.
- A failed correct-sample attribution prevents publication of a partial policy.
- Generated artifacts stay under `results/AgentAttrPrune`.

## Installation

Requirements:

- Python 3.10 or newer;
- PyTorch with CUDA support;
- Transformers 4.51 or newer for Qwen3;
- Accelerate for `device_map=auto`;
- local Qwen3 model weights.

```bash
conda create -n agentattrprune python=3.10 -y
conda activate agentattrprune
pip install -e .
```

Install development dependencies with:

```bash
pip install -e '.[test]'
```

The repository defaults to:

```text
/egr/research-optml/ruoyu.chen/Models/Qwen3-8B
```

Override it with `MODEL_PATH`. See [data/README.md](data/README.md) for dataset
placement and download commands.

## Running MMLU

Run all commands from the repository root.

### Dry run

Print resolved commands without loading a model:

```bash
DRY_RUN=1 ./experiments_attrprune/run_mmlu.sh all
```

### One-sample attribution

Generate all worker-round and reviewer artifacts for one dev sample:

```bash
RUN_NAME=mmlu/qwen3_8b/sample_0 \
DEV_START_INDEX=0 \
MAX_NEW_TOKENS=512 \
./experiments_attrprune/run_mmlu.sh one-sample
```

### Full dev calibration only

`calibrate` runs MMLU dev trace collection, attributes every node in every
selected sample, and aggregates correct-only matrices. It does not run val or
test evaluation.

Example for eight visible A6000 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CONDA_ENV=agentprune \
MMLU_DATA_DIR=/path/to/MMLU/data \
RUN_NAME=mmlu/qwen3_8b/full_dev_a6000x8_tok2048 \
DEV_START_INDEX=0 \
DEV_LIMIT=285 \
MAX_NEW_TOKENS=2048 \
ATTR_BATCH_SIZE=4 \
TARGET_RECOVERY=0.95 \
SALIENCY_FLOOR=0.4 \
./run_full_mmlu.sh calibrate
```

`MAX_NEW_TOKENS` is only a ceiling. Generation stops normally at EOS. A node
that reaches the ceiling is marked as truncated, and its sample is rejected
from formal calibration. Leaving `TARGET_TOKEN_LIMIT` unset explains all tokens
that were actually generated.

`device_map=auto` provides model sharding, not data parallelism. A Qwen3-8B
model may fit on one 48 GB A6000, so making eight GPUs visible does not
guarantee that all eight devices will be saturated.

### Full dev-to-val protocol

Run dev calibration, dense/binary/token-ratio val evaluation, and comparison:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./experiments_attrprune/run_mmlu.sh all
```

### Individual stages

```bash
# Dense MMLU dev traces
./experiments_attrprune/run_mmlu.sh traces

# Per-node attribution, sample matrices, correct-only aggregate, policies
./experiments_attrprune/run_mmlu.sh attribution

# Rebuild aggregate matrices and policies from cached events
./experiments_attrprune/run_mmlu.sh aggregate

# Rebuild sentence SVGs, insertion curves, and HTML without loading the model
./experiments_attrprune/run_mmlu.sh render

# Frozen-policy validation
./experiments_attrprune/run_mmlu.sh eval-dense
./experiments_attrprune/run_mmlu.sh eval-binary
./experiments_attrprune/run_mmlu.sh eval-token-ratio
./experiments_attrprune/run_mmlu.sh eval-word-ratio

# Standard evaluation bundle and summary
./experiments_attrprune/run_mmlu.sh eval
./experiments_attrprune/run_mmlu.sh compare
```

Resume completed trace samples with:

```bash
RESUME=1 ./experiments_attrprune/run_mmlu.sh traces
```

Render cached event JSON again with a different display floor:

```bash
SALIENCY_FLOOR=0.4 ./experiments_attrprune/run_mmlu.sh render
```

Use a new `RUN_NAME` whenever the model, data snapshot, worker configuration,
round count, or generation settings change.

### Static MMLU pruning-rate sweep

After `aggregate_edge_matrices.json` has been generated, static policies can
be built and evaluated at several pruning rates without rerunning attribution.
The original allocation remains the default: it rounds and applies the
requested rate independently to each round and each edge type
(`spatial`/`temporal`).

The optional global allocation instead flattens every observed spatial and
temporal edge across all rounds, sorts them by importance, and applies one
pruning budget to the whole set:

```bash
PRUNING_SCOPE=global \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_static_mmlu_pruning_sweep.sh all 5 10 20 30 40 50
```

Generate only the global policy JSON files with:

```bash
PRUNING_SCOPE=global \
./run_static_mmlu_pruning_sweep.sh prepare 5 10 20 30 40 50
```

Omit `PRUNING_SCOPE` or set it to `per_round_edge_type` to reproduce the old
behavior. Global policies are written below
`attribution/static_pruning/token_ratio_conditional_mean_global/`, and their
evaluations use `_global`-suffixed names, so old policies, checkpoints, and
summaries are not overwritten.

## Running GSM8K and HumanEval

Download all missing files:

```bash
python -m AgentAttrPrune.datasets.download all --data-root data
```

The root launchers default to eight visible GPUs, Qwen3-8B, two rounds,
`MAX_NEW_TOKENS=2048`, and the complete local dataset selections. `calibrate`
uses only the calibration split and writes all per-node attribution SVG/HTML,
sample matrices, aggregate heatmaps, and policies:

```bash
# All 7,473 GSM8K train samples; no test evaluation
./run_full_gsm8k.sh calibrate

# All 80 examples in the fixed HumanEval calibration partition; no test evaluation
./run_full_humaneval.sh calibrate
```

Use `all` to append frozen-policy evaluation on GSM8K test or the disjoint
HumanEval test partition:

```bash
./run_full_gsm8k.sh all
./run_full_humaneval.sh all
```

The lightweight wrappers expose the same stages and environment controls:

```bash
# GSM8K train calibration -> test evaluation
DRY_RUN=1 ./experiments_attrprune/run_gsm8k.sh all
./experiments_attrprune/run_gsm8k.sh one-sample

# Fixed HumanEval dev calibration -> disjoint test evaluation
DRY_RUN=1 ./experiments_attrprune/run_humaneval.sh all
./experiments_attrprune/run_humaneval.sh one-sample
```

The shared entrypoint is also available directly:

```bash
BENCHMARK=gsm8k ./experiments_attrprune/run_benchmark.sh calibrate
BENCHMARK=humaneval ./experiments_attrprune/run_benchmark.sh eval-dense
```

Use `DATA_DIR`, `CALIBRATION_LIMIT`, `EVALUATION_LIMIT`,
`CALIBRATION_SPLIT`, and `EVALUATION_SPLIT` with the shared entrypoint.
HumanEval correctness runs generated Python in a limited subprocess; this is
not a security boundary for hostile code.

## Common environment variables

| Variable | Shell default | Meaning |
|---|---:|---|
| `MODEL_PATH` | local Qwen3-8B path | Generation and attribution model |
| `MMLU_DATA_DIR` | `data/MMLU` | MMLU root or directory containing `data/` |
| `GSM8K_DATA_DIR` | `data/GSM8K` | Directory containing `train.jsonl` and `test.jsonl` |
| `HUMANEVAL_DATA_DIR` | `data/HumanEval` | Directory containing `humaneval-py.jsonl` |
| `RESULTS_ROOT` | `results/AgentAttrPrune` | Generated-artifact root |
| `RUN_NAME` | derived from model/seed | Run subdirectory |
| `CALIBRATION_START_INDEX` | `0` | Generic runner's first calibration sample |
| `CALIBRATION_LIMIT` | benchmark-specific | Generic runner's calibration sample count |
| `EVALUATION_LIMIT` | benchmark-specific | Generic runner's frozen-policy evaluation count |
| `DEV_START_INDEX` | `0` | First shuffled dev sample |
| `DEV_LIMIT` | `285` | Dev calibration sample count |
| `VAL_LIMIT` | `1531` | Val evaluation sample count |
| `NUM_WORKERS` | `5` | Worker count |
| `NUM_ROUNDS` | `2` | Reasoning rounds |
| `TEMPERATURE` | `0` | Deterministic generation temperature |
| `MAX_NEW_TOKENS` | `2048` | Per-node generation ceiling; generation still stops at EOS |
| `ATTR_BATCH_SIZE` | `4` | Candidate inference batch size |
| `TARGET_TOKEN_LIMIT` | unset | Optional attribution-target prefix |
| `TARGET_RECOVERY` | `0.95` | Early-stop recovery threshold |
| `SALIENCY_FLOOR` | `0.4` | Display-only selected-prefix floor |
| `POLICY_PRUNING_RATE` | `0.25` | Per-round, per-edge-type pruning rate |
| `RESUME` | `0` | Reuse completed trace attempts and validated evaluation checkpoints |
| `OVERWRITE_ATTRIBUTION` | `0` | Recompute cached attribution events |
| `KEEP_GOING` | `0` | Collect failures but still reject a partial policy |
| `DRY_RUN` | `0` | Print commands without executing them |

All `run_full_*.sh` launchers default the generation ceiling to `2048` unless
`MAX_NEW_TOKENS` is set explicitly.

Show every stage and override with:

```bash
./experiments_attrprune/run_mmlu.sh --help
```

## Output layout

```text
results/AgentAttrPrune/<benchmark>/qwen3_8b/seed_<seed>/
├── run_config.json
├── calibration/
│   ├── attempts.jsonl
│   ├── correct_results.jsonl
│   ├── summary.json
│   └── traces/sample_XXXXX/trace.json
├── attribution/
│   ├── manifest.json
│   ├── aggregate_edge_matrices.json
│   ├── aggregate_token_ratio_matrices.png
│   ├── pruned_policy_token_ratio.json
│   ├── pruned_policy_binary.json
│   ├── pruned_policy_word_ratio.json
│   └── sample_XXXXX/
│       ├── node_attribution_index.json
│       ├── event_*.json
│       ├── event_*.svg
│       ├── event_*.insertion_curve.svg
│       ├── event_*.html
│       ├── sample_edge_matrices.json
│       └── sample_token_ratio_matrices.png
├── evaluation/
│   ├── dense_val/
│   │   ├── evaluation_state.json
│   │   ├── predictions.jsonl
│   │   └── evaluation_summary.json
│   ├── binary_val/
│   └── token_ratio_val/
├── comparison_summary.json
└── comparison_summary.csv
```

The run configuration records the model, benchmark path and content hashes, worker
order, rounds, decoding parameters, topology, seed, and fingerprint. Cached
events, policies, and evaluations are rejected when their fingerprints do not
match.

Generated data is ignored by Git. See
[results/AgentAttrPrune/README.md](results/AgentAttrPrune/README.md).

During evaluation, `predictions.jsonl` is atomically rewritten and synced after
each completed sample. A rerun with `--resume` validates
`evaluation_state.json`, skips all completed sample indices (including wrong
predictions), and evaluates only the missing indices. The final summary is not
written until the requested selection is complete; accuracy and token/call
totals are then recomputed from the full persisted record set. The evaluation
identity includes the run fingerprint, exact policy hash, dataset range, seed,
and trace mode, preventing accidental mixtures across configurations.

## Fair-comparison protocol

1. Learn policies only from baseline-correct calibration traces.
2. Evaluate every policy on the complete selected val split, including samples
   that the dense baseline answers incorrectly.
3. Reuse the same graph executor, worker order, roles, prompts, reviewer, and
   generation settings for dense and pruned modes.
4. Keep reviewer aggregation dense.
5. Require matching run fingerprints and dataset identities.
6. Report candidate/retained edges, effective edge retention, prompt/generated
   tokens, runtime, and accuracy.
7. Do not describe attribution importance as a calibrated causal probability.

Future baselines must follow [baselines/README.md](baselines/README.md).

## Tests

```bash
python -m pytest -q
bash -n experiments_attrprune/run_mmlu.sh
bash -n run_full_mmlu.sh
python -m experiments_attrprune.collect_correct --help
python -m experiments_attrprune.attribute_edges --help
python -m experiments_attrprune.evaluate_mmlu --help
python -m experiments_attrprune.compare --help
```

The unit and stub end-to-end tests do not launch the full 285-sample GPU
calibration.

## Relationship to AgentPrune

AgentAttrPrune adopts AgentPrune's fully connected multi-agent candidate
topology and pruning-rate comparison idea, but learns deterministic,
round-specific policies from sentence attribution.

If you use this repository, also cite the original
[AgentPrune paper](https://openreview.net/forum?id=LkzuPorQ5L):

```bibtex
@inproceedings{zhang2025cut,
  title={Cut the Crap: An Economical Communication Pipeline for {LLM}-based Multi-Agent Systems},
  author={Guibin Zhang and Yanwei Yue and Zhixun Li and Sukwon Yun and Guancheng Wan and Kun Wang and Dawei Cheng and Jeffrey Xu Yu and Tianlong Chen},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=LkzuPorQ5L}
}
```
