# Round-Specific Edge Pruning from Sentence Attribution

## 1. Objective

A receiving agent in a multi-agent system often observes complete outputs from
several upstream agents. In many cases, a small subset of those incoming
sentences is sufficient to reproduce the receiving node's original output
score. AgentAttrPrune finds such a subset for every generation event and turns
the selected content into communication-edge importance.

The current implementation has six goals:

1. preserve a complete, readable attribution bundle for every worker-round
   node and final reviewer in every selected MMLU dev sample;
2. remove only incoming agent-output or temporal-history sentences while
   keeping fixed prompt content intact;
3. recover the receiving node's complete recorded output score with an exact
   greedy sentence search and early stopping;
4. compute edge importance from the fraction of source-output tokenizer tokens
   selected by attribution;
5. store separate round-specific spatial and temporal matrices with orientation
   `[round][source][target]`;
6. learn policies from baseline-correct dev samples and evaluate the frozen
   policy on a complete, disjoint evaluation split.

Primary implementation modules:

- `AgentAttrPrune/attribution/`: sentence masks, teacher-forced scoring, greedy
  search, edge metrics, matrices, aggregation, and visualizations;
- `AgentAttrPrune/graph/`, `tracing/`, and `pruning/`: shared dense/pruned graph
  execution, exact trace capture, and round-policy replay;
- `experiments_attrprune/collect_correct.py`: dense MMLU dev traces;
- `experiments_attrprune/attribute_edges.py`: per-node attribution, sample
  matrices, correct-only aggregation, and policy publication;
- `experiments_attrprune/evaluate_mmlu.py`: complete dense/pruned evaluation.

## 2. Data model and notation

Let \(x\) be a recorded MMLU sample and \(e\) one receiving generation event.
The event stores:

- receiving agent \(b_e\) and target round \(r_e\);
- exact `input_messages` and, when available, `rendered_prompt`;
- zero or more incoming-edge records;
- original output token sequence
  \(y=(y_0,\ldots,y_{L-1})\);
- generation termination metadata.

Incoming record \(q\) defines an observed edge:

\[
q=(\tau_q,a_q,b_e,r_q^{src},r_e),
\]

where \(\tau_q\in\{\text{spatial},\text{temporal}\}\) and \(a_q\) is the source
agent. Spatial messages are generated within the current round. Temporal
messages carry a previous-round output into the current round.

All matrices use:

\[
M[r,a,b]=\text{importance of source }a\rightarrow\text{target }b.
\]

Spatial and temporal matrices are always stored as separate
\(R\times N\times N\) tensors.

## 3. Dense graph and trace requirements

The default graph contains five workers and two reasoning rounds:

- spatial candidates form an ordered same-round DAG;
- round 0 contains no temporal edges;
- later rounds contain a complete previous-to-current `N x N` temporal
  candidate set, including self-temporal communication;
- the final reviewer observes all final-round worker outputs.

For \(N\) workers and \(R\) rounds, a complete trace contains:

\[
NR+1
\]

generation events. The extra event is the final reviewer. The default
configuration therefore requires 11 events per sample.

Schema-v1 traces record exact prompt and generated token IDs. The HF backend
also records:

- `finish_reason`;
- `hit_max_new_tokens`;
- top-level `generation_complete`;
- `truncated_event_ids`.

If any event reaches the generation ceiling without EOS, the trace remains
available for diagnosis but the sample is rejected from formal attribution and
policy calibration. Older traces that lack explicit termination fields are
checked conservatively against their recorded `llm_call.max_tokens`.

## 4. Calibration split and correctness filtering

The formal protocol uses:

- MMLU `dev` for dense traces and policy calibration;
- MMLU `val` for complete evaluation after the policy is frozen;
- MMLU `test` only for optional final reporting.

Per-node attribution defaults to `sample_scope=all`, so incorrect dev samples
still receive complete diagnostic artifacts. Before aggregation, correctness is
verified again:

\[
\operatorname{normalize}(\hat y_x)
=\operatorname{normalize}(y_x^{gold}).
\]

Incorrect sample matrices are marked `correct_verified=false` and excluded by
`aggregate_sample_matrices`. Using `--sample-scope correct` skips their
diagnostic attribution but does not change the aggregation rule.

Calibrating on val-correct samples and then reporting results on the same val
split is leakage and is not supported by the formal protocol.

## 5. Event coverage

The default `--node-scope all` processes:

- every worker in every reasoning round;
- the final reviewer.

All-node mode validates the expected \(NR+1\) event count. A missing event makes
the sample incomplete. A node with no incoming agent/history text still emits
complete input/output artifacts with status `no_attributable_input` and empty
sentence and edge-score arrays.

Compatibility modes:

- `--node-scope workers`: omit the reviewer;
- `--node-scope multi-input`: process only events with at least
  `--min-incoming` incoming records.

`--min-incoming` defaults to 2 but only filters event selection in
`multi-input` mode. It does not suppress one-input nodes when
`--node-scope all` is used.

Reviewer attribution is a required diagnostic artifact in all-node mode, but
the reviewer is not a worker. Reviewer edges never enter worker `N x N`
matrices or pruning masks.

## 6. Incoming-only sentence scope

For event \(e\), the implementation first obtains the exact rendered prompt.
It prefers the trace's recorded `rendered_prompt`; otherwise it rerenders
`input_messages` with the current tokenizer chat template and disables Qwen3
thinking by default.

For every incoming source output:

1. locate user-message content spans in rendered-chat order;
2. find the latest feasible, ordered, non-overlapping exact matches for every
   incoming source output;
3. split only those source-output spans into sentences;
4. map sentence character spans to prompt-token offsets;
5. assign a tokenizer token crossing a sentence boundary to the sentence with
   the largest character overlap.

The exact ordered matching distinguishes repeated spatial/temporal text and
avoids attributing an identical phrase in the task or system prompt.

Let candidate sentence token sets be
\(V=\{v_1,\ldots,v_m\}\). Fixed visible tokens are:

\[
F=P\setminus\bigcup_{j=1}^{m}v_j.
\]

\(F\) contains:

- system messages;
- the MMLU question and answer options;
- source labels such as `Agent ... output is:`;
- separators;
- chat-template and generation-prompt tokens.

Fixed content is never ranked, masked, or credited to an edge.

## 7. Target output and teacher forcing

The default target configuration is:

```text
target_mode     = all
score_type      = logit
score_reduction = sum
```

`--target-token-limit N` optionally explains only the first \(N\) generated
tokens. Leaving it unset explains the complete recorded output.

`--target-mode decision` is a cheaper MMLU diagnostic mode that finds the final
standalone A/B/C/D token. Because teacher forcing fixes the entire reasoning
prefix preceding that token, decision-only attribution can underestimate the
effect of incoming messages. It is not the formal default.

Exact `generated_ids` from the trace are preferred. If they are unavailable,
the output is retokenized and the event records
`target.id_source=retokenized_output`. Formal calibration rejects a generation
model/tokenizer mismatch. When prompt token IDs are recorded, they are also
checked against the current tokenizer.

For a selected sentence set \(S\subseteq V\), the visible prompt is:

\[
P_S=F\cup\bigcup_{v_j\in S}v_j.
\]

Hidden sentence tokens are deleted and the remaining prompt is compacted. A
batch of different prompt lengths is left-padded with explicit attention masks
and position IDs.

For output positions \(T\), the original output is teacher-forced:

\[
z_t^{(S)}=f_\theta(P_S,y_{<t}),\qquad t\in T.
\]

The target-token raw logit, log-probability, and probability are:

\[
\ell_t^{(S)}=z_t^{(S)}[y_t],
\]

\[
q_t^{(S)}=\ell_t^{(S)}-\log\sum_c\exp z_t^{(S)}[c],
\]

\[
p_t^{(S)}=\exp q_t^{(S)}.
\]

The formal score is:

\[
s(S)=\sum_{t\in T}\ell_t^{(S)}.
\]

Regardless of the selected primary score, endpoint diagnostics store
`logit_sum`, `logit_mean`, `logprob_sum`, and `logprob_mean`.

When Qwen3 supports `logits_to_keep`, the scorer requests only the required
vocabulary-logit positions. Arbitrary non-contiguous targets under Accelerate
multi-GPU sharding fall back to full sequence logits to avoid device-index
errors. Models that do not support `logits_to_keep` also fall back safely.

## 8. Exact greedy search and early stopping

Compute the two endpoints:

\[
s_0=s(\varnothing),\qquad s_*=s(V),\qquad
\Delta=|s_*-s_0|.
\]

Numerical tolerance is:

\[
\epsilon=\epsilon_{abs}+\epsilon_{rel}|s_*|,
\]

with defaults:

```text
absolute_atol = 1e-4
relative_rtol = 1e-3
```

If \(\Delta\le\epsilon\), the fixed prompt already matches the full input. The
algorithm selects the empty set and returns
`fixed_prompt_already_matches_full`.

Otherwise, start from \(S_0=\varnothing\). At step \(k\), score **every
remaining sentence candidate** and choose the candidate closest to the full
input score:

\[
j_k=\arg\min_{j\notin S_{k-1}}
\left|s(S_{k-1}\cup\{v_j\})-s_*\right|,
\]

\[
S_k=S_{k-1}\cup\{v_{j_k}\}.
\]

Gap and recovery are:

\[
g_k=|s(S_k)-s_*|,
\]

\[
\rho_k=\operatorname{clip}\left(
1-\frac{g_k}{\max(\Delta,10^{-12})},0,1
\right).
\]

A step qualifies as close when:

\[
\rho_k\ge\rho_{target}
\quad\text{or}\quad
g_k\le\epsilon.
\]

The default recovery threshold is \(0.95\). `--patience P` requires \(P\)
consecutive qualifying selections. After validation, the returned set is still
the shortest prefix that first entered the close region. The current default is
`patience=1`.

Stop reasons:

- `fixed_prompt_already_matches_full`;
- `near_full_score`;
- `all_sentences_selected`;
- `max_selected_sentences`.

Hitting an explicit `max_selected_sentences` before recovery is incomplete and
excluded from calibration unless `--accept-incomplete` is explicitly enabled.

The stored curve contains the selected sentence, score, gap, recovery,
sentence fraction, and attributable-token fraction for every evaluated greedy
step. Edge scores always use the final returned
`selected_region_indices`.

### 8.1 No legacy search window

The current implementation does **not** use the approximate search mechanism
from `efficient-batch-inference-text.py`.

Specifically:

- there is no `search_scope` shortlist;
- there is no `pending_samples` opposite/deletion tail search;
- there is no periodic `update_step` reranking;
- every greedy step globally rescans all remaining candidates.

`batch_size=4` only splits candidate variants into model batches. It does not
reduce search scope. If an event has \(m\) sentences and stops after \(K\)
selections, the candidate-forward count is approximately:

\[
m+(m-1)+\cdots+(m-K+1).
\]

This exact search is more expensive than the legacy approximate search but
preserves the stated closest-to-full greedy objective.

## 9. Selected-prefix saliency

Sentence saliency is computed after search without additional model forwards.
For the final selected prefix:

\[
d_k=|score_k-score_{k-1}|,\qquad
C_k=\sum_{i=1}^{k}d_i,
\]

where `score_0` is the baseline score.

For \(K>1\):

\[
\operatorname{saliency}_k
=f+(1-f)\frac{C_K-C_k}{C_K-C_1},
\]

where \(f=0.4\) by default. Thus the first greedy selection is `1.0` and the
last selection is `0.4`. A single selected sentence receives `1.0`. If the
cumulative differences degenerate to zero, the implementation falls back to a
linear sequence from `1.0` to the floor.

Only the final selected prefix participates. Additional steps used only for
patience validation keep `null` saliency. Unselected candidates also have
`null` saliency.

`--saliency-floor` and `SALIENCY_FLOOR` affect JSON post-processing and display
only. They do not alter search, early stopping, edge metrics, matrices, or
policies.

## 10. Edge metrics

For edge \(q\), let \(V_q\) contain all its candidate sentences and
\(S_q=S\cap V_q\) contain selected sentences.

### 10.1 Binary ablation

A sentence is substantive when it contains at least \(w_{min}\) words:

\[
B_q=\mathbf{1}\left[
\sum_{v\in S_q}
\mathbf{1}[\operatorname{words}(v)\ge w_{min}]
\ge k_{min}
\right].
\]

Defaults:

```text
min_sentence_words     = 4
min_selected_sentences = 2
```

This prevents fragments such as `Decision: B` from satisfying the binary rule
on their own.

### 10.2 Word-ratio ablation

\[
R_q=\min\left(
1,
\frac{\sum_{v\in S_q}\operatorname{words}(v)}
{\max(1,\operatorname{words}(o_q))}
\right).
\]

Short selected sentences still contribute to the word-ratio numerator. English
words use a simple regular expression and CJK characters are counted
individually.

### 10.3 Primary token-ratio metric

Each visible source output is tokenized separately with source-local offsets.
A token crossing a sentence boundary is assigned to exactly one sentence.

\[
T_q=\min\left(
1,
\frac{
\left|\bigcup_{v\in S_q}\operatorname{SourceTokens}(v)\right|
}{
\max(1,\operatorname{TokenCount}(o_q))
}
\right).
\]

This is the primary matrix and policy metric. System prompts, tasks, headers,
chat-template tokens, and special tokens are excluded from both numerator and
denominator. `source_generated_token_count` is retained for auditing, but the
matrix denominator uses the exact source text visible to the receiver.

## 11. Per-sample matrices

Every sample stores the following tensors for both spatial and temporal edge
types:

```text
binary
word_ratio
token_ratio
opportunity
candidate
```

All use `[round][source][target]`.

`candidate=1` means the dense trace contains that worker edge. `opportunity=1`
means an attributed event produced an observed edge score. Candidate topology
is inferred from every recorded worker input, including nodes omitted by
restricted attribution scopes. This prevents `unobserved=keep` from creating
edges that did not exist in the dense graph.

If the same matrix cell appears more than once in a sample, its metric takes the
maximum and the opportunity/candidate indicators remain 1.

Reviewer attribution never enters these worker matrices.

## 12. Correct-only aggregation

Let \(\mathcal C\) be the verified-correct calibration set with size \(n\).

Sample mean:

\[
\bar M_{sample}[r,a,b]
=\frac{1}{n}\sum_{x\in\mathcal C}M^x[r,a,b].
\]

This treats a sample without attribution opportunity for an edge as zero.

Conditional mean:

\[
\bar M_{conditional}[r,a,b]
=
\frac{\sum_{x\in\mathcal C}M^x[r,a,b]}
{\sum_{x\in\mathcal C}O^x[r,a,b]},
\]

with zero when the denominator is zero.

The aggregate stores:

- `sample_mean`;
- `conditional_mean`;
- `sum`;
- `opportunity_count`;
- `candidate_count`.

Policies use `conditional_mean` by default. Aggregation rejects incompatible
event-selection configurations, including mixed node scopes, event filters,
or reviewer selection.

Binary conditional mean is an empirical frequency. Token/word conditional mean
is an average retained-content ratio. Neither is a calibrated causal
probability.

## 13. Deterministic policy construction

Policies are deterministic 0/1 masks. Two construction modes are supported:

1. threshold: retain observed edges whose importance meets the threshold;
2. pruning rate: within each round and edge type, remove the lowest-importance
   observed edges.

For pruning rate \(p>0\), the implementation removes:

\[
\min\left(
|E|,
\max\left(1,\operatorname{round}(p|E|)\right)
\right)
\]

observed edges per round and edge type. A zero rate removes zero edges. Ties use
stable source/target ordering.

`unobserved=keep` is the conservative default: a dense candidate edge with no
attribution evidence is retained. `unobserved=drop` is available explicitly.
Edges outside the candidate topology always remain zero.

Structural constraints:

- spatial diagonals are zero;
- all round-0 temporal masks are zero;
- spatial masks are projected to a DAG when necessary, prioritizing observed
  high-importance edges;
- temporal masks may include self edges and do not undergo same-round cycle
  checks;
- reviewer aggregation remains dense.

The policy records worker IDs, agent types, and roles in matrix order. Evaluation
restores this metadata rather than silently replacing heterogeneous workers.

## 14. Artifacts and schemas

Default attribution output:

```text
attribution/
├── manifest.json
├── sample_00001/
│   ├── node_attribution_index.json
│   ├── event_000_r00_knowledgeable_expert.json
│   ├── event_000_r00_knowledgeable_expert.svg
│   ├── event_000_r00_knowledgeable_expert.insertion_curve.svg
│   ├── event_000_r00_knowledgeable_expert.html
│   ├── ...
│   ├── sample_edge_matrices.json
│   └── sample_token_ratio_matrices.png
├── ...
├── aggregate_edge_matrices.json
├── aggregate_token_ratio_matrices.png
├── pruned_policy_token_ratio.json
├── pruned_policy_binary.json
└── pruned_policy_word_ratio.json
```

Current writers emit `schema_version: 1`. Legacy traces without a schema version
may be read for diagnosis when explicitly allowed, but they cannot be mixed
into a formal schema-v1 aggregate.

### 14.1 Event JSON

Abbreviated schema:

```json
{
  "schema_version": 1,
  "sample_id": "sample_00001",
  "correct_verified": true,
  "status": "attributed",
  "scope": "incoming_agent_output_sentences_only",
  "event": {
    "event_id": 2,
    "round": 0,
    "agent_id": "mathematician",
    "incoming_count": 2
  },
  "input": {
    "messages": [],
    "rendered_prompt": "...",
    "input_ids": []
  },
  "output": {
    "text": "...",
    "generated_ids": []
  },
  "target": {
    "mode": "all",
    "id_source": "trace_generated_ids",
    "selected_indices": [],
    "selected_ids": [],
    "selected_tokens": []
  },
  "incoming_edges": [],
  "sentences": [
    {
      "index": 4,
      "selected": true,
      "selection_step": 1,
      "saliency": 1.0
    }
  ],
  "attribution": {
    "score_type": "logit",
    "reduction": "sum",
    "baseline_score": 8.1,
    "full_score": 11.7,
    "selected_region_indices": [4, 1],
    "curve": [],
    "selected_recovery": 0.97,
    "stop_reason": "near_full_score"
  },
  "edge_scores": [],
  "artifacts": {
    "json": "...",
    "svg": "...",
    "insertion_curve_svg": "...",
    "html": "..."
  }
}
```

Every event JSON is self-contained: the user does not need to reopen the
original trace to recover complete input/output text or incoming source
messages.

### 14.2 Node index

`node_attribution_index.json` records:

- expected and actual event counts;
- completeness status;
- correctness;
- event ID, round, agent, role, attribution status;
- JSON, HTML, sentence-SVG, and insertion-curve paths.

Missing any required all-node event makes the sample incomplete.

### 14.3 Sample matrix JSON

```json
{
  "correct_verified": true,
  "agents": ["knowledgeable_expert", "critic", "mathematician"],
  "agent_types": ["AnalyzeAgent", "AnalyzeAgent", "AnalyzeAgent"],
  "agent_roles": ["Knowlegable Expert", "Critic", "Mathematician"],
  "matrix_orientation": "[round][source][target]",
  "num_rounds": 2,
  "matrices": {
    "binary": {"spatial": [], "temporal": []},
    "token_ratio": {"spatial": [], "temporal": []},
    "word_ratio": {"spatial": [], "temporal": []},
    "opportunity": {"spatial": [], "temporal": []},
    "candidate": {"spatial": [], "temporal": []}
  }
}
```

### 14.4 Aggregate and policy JSON

Aggregate metric paths:

```text
metrics.<token_ratio|binary|word_ratio>.<spatial|temporal>.
  sample_mean
  conditional_mean
  sum
  opportunity_count
  candidate_count
```

Policy files record:

- metric and aggregate statistic;
- threshold or pruning rate;
- unobserved-edge strategy;
- `masks.spatial` and `masks.temporal`;
- worker agent types and roles;
- observed/candidate/retained counts;
- effective candidate pruning rate;
- any edges removed by DAG projection.

## 15. Visualizations

### 15.1 Sentence attribution SVG

`event_*.svg` uses a single-column readable layout:

- fixed prompt text is gray;
- unselected candidate sentences are gray;
- selected sentences use a continuous orange gradient from the configured
  saliency floor to 1.0;
- the receiving node's complete output is green;
- each selected sentence shows its greedy step and saliency.

There is no token-level panel.

### 15.2 Insertion curve SVG

`event_*.insertion_curve.svg` plots:

- x-axis: retained significant sentences as a percentage of all candidates;
- y-axis: the actual teacher-forced insertion score;
- black curve: observed greedy prefix only;
- orange point and vertical line: early-stop location;
- annotation: selected sentence count, sentence percentage, source-output token
  percentage, and recovery;
- horizontal dashed line: target recovery score;
- isolated blue diamond at 100%: full-input endpoint.

The plot never fabricates scores after early stopping. A no-input event receives
an explicit no-search visualization.

### 15.3 HTML and matrix heatmaps

`event_*.html` contains the upstream edge graph, complete input/output,
highlighted sentences, search metadata, and insertion curve.

`sample_token_ratio_matrices.png` and
`aggregate_token_ratio_matrices.png` display per-round spatial and temporal
heatmaps.

The `render` stage recomputes display saliency and rebuilds SVG/HTML artifacts
from cached event JSON without loading the model.

## 16. CLI workflow

Run commands from the repository root.

### 16.1 Dry run

```bash
DRY_RUN=1 ./experiments_attrprune/run_mmlu.sh all
```

### 16.2 One sample

```bash
RUN_NAME=mmlu/qwen3_8b/sample_0 \
DEV_START_INDEX=0 \
MAX_NEW_TOKENS=512 \
./experiments_attrprune/run_mmlu.sh one-sample
```

### 16.3 Complete dev calibration only

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

`calibrate` runs only dev trace collection, per-node attribution, sample
matrices, correct-only aggregation, heatmaps, and policy generation. It does
not run val or test.

### 16.4 Full dev-to-val workflow

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./experiments_attrprune/run_mmlu.sh all
```

Equivalent stages:

```bash
./experiments_attrprune/run_mmlu.sh traces
./experiments_attrprune/run_mmlu.sh attribution
./experiments_attrprune/run_mmlu.sh eval
./experiments_attrprune/run_mmlu.sh compare
```

Cached-artifact stages:

```bash
./experiments_attrprune/run_mmlu.sh aggregate
./experiments_attrprune/run_mmlu.sh render
```

Evaluation stages:

```bash
./experiments_attrprune/run_mmlu.sh eval-dense
./experiments_attrprune/run_mmlu.sh eval-binary
./experiments_attrprune/run_mmlu.sh eval-token-ratio
./experiments_attrprune/run_mmlu.sh eval-word-ratio
```

Show the complete environment-variable interface:

```bash
./experiments_attrprune/run_mmlu.sh --help
```

Cached event JSON is reused only when the algorithm version, model/tokenizer,
actual model dtype, trace content, exact token IDs, and attribution
configuration match. A failed correct-sample event prevents partial policy
publication even when `KEEP_GOING=1` collects additional failures.

## 17. Qwen3 environment and GPU behavior

The repository expects Transformers 4.51 or newer. Install dependencies with:

```bash
pip install -e .
```

Defaults:

```text
torch_dtype = auto
device_map  = auto
```

`CUDA_VISIBLE_DEVICES` controls devices available to Accelerate.
`device_map=auto` is model sharding, not data parallelism. A Qwen3-8B model may
fit on one 48 GB A6000, so eight visible GPUs do not guarantee eight-GPU
saturation.

Attribution scores many perturbed prompts. When memory is insufficient:

1. reduce `ATTR_BATCH_SIZE`;
2. optionally limit the attribution target with `TARGET_TOKEN_LIMIT`;
3. use `target_mode=decision` only for diagnostic experiments.

Increasing `MAX_NEW_TOKENS` does not force longer output; generation still
stops at EOS. However, leaving `TARGET_TOKEN_LIMIT` unset means an unusually
long completed output is fully teacher-forced during attribution and can
substantially increase cost.

## 18. Interpretation boundaries and limitations

1. **Not a causal proof.** The method finds observed incoming sentences
   sufficient to reproduce a recorded output-token score. It does not prove
   universal necessity under every counterfactual generation.
2. **No regeneration during attribution.** Prompt variants teacher-force the
   original output instead of sampling new answers.
3. **Raw logits are model-specific.** Raw-logit sums support within-event
   endpoint comparison but should not be compared directly across models.
4. **Decision-token attribution has a strong fixed-prefix assumption.**
5. **Retokenization is a fallback.** Missing generated token IDs may change
   boundary-space tokenization; artifacts disclose the fallback.
6. **Exact source-text alignment is required.** Normalized or edited traces may
   fail safe alignment.
7. **Truncated outputs are rejected.** Increase `MAX_NEW_TOKENS` and use a new
   `RUN_NAME`.
8. **Sentence splitting is heuristic.** Punctuation and newlines define
   boundaries; word counts are not linguistic tokenization.
9. **Reviewer edges are not pruned.**
10. **Worker order is part of the schema.** Arbitrary-DAG calibration should
    not rely on legacy traces without explicit node indices.
11. **Correct-only calibration has selection bias.** Report complete-split
    accuracy and cost.
12. **Exact global greedy search is expensive.** Early stopping and batching
    reduce wall time but not the underlying candidate count.
13. **Generated artifacts can be large.** Keep them outside source control.

## 19. Validation

Unit and stub end-to-end tests:

```bash
python -m pytest -q
bash -n experiments_attrprune/run_mmlu.sh
bash -n run_full_mmlu.sh
```

The tests cover:

- incoming-only masking and fixed-prompt preservation;
- repeated spatial/temporal source disambiguation;
- exact greedy selection, recovery, tolerance, patience, and stopping;
- token deduplication and all three edge metrics;
- early-stop saliency and degenerate fallbacks;
- complete per-node JSON/HTML/SVG/insertion-curve artifacts;
- no-input nodes;
- sample and aggregate matrix orientation;
- incorrect-sample exclusion from aggregation;
- teacher-forced logit/log-probability scoring;
- round-specific mask replay;
- role restoration;
- round-0 temporal constraints;
- spatial DAG projection and external-policy validation;
- truncation detection;
- stub end-to-end pipeline behavior.

A real-model smoke should first use `one-sample`, verify all \(NR+1\) events,
then use `render` to validate the cached reports before starting complete dev
calibration.
