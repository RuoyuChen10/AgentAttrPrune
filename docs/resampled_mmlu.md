# MMLU: independent dev40 calibration cohorts

Run on **otml-seit2**, inside tmux:

```bash
cd /egr/research-optml/ruoyu.chen/Project/AgentAttrPrune
CUDA_VISIBLE_DEVICES=6,7 ./run_mmlu_dev40_resampling.sh 1 2
```

The launcher runs cohort 1, then cohort 2, using the same two GPUs. Each cohort
collects 40 Dense traces, attributes every node with the reference
correctness-aware settings, aggregates the results, and evaluates global static
pruning at **25%, 50%, 60%, 70%, 80%, 90%**, in that order. Each stage and
validation rate runs in a separate process. This command does not draw curves.

Each new cohort samples 40 distinct questions independently from the 285-question
dev split. Cohorts may overlap. Calibration seeds 1 and 2 select the questions;
generation and validation retain the reference seed 0. The original cohort
(`dev40_correctness_aware`, the first 40 shuffled dev records) is read only.

Settings remain Qwen3-8B, canonical worker order, two rounds, temperature 0,
2048 generated tokens, thinking off, serial calibration generation, attribution
batch 4, and validation batch 4 on all 1,531 validation questions. The complete
reference attribution configuration and reviewer aliases are inherited.

Pruning is static: all retained edges execute on every question. There is no
sigmoid or Bernoulli edge sampling. The six rates retain 34, 23, 18, 14, 9, and 5
worker edges respectively, following the existing rounding rule. The reviewer's
five inputs are not pruned. If Round 1 has no spatial **and** no temporal edges,
its worker calls are skipped and the reviewer reads all five Round 0 outputs.

Results are isolated under:

```text
results/AgentAttrPrune/mmlu/qwen3_8b/
  dev40_resample_seed1_correctness_aware/
  dev40_resample_seed2_correctness_aware/
```

Each directory contains frozen question/order plans, calibration traces,
attributions, policies, six independent validation checkpoints, and a final
`sweep_summary.json` / `sweep_summary.csv`. Logs are under
`logs/mmlu_dev40_resampling/seed_1/` and `seed_2/`.

Restart the same command to resume. To run only cohort 2, supply `2`. Completed
traces are verified and reused. Changed inputs, missing node attributions, failed
runs, and altered policies prevent publication as a complete result. A failed
correctness rescue contributes zero while remaining in the observed denominator.

CPU preparation and command inspection:

```bash
STAGE=prepare ./run_mmlu_dev40_resampling.sh 1 2
DRY_RUN=1 ./run_mmlu_dev40_resampling.sh 1 2
```

`STAGE` also accepts `collect`, `attribute`, `aggregate`, `policies`, `eval`, and
`summarize`. The default `all` performs the complete pipeline. The launcher never
starts, stops, or modifies other tmux sessions or experiments.
