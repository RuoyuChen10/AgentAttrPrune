# AgentAttrPrune results

All generated experiment artifacts are written below this directory. A typical
run is:

```text
results/AgentAttrPrune/mmlu/qwen3_8b/seed_0/
```

Each run may contain `run_config.json`, `calibration/`, `attribution/`,
`evaluation/`, `comparison_summary.json`, and `comparison_summary.csv`.
Experiment code must not write runtime artifacts into `AgentAttrPrune/`,
`experiments_attrprune/`, or `tests/`.

Everything below this guide is ignored by Git. Archive results in dedicated
experiment storage or as explicit external artifacts rather than committing
large traces, logits, generated text, JSON bundles, SVG/HTML reports, or matrix
images to the source repository.
