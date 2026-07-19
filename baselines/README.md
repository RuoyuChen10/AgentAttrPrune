# Baseline integration contract

`baselines/` is reserved for future external comparison methods, such as the
original AgentPrune trainable edge optimizer. The current release does not copy
or wrap those implementations.

Any baseline added to the primary comparison must satisfy all of the following
requirements:

1. Read `run_config.json` from the active run and reuse the exact model, MMLU
   split, sample range, seed, worker IDs and role order, round count, prompts,
   reviewer, and generation settings.
2. Use the same evaluator as `experiments_attrprune/evaluate_mmlu.py` on the
   complete selected split. A baseline must not evaluate only baseline-correct
   samples or use a different answer parser.
3. Emit the same evaluation schema as the dense, binary, word-ratio, and
   token-ratio modes. At minimum, report accuracy, correct/total counts,
   prompt/generated token counts, runtime, candidate/retained edge counts, and
   effective edge retention.
4. Write artifacts under
   `evaluation/<baseline_name>_<split>/` inside the current run. Do not scatter
   generated files across the source tree or repository root.
5. Save the resolved policy/topology and every baseline-specific setting, then
   verify the shared run fingerprint before comparison.

Baseline-specific training data, search budget, checkpoints, and
hyperparameters must be recorded separately and must never overwrite the shared
`run_config.json`.
