# Benchmark data

Dataset files are not committed to this repository. The default MMLU location
is:

```text
data/MMLU/
```

The other supported locations are:

```text
data/GSM8K/train.jsonl
data/GSM8K/test.jsonl
data/HumanEval/humaneval-py.jsonl
```

Download every supported benchmark:

```bash
python -m AgentAttrPrune.datasets.download all --data-root data
```

The GSM8K downloader uses the original OpenAI train and test JSONL files. The
HumanEval downloader uses the Python benchmark file distributed by AgentPrune.
That HumanEval file has no official calibration split, so this repository makes
a deterministic, disjoint 80/81 `dev`/`test` partition with seed 888.

The loader accepts either a directory containing `dev/`, `val/`, and `test/`
directly or the original archive layout in which those directories are under
`data/`.

The formal experiment selections are:

| Benchmark | Calibration and attribution | Frozen-policy evaluation |
|---|---|---|
| MMLU | `dev` | `val` |
| GSM8K | `train` | `test` |
| HumanEval | fixed seed-888 `dev` half | disjoint fixed `test` half |

Evaluation records never contribute to attribution aggregation or policy
construction.

Point the shell pipeline at an existing copy:

```bash
MMLU_DATA_DIR=/path/to/MMLU/data \
./experiments_attrprune/run_mmlu.sh calibrate
```

Or call the trace collector directly:

```bash
python -m experiments_attrprune.collect_correct \
  --data-dir /path/to/MMLU/data \
  --split dev
```

Do not commit questions, extracted CSV files, symlinks to private datasets,
download caches, or generated dataset metadata under this directory.

HumanEval evaluation executes model-generated Python in a subprocess with time,
memory, file-size, and open-file limits. These limits prevent common accidental
runaway programs but are not a security boundary against deliberately hostile
code; use the evaluator only for trusted local benchmark runs.
