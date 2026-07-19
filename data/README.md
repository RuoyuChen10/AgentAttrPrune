# MMLU data

Dataset files are not committed to this repository. The default MMLU location
is:

```text
data/MMLU/
```

The loader accepts either a directory containing `dev/`, `val/`, and `test/`
directly or the original archive layout in which those directories are under
`data/`.

The formal experiment protocol uses:

- `dev`: dense trace collection, node attribution, and correct-only policy
  calibration;
- `val`: complete, leakage-free evaluation after the policy is frozen;
- `test`: optional final reporting only.

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
