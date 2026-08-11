"""Benchmark-generic evaluation entrypoint.

The implementation remains in :mod:`evaluate_mmlu` so existing commands and
imports stay compatible.
"""

from experiments_attrprune.evaluate_mmlu import main


if __name__ == "__main__":
    main()
