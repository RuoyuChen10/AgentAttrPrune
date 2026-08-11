from __future__ import annotations

import argparse
from pathlib import Path

from .benchmarks import BENCHMARKS
from .gsm8k import download_gsm8k
from .humaneval import download_humaneval
from .mmlu import download_mmlu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument(
        "benchmark",
        choices=(*BENCHMARKS, "all"),
        help="Dataset to download.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Parent directory containing MMLU, GSM8K, and HumanEval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    for name in names:
        if name == "mmlu":
            path = download_mmlu(args.data_root / "MMLU")
        elif name == "gsm8k":
            path = download_gsm8k(args.data_root / "GSM8K")
        else:
            path = download_humaneval(args.data_root / "HumanEval")
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
