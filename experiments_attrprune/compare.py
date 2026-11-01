"""Compare dense, binary, and primary token-ratio evaluations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from AgentAttrPrune.datasets import BENCHMARKS
from experiments_attrprune.common import (
    DEFAULT_MODEL,
    config_fingerprint,
    load_json,
    resolve_run_dir,
    write_json,
)


REQUIRED_SUMMARY_FIELDS = {
    "split",
    "start_index",
    "stop_index_exclusive",
    "dataset_size",
    "accuracy",
    "correct",
    "total",
    "prompt_tokens",
    "generated_tokens",
    "elapsed_seconds",
    "retained_edges",
    "candidate_edges",
    "edge_retention",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fair-comparison summary for one AgentAttrPrune run."
    )
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="mmlu")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-name", default=str(DEFAULT_MODEL))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_comparison(run_dir: Path) -> dict:
    run_config = load_json(run_dir / "run_config.json")
    expected_fingerprint = run_config["fingerprint"]
    if config_fingerprint(run_config) != expected_fingerprint:
        raise ValueError("run_config.json fingerprint is invalid")
    evaluation_split = str(
        (run_config.get("splits") or {}).get("evaluation", "val")
    )
    evaluations = tuple(
        f"{mode}_{evaluation_split}" for mode in ("dense", "binary", "token_ratio")
    )
    expected_modes = {
        f"{mode}_{evaluation_split}": mode
        for mode in ("dense", "binary", "token_ratio")
    }
    summaries = {}
    for name in evaluations:
        path = run_dir / "evaluation" / name / "evaluation_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation summary: {path}")
        summary = load_json(path)
        if summary.get("run_fingerprint") != expected_fingerprint:
            raise ValueError(
                f"{name} was produced from run fingerprint "
                f"{summary.get('run_fingerprint')!r}, expected {expected_fingerprint!r}"
            )
        if summary.get("mode") != expected_modes[name]:
            raise ValueError(
                f"{name} declares mode {summary.get('mode')!r}, "
                f"expected {expected_modes[name]!r}"
            )
        missing = sorted(REQUIRED_SUMMARY_FIELDS - summary.keys())
        if missing:
            raise ValueError(f"{name} is missing required fields: {missing}")
        summaries[name] = summary

    selection_fields = (
        "split",
        "start_index",
        "stop_index_exclusive",
        "dataset_size",
        "total",
    )
    dense_name = f"dense_{evaluation_split}"
    binary_name = f"binary_{evaluation_split}"
    token_ratio_name = f"token_ratio_{evaluation_split}"
    dense_selection = {
        field: summaries[dense_name].get(field) for field in selection_fields
    }
    for name in (binary_name, token_ratio_name):
        if summaries[name].get("eval_batch_size", 1) != summaries[dense_name].get(
            "eval_batch_size", 1
        ):
            raise ValueError(
                f"{name} eval_batch_size differs from {dense_name}; "
                "use the same inference batching for policy comparisons"
            )
        selection = {field: summaries[name].get(field) for field in selection_fields}
        if selection != dense_selection:
            raise ValueError(
                f"{name} evaluation selection differs from {dense_name}: "
                f"{selection!r} vs {dense_selection!r}"
            )
        if summaries[name]["candidate_edges"] != summaries[dense_name][
            "candidate_edges"
        ]:
            raise ValueError(
                f"{name} candidate topology differs from {dense_name}"
            )

    rows = []
    for name, summary in summaries.items():
        rows.append(
            {
                "evaluation": name,
                "accuracy": summary["accuracy"],
                "correct": summary["correct"],
                "total": summary["total"],
                "prompt_tokens": summary["prompt_tokens"],
                "generated_tokens": summary["generated_tokens"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "eval_batch_size": summary.get("eval_batch_size", 1),
                "retained_edges": summary["retained_edges"],
                "candidate_edges": summary["candidate_edges"],
                "edge_retention": summary["edge_retention"],
            }
        )

    dense = summaries[dense_name]
    for name in (binary_name, token_ratio_name):
        summary = summaries[name]
        summary["delta_vs_dense"] = {
            "accuracy": summary["accuracy"] - dense["accuracy"],
            "prompt_tokens": summary.get("prompt_tokens", 0)
            - dense.get("prompt_tokens", 0),
            "generated_tokens": summary.get("generated_tokens", 0)
            - dense.get("generated_tokens", 0),
            "elapsed_seconds": summary.get("elapsed_seconds", 0.0)
            - dense.get("elapsed_seconds", 0.0),
        }

    return {
        "schema_version": 1,
        "run_fingerprint": expected_fingerprint,
        "benchmark": str(run_config.get("benchmark", "mmlu")),
        "run_config": str(run_dir / "run_config.json"),
        "fairness_contract": {
            "same_model": True,
            "same_data_snapshot": True,
            "same_workers_and_roles": True,
            "same_rounds": True,
            "same_decoding": True,
            "same_eval_batch_size": True,
            "same_evaluation_split": True,
            "same_evaluation_selection": True,
            "same_graph_executor": True,
        },
        "evaluations": summaries,
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or resolve_run_dir(
        model_name=args.model_name,
        seed=args.seed,
        results_root=args.results_root,
        run_name=args.run_name,
        benchmark=args.benchmark,
    )
    comparison = build_comparison(run_dir)
    json_path = write_json(run_dir / "comparison_summary.json", comparison)
    csv_path = write_csv(run_dir / "comparison_summary.csv", comparison["rows"])
    print(f"Saved comparison JSON to {json_path}")
    print(f"Saved comparison CSV to {csv_path}")


if __name__ == "__main__":
    main()
