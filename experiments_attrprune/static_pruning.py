"""Build and summarize a frozen attribution-based pruning-rate sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from AgentAttrPrune.attribution import build_pruned_masks
from AgentAttrPrune.pruning import RoundPolicy
from experiments_attrprune.common import config_fingerprint, load_json, write_json


DEFAULT_PERCENTAGES = (5, 10, 20, 30, 40, 50)
SELECTION_FIELDS = (
    "benchmark",
    "split",
    "start_index",
    "stop_index_exclusive",
    "dataset_size",
    "total",
)


def _validated_percentages(values: Sequence[int]) -> tuple[int, ...]:
    percentages = tuple(int(value) for value in values)
    if not percentages:
        raise ValueError("At least one pruning percentage is required")
    if any(value < 0 or value > 100 for value in percentages):
        raise ValueError("Pruning percentages must be integers in [0, 100]")
    if len(set(percentages)) != len(percentages):
        raise ValueError("Pruning percentages must be unique")
    return percentages


def _scope_suffix(pruning_scope: str) -> str:
    if pruning_scope not in {"per_round_edge_type", "global"}:
        raise ValueError(
            "pruning_scope must be per_round_edge_type or global"
        )
    return "" if pruning_scope == "per_round_edge_type" else "_global"


def _sweep_root(
    run_dir: Path,
    metric: str,
    statistic: str,
    pruning_scope: str,
) -> Path:
    return (
        run_dir
        / "attribution"
        / "static_pruning"
        / f"{metric}_{statistic}{_scope_suffix(pruning_scope)}"
    )


def _policy_label(percentage: int) -> str:
    return f"prune_{percentage:02d}pct"


def _evaluation_name(
    metric: str,
    statistic: str,
    percentage: int,
    pruning_scope: str,
) -> str:
    return (
        f"static_{metric}_{statistic}{_scope_suffix(pruning_scope)}_"
        f"{_policy_label(percentage)}"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sources(
    run_config: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    *,
    metric: str,
    statistic: str,
) -> None:
    declared_fingerprint = run_config.get("fingerprint")
    if not declared_fingerprint or config_fingerprint(run_config) != declared_fingerprint:
        raise ValueError("run_config.json has a missing or invalid fingerprint")
    if aggregate.get("run_fingerprint") != declared_fingerprint:
        raise ValueError("Aggregate run fingerprint does not match run_config.json")
    if list(aggregate.get("agents") or ()) != [
        str(item["agent_id"]) for item in run_config["workers"]
    ]:
        raise ValueError("Aggregate worker order does not match run_config.json")
    if list(aggregate.get("agent_roles") or ()) != [
        str(item["role"]) for item in run_config["workers"]
    ]:
        raise ValueError("Aggregate worker roles do not match run_config.json")
    if list(aggregate.get("agent_types") or ()) != [
        str(item.get("agent_type", "AnalyzeAgent"))
        for item in run_config["workers"]
    ]:
        raise ValueError("Aggregate worker types do not match run_config.json")
    if int(aggregate.get("num_rounds", 0)) != int(run_config["num_rounds"]):
        raise ValueError("Aggregate round count does not match run_config.json")
    if metric not in (aggregate.get("metrics") or {}):
        raise ValueError(f"Aggregate does not contain metric {metric!r}")
    if statistic not in {"conditional_mean", "sample_mean"}:
        raise ValueError("statistic must be conditional_mean or sample_mean")
    if int(aggregate.get("num_correct_samples", 0)) < 1:
        raise ValueError("Aggregate contains no verified-correct calibration samples")


def build_static_sweep(
    run_dir: str | Path,
    *,
    percentages: Sequence[int] = DEFAULT_PERCENTAGES,
    metric: str = "token_ratio",
    statistic: str = "conditional_mean",
    unobserved: str = "keep",
    pruning_scope: str = "per_round_edge_type",
    aggregate_json: str | Path | None = None,
) -> dict[str, Any]:
    """Create one immutable policy per requested pruning percentage."""

    run_dir = Path(run_dir).resolve()
    percentages = _validated_percentages(percentages)
    run_config_path = run_dir / "run_config.json"
    aggregate_path = Path(aggregate_json).resolve() if aggregate_json else (
        run_dir / "attribution" / "aggregate_edge_matrices.json"
    )
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Missing run configuration: {run_config_path}")
    if not aggregate_path.is_file():
        raise FileNotFoundError(f"Missing aggregate attribution matrix: {aggregate_path}")

    run_config = load_json(run_config_path)
    aggregate = load_json(aggregate_path)
    _validate_sources(
        run_config,
        aggregate,
        metric=metric,
        statistic=statistic,
    )

    split = str((run_config.get("splits") or {}).get("evaluation", "val"))
    aggregate_sha256 = _file_sha256(aggregate_path)
    output_dir = _sweep_root(run_dir, metric, statistic, pruning_scope)
    policy_dir = output_dir / "policies"
    entries: list[dict[str, Any]] = []

    for percentage in percentages:
        pruning_rate = percentage / 100.0
        payload = build_pruned_masks(
            aggregate,
            metric=metric,
            statistic=statistic,
            threshold=None,
            pruning_rate=pruning_rate,
            unobserved=unobserved,
            pruning_scope=pruning_scope,
        )
        label = _policy_label(percentage)
        evaluation_name = _evaluation_name(
            metric,
            statistic,
            percentage,
            pruning_scope,
        )
        payload["static_sweep"] = {
            "label": label,
            "requested_pruning_percent": percentage,
            "aggregate_json": str(aggregate_path),
            "aggregate_sha256": aggregate_sha256,
        }
        # Exercise the public policy validator before publishing the file.
        policy = RoundPolicy.from_dict(payload)
        policy_path = (policy_dir / f"{label}.json").resolve()
        policy.to_json(policy_path)

        retention = payload["retention"]
        candidate_edges = sum(
            int(retention[edge_type]["candidate_edges"])
            for edge_type in ("spatial", "temporal")
        )
        retained_edges = sum(
            int(retention[edge_type]["retained_candidate_edges"])
            for edge_type in ("spatial", "temporal")
        )
        effective_rate = (
            1.0 - retained_edges / candidate_edges if candidate_edges else 0.0
        )
        entries.append(
            {
                "label": label,
                "requested_pruning_percent": percentage,
                "requested_pruning_rate": pruning_rate,
                "effective_candidate_pruning_rate": effective_rate,
                "candidate_edges": candidate_edges,
                "retained_edges": retained_edges,
                "pruned_edges": candidate_edges - retained_edges,
                "retention_by_edge_type": retention,
                "policy_json": str(policy_path),
                "evaluation_name": evaluation_name,
                "evaluation_directory": str(
                    (run_dir / "evaluation" / f"{evaluation_name}_{split}").resolve()
                ),
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "static_attribution_pruning_sweep",
        "run_fingerprint": run_config["fingerprint"],
        "benchmark": str(run_config.get("benchmark", "mmlu")),
        "run_dir": str(run_dir),
        "run_config_json": str(run_config_path),
        "aggregate_json": str(aggregate_path),
        "aggregate_sha256": aggregate_sha256,
        "calibration_num_correct_samples": int(aggregate["num_correct_samples"]),
        "metric": metric,
        "statistic": statistic,
        "unobserved": unobserved,
        "pruning_scope": (
            "observed_attributed_edges_global"
            if pruning_scope == "global"
            else "observed_attributed_edges_per_round_edge_type"
        ),
        "pruning_allocation": pruning_scope,
        "evaluation_split": split,
        "percentages": list(percentages),
        "policies": entries,
    }
    manifest_path = write_json(output_dir / "manifest.json", manifest)
    manifest["manifest_json"] = str(manifest_path.resolve())
    return manifest


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError("Cannot write an empty sweep CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def summarize_static_sweep(manifest_json: str | Path) -> dict[str, Any]:
    """Validate all sweep evaluations and publish compact JSON/CSV results."""

    manifest_path = Path(manifest_json).resolve()
    manifest = load_json(manifest_path)
    if manifest.get("kind") != "static_attribution_pruning_sweep":
        raise ValueError("Not a static attribution pruning sweep manifest")
    run_dir = Path(manifest["run_dir"])
    expected_fingerprint = manifest["run_fingerprint"]
    selection: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    evaluations: dict[str, Any] = {}

    for entry in manifest.get("policies") or []:
        evaluation_dir = Path(entry["evaluation_directory"])
        summary_path = evaluation_dir / "evaluation_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing sweep evaluation summary: {summary_path}")
        summary = load_json(summary_path)
        if summary.get("run_fingerprint") != expected_fingerprint:
            raise ValueError(f"{summary_path} belongs to a different run")
        if summary.get("mode") != manifest["metric"]:
            raise ValueError(f"{summary_path} declares the wrong evaluation mode")
        if summary.get("evaluation_name") != entry["evaluation_name"]:
            raise ValueError(f"{summary_path} declares the wrong evaluation name")
        if Path(summary["source_policy_json"]).resolve() != Path(
            entry["policy_json"]
        ).resolve():
            raise ValueError(f"{summary_path} used a different policy")
        current_selection = {field: summary.get(field) for field in SELECTION_FIELDS}
        if selection is None:
            selection = current_selection
        elif current_selection != selection:
            raise ValueError("Sweep evaluations do not use an identical data selection")
        if int(summary["candidate_edges"]) != int(entry["candidate_edges"]):
            raise ValueError(f"{summary_path} candidate edge count disagrees with policy")
        if int(summary["retained_edges"]) != int(entry["retained_edges"]):
            raise ValueError(f"{summary_path} retained edge count disagrees with policy")

        requested_rate = float(entry["requested_pruning_rate"])
        effective_rate = 1.0 - float(summary["edge_retention"])
        if not math.isclose(
            effective_rate,
            float(entry["effective_candidate_pruning_rate"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{summary_path} effective pruning rate disagrees with policy")
        row = {
            "label": entry["label"],
            "requested_pruning_percent": int(entry["requested_pruning_percent"]),
            "requested_pruning_rate": requested_rate,
            "effective_pruning_percent": 100.0 * effective_rate,
            "effective_pruning_rate": effective_rate,
            "pruned_edges": int(summary["candidate_edges"])
            - int(summary["retained_edges"]),
            "retained_edges": int(summary["retained_edges"]),
            "candidate_edges": int(summary["candidate_edges"]),
            "accuracy": float(summary["accuracy"]),
            "accuracy_percent": 100.0 * float(summary["accuracy"]),
            "correct": int(summary["correct"]),
            "total": int(summary["total"]),
            "prompt_tokens": int(summary["prompt_tokens"]),
            "generated_tokens": int(summary["generated_tokens"]),
            "elapsed_seconds": float(summary["elapsed_seconds"]),
            "policy_json": entry["policy_json"],
            "evaluation_summary_json": str(summary_path.resolve()),
        }
        rows.append(row)
        evaluations[entry["label"]] = summary

    if not rows:
        raise ValueError("Sweep manifest contains no policies")
    split = manifest["evaluation_split"]
    pruning_scope = str(
        manifest.get("pruning_allocation", "per_round_edge_type")
    )
    output_dir = (
        run_dir
        / "evaluation"
        / (
            f"static_{manifest['metric']}_{manifest['statistic']}"
            f"{_scope_suffix(pruning_scope)}_sweep_{split}"
        )
    )
    summary = {
        "schema_version": 1,
        "kind": "static_attribution_pruning_sweep_summary",
        "run_fingerprint": expected_fingerprint,
        "benchmark": manifest["benchmark"],
        "metric": manifest["metric"],
        "statistic": manifest["statistic"],
        "pruning_scope": manifest["pruning_scope"],
        "pruning_allocation": pruning_scope,
        "manifest_json": str(manifest_path),
        "selection": selection,
        "rows": rows,
        "evaluations": evaluations,
    }
    json_path = write_json(output_dir / "summary.json", summary)
    csv_path = _write_csv(output_dir / "summary.csv", rows)
    summary["summary_json"] = str(json_path.resolve())
    summary["summary_csv"] = str(csv_path.resolve())
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or summarize a static attribution-pruning rate sweep."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Build rate-specific policies from an existing aggregate matrix."
    )
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument(
        "--percentages",
        nargs="+",
        type=int,
        default=list(DEFAULT_PERCENTAGES),
    )
    prepare.add_argument(
        "--metric",
        choices=("binary", "token_ratio", "word_ratio"),
        default="token_ratio",
    )
    prepare.add_argument(
        "--statistic",
        choices=("conditional_mean", "sample_mean"),
        default="conditional_mean",
    )
    prepare.add_argument("--unobserved", choices=("keep", "drop"), default="keep")
    prepare.add_argument(
        "--pruning-scope",
        choices=("per_round_edge_type", "global"),
        default="per_round_edge_type",
        help=(
            "Allocate the requested pruning rate independently per matrix "
            "(legacy behavior) or once across all spatial/temporal matrices."
        ),
    )
    prepare.add_argument("--aggregate-json", type=Path, default=None)

    summarize = subparsers.add_parser(
        "summarize", help="Validate completed evaluations and write summary JSON/CSV."
    )
    summarize.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        manifest = build_static_sweep(
            args.run_dir,
            percentages=args.percentages,
            metric=args.metric,
            statistic=args.statistic,
            unobserved=args.unobserved,
            pruning_scope=args.pruning_scope,
            aggregate_json=args.aggregate_json,
        )
        print(f"Prepared {len(manifest['policies'])} policies")
        print(f"Manifest: {manifest['manifest_json']}")
        for entry in manifest["policies"]:
            print(
                f"  {entry['label']}: pruned {entry['pruned_edges']}/"
                f"{entry['candidate_edges']} edges -> {entry['policy_json']}"
            )
        return
    summary = summarize_static_sweep(args.manifest)
    print(f"Summary JSON: {summary['summary_json']}")
    print(f"Summary CSV: {summary['summary_csv']}")


if __name__ == "__main__":
    main()
