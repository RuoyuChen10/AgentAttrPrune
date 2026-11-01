"""Build and summarize a frozen attribution-based pruning-rate sweep."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from AgentAttrPrune.attribution import build_pruned_masks
from AgentAttrPrune.datasets import benchmark_definition
from AgentAttrPrune.pruning import RoundPolicy
from experiments_attrprune.common import (
    config_fingerprint,
    load_json,
    prompt_contract,
    validate_prompt_contract,
    write_json,
)


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


def _calibration_metadata(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Read explicit calibration counts, retaining the legacy correct-only contract."""

    scope = str(aggregate.get("aggregation_scope", "correct"))
    correct = int(aggregate.get("num_correct_samples", 0))
    total = int(aggregate.get("num_aggregated_samples", correct))
    if scope not in {"correct", "all"}:
        raise ValueError("Aggregate aggregation_scope must be correct or all")
    if total < 1:
        raise ValueError("Aggregate contains no aggregated calibration samples")
    if correct < 0 or correct > total or (scope == "correct" and correct != total):
        raise ValueError("Aggregate calibration sample counts are inconsistent")
    return {
        "calibration_num_aggregated_samples": total,
        "calibration_num_correct_samples": correct,
        "calibration_aggregation_scope": scope,
        "calibration_attribution_mode": str(
            (aggregate.get("attribution_configuration") or {}).get(
                "attribution_mode", "original"
            )
        ),
    }


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
    _calibration_metadata(aggregate)


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
    calibration_manifest_path = run_dir / "attribution" / "manifest.json"
    if calibration_manifest_path.is_file():
        calibration_manifest = load_json(calibration_manifest_path)
        if (
            calibration_manifest.get("failures")
            or calibration_manifest.get("complete") is False
            or calibration_manifest.get("aggregation_complete") is False
        ):
            raise ValueError(
                "Calibration attribution is incomplete or has failures; refusing "
                "to prepare policies from a potentially stale aggregate. "
                f"Resume calibration first: {calibration_manifest_path}"
            )
        published_hash = calibration_manifest.get("aggregate_sha256")
        if published_hash and published_hash != _file_sha256(aggregate_path):
            raise ValueError(
                "Aggregate does not match the completed calibration publication; "
                "refusing to prepare policies from a stale or changed aggregate"
            )

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
        **_calibration_metadata(aggregate),
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


def transfer_static_sweep(
    source_run: str | Path,
    target_run: str | Path,
    *,
    percentages: Sequence[int],
    metric: str = "token_ratio",
    statistic: str = "conditional_mean",
    pruning_scope: str = "global",
) -> dict[str, Any]:
    """Reuse frozen masks in a new run using the current prompt contract.

    This is an evaluation-only transfer: no calibration artifacts are copied,
    and every transferred policy records its source fingerprint and file hash.
    The target directory must not already exist.
    """

    source_run = Path(source_run).resolve()
    target_run = Path(target_run).resolve()
    percentages = _validated_percentages(percentages)
    if target_run.exists():
        raise FileExistsError(
            f"Target run already exists; refusing to overwrite it: {target_run}"
        )

    source_config_path = source_run / "run_config.json"
    if not source_config_path.is_file():
        raise FileNotFoundError(f"Missing source run configuration: {source_config_path}")
    source_config = load_json(source_config_path)
    source_fingerprint = source_config.get("fingerprint")
    if not source_fingerprint or config_fingerprint(source_config) != source_fingerprint:
        raise ValueError("Source run_config.json has a missing or invalid fingerprint")

    benchmark = str(source_config.get("benchmark", "mmlu")).lower()
    configured_prompt = prompt_contract(benchmark)
    prompt_set = benchmark_definition(benchmark).prompt_set
    prepare_reviewer_ids = getattr(prompt_set, "prepare_reviewer_agent_ids", None)
    if configured_prompt is None or prepare_reviewer_ids is None:
        raise ValueError(
            f"{benchmark} does not declare a transferable reviewer-ID prompt contract"
        )

    target_config = copy.deepcopy(source_config)
    target_config["prompt"] = {
        **configured_prompt,
        "reviewer_agent_ids": prepare_reviewer_ids(
            [str(worker["agent_id"]) for worker in source_config["workers"]]
        ),
    }
    target_config["fingerprint"] = config_fingerprint(target_config)
    validate_prompt_contract(target_config, benchmark=benchmark)
    target_fingerprint = target_config["fingerprint"]

    source_policy_dir = (
        _sweep_root(source_run, metric, statistic, pruning_scope) / "policies"
    )
    target_sweep_root = _sweep_root(target_run, metric, statistic, pruning_scope)
    target_policy_dir = target_sweep_root / "policies"
    split = str((target_config.get("splits") or {}).get("evaluation", "val"))
    expected_agents = [str(worker["agent_id"]) for worker in source_config["workers"]]
    expected_roles = [str(worker["role"]) for worker in source_config["workers"]]
    expected_types = [
        str(worker.get("agent_type", "AnalyzeAgent"))
        for worker in source_config["workers"]
    ]

    transferred: list[tuple[Path, dict[str, Any]]] = []
    entries: list[dict[str, Any]] = []
    aggregate_paths: set[Path] = set()
    for percentage in percentages:
        label = _policy_label(percentage)
        source_policy_path = source_policy_dir / f"{label}.json"
        if not source_policy_path.is_file():
            raise FileNotFoundError(f"Missing source policy: {source_policy_path}")
        source_payload = load_json(source_policy_path)
        source_policy = RoundPolicy.from_dict(source_payload)
        source_policy.validate_agents(expected_agents)
        if list(source_policy.agent_roles) != expected_roles:
            raise ValueError(f"Source policy roles do not match: {source_policy_path}")
        if list(source_policy.agent_types) != expected_types:
            raise ValueError(f"Source policy types do not match: {source_policy_path}")
        if source_policy.num_rounds != int(source_config["num_rounds"]):
            raise ValueError(f"Source policy round count does not match: {source_policy_path}")
        if source_policy.metadata.get("run_fingerprint") != source_fingerprint:
            raise ValueError(f"Source policy fingerprint does not match: {source_policy_path}")
        if source_policy.metadata.get("metric") != metric:
            raise ValueError(f"Source policy metric does not match: {source_policy_path}")
        if source_policy.metadata.get("statistic") != statistic:
            raise ValueError(f"Source policy statistic does not match: {source_policy_path}")
        if source_policy.metadata.get("pruning_allocation") != pruning_scope:
            raise ValueError(
                f"Source policy pruning allocation does not match: {source_policy_path}"
            )

        static_sweep = source_payload.get("static_sweep") or {}
        aggregate_path = Path(
            static_sweep.get(
                "aggregate_json",
                source_run / "attribution" / "aggregate_edge_matrices.json",
            )
        ).resolve()
        if not aggregate_path.is_file():
            raise FileNotFoundError(f"Missing source aggregate: {aggregate_path}")
        aggregate_sha256 = _file_sha256(aggregate_path)
        declared_aggregate_sha256 = static_sweep.get("aggregate_sha256")
        if declared_aggregate_sha256 and declared_aggregate_sha256 != aggregate_sha256:
            raise ValueError(
                f"Source aggregate hash does not match policy: {source_policy_path}"
            )
        aggregate_paths.add(aggregate_path)

        payload = copy.deepcopy(source_payload)
        payload["run_fingerprint"] = target_fingerprint
        payload["policy_transfer"] = {
            "kind": "frozen_policy_replay_with_current_prompt",
            "calibration_reused": True,
            "source_run": str(source_run),
            "source_run_fingerprint": source_fingerprint,
            "source_policy_json": str(source_policy_path.resolve()),
            "source_policy_sha256": _file_sha256(source_policy_path),
        }
        RoundPolicy.from_dict(payload)
        target_policy_path = (target_policy_dir / f"{label}.json").resolve()
        transferred.append((target_policy_path, payload))

        retention = payload["retention"]
        candidate_edges = sum(
            int(retention[edge_type]["candidate_edges"])
            for edge_type in ("spatial", "temporal")
        )
        retained_edges = sum(
            int(retention[edge_type]["retained_candidate_edges"])
            for edge_type in ("spatial", "temporal")
        )
        evaluation_name = _evaluation_name(metric, statistic, percentage, pruning_scope)
        entries.append(
            {
                "label": label,
                "requested_pruning_percent": percentage,
                "requested_pruning_rate": percentage / 100.0,
                "effective_candidate_pruning_rate": (
                    1.0 - retained_edges / candidate_edges if candidate_edges else 0.0
                ),
                "candidate_edges": candidate_edges,
                "retained_edges": retained_edges,
                "pruned_edges": candidate_edges - retained_edges,
                "retention_by_edge_type": retention,
                "policy_json": str(target_policy_path),
                "evaluation_name": evaluation_name,
                "evaluation_directory": str(
                    (
                        target_run / "evaluation" / f"{evaluation_name}_{split}"
                    ).resolve()
                ),
            }
        )

    if len(aggregate_paths) != 1:
        raise ValueError("Transferred policies do not share one source aggregate")
    aggregate_path = aggregate_paths.pop()
    aggregate = load_json(aggregate_path)
    if aggregate.get("run_fingerprint") != source_fingerprint:
        raise ValueError("Source aggregate fingerprint does not match source run")

    manifest = {
        "schema_version": 1,
        "kind": "static_attribution_pruning_sweep",
        "run_fingerprint": target_fingerprint,
        "benchmark": benchmark,
        "run_dir": str(target_run),
        "run_config_json": str((target_run / "run_config.json").resolve()),
        "aggregate_json": str(aggregate_path),
        "aggregate_sha256": _file_sha256(aggregate_path),
        **_calibration_metadata(aggregate),
        "metric": metric,
        "statistic": statistic,
        "unobserved": str(transferred[0][1].get("unobserved", "keep")),
        "pruning_scope": (
            "observed_attributed_edges_global"
            if pruning_scope == "global"
            else "observed_attributed_edges_per_round_edge_type"
        ),
        "pruning_allocation": pruning_scope,
        "evaluation_split": split,
        "percentages": list(percentages),
        "policy_transfer": {
            "kind": "frozen_policy_replay_with_current_prompt",
            "calibration_reused": True,
            "source_run": str(source_run),
            "source_run_fingerprint": source_fingerprint,
        },
        "policies": entries,
    }

    target_run.mkdir(parents=True, exist_ok=False)
    write_json(target_run / "run_config.json", target_config)
    for target_policy_path, payload in transferred:
        payload.update(_calibration_metadata(aggregate))
        write_json(target_policy_path, payload)
    manifest_path = write_json(target_sweep_root / "manifest.json", manifest)
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
    eval_batch_size: int | None = None
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
        state_paths = {evaluation_dir / "evaluation_state.json"}
        if summary.get("evaluation_state_json"):
            state_paths.add(Path(summary["evaluation_state_json"]))
        existing_state_paths = sorted(path for path in state_paths if path.is_file())
        if existing_state_paths:
            policy_payload = RoundPolicy.from_json(entry["policy_json"]).to_dict()
            policy_fingerprint = hashlib.sha256(json.dumps(
                policy_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            for state_path in existing_state_paths:
                state = load_json(state_path)
                if state.get("policy_fingerprint") != policy_fingerprint:
                    raise ValueError(
                        f"{summary_path} evaluation policy fingerprint does not match "
                        "the current prepared policy; rerun evaluation with a new name "
                        "or restore the evaluated policy"
                    )
        elif manifest.get("calibration_attribution_mode") == "correctness-aware":
            raise ValueError(
                f"Missing evaluation_state.json for correctness-aware sweep: {evaluation_dir}"
            )
        current_selection = {field: summary.get(field) for field in SELECTION_FIELDS}
        if selection is None:
            selection = current_selection
        elif current_selection != selection:
            raise ValueError("Sweep evaluations do not use an identical data selection")
        current_batch_size = int(summary.get("eval_batch_size", 1))
        if current_batch_size < 1:
            raise ValueError(f"{summary_path} declares an invalid eval_batch_size")
        if eval_batch_size is None:
            eval_batch_size = current_batch_size
        elif current_batch_size != eval_batch_size:
            raise ValueError("Sweep evaluations do not use an identical eval_batch_size")
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
            "eval_batch_size": current_batch_size,
            "prompt_tokens": int(summary["prompt_tokens"]),
            "generated_tokens": int(summary["generated_tokens"]),
            "elapsed_seconds": float(summary["elapsed_seconds"]),
            "batch_metadata_samples": int(
                summary.get("generation_batching", {}).get("recorded_samples", 0)
            ),
            "samples_without_batch_metadata": int(
                summary.get("generation_batching", {}).get(
                    "samples_without_batch_metadata", summary["total"]
                )
            ),
            "oom_fallback_samples": int(
                summary.get("generation_batching", {}).get("oom_fallback_samples", 0)
            ),
            "oom_fallback_calls": int(
                summary.get("generation_batching", {}).get("oom_fallback_calls", 0)
            ),
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
        "eval_batch_size": eval_batch_size,
        "calibration_num_aggregated_samples": int(manifest.get(
            "calibration_num_aggregated_samples",
            manifest.get("calibration_num_correct_samples", 0),
        )),
        "calibration_num_correct_samples": int(manifest.get("calibration_num_correct_samples", 0)),
        "calibration_aggregation_scope": str(manifest.get("calibration_aggregation_scope", "correct")),
        "calibration_attribution_mode": str(manifest.get("calibration_attribution_mode", "original")),
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

    transfer = subparsers.add_parser(
        "transfer",
        help=(
            "Reuse frozen policies in a new run bound to the current prompt "
            "contract, without rerunning calibration."
        ),
    )
    transfer.add_argument("--source-run", type=Path, required=True)
    transfer.add_argument("--target-run", type=Path, required=True)
    transfer.add_argument("--percentages", nargs="+", type=int, required=True)
    transfer.add_argument(
        "--metric",
        choices=("binary", "token_ratio", "word_ratio"),
        default="token_ratio",
    )
    transfer.add_argument(
        "--statistic",
        choices=("conditional_mean", "sample_mean"),
        default="conditional_mean",
    )
    transfer.add_argument(
        "--pruning-scope",
        choices=("per_round_edge_type", "global"),
        default="global",
    )

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
    if args.command == "transfer":
        manifest = transfer_static_sweep(
            args.source_run,
            args.target_run,
            percentages=args.percentages,
            metric=args.metric,
            statistic=args.statistic,
            pruning_scope=args.pruning_scope,
        )
        print(f"Transferred {len(manifest['policies'])} frozen policies")
        print(f"Source run: {manifest['policy_transfer']['source_run']}")
        print(f"Target run: {manifest['run_dir']}")
        print(f"Manifest: {manifest['manifest_json']}")
        return
    summary = summarize_static_sweep(args.manifest)
    print(f"Summary JSON: {summary['summary_json']}")
    print(f"Summary CSV: {summary['summary_csv']}")


if __name__ == "__main__":
    main()
