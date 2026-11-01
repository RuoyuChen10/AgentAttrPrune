"""Import the frozen repeat_00 rollout of every question without model inference.

The source may still be attributing other repeats. Its immutable generation
artifacts are SHA-checked; mutable attribution files are read as stable atomic
snapshots. Only validated completed node caches are relocated. No source lock is
taken and no source file is written.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import uuid

import numpy as np

from experiments_attrprune import random_order_mmlu as experiment
from experiments_attrprune.common import DEFAULT_RESULTS_ROOT, ExperimentPaths, load_json, write_json


DEFAULT_RUN = DEFAULT_RESULTS_ROOT / "mmlu/qwen3_8b/dev40_random_order1_correctness_aware"
MANIFEST_NAME = "subset_import_manifest.json"


def stable_snapshot(path: Path, expected_sha: str | None = None) -> tuple[dict, str]:
    """Reject changed/replaced files rather than combine different file versions."""
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    checksum = hashlib.sha256(raw).hexdigest()
    if (identity(before) != identity(after) or experiment.file_sha(path) != checksum
            or (expected_sha is not None and checksum != expected_sha)):
        raise ValueError(f"Source artifact changed during snapshot: {path}")
    return json.loads(raw), checksum


def _context(source_config: dict, run_dir: Path, repeats: int):
    settings = source_config["random_order_experiment"]
    return experiment.prepare_context(experiment.parse_args([
        "--stage", "collect", "--reference-run", settings["reference"]["run_dir"],
        "--run-dir", str(run_dir), "--repeats", str(repeats),
        "--order-seed", str(settings["order_seed"]),
        "--attr-batch-size", str(settings["attribution_batch_size"]), "--resume",
    ]))


def _target_context(source_config: dict, run_dir: Path):
    # prepare_context normally rejects a nonempty directory without run_config.
    # Compute its exact expected config/plan against an unused path (never
    # created), then validate actual destination files ourselves. This permits
    # recovery after the import identity marker is atomically published but
    # before both base metadata files have been written.
    reference = Path(source_config["random_order_experiment"]["reference"]["run_dir"]).resolve()
    if run_dir == reference or run_dir in reference.parents or reference in run_dir.parents:
        raise ValueError("Destination must be outside the fixed-order reference run")
    unused = run_dir.parent / f".{run_dir.name}.subset-check-{uuid.uuid4().hex}"
    ctx = replace(_context(source_config, unused, 1), paths=ExperimentPaths(run_dir))
    for path, expected in ((ctx.paths.run_config, ctx.config), (run_dir / "rollout_plan.json", ctx.plan)):
        if path.exists() and load_json(path) != expected:
            raise ValueError(f"Destination configuration or plan differs: {path}")
    return ctx


def _metadata_scorer(configuration: dict):
    """Supply only recorded backend identity to the existing cache-key builder.

    This object deliberately has neither a scoring nor a generation method and
    cannot load a tokenizer or a model. Full attribution settings and algorithm
    versions are still computed by EdgeAttributionCalibrator itself.
    """
    vocab = configuration["tokenizer_vocab_size"]

    def tokenizer_length(self):
        if vocab is None:
            raise TypeError("The recorded tokenizer does not expose a vocabulary size")
        return int(vocab)

    tokenizer_type = type(configuration["tokenizer_class"], (), {"__len__": tokenizer_length})
    tokenizer = tokenizer_type()
    tokenizer.name_or_path = configuration["tokenizer_name"]
    model = SimpleNamespace(
        config=SimpleNamespace(_name_or_path=configuration["model_config_name"]),
        parameters=lambda: iter([SimpleNamespace(dtype=configuration["model_dtype"])]),
    )
    return SimpleNamespace(model_name=configuration["model_name"], model=model, tokenizer=tokenizer)


def validate_event_cache(ctx, rollout: dict, trace: dict, source: dict, cached: dict) -> None:
    """Validate input identity, algorithm settings, completion, and edge coverage."""
    from AgentAttrPrune.attribution.calibrator import EdgeAttributionCalibrator, _canonical_model_name
    from AgentAttrPrune.attribution.correctness import parse_worker_decision
    from AgentAttrPrune.attribution.edge import METRICS

    configuration = cached.get("configuration") or {}
    if _canonical_model_name(configuration.get("model_name")) != _canonical_model_name(ctx.config["model_name"]):
        raise ValueError("Cached attribution model differs from the run model")
    calibrator = EdgeAttributionCalibrator(
        ctx.paths.traces, ctx.paths.attribution, model_name=ctx.config["model_name"],
        config=experiment.attribution_config(ctx), scorer=_metadata_scorer(configuration),
    )
    if configuration != calibrator._event_configuration(source, trace):
        raise ValueError("Stale event SHA or incompatible attribution configuration")
    event_identity = {key: source.get(key) for key in ("event_id", "round", "agent_id", "agent_type", "role")}
    event_identity["incoming_count"] = len(source.get("incoming") or [])
    if (cached.get("schema_version") != 1 or cached.get("event") != event_identity
            or cached.get("sample_id") != rollout["rollout_id"]
            or cached.get("trace_json") != str(experiment.trace_path_for(ctx, rollout).resolve())
            or cached.get("sample") != trace["sample"]
            or cached.get("correct_verified") != trace["final"]["correct"]):
        raise ValueError("Cached event provenance does not match its trace")
    event_path = ctx.paths.attribution / rollout["rollout_id"] / f"{calibrator._event_slug(source)}.json"
    expected_artifacts = {
        "json": event_path, "svg": event_path.with_suffix(".svg"),
        "visualization": event_path.with_suffix(".svg"), "html": event_path.with_suffix(".html"),
        "insertion_curve_svg": event_path.with_suffix(".insertion_curve.svg"),
        "progress_json": event_path.parent / "diagnostics" / event_path.stem / "progress.json",
    }
    for key, path in expected_artifacts.items():
        value = cached.get("artifacts", {}).get(key)
        if value is not None and value != str(path.resolve()):
            raise ValueError("Cached artifact paths point outside their event")
    original_input = cached.get("input") or {}
    if any(original_input.get(key) != source.get(trace_key) for key, trace_key in (
        ("messages", "input_messages"), ("input_ids", "input_ids"), ("attention_mask", "attention_mask"),
    )):
        raise ValueError("Cached event input differs from its frozen generation")
    output = cached.get("output") or {}
    if (output.get("text") != source["output"]
            or output.get("generated_ids") != np.asarray(source["generated_ids"]).reshape(-1).tolist()):
        raise ValueError("Cached event output differs from its frozen generation")
    incoming = source.get("incoming") or []
    status = cached.get("status")
    attribution = cached.get("attribution") or {}
    if not incoming:
        if status != "no_attributable_input" or attribution.get("performed") is not False or cached.get("edge_scores"):
            raise ValueError("No-input cache is incomplete or contains edge scores")
        return
    worker_wrong = (source["agent_id"] in ctx.plan["agents"]
                    and parse_worker_decision(source["output"]) != trace["sample"]["answer"])
    if attribution.get("performed") is not True:
        raise ValueError("Node attribution was not completed")
    if worker_wrong:
        rescue = cached.get("rescue") or {}
        verification = rescue.get("verification") or {}
        sentences = cached.get("sentences") or []
        if (status not in {"rescued", "rescue_failed"}
                or attribution.get("early_stop") is not False
                or attribution.get("stop_reason") != "all_sentences_ranked"
                or len(attribution.get("curve") or []) != len(sentences)
                or sorted(attribution.get("ordered_region_indices") or []) != list(range(len(sentences)))
                or verification.get("hit_max_new_tokens") is not False
                or verification.get("correct") != (status == "rescued")
                or rescue.get("success") != (status == "rescued")):
            raise ValueError("Gold-prefix cache has incomplete search or verification")
    elif status != "attributed" or attribution.get("stop_reason") not in {
        "fixed_prompt_already_matches_full", "near_full_score", "all_sentences_selected",
    }:
        raise ValueError("Original attribution cache is incomplete")
    scores = cached.get("edge_scores") or []
    expected = [(edge["edge_type"], edge["agent_id"], source["agent_id"], edge["round"], source["round"])
                for edge in incoming]
    actual = [(edge.get("edge_type"), edge.get("source_agent"), edge.get("target_agent"),
               edge.get("source_round"), edge.get("target_round")) for edge in scores]
    if actual != expected:
        raise ValueError("Cached scores do not cover exactly the observed incoming edges")
    for edge in scores:
        values = np.asarray([edge.get(metric, float("nan")) for metric in METRICS], dtype=float)
        if (edge.get("opportunity") != 1 or not np.isfinite(values).all()
                or (values < 0).any() or (values > 1).any()
                or (status == "rescue_failed" and values.any())):
            raise ValueError("Invalid observed edge score in completed cache")


def _freeze(path: Path, payload: dict) -> None:
    if path.exists():
        if load_json(path) != payload:
            raise ValueError(f"Refusing to overwrite an existing destination artifact: {path}")
    else:
        write_json(path, payload)


def _relocate_cache(cached: dict, source: dict, target_trace: dict, trace_path: Path,
                    event_path: Path, progress: dict | None) -> dict:
    relocated = copy.deepcopy(cached)
    relocated["trace_json"] = str(trace_path.resolve())
    relocated["configuration"]["event_sha256"] = experiment.event_digest(source, target_trace)
    relocated["artifacts"] = {
        "json": str(event_path.resolve()), "svg": str(event_path.with_suffix(".svg").resolve()),
        "visualization": str(event_path.with_suffix(".svg").resolve()),
        "html": str(event_path.with_suffix(".html").resolve()),
        "insertion_curve_svg": str(event_path.with_suffix(".insertion_curve.svg").resolve()),
    }
    if "progress_json" in cached.get("artifacts", {}):
        relocated["artifacts"]["progress_json"] = (
            str((event_path.parent / "diagnostics" / event_path.stem / "progress.json").resolve())
            if progress is not None else None
        )
    return relocated


def _render_cache(cached: dict, path: Path) -> None:
    from AgentAttrPrune.attribution.visualization import (
        write_event_html, write_event_insertion_curve_svg, write_event_svg,
    )
    write_event_svg(cached, path.with_suffix(".svg"))
    write_event_insertion_curve_svg(cached, path.with_suffix(".insertion_curve.svg"))
    write_event_html(cached, path.with_suffix(".html"))


def _complete_sample(ctx, rollout: dict, trace: dict, cached_events: list[dict]) -> dict:
    """Rebuild only deterministic CPU indexes/matrices, without touching scores."""
    from AgentAttrPrune.attribution.calibrator import EdgeAttributionCalibrator
    from AgentAttrPrune.attribution.edge import build_sample_matrices
    from AgentAttrPrune.attribution.visualization import write_sample_matrix_png

    folder = ctx.paths.attribution / rollout["rollout_id"]
    index_path = folder / "node_attribution_index.json"
    index = {
        "schema_version": 1, "sample_id": rollout["rollout_id"], "sample_index": rollout["sample_index"],
        "correct_verified": trace["final"]["correct"], "node_scope": "all",
        "expected_event_count": len(trace["events"]), "actual_event_count": len(cached_events), "complete": True,
        "events": [{
            **{key: event["event"][key] for key in ("event_id", "round", "agent_id", "role")},
            "status": event["status"],
            **{key: event["artifacts"][key] for key in ("json", "svg", "visualization", "insertion_curve_svg", "html")},
        } for event in cached_events],
    }
    _freeze(index_path, index)
    matrix = build_sample_matrices(trace, cached_events)
    calibrator = EdgeAttributionCalibrator(ctx.paths.traces, ctx.paths.attribution,
                                           config=experiment.attribution_config(ctx))
    matrix["selection_configuration"] = calibrator._selection_configuration()
    matrix["event_attribution_files"] = [event["artifacts"]["json"] for event in cached_events]
    matrix["node_attribution_index"] = str(index_path.resolve())
    matrix_path, png_path = folder / "sample_edge_matrices.json", folder / "sample_token_ratio_matrices.png"
    matrix["visualizations"] = {"token_ratio_png": str(png_path.resolve())}
    for edge_type in ("spatial", "temporal"):
        if matrix["matrices"]["candidate"][edge_type] != matrix["matrices"]["opportunity"][edge_type]:
            raise ValueError("A completed sample is missing observed edge attribution")
    _freeze(matrix_path, matrix)
    if not png_path.exists():
        write_sample_matrix_png(matrix, png_path)
    return {
        "sample_id": rollout["rollout_id"], "trace_json": str(experiment.trace_path_for(ctx, rollout).resolve()),
        "correct_verified": trace["final"]["correct"], "event_count": len(cached_events),
        "node_attribution_index": str(index_path.resolve()), "sample_matrix": str(matrix_path.resolve()),
        "sample_matrix_png": str(png_path.resolve()),
    }


def import_repeat_zero(source_run: Path, run_dir: Path, *, resume: bool = False, dry_run: bool = False) -> dict:
    """Create an independently resumable one-repeat run; never run inference."""
    from AgentAttrPrune.attribution.calibrator import EdgeAttributionCalibrator

    source_run, run_dir = Path(source_run).resolve(), Path(run_dir).resolve()
    if source_run == run_dir or source_run in run_dir.parents or run_dir in source_run.parents:
        raise ValueError("Source and destination must be independent directories")
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError("Destination already exists; use --resume for this import")
    anchors = {}
    snapshots = {}
    for name in ("run_config.json", "rollout_plan.json", "calibration/selection_manifest.json"):
        snapshots[name], anchors[name] = stable_snapshot(source_run / name)
    source_config = snapshots["run_config.json"]
    source_plan = snapshots["rollout_plan.json"]
    source_ctx = _context(source_config, source_run, int(source_plan["repeats_per_question"]))
    if source_ctx.config != source_config or source_ctx.plan != source_plan:
        raise ValueError("Source configuration or plan changed")
    selection = experiment.validate_selection(source_ctx)
    if selection != snapshots["calibration/selection_manifest.json"]:
        raise ValueError("Source selection changed")
    target_ctx = _target_context(source_config, run_dir)
    selected = [rollout for rollout in source_plan["rollouts"] if rollout["repeat_index"] == 0]
    if selected != target_ctx.plan["rollouts"]:
        raise ValueError("repeat_00 subset does not exactly match the destination rollout plan")
    source_settings, target_settings = copy.deepcopy(source_config), copy.deepcopy(target_ctx.config)
    for settings in (source_settings, target_settings):
        settings.pop("fingerprint")
        for key in ("repeats_per_question", "plan_sha256"):
            settings["random_order_experiment"].pop(key)
    if source_settings != target_settings:
        raise ValueError("Only repetition budget and plan identity may change during import")
    identity = {
        "source_run": str(source_run), "source_fingerprint": source_config["fingerprint"],
        "source_artifact_sha256": anchors, "destination_run": str(run_dir),
        "destination_fingerprint": target_ctx.config["fingerprint"],
        "destination_plan_sha256": experiment.digest(target_ctx.plan),
        "selection_rule": {"repeat_index": 0, "depends_on_correctness": False},
    }
    manifest_path = run_dir / MANIFEST_NAME
    previous_import = load_json(manifest_path) if manifest_path.exists() else None
    if run_dir.exists() and any(p.name != ".random_order.lock" for p in run_dir.iterdir()):
        if previous_import is None or previous_import.get("identity") != identity:
            raise ValueError("Destination is not this verified subset import; refusing to overwrite it")
    records = {item["rollout_id"]: item for item in selection["samples"]}
    traces, cache_items, skipped = [], [], []
    for rollout in selected:
        record = records[rollout["rollout_id"]]
        source_path = experiment.trace_path_for(source_ctx, rollout)
        original, source_sha = stable_snapshot(source_path, record["sha256"])
        experiment.validate_trace(source_ctx, rollout, original)
        target_trace = copy.deepcopy(original)
        target_trace["config"].update(target_ctx.config)
        target_trace["source_provenance"] = {
            "trace_json": str(source_path), "sha256": source_sha,
            "run_fingerprint": source_config["fingerprint"], "selection_rule": "repeat_index == 0",
        }
        experiment.validate_trace(target_ctx, rollout, target_trace)
        target_path = experiment.trace_path_for(target_ctx, rollout)
        if target_path.exists() and load_json(target_path) != target_trace:
            raise ValueError(f"Destination trace differs; refusing to overwrite {target_path}")
        traces.append((rollout, target_trace, target_path, source_path, source_sha))
        for source in original["events"]:
            slug = EdgeAttributionCalibrator._event_slug(source)
            source_event_path = source_ctx.paths.attribution / rollout["rollout_id"] / f"{slug}.json"
            target_event_path = target_ctx.paths.attribution / rollout["rollout_id"] / f"{slug}.json"
            if target_event_path.exists():
                cached, target_sha = stable_snapshot(target_event_path)
                validate_event_cache(target_ctx, rollout, target_trace, source, cached)
                cache_items.append((target_event_path, cached, None, None, target_sha, "preserved"))
                continue
            if not source_event_path.exists():
                continue
            try:
                cached, cache_sha = stable_snapshot(source_event_path)
                validate_event_cache(source_ctx, rollout, original, source, cached)
            except (ValueError, KeyError, TypeError, OSError) as error:
                skipped.append({"source_event": str(source_event_path), "reason": str(error)})
                continue
            progress = None
            progress_path = source_event_path.parent / "diagnostics" / slug / "progress.json"
            if progress_path.exists():
                try:
                    progress, _ = stable_snapshot(progress_path)
                    if progress.get("configuration") != cached["configuration"]:
                        progress = None
                except (ValueError, OSError):
                    progress = None
            relocated = _relocate_cache(cached, source, target_trace, target_path, target_event_path, progress)
            validate_event_cache(target_ctx, rollout, target_trace, source, relocated)
            if progress is not None:
                progress["configuration"] = copy.deepcopy(relocated["configuration"])
            cache_items.append((target_event_path, relocated, progress, str(source_event_path), cache_sha, "imported"))

    def recheck_source():
        for name, checksum in anchors.items():
            if experiment.file_sha(source_run / name) != checksum:
                raise ValueError(f"Frozen source artifact changed: {name}")
        for _, _, _, source_path, checksum in traces:
            if experiment.file_sha(source_path) != checksum:
                raise ValueError(f"Frozen source trace changed: {source_path}")

    recheck_source()
    previous_events = {item["destination_event_json"]: item
                       for item in (previous_import or {}).get("event_provenance", [])}
    event_provenance = []
    for path, _, _, source_path, source_sha, action in cache_items:
        record = {"destination_event_json": str(path), "action": action,
                  "source_event_json": source_path, "snapshot_sha256": source_sha}
        if action == "preserved" and str(path) in previous_events:
            record = {**previous_events[str(path)], "action": "preserved"}
        event_provenance.append(record)
    report = {
        "schema_version": 1, "identity": identity, "num_questions": len(selected),
        "num_rollouts": len(selected), "repeats_per_question": 1,
        "num_reusable_events": len(cache_items), "skipped_caches": skipped,
        # Persist provenance before publishing the first cache, including when
        # an import stops between individual atomic cache writes.
        "event_provenance": event_provenance,
        "trace_provenance": [{"rollout_id": rollout["rollout_id"], "source_trace_json": str(source_path),
                              "source_sha256": checksum, "destination_trace_json": str(target_path)}
                             for rollout, _, target_path, source_path, checksum in traces],
        "complete": False,
    }
    if dry_run:
        return {**report, "dry_run": True}
    # Do not take the source lock: collection is frozen while attribution may run.
    with experiment.run_lock(run_dir):
        recheck_source()
        target_ctx = _target_context(source_config, run_dir)
        if manifest_path.exists() and load_json(manifest_path).get("identity") != identity:
            raise ValueError("Destination import identity changed while waiting for its lock")
        # Checkpoint newly discovered source-cache provenance on every resume,
        # before any of those caches can become visible in the destination.
        write_json(manifest_path, report)
        # Persist base config before traces to make interrupted imports resumable.
        _freeze(target_ctx.paths.run_config, target_ctx.config)
        _freeze(run_dir / "rollout_plan.json", target_ctx.plan)
        for _, trace, target_path, _, _ in traces:
            _freeze(target_path, trace)
        if any(not target_path.is_file() for _, _, target_path, _, _ in traces):
            raise RuntimeError("All destination traces must exist before CPU-only collection recovery")
        # A non-model sentinel is a second guard: accidental generation cannot
        # instantiate HFChat even if collect_run's orphan recovery changes later.
        asyncio.run(experiment.collect_run(target_ctx, llm=SimpleNamespace()))
        for path, cached, progress, source_path, source_sha, action in cache_items:
            if path.exists():
                if load_json(path) != cached:
                    raise ValueError(f"Destination cache changed; refusing to overwrite {path}")
            else:
                if progress is not None:
                    _freeze(Path(cached["artifacts"]["progress_json"]), progress)
                _render_cache(cached, path)
                _freeze(path, cached)
        complete_samples = []
        for rollout, trace, _, _, _ in traces:
            cached_events = []
            for source in trace["events"]:
                path = target_ctx.paths.attribution / rollout["rollout_id"] / f"{EdgeAttributionCalibrator._event_slug(source)}.json"
                if not path.exists():
                    break
                cached = load_json(path)
                validate_event_cache(target_ctx, rollout, trace, source, cached)
                cached_events.append(cached)
            if len(cached_events) == len(trace["events"]):
                complete_samples.append(_complete_sample(target_ctx, rollout, trace, cached_events))
        attribution_manifest = target_ctx.paths.attribution / "manifest.json"
        old_manifest = load_json(attribution_manifest) if attribution_manifest.exists() else None
        current_manifest = copy.deepcopy(old_manifest) if old_manifest is not None else {
                "schema_version": 1, "run_dir": str(target_ctx.paths.traces),
                "output_dir": str(target_ctx.paths.attribution), "model_name": target_ctx.config["model_name"],
                "configuration": experiment.attribution_config(target_ctx).to_dict(),
                "samples": [], "skipped_incorrect": [], "failures": [],
                "complete": False, "aggregation_complete": False,
        }
        if current_manifest.get("configuration") != experiment.attribution_config(target_ctx).to_dict():
            raise ValueError("Existing destination attribution manifest uses different settings")
        changed = current_manifest.get("samples") != complete_samples
        current_manifest["samples"] = complete_samples
        current_manifest["complete"] = (len(complete_samples) == len(selected)
                                        and not current_manifest.get("failures")
                                        and not current_manifest.get("skipped_incorrect"))
        published_path = target_ctx.paths.attribution / "aggregate_edge_matrices.json"
        valid_publication = (not changed and current_manifest["complete"]
                             and current_manifest.get("aggregation_complete") is True
                             and published_path.exists()
                             and current_manifest.get("aggregate_sha256") == experiment.file_sha(published_path)
                             and load_json(published_path).get("run_fingerprint") == target_ctx.config["fingerprint"])
        if not valid_publication:
            current_manifest["aggregation_complete"] = False
            current_manifest.pop("aggregate_sha256", None)
        if current_manifest != old_manifest:
            write_json(attribution_manifest, current_manifest)
        recheck_source()
        experiment.validate_selection(target_ctx)
        report.update(complete=True, num_complete_attribution_samples=len(complete_samples),
                      event_provenance=event_provenance)
        write_json(manifest_path, report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=experiment.DEFAULT_RUN)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = import_repeat_zero(args.source_run, args.run_dir, resume=args.resume, dry_run=args.dry_run)
    print(json.dumps({
        **{key: report.get(key) for key in ("complete", "dry_run", "num_questions", "num_rollouts",
                                           "num_reusable_events", "num_complete_attribution_samples")},
        "skipped_caches": len(report["skipped_caches"]),
        "manifest_json": str(args.run_dir.resolve() / MANIFEST_NAME),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
