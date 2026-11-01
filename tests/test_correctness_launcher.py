from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments_attrprune.attribute_edges import validate_calibration_selection
from experiments_attrprune.common import ExperimentPaths


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "run_mmlu_dev40_correctness_aware.sh"


def _launch(tmp_path, *args, dry_run=True, **overrides):
    env = dict(os.environ)
    for key in ("CUDA_VISIBLE_DEVICES", "RUN_DIR", "RUN_NAME", "PRUNING_PERCENTAGES"):
        env.pop(key, None)
    env.update(RESULTS_ROOT=str(tmp_path / "results"), DRY_RUN="1" if dry_run else "0")
    env.update(overrides)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args], env=env, text=True, capture_output=True,
        check=False,
    )


def test_default_stage_only_calibrates_complete_dev40_selection(tmp_path):
    result = _launch(
        tmp_path, DEV_LIMIT="285", DEV_START_INDEX="10", ATTR_SAMPLE_SCOPE="correct",
        ATTR_NODE_SCOPE="workers", ATTR_SAMPLE_LIMIT="1", ATTR_EVENT_LIMIT="2",
    )
    assert result.returncode == 0, result.stderr
    assert "experiments_attrprune.collect_correct" in result.stdout
    assert "--start-index 0 --limit 40" in result.stdout
    assert "--attribution-mode correctness-aware --aggregation-scope all" in result.stdout
    assert "--sample-scope all --node-scope all" in result.stdout
    assert "--expected-sample-count 40 --expected-start-index 0" in result.stdout
    assert "--sample-limit" not in result.stdout
    assert "--event-limit" not in result.stdout
    assert "evaluate_mmlu" not in result.stdout
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("stage", ["calibrate", "eval"])
def test_gpu_stages_require_explicit_gpu_selection(tmp_path, stage):
    result = _launch(tmp_path, stage, dry_run=False)
    assert result.returncode == 2
    assert "CUDA_VISIBLE_DEVICES" in result.stderr
    assert not (tmp_path / "results").exists()


def test_prepare_and_eval_are_separate_and_preserve_requested_ratios(tmp_path):
    prepared = _launch(tmp_path, "prepare", "0", "50", MODEL_PATH="/no/model")
    assert prepared.returncode == 0, prepared.stderr
    assert "static_pruning prepare" in prepared.stdout
    assert "--pruning-scope global --percentages 0 50" in prepared.stdout
    assert "evaluate_mmlu" not in prepared.stdout
    evaluated = _launch(tmp_path, "eval", "0", "50", SKIP_COMPLETED="1")
    assert evaluated.returncode == 0, evaluated.stderr
    assert "prune_00pct.json" in evaluated.stdout
    assert "prune_50pct.json" in evaluated.stdout
    assert evaluated.stdout.count("--eval-batch-size 4") == 2
    assert "collect_correct" not in evaluated.stdout
    assert "attribute_edges" not in evaluated.stdout
    assert "Skip completed:     0" in evaluated.stdout


@pytest.mark.parametrize("complete,failures", [(False, []), (True, [{"error": "failed"}])])
def test_postcalibration_stages_reject_failed_or_incomplete_manifest(
    tmp_path, complete, failures
):
    attribution = (
        tmp_path / "results" / "mmlu" / "qwen3_8b" / "dev40_correctness_aware" / "attribution"
    )
    attribution.mkdir(parents=True)
    (attribution / "manifest.json").write_text(json.dumps({
        "complete": complete, "failures": failures,
        "samples": [{"sample_id": f"sample_{index:05d}"} for index in range(40)],
    }))
    (attribution / "aggregate_edge_matrices.json").write_text(json.dumps({
        "num_aggregated_samples": 40, "aggregation_scope": "all",
        "attribution_configuration": {"attribution_mode": "correctness-aware"},
    }))
    result = _launch(
        tmp_path, "prepare", "0", dry_run=False, CONDA_ENV="", PYTHON_BIN=sys.executable,
    )
    assert result.returncode != 0
    assert "manifest is incomplete or failed" in result.stderr
    assert "static_pruning prepare" not in result.stdout


def _published_aggregate(tmp_path):
    attribution = (
        tmp_path / "results" / "mmlu" / "qwen3_8b" / "dev40_correctness_aware" / "attribution"
    )
    attribution.mkdir(parents=True)
    aggregate_path = attribution / "aggregate_edge_matrices.json"
    aggregate_path.write_text(json.dumps({
        "num_aggregated_samples": 40, "aggregation_scope": "all",
        "attribution_configuration": {"attribution_mode": "correctness-aware"},
    }))
    digest = hashlib.sha256(aggregate_path.read_bytes()).hexdigest()
    manifest = {
        "complete": True, "failures": [], "aggregation_complete": True,
        "aggregate_sha256": digest,
        "samples": [{"sample_id": f"sample_{index:05d}"} for index in range(40)],
    }
    (attribution / "manifest.json").write_text(json.dumps(manifest))
    return attribution, digest, manifest


@pytest.mark.parametrize("problem", ["unfinished", "changed"])
def test_postcalibration_rejects_unpublished_or_changed_aggregate(tmp_path, problem):
    attribution, _, manifest = _published_aggregate(tmp_path)
    if problem == "unfinished":
        manifest["aggregation_complete"] = False
    else:
        manifest["aggregate_sha256"] = "obsolete"
    (attribution / "manifest.json").write_text(json.dumps(manifest))
    result = _launch(
        tmp_path, "prepare", "0", dry_run=False, CONDA_ENV="", PYTHON_BIN=sys.executable,
    )
    assert result.returncode != 0
    assert "publication is incomplete or changed" in result.stderr


@pytest.mark.parametrize("stage", ["eval", "summarize"])
@pytest.mark.parametrize("stale_source", ["sweep", "policy"])
def test_evaluation_rejects_stale_sweep_or_policy_hash(tmp_path, stage, stale_source):
    attribution, digest, _ = _published_aggregate(tmp_path)
    sweep = attribution / "static_pruning" / "token_ratio_conditional_mean_global"
    policy = sweep / "policies" / "prune_00pct.json"
    policy.parent.mkdir(parents=True)
    policy.write_text(json.dumps({"static_sweep": {
        "aggregate_sha256": "old" if stale_source == "policy" else digest,
    }}))
    (sweep / "manifest.json").write_text(json.dumps({
        "aggregate_sha256": "old" if stale_source == "sweep" else digest,
        "policies": [{"requested_pruning_percent": 0, "policy_json": str(policy)}],
    }))
    ratios = ("0",) if stage == "eval" else ()
    result = _launch(
        tmp_path, stage, *ratios, dry_run=False, CONDA_ENV="", PYTHON_BIN=sys.executable,
        CUDA_VISIBLE_DEVICES="0",
    )
    assert result.returncode != 0
    assert "stale aggregate; rerun prepare" in result.stderr
    assert "evaluate_mmlu" not in result.stdout


def _selection_args(**overrides):
    values = dict(
        expected_sample_count=2, expected_start_index=0, render_only=False,
        sample_glob="sample_*", sample_limit=None, sample_scope="all",
        node_scope="all", event_limit=None, event_id=None, allow_legacy_traces=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _trace(paths, index):
    destination = paths.traces / f"sample_{index:05d}" / "trace.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"config": {"sample_index": index}}))
    return destination


def test_fixed_selection_rejects_missing_extra_or_mutated_traces(tmp_path):
    paths = ExperimentPaths(tmp_path)
    first = _trace(paths, 0)
    with pytest.raises(ValueError, match="missing"):
        validate_calibration_selection(_selection_args(), paths)
    _trace(paths, 1)
    extra = _trace(paths, 2)
    with pytest.raises(ValueError, match="unexpected"):
        validate_calibration_selection(_selection_args(), paths)
    extra.unlink()
    validate_calibration_selection(_selection_args(), paths)
    manifest = json.loads((paths.calibration / "selection_manifest.json").read_text())
    assert len(manifest["samples"]) == 2
    validate_calibration_selection(_selection_args(), paths)
    first.write_text(json.dumps({"config": {"sample_index": 0}, "changed": True}))
    with pytest.raises(ValueError, match="contents changed"):
        validate_calibration_selection(_selection_args(), paths)


def test_fixed_selection_rejects_partial_attribution_and_mislabelled_trace(tmp_path):
    paths = ExperimentPaths(tmp_path)
    first = _trace(paths, 0)
    _trace(paths, 1)
    with pytest.raises(ValueError, match="all selected samples and all nodes"):
        validate_calibration_selection(_selection_args(node_scope="workers"), paths)
    first.write_text(json.dumps({"config": {"sample_index": 1}}))
    with pytest.raises(ValueError, match="does not match its directory"):
        validate_calibration_selection(_selection_args(), paths)


def test_fixed_selection_rejects_partial_aggregate_manifest(tmp_path):
    paths = ExperimentPaths(tmp_path)
    _trace(paths, 0)
    _trace(paths, 1)
    paths.attribution.mkdir(parents=True)
    (paths.attribution / "manifest.json").write_text(json.dumps({
        "samples": [{"sample_matrix": str(
            paths.attribution / "sample_00000" / "sample_edge_matrices.json"
        )}],
        "failures": [],
    }))
    with pytest.raises(ValueError, match="matrices for every selected sample"):
        validate_calibration_selection(_selection_args(aggregate_only=True), paths)
