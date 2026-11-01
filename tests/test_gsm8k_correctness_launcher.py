from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments_attrprune.common import build_run_config, config_fingerprint


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "run_gsm8k_train40_correctness_aware.sh"
RUN_NAME = "gsm8k/qwen3_8b/train40_correctness_aware"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _launch(tmp_path, *args, dry_run=True, **overrides):
    env = dict(os.environ)
    for key in ("CUDA_VISIBLE_DEVICES", "RUN_DIR", "RUN_NAME", "PRUNING_PERCENTAGES"):
        env.pop(key, None)
    # Guard all subprocess tests from ever loading a model, including on a
    # preflight regression. The real preflight and CPU prepare still execute.
    shim = tmp_path / "python_without_model"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:3] == ['-m', 'experiments_attrprune.evaluate_benchmark']:\n"
        "    print('STUB_EVALUATION ' + json.dumps({'args': sys.argv[3:], "
        "'gpus': os.environ.get('CUDA_VISIBLE_DEVICES')}))\n"
        "    raise SystemExit(0)\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    shim.chmod(0o755)
    env.update(
        RESULTS_ROOT=str(tmp_path / "results"),
        DRY_RUN="1" if dry_run else "0",
        CONDA_ENV="", PYTHON_BIN=str(shim),
    )
    env.update(overrides)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args], env=env, text=True,
        capture_output=True, check=False,
    )


def _published_run(tmp_path):
    run_dir = tmp_path / "results" / RUN_NAME
    config = build_run_config(
        benchmark="gsm8k", model_name="/models/Qwen3-8B", num_workers=4,
        num_rounds=2, temperature=0.0, max_new_tokens=2048, seed=0,
        calibration_split="train", evaluation_split="test",
        dataset_identity={"benchmark": "gsm8k", "shuffle_seed": 888},
    )
    _write(run_dir / "run_config.json", config)
    selection = []
    for index in range(40):
        sample_id = f"sample_{index:05d}"
        trace = run_dir / "calibration" / "traces" / sample_id / "trace.json"
        _write(trace, {"config": {**config, "sample_index": index}})
        selection.append({
            "sample_id": sample_id, "sample_index": index,
            "trace_json": str(trace.resolve()),
            "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        })
    _write(run_dir / "calibration" / "selection_manifest.json", {
        "schema_version": 1, "expected_sample_count": 40,
        "expected_start_index": 0, "samples": selection,
    })
    spatial = [[[float(source < target) for target in range(4)]
                for source in range(4)] for _ in range(2)]
    temporal = [[[float(round_index == 1) for _ in range(4)]
                 for _ in range(4)] for round_index in range(2)]

    def scale(array, value):
        return [[[item * value for item in row] for row in matrix] for matrix in array]

    def metric(candidates, importance):
        return {
            "conditional_mean": scale(candidates, importance),
            "sample_mean": scale(candidates, importance),
            "sum": scale(candidates, importance * 40),
            "opportunity_count": scale(candidates, 40),
            "candidate_count": scale(candidates, 40),
        }

    aggregate = {
        "schema_version": 1, "run_fingerprint": config["fingerprint"],
        "num_aggregated_samples": 40, "num_correct_samples": 0,
        "aggregation_scope": "all", "num_rounds": 2,
        "matrix_orientation": "[round][source][target]",
        "agents": [worker["agent_id"] for worker in config["workers"]],
        "agent_roles": [worker["role"] for worker in config["workers"]],
        "agent_types": [worker["agent_type"] for worker in config["workers"]],
        "attribution_configuration": {"attribution_mode": "correctness-aware"},
        "metrics": {"token_ratio": {
            "spatial": metric(spatial, 0.8), "temporal": metric(temporal, 0.1),
        }},
    }
    attribution = run_dir / "attribution"
    _write(attribution / "aggregate_edge_matrices.json", aggregate)
    digest = hashlib.sha256((attribution / "aggregate_edge_matrices.json").read_bytes()).hexdigest()
    _write(attribution / "manifest.json", {
        "complete": True, "aggregation_complete": True, "failures": [],
        "aggregate_sha256": digest,
        "samples": [{"sample_id": item["sample_id"]} for item in selection],
    })
    return run_dir


def _mutate(path, change):
    payload = json.loads(path.read_text())
    change(payload)
    _write(path, payload)


def test_default_calibrates_only_fixed_train40_and_overrides_inherited_selection(tmp_path):
    result = _launch(
        tmp_path, BENCHMARK="mmlu", DATA_DIR="/wrong/mmlu", CALIBRATION_SPLIT="dev",
        CALIBRATION_LIMIT="7473", CALIBRATION_START_INDEX="10", NUM_WORKERS="5",
        ATTR_SAMPLE_SCOPE="correct", ATTR_NODE_SCOPE="workers",
        ATTR_SAMPLE_LIMIT="1", ATTR_EVENT_LIMIT="2", TARGET_TOKEN_LIMIT="3",
    )
    assert result.returncode == 0, result.stderr
    assert "experiments_attrprune.collect_correct --benchmark gsm8k" in result.stdout
    assert "--split train --evaluation-split test --start-index 0 --limit 40" in result.stdout
    assert "--num-workers 4 --num-rounds 2 --temperature 0 --max-new-tokens 2048" in result.stdout
    assert "--attribution-mode correctness-aware --aggregation-scope all" in result.stdout
    assert "--sample-scope all --node-scope all" in result.stdout
    assert "--expected-sample-count 40 --expected-start-index 0" in result.stdout
    for absent in ["--sample-limit", "--event-limit", "--target-token-limit", "evaluate_benchmark", "/wrong/mmlu"]:
        assert absent not in result.stdout
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("stage", ["calibrate", "eval"])
def test_gpu_stages_require_explicit_devices_before_any_artifacts(tmp_path, stage):
    result = _launch(tmp_path, stage, dry_run=False)
    assert result.returncode == 2
    assert "CUDA_VISIBLE_DEVICES" in result.stderr
    assert not (tmp_path / "results").exists()


def test_default_eval_ratios_and_explicit_dense_use_same_batch_and_resume(tmp_path):
    default = _launch(tmp_path, "eval", CUDA_VISIBLE_DEVICES="6,7")
    assert default.returncode == 0, default.stderr
    assert default.stdout.count("--benchmark gsm8k") == 3
    for percentage in [50, 60, 80]:
        assert f"prune_{percentage:02d}pct.json" in default.stdout
    assert default.stdout.count("--split test --start-index 0 --limit 1319") == 3
    assert default.stdout.count("--eval-batch-size 4") == 3
    assert default.stdout.count("--resume") == 3
    custom = _launch(tmp_path, "eval", "0", "50", EVAL_BATCH_SIZE="2")
    assert custom.returncode == 0, custom.stderr
    assert "prune_00pct.json" in custom.stdout
    assert custom.stdout.count("--eval-batch-size 2") == 2
    assert "collect_correct" not in custom.stdout
    assert "attribute_edges" not in custom.stdout


@pytest.mark.parametrize("percentage", ["-1", "101", "1.5", "wrong"])
def test_invalid_ratio_rejected_even_in_dry_run(tmp_path, percentage):
    result = _launch(tmp_path, "prepare", percentage)
    assert result.returncode == 2
    assert "integers from 0 to 100" in result.stderr


def test_prepare_all_wrong_reviewers_cpu_only_builds_global_masks_and_dense28(tmp_path):
    run_dir = _published_run(tmp_path)
    result = _launch(tmp_path, "prepare", "0", "50", "60", "80", dry_run=False)
    assert result.returncode == 0, result.stderr
    assert "STUB_EVALUATION" not in result.stdout
    sweep = run_dir / "attribution/static_pruning/token_ratio_conditional_mean_global"
    manifest = json.loads((sweep / "manifest.json").read_text())
    assert manifest["calibration_num_aggregated_samples"] == 40
    assert manifest["calibration_num_correct_samples"] == 0
    assert manifest["pruning_allocation"] == "global"
    assert [item["pruned_edges"] for item in manifest["policies"]] == [0, 14, 17, 22]
    assert manifest["policies"][0]["retained_edges"] == 28
    half = manifest["policies"][1]["retention_by_edge_type"]
    assert half["spatial"]["retained_candidate_edges"] == 12
    assert half["temporal"]["retained_candidate_edges"] == 2
    assert not (run_dir / "evaluation").exists()


@pytest.mark.parametrize("problem", ["incomplete", "failed", "unfinished", "changed", "missing_sample"])
def test_prepare_rejects_incomplete_calibration_publication(tmp_path, problem):
    run_dir = _published_run(tmp_path)
    manifest = run_dir / "attribution/manifest.json"
    changes = {
        "incomplete": lambda d: d.update(complete=False),
        "failed": lambda d: d.update(failures=[{"error": "OOM"}]),
        "unfinished": lambda d: d.update(aggregation_complete=False),
        "changed": lambda d: d.update(aggregate_sha256="old"),
        "missing_sample": lambda d: d["samples"].pop(),
    }
    _mutate(manifest, changes[problem])
    result = _launch(tmp_path, "prepare", dry_run=False)
    assert result.returncode != 0
    assert "incomplete or" in result.stderr
    assert "static_pruning prepare" not in result.stdout


@pytest.mark.parametrize("problem", ["benchmark", "split", "shuffle", "fingerprint"])
def test_prepare_rejects_incompatible_run_identity(tmp_path, problem):
    run_dir = _published_run(tmp_path)

    def change(config):
        if problem == "benchmark":
            config["benchmark"] = "mmlu"
        elif problem == "split":
            config["splits"]["calibration"] = "test"
        elif problem == "shuffle":
            config["dataset"]["shuffle_seed"] = 42
        else:
            config["fingerprint"] = "invalid"
        if problem != "fingerprint":
            config["fingerprint"] = config_fingerprint(config)

    _mutate(run_dir / "run_config.json", change)
    result = _launch(tmp_path, "prepare", dry_run=False)
    assert result.returncode != 0
    assert "static_pruning prepare" not in result.stdout


@pytest.mark.parametrize("problem", ["changed_trace", "extra_trace", "wrong_selection"])
def test_prepare_rejects_changed_frozen_selection(tmp_path, problem):
    run_dir = _published_run(tmp_path)
    traces = run_dir / "calibration/traces"
    if problem == "changed_trace":
        _mutate(traces / "sample_00000/trace.json", lambda d: d.update(changed=True))
    elif problem == "extra_trace":
        _write(traces / "sample_00040/trace.json", {"config": {"sample_index": 40}})
    else:
        _mutate(run_dir / "calibration/selection_manifest.json", lambda d: d.update(expected_start_index=1))
    result = _launch(tmp_path, "prepare", dry_run=False)
    assert result.returncode != 0
    assert "Frozen calibration" in result.stderr
    assert "static_pruning prepare" not in result.stdout


@pytest.mark.parametrize("stage", ["eval", "summarize"])
@pytest.mark.parametrize("problem", ["sweep_hash", "policy_hash", "policy_identity", "sweep_scope"])
def test_postprepare_rejects_stale_or_incompatible_policy_before_eval(tmp_path, stage, problem):
    run_dir = _published_run(tmp_path)
    prepared = _launch(tmp_path, "prepare", "50", dry_run=False)
    assert prepared.returncode == 0, prepared.stderr
    sweep = run_dir / "attribution/static_pruning/token_ratio_conditional_mean_global"
    if problem == "sweep_hash":
        _mutate(sweep / "manifest.json", lambda d: d.update(aggregate_sha256="old"))
    elif problem == "policy_hash":
        _mutate(sweep / "policies/prune_50pct.json", lambda d: d["static_sweep"].update(aggregate_sha256="old"))
    elif problem == "policy_identity":
        _mutate(sweep / "policies/prune_50pct.json", lambda d: d.update(run_fingerprint="old"))
    else:
        _mutate(sweep / "manifest.json", lambda d: d.update(pruning_allocation="per_round_edge_type"))
    args = ("50",) if stage == "eval" else ()
    result = _launch(tmp_path, stage, *args, dry_run=False, CUDA_VISIBLE_DEVICES="6,7")
    assert result.returncode != 0
    assert "rerun prepare" in result.stderr
    assert "STUB_EVALUATION" not in result.stdout


def test_valid_eval_validates_publication_then_passes_gpu_batch_and_resume(tmp_path):
    _published_run(tmp_path)
    prepared = _launch(tmp_path, "prepare", dry_run=False)
    assert prepared.returncode == 0, prepared.stderr
    result = _launch(tmp_path, "eval", dry_run=False, CUDA_VISIBLE_DEVICES="6,7", SKIP_COMPLETED="1")
    assert result.returncode == 0, result.stderr
    invocations = [json.loads(line.split(" ", 1)[1]) for line in result.stdout.splitlines()
                   if line.startswith("STUB_EVALUATION ")]
    assert len(invocations) == 3
    for call, percentage in zip(invocations, [50, 60, 80]):
        assert call["gpus"] == "6,7"
        args = call["args"]
        assert args[args.index("--eval-batch-size") + 1] == "4"
        assert args[args.index("--split") + 1] == "test"
        assert args[args.index("--limit") + 1] == "1319"
        assert args[args.index("--policy-json") + 1].endswith(f"prune_{percentage}pct.json")
        assert "--resume" in args
    assert "static_pruning summarize" not in result.stdout
