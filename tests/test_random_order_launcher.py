from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "run_mmlu_dev40_random_order.sh"
RUN_NAME = "mmlu/qwen3_8b/dev40_random_order10_correctness_aware"


def _launch(tmp_path, *args, dry_run=True, **overrides):
    env = dict(os.environ)
    for key in (
        "CUDA_VISIBLE_DEVICES", "RUN_DIR", "RUN_NAME", "REFERENCE_RUN",
        "ORDER_SEED", "ATTR_BATCH_SIZE", "RESUME", "CONDA_BIN",
    ):
        env.pop(key, None)
    # Never enter model-loading code, even if a launcher's preflight regresses.
    shim = tmp_path / "python recorder"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], "
        "'gpus': os.environ.get('CUDA_VISIBLE_DEVICES')}))\n"
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


def _command(result):
    assert result.returncode == 0, result.stderr
    return shlex.split(result.stdout.strip())


def _value(command, option):
    return command[command.index(option) + 1]


def test_dry_run_defaults_to_new_calibration_and_readonly_reference(tmp_path):
    result = _launch(
        tmp_path, DEV_LIMIT="285", NUM_ROUNDS="5", TEMPERATURE="1",
        MODEL_PATH="/unrelated/model", ATTR_SAMPLE_LIMIT="1",
    )
    command = _command(result)
    assert command[1:3] == ["-m", "experiments_attrprune.random_order_mmlu"]
    assert _value(command, "--stage") == "calibrate"
    assert _value(command, "--reference-run") == str(
        REPO_ROOT / "results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware"
    )
    assert _value(command, "--run-dir") == str(tmp_path / "results" / RUN_NAME)
    assert _value(command, "--order-seed") == "0"
    assert _value(command, "--attr-batch-size") == "4"
    assert "--resume" in command
    assert "--dry-run" in command
    assert "/unrelated/model" not in command
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("stage", ["collect", "attribute", "calibrate"])
def test_gpu_stage_rejects_missing_devices_before_dispatch(tmp_path, stage):
    result = _launch(tmp_path, stage, dry_run=False)
    assert result.returncode == 2
    assert "CUDA_VISIBLE_DEVICES" in result.stderr
    assert result.stdout == ""
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("stage", ["aggregate", "visualize"])
def test_cpu_stages_dispatch_without_gpu_selection(tmp_path, stage):
    result = _launch(tmp_path, stage, dry_run=False)
    assert result.returncode == 0, result.stderr
    invocation = json.loads(result.stdout.splitlines()[-1])
    assert invocation["gpus"] is None
    assert _value(invocation["args"], "--stage") == stage
    assert "--dry-run" not in invocation["args"]


def test_custom_paths_seed_batch_and_gpu_reach_cli_as_literal_arguments(tmp_path):
    reference = str(tmp_path / "reference with spaces")
    name = "custom run/literal $value"
    result = _launch(
        tmp_path, "collect", dry_run=False, CUDA_VISIBLE_DEVICES="6,7",
        REFERENCE_RUN=reference, RUN_NAME=name, ORDER_SEED="17",
        ATTR_BATCH_SIZE="2", RESUME="0",
    )
    assert result.returncode == 0, result.stderr
    invocation = json.loads(result.stdout.splitlines()[-1])
    command = invocation["args"]
    assert invocation["gpus"] == "6,7"
    assert _value(command, "--reference-run") == reference
    assert _value(command, "--run-dir") == str(tmp_path / "results" / name)
    assert _value(command, "--order-seed") == "17"
    assert _value(command, "--attr-batch-size") == "2"
    assert "--resume" not in command


def test_conda_mode_is_supported_without_executing_conda_in_dry_run(tmp_path):
    command = _command(_launch(
        tmp_path, CONDA_ENV="agentprune", CONDA_BIN="/missing/conda",
    ))
    assert command[:6] == [
        "/missing/conda", "run", "--no-capture-output", "-n", "agentprune", "python",
    ]


@pytest.mark.parametrize("args", [("eval",), ("prepare",), ("collect", "extra")])
def test_unrelated_stages_or_extra_arguments_cannot_start_work(tmp_path, args):
    result = _launch(tmp_path, *args)
    assert result.returncode == 2
    assert not (tmp_path / "results").exists()


def test_conflicting_inherited_run_dir_is_rejected(tmp_path):
    result = _launch(tmp_path, RUN_DIR="/existing/other/experiment")
    assert result.returncode == 2
    assert "Set RUN_NAME and RESULTS_ROOT" in result.stderr
    assert not (tmp_path / "results").exists()
