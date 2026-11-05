"""Launch GSM8K train40 cohorts without changing generation seeds."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_gsm8k_train40_resampling.sh"


def invoke(tmp_path, *seeds, dry=True, **options):
    shim = tmp_path / "python shim"
    shim.write_text(f"#!{sys.executable}\nimport json,sys\nprint('SHIM '+json.dumps(sys.argv[1:]))\n")
    shim.chmod(0o755)
    env = dict(os.environ)
    for name in ("CUDA_VISIBLE_DEVICES", "STAGE", "REFERENCE_RUN", "RUNS_ROOT"):
        env.pop(name, None)
    env.update(PYTHON_BIN=str(shim), LOG_ROOT=str(tmp_path / "logs"),
               DRY_RUN="1" if dry else "0")
    env.update(options)
    return subprocess.run(["bash", str(SCRIPT), *seeds], env=env,
                          text=True, capture_output=True)


def test_two_cohorts_six_rates_isolate_processes_and_checkpoints(tmp_path):
    result = invoke(tmp_path, "1", "2")
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    runner = commands
    assert len(runner) == 24
    for start, seed in ((0, "1"), (12, "2")):
        cohort = runner[start:start+12]
        assert [c[c.index("--stage")+1] for c in cohort] == [
            "prepare", "collect", "attribute", "aggregate", "policies", *(["eval"]*6), "summarize"]
        assert [c[c.index("--pruning-percent")+1] for c in cohort if "--pruning-percent" in c] == ["25", "50", "60", "70", "80", "90"]
        assert all(c[c.index("--calibration-seed")+1] == seed for c in cohort)
        assert all("--seed" not in c and "--resume" in c and "--dry-run" in c for c in cohort)
        assert all(c[c.index("--run-dir")+1].endswith(f"train40_resample_seed{seed}_correctness_aware") for c in cohort)
    assert all("experiments_attrprune.resampled_curve" not in c for c in commands)
    assert not (tmp_path / "logs").exists()


def test_cpu_prepare_preserves_literal_paths_and_logs_each_seed(tmp_path):
    run_root = str(tmp_path / "runs $literal `name`")
    result = invoke(tmp_path, "1", "2", dry=False, STAGE="prepare", RUNS_ROOT=run_root)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line[5:]) for line in result.stdout.splitlines() if line.startswith("SHIM ")]
    assert len(calls) == 2
    assert all(c[c.index("--stage")+1] == "prepare" for c in calls)
    assert all(c[c.index("--run-dir")+1].startswith(run_root+"/") for c in calls)
    assert len(list((tmp_path / "logs").glob("seed_*/*.log"))) == 2


def test_failed_stage_stops_later_rates_and_second_seed(tmp_path):
    failure = tmp_path / "failure"
    failure.write_text("#!/bin/sh\nexit 7\n")
    failure.chmod(0o755)
    result = invoke(tmp_path, "1", "2", dry=False, CUDA_VISIBLE_DEVICES="6,7", PYTHON_BIN=str(failure))
    assert result.returncode == 7 and "exit_status=7" in result.stdout
    assert "stage=collect" not in result.stdout
    assert not (tmp_path / "logs/seed_2").exists()


@pytest.mark.parametrize("seeds", [("0",), ("1", "1"), ("3",)])
def test_seed_validation_prevents_baseline_overwrite(tmp_path, seeds):
    result = invoke(tmp_path, *seeds, dry=False, CUDA_VISIBLE_DEVICES="6,7")
    assert result.returncode == 2 and "SHIM " not in result.stdout


def test_gpu_stages_require_explicit_devices(tmp_path):
    result = invoke(tmp_path, "1", dry=False)
    assert result.returncode == 2 and "CUDA_VISIBLE_DEVICES" in result.stderr


def test_eval_only_preserves_all_rates_for_requested_cohort(tmp_path):
    result = invoke(tmp_path, "2", STAGE="eval")
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 6
    assert all(c[c.index("--calibration-seed")+1] == "2" for c in commands)
