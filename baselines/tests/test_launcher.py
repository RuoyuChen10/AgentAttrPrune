"""Exercise process isolation and path handling without model dependencies."""
import importlib.util
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("baseline_launcher", Path(__file__).resolve().parents[1] / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "baselines"
    shared = tmp_path / "shared" / "experiments_attrprune"
    shared.mkdir(parents=True)
    for name in ("baseline_mmlu_alignment.py", "baseline_gsm8k_alignment.py"):
        (shared / name).touch()
    for method, benchmarks in module.METHODS.items():
        package = root / method / "experiments"
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
        for name in benchmarks.values():
            (package / (name.rsplit(".", 1)[1] + ".py")).write_text(
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "print('CHILD ' + json.dumps({'method': Path.cwd().name, 'cwd': os.getcwd(), "
                "'argv': sys.argv[1:], 'path': os.environ['PYTHONPATH'], "
                "'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'), "
                "'shared': os.environ['BASELINES_ATTRPRUNE_ROOT']}))\n"
            )
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setenv("BASELINES_ATTRPRUNE_ROOT", str(shared.parent))
    monkeypatch.setenv("PYTHONPATH", "/old/baseline/experiments")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.chdir(tmp_path)
    return module, tmp_path


def test_all_runs_isolated_children_and_fixes_batch4(launcher, capfd):
    module, tmp = launcher
    assert module.main(["all", "mmlu", "--dry-run", "--output-root", "my outputs"]) == 0
    children = [json.loads(line[6:]) for line in capfd.readouterr().out.splitlines() if line.startswith("CHILD ")]
    assert [row["method"] for row in children] == list(module.METHODS)
    for row in children:
        assert row["path"].split(os.pathsep) == [str(module.ROOT / row["method"]), str(tmp / "shared")]
        assert row["cuda"] == ""
        args = row["argv"]
        assert args[args.index("--run-dir") + 1] == str(tmp / "my outputs" / row["method"] / "mmlu/aligned_seed0_t0_batch4")
        assert "--dry-run" in args
    dropout = children[1]["argv"]
    assert dropout[dropout.index("--eval-batch-size") + 1] == "4"
    assert dropout[dropout.index("--llm-batch-size") + 1] == "4"
    assert not (tmp / "my outputs").exists()


def test_all_stops_after_child_failure(launcher, capfd):
    module, _ = launcher
    target = module.ROOT / "agentdropout/experiments/run_gsm8k_attrprune_seed0.py"
    target.write_text("raise SystemExit(7)\n")
    assert module.main(["all", "gsm8k", "--dry-run"]) == 7
    assert "[safesieve/gsm8k]" not in capfd.readouterr().out


def test_relative_paths_resolved_before_child_changes_directory(launcher, capfd):
    module, tmp = launcher
    assert module.main(["agentprune", "gsm8k", "--stage", "prepare", "--run-dir", "my run",
                        "--reference-run", "reference", "--resume"]) == 0
    row = next(json.loads(line[6:]) for line in capfd.readouterr().out.splitlines() if line.startswith("CHILD "))
    args = row["argv"]
    assert args[args.index("--reference-run") + 1] == str(tmp / "reference")
    assert args[args.index("--run-dir") + 1] == str(tmp / "my run")
    assert "--resume" in args
    assert row["cuda"] == ""


def test_refuses_shared_output_directory_for_all(launcher):
    module, _ = launcher
    with pytest.raises(SystemExit) as error:
        module.main(["all", "mmlu", "--run-dir", "collision", "--dry-run"])
    assert error.value.code == 2
