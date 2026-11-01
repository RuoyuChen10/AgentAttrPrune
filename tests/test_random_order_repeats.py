"""Budget changes preserve existing orders and require independent run identity."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

from experiments_attrprune import random_order_mmlu as experiment
from test_pipeline_stub import VisibleTokenScorer
from test_random_order_mmlu import CompleteStub, small_context


def test_default_budget_is_still_ten_repeats():
    assert experiment.parse_args([]).repeats == 10
    implicit = experiment.make_rollout_plan(list("abcde"), list(range(40)))
    explicit = experiment.make_rollout_plan(list("abcde"), list(range(40)), repeats=10)
    assert implicit == explicit
    assert implicit["num_rollouts"] == 400


@pytest.mark.parametrize("order_seed", [0, 3, 888])
def test_one_repeat_is_exactly_repeat_zero_of_ten(order_seed):
    agents = list("abcde")
    ten = experiment.make_rollout_plan(agents, list(range(40)), repeats=10, order_seed=order_seed)
    one = experiment.make_rollout_plan(agents, list(range(40)), repeats=1, order_seed=order_seed)
    assert one["num_questions"] == 40
    assert one["num_rollouts"] == 40
    assert one["repeats_per_question"] == 1
    assert one["rollouts"] == [r for r in ten["rollouts"] if r["repeat_index"] == 0]
    assert any(r["round_orders"][0] != r["round_orders"][1] for r in one["rollouts"])


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_cli_rejects_nonpositive_or_noninteger_repeats(value):
    with pytest.raises(SystemExit) as error:
        experiment.parse_args(["--repeats", value])
    assert error.value.code == 2


def test_explicit_ten_preserves_default_configuration_and_fingerprint(small_context, monkeypatch):
    _, _, original_args = small_context
    monkeypatch.setattr(experiment, "REPEATS", 10)
    argv = ["--stage", "collect", "--reference-run", str(original_args.reference_run),
            "--run-dir", str(original_args.run_dir), "--resume"]
    implicit = experiment.prepare_context(experiment.parse_args(argv))
    explicit = experiment.prepare_context(experiment.parse_args(argv + ["--repeats", "10"]))
    assert implicit.plan == explicit.plan
    assert implicit.config == explicit.config
    experiment.freeze_json(implicit.paths.run_config, implicit.config)
    experiment.freeze_json(implicit.paths.run_dir / "rollout_plan.json", implicit.plan)
    assert experiment.prepare_context(experiment.parse_args(argv)).config == explicit.config


def test_resume_rejects_changed_budget_but_new_run_keeps_same_first_orders(small_context):
    ctx, _, args = small_context
    experiment.freeze_json(ctx.paths.run_config, ctx.config)
    experiment.freeze_json(ctx.paths.run_dir / "rollout_plan.json", ctx.plan)
    single_args = copy.copy(args)
    single_args.repeats = 1
    with pytest.raises(ValueError, match="configuration/reference changed"):
        experiment.prepare_context(single_args)
    single_args.run_dir = args.run_dir.with_name("single")
    single = experiment.prepare_context(single_args)
    assert single.config["fingerprint"] != ctx.config["fingerprint"]
    assert single.plan["rollouts"] == [r for r in ctx.plan["rollouts"] if r["repeat_index"] == 0]
    assert experiment.load_json(ctx.paths.run_config) == ctx.config


def test_single_repeat_counts_come_from_plan_through_aggregation(small_context, monkeypatch):
    _, tokenizer, args = small_context
    args.repeats = 1
    ctx = experiment.prepare_context(args)
    # Changing the module default cannot change the already frozen run budget.
    monkeypatch.setattr(experiment, "REPEATS", 10)
    llm = CompleteStub(tokenizer)
    summary = asyncio.run(experiment.collect_run(ctx, llm=llm))
    selection = experiment.validate_selection(ctx)
    manifest = experiment.attribute_run(ctx, scorer=VisibleTokenScorer(tokenizer))
    aggregate = experiment.aggregate_run(ctx)
    assert llm.calls == 10  # Two questions, two workers x two rounds, one reviewer.
    assert len(manifest["samples"]) == 2
    for artifact in (summary, selection, aggregate):
        assert artifact["num_questions"] == 2
        assert artifact["repeats_per_question"] == 1
        assert artifact["num_rollouts"] == 2
    assert aggregate["num_aggregated_samples"] == 2
    assert not list(ctx.paths.run_dir.rglob("*policy*.json"))


def test_single_repeat_dry_run_reports_requested_budget_without_writing(small_context, capsys):
    ctx, _, args = small_context
    experiment.main(["--reference-run", str(args.reference_run), "--run-dir", str(args.run_dir),
                     "--repeats", "1", "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["repeats_per_question"] == 1
    assert result["num_rollouts"] == 2
    assert not ctx.paths.run_dir.exists()


def test_calibrate_forwards_repeats_to_every_stage(small_context, monkeypatch):
    _, _, args = small_context
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    commands = []
    monkeypatch.setattr(experiment.subprocess, "run", lambda command, **kwargs: commands.append(command))
    experiment.main(["--reference-run", str(args.reference_run), "--run-dir", str(args.run_dir),
                     "--repeats", "1", "--resume"])
    assert len(commands) == 4
    assert all(command[command.index("--repeats") + 1] == "1" for command in commands)
