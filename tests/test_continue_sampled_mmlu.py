"""CPU checks for preserving serial answers while batching unfinished questions."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from AgentAttrPrune.pruning.sampled import sample_graph
from experiments_attrprune import continue_sampled_mmlu as continuation
from experiments_attrprune import evaluate_sampled_mmlu as legacy
from experiments_attrprune.common import load_json, write_json
from test_evaluate_sampled_mmlu import setup as sampled_setup
from test_random_order_mmlu import CompleteStub


class BatchStub(CompleteStub):
    """Tracks concurrent requests and exercises per-task trace contexts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = []
        self.active = 0
        self.max_active = 0

    @asynccontextmanager
    async def async_batching(self, size):
        self.batches.append(size)
        yield

    async def agen(self, *args, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.002)
            return await super().agen(*args, **kwargs)
        finally:
            self.active -= 1


def snapshot(root):
    return {path.relative_to(root): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def run(args, llm=None):
    return asyncio.run(continuation.evaluate(args, llm=llm))


@pytest.fixture
def partial(sampled_setup, request):
    args = sampled_setup.args("--pruning-percent", "60", "--limit", "6",
                              "--pruning-scope", getattr(request, "param", "global"))
    with pytest.raises(RuntimeError, match="failed after"):
        asyncio.run(legacy.evaluate(args, llm=CompleteStub(sampled_setup.tokenizer, fail_after=22)))
    ctx = legacy.prepare_context(args)
    originals = legacy._read_jsonl(ctx.output_dir / "predictions.jsonl")
    assert len(originals) == 2

    def continuation_args(*extra):
        return continuation.parse_args([
            "--source-evaluation", str(ctx.output_dir), "--eval-batch-size", "4",
            "--resume", *extra,
        ])

    return SimpleNamespace(source=ctx.output_dir, legacy=ctx, originals=originals,
        frozen=snapshot(ctx.output_dir), tokenizer=sampled_setup.tokenizer,
        args=continuation_args)


def test_dry_run_and_prepare_freeze_import_without_loading_model(partial, monkeypatch):
    import AgentAttrPrune.llm as llm_module
    monkeypatch.setattr(llm_module, "HFChat", lambda *a, **k: pytest.fail("Preparation loaded a model"))
    ctx = continuation.prepare_context(partial.args())
    assert ctx.imported_records == {r["sample_index"]: r for r in partial.originals}
    assert ctx.manifest["imported_sample_count"] == 2
    assert ctx.manifest["new_eval_batch_size"] == 4
    assert ctx.current.config["fingerprint"] != ctx.legacy.config["fingerprint"]
    assert ctx.current.policy == ctx.legacy.policy
    assert ctx.current.selection == ctx.legacy.selection
    assert not ctx.output_dir.exists()
    run(partial.args("--dry-run"))
    assert not ctx.output_dir.exists()
    prepared = run(partial.args("--prepare-only"))
    assert prepared["prepared"]
    assert ctx.output_dir.is_dir()
    frozen = snapshot(ctx.output_dir)
    run(partial.args("--prepare-only"))
    assert snapshot(ctx.output_dir) == frozen
    assert snapshot(partial.source) == partial.frozen


@pytest.mark.parametrize("partial", ["global", "uniform"], indirect=True)
def test_batch4_runs_only_missing_questions_preserving_sampling_and_answers(partial, monkeypatch):
    llm = BatchStub(partial.tokenizer)
    summary = run(partial.args(), llm)
    assert summary["complete"] and summary["total"] == summary["correct"] == 6
    assert summary["llm_calls"] == 66
    assert summary["eval_batch_size"] == "mixed"
    assert summary["continuation_eval_batch_size"] == 4
    assert summary["batch_size_counts"] == {"1": 2, "4": 4}
    assert llm.calls == 44 and llm.max_active == 4 and llm.active == 0
    assert llm.batches == [4]
    ctx = continuation.prepare_context(partial.args())
    records = legacy._read_jsonl(ctx.output_dir / "predictions.jsonl")
    assert len(records) == 6 and sorted(r["sample_index"] for r in records) == list(range(6))
    assert records[:2] == partial.originals
    for record in records:
        seed = legacy.sampling_seed(ctx.legacy.config["seed"], record["sample_index"])
        assert record["sampling_seed"] == seed
        assert record["realized_graph"] == sample_graph(ctx.legacy.policy, seed=seed)
        trace = load_json(record["trace_json"])
        assert trace["sample"] == ctx.legacy.samples[record["sample_index"]]
        assert len(trace["events"]) == 11
        assert trace["config"]["prompt"] == ctx.legacy.source_config["prompt"]
        assert trace["config"]["eval_batch_size"] == (1 if record["sample_index"] < 2 else 4)
    assert snapshot(partial.source) == partial.frozen
    frozen = snapshot(ctx.output_dir)
    import AgentAttrPrune.llm as llm_module
    monkeypatch.setattr(llm_module, "HFChat", lambda *a, **k: pytest.fail("Complete resume loaded a model"))
    assert run(partial.args()) == summary
    assert snapshot(ctx.output_dir) == frozen
    assert snapshot(partial.source) == partial.frozen


def test_original_writer_lock_blocks_reading_a_moving_checkpoint(partial):
    with (partial.source / ".random_order.lock").open("rb+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises((RuntimeError, BlockingIOError)):
            continuation.prepare_context(partial.args())
    assert snapshot(partial.source) == partial.frozen


@pytest.mark.parametrize("target", ["trace", "prediction", "attempts", "policy", "selection", "config"])
def test_source_change_after_prepare_is_rejected_before_any_generation(partial, target):
    run(partial.args("--prepare-only"))
    if target == "trace":
        path = Path(partial.originals[0]["trace_json"])
        value = load_json(path)
        value["events"][-1]["output"] = "B"
        write_json(path, value)
    elif target == "prediction":
        path = partial.source / "predictions.jsonl"
        path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    elif target == "attempts":
        path = partial.source / "attempts.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps({"sample_index": 2, "status": "started"}) + "\n")
    else:
        name = {"policy": "sampling_policy.json", "selection": "selection_manifest.json",
                "config": "evaluation_config.json"}[target]
        path = partial.source / name
        value = load_json(path)
        value["changed_after_migration"] = True
        write_json(path, value)
    llm = BatchStub(partial.tokenizer)
    with pytest.raises(ValueError):
        run(partial.args(), llm)
    assert llm.calls == 0


@pytest.mark.parametrize("target", ["source", "child"])
def test_continuation_cannot_write_inside_legacy_output(partial, target):
    output = partial.source if target == "source" else partial.source / "continuation"
    with pytest.raises(ValueError):
        run(partial.args("--output-dir", str(output), "--prepare-only"))
    assert snapshot(partial.source) == partial.frozen


def test_foreign_destination_is_preserved(partial, tmp_path):
    output = tmp_path / "foreign"
    output.mkdir()
    (output / "important.txt").write_text("preserve me")
    with pytest.raises((ValueError, FileExistsError)):
        run(partial.args("--output-dir", str(output), "--prepare-only"))
    assert snapshot(output) == {Path("important.txt"): b"preserve me"}


def test_batch_failure_never_becomes_an_answer_and_resume_runs_missing_only(partial):
    failed = BatchStub(partial.tokenizer, fail_after=8)
    with pytest.raises(RuntimeError):
        run(partial.args(), failed)
    assert failed.active == 0
    ctx = continuation.prepare_context(partial.args())
    assert not (ctx.output_dir / "evaluation_summary.json").exists()
    records = legacy._read_jsonl(ctx.output_dir / "predictions.jsonl")
    assert records == partial.originals
    assert load_json(ctx.output_dir / "progress.json")["complete"] is False
    resumed = BatchStub(partial.tokenizer)
    summary = run(partial.args(), resumed)
    assert summary["complete"] and summary["total"] == 6
    assert summary["failed_attempts"] >= 1
    assert resumed.calls == 44
    assert snapshot(partial.source) == partial.frozen


def test_successful_orphan_continuation_traces_recover_without_replaying(partial, monkeypatch):
    original_writer = continuation._write_jsonl_atomic
    interrupted = False

    def interrupt_after_success(path, records):
        nonlocal interrupted
        if path.name == "predictions.jsonl" and len(records) > 2 and not interrupted:
            interrupted = True
            raise RuntimeError("checkpoint interrupted after complete trace")
        return original_writer(path, records)

    monkeypatch.setattr(continuation, "_write_jsonl_atomic", interrupt_after_success)
    with pytest.raises(RuntimeError, match="checkpoint interrupted"):
        run(partial.args(), BatchStub(partial.tokenizer))
    ctx = continuation.prepare_context(partial.args())
    traces = {path: path.read_bytes() for path in (ctx.output_dir / "traces").glob("sample_*/trace.json")}
    assert traces and not (ctx.output_dir / "evaluation_summary.json").exists()
    monkeypatch.setattr(continuation, "_write_jsonl_atomic", original_writer)
    llm = BatchStub(partial.tokenizer)
    summary = run(partial.args(), llm)
    assert summary["complete"] and summary["total"] == 6
    assert llm.calls == (4 - len(traces)) * 11
    assert all(path.read_bytes() == data for path, data in traces.items())
    attempts = legacy._read_jsonl(ctx.output_dir / "attempts.jsonl")
    assert any(attempt["status"] == "recovered" for attempt in attempts)
    assert snapshot(partial.source) == partial.frozen


def test_original_orphan_trace_is_imported_without_rewriting_legacy_checkpoint(partial):
    # A serial process can stop after its trace is committed but before its
    # predictions checkpoint is replaced. Preserve this successful answer too.
    legacy._write_jsonl_atomic(partial.source / "predictions.jsonl", partial.originals[:1])
    frozen = snapshot(partial.source)
    llm = BatchStub(partial.tokenizer)
    summary = run(partial.args(), llm)
    assert summary["batch_size_counts"] == {"1": 2, "4": 4}
    assert llm.calls == 44
    records = legacy._read_jsonl(Path(summary["predictions_jsonl"]))
    assert records[:2] == partial.originals
    assert snapshot(partial.source) == frozen


@pytest.mark.parametrize("target", ["duplicate", "missing_trace", "unknown_trace", "changed_trace", "config"])
def test_new_checkpoint_corruption_cannot_be_published_as_complete(partial, target):
    summary = run(partial.args(), BatchStub(partial.tokenizer))
    output = Path(summary["predictions_jsonl"]).parent
    records = legacy._read_jsonl(output / "predictions.jsonl")
    if target == "duplicate":
        legacy._write_jsonl_atomic(output / "predictions.jsonl", records + [records[-1]])
    elif target == "missing_trace":
        Path(records[-1]["trace_json"]).unlink()
    elif target == "unknown_trace":
        trace = load_json(records[-1]["trace_json"])
        write_json(output / "traces/sample_99999/trace.json", trace)
    elif target == "changed_trace":
        trace = load_json(records[-1]["trace_json"])
        trace["events"][-1]["output"] = "B"
        write_json(Path(records[-1]["trace_json"]), trace)
    else:
        config = load_json(output / "evaluation_config.json")
        config["eval_batch_size"] = 2
        write_json(output / "evaluation_config.json", config)
    llm = BatchStub(partial.tokenizer)
    with pytest.raises((ValueError, FileNotFoundError)):
        run(partial.args(), llm)
    assert llm.calls == 0
    assert snapshot(partial.source) == partial.frozen
