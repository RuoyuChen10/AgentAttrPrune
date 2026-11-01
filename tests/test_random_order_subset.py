"""CPU-only tests for safe repeat_00 trace and attribution-cache relocation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from experiments_attrprune import random_order_mmlu as experiment
from experiments_attrprune import random_order_subset as subset
from experiments_attrprune.common import load_json, write_json
from test_random_order_mmlu import CompleteStub, FailingRescueScorer, small_context
from test_pipeline_stub import VisibleTokenScorer


@pytest.fixture
def source_run(small_context, monkeypatch):
    ctx, tokenizer, _ = small_context
    asyncio.run(experiment.collect_run(ctx, llm=CompleteStub(tokenizer)))
    experiment.attribute_run(ctx, scorer=VisibleTokenScorer(tokenizer))

    def no_inference(*args, **kwargs):
        raise AssertionError("Importer must never load or call a model")

    monkeypatch.setattr("AgentAttrPrune.llm.HFChat", no_inference)
    monkeypatch.setattr("AgentAttrPrune.attribution.scoring.TeacherForcedTextScorer.from_pretrained", no_inference)
    return ctx, tokenizer


def _hashes(root):
    return {p: experiment.file_sha(p) for p in root.rglob("*") if p.is_file()}


def test_import_preserves_generation_and_reuses_caches_without_inference(source_run, tmp_path):
    source, tokenizer = source_run
    original = _hashes(source.paths.run_dir)
    target_dir = tmp_path / "once"
    dry = subset.import_repeat_zero(source.paths.run_dir, target_dir, dry_run=True)
    assert dry["num_rollouts"] == 2 and dry["num_reusable_events"] == 10
    assert not target_dir.exists()
    report = subset.import_repeat_zero(source.paths.run_dir, target_dir)
    assert report["complete"] is True
    assert report["num_complete_attribution_samples"] == 2
    assert report["skipped_caches"] == []
    assert report["identity"]["selection_rule"] == {"repeat_index": 0, "depends_on_correctness": False}
    target = subset._context(source.config, target_dir, 1)
    for rollout in target.plan["rollouts"]:
        old = load_json(experiment.trace_path_for(source, rollout))
        new = load_json(experiment.trace_path_for(target, rollout))
        assert new["events"] == old["events"]
        assert new["sample"] == old["sample"] and new["final"] == old["final"]
        assert new["config"]["round_orders"] == old["config"]["round_orders"]
        assert new["config"]["fingerprint"] != old["config"]["fingerprint"]
        assert new["source_provenance"]["sha256"] == experiment.file_sha(experiment.trace_path_for(source, rollout))
    scorer = VisibleTokenScorer(tokenizer)

    def should_be_cached(*args, **kwargs):
        raise AssertionError("Imported complete node caches must not be recomputed")

    scorer.score_teacher_forced_targets = should_be_cached
    assert experiment.attribute_run(target, scorer=scorer)["complete"] is True
    assert experiment.aggregate_run(target)["num_rollouts"] == 2
    assert _hashes(source.paths.run_dir) == original


def test_resume_preserves_destination_event_bytes_and_original_provenance(source_run, tmp_path):
    source, _ = source_run
    target = tmp_path / "once"
    first = subset.import_repeat_zero(source.paths.run_dir, target)
    event_hashes = {p: experiment.file_sha(p) for p in (target / "attribution").glob("sample_*/event_*.json")}
    with pytest.raises(FileExistsError, match="--resume"):
        subset.import_repeat_zero(source.paths.run_dir, target)
    second = subset.import_repeat_zero(source.paths.run_dir, target, resume=True)
    assert {p: experiment.file_sha(p) for p in event_hashes} == event_hashes
    for before, after in zip(first["event_provenance"], second["event_provenance"]):
        assert after["action"] == "preserved"
        assert after["source_event_json"] == before["source_event_json"]
        assert after["snapshot_sha256"] == before["snapshot_sha256"]


@pytest.mark.parametrize("interrupted_path", [subset.MANIFEST_NAME, "run_config.json", "rollout_plan.json"])
def test_import_recovers_each_initial_metadata_publication_window(source_run, tmp_path, monkeypatch, interrupted_path):
    source, _ = source_run
    target = tmp_path / "once"
    original_write = subset.write_json

    def write_then_interrupt(path, payload):
        result = original_write(path, payload)
        if Path(path).parent == target and Path(path).name == interrupted_path:
            raise RuntimeError("simulated interruption after atomic metadata write")
        return result

    monkeypatch.setattr(subset, "write_json", write_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        subset.import_repeat_zero(source.paths.run_dir, target)
    monkeypatch.setattr(subset, "write_json", original_write)
    report = subset.import_repeat_zero(source.paths.run_dir, target, resume=True)
    assert report["complete"] is True and report["num_complete_attribution_samples"] == 2


@pytest.mark.parametrize("artifact", ["trace", "plan", "selection"])
def test_frozen_source_changes_are_rejected_before_writing_destination(source_run, tmp_path, artifact):
    source, _ = source_run
    if artifact == "trace":
        path = next(source.paths.traces.glob("sample_*/trace.json"))
        path.write_text(path.read_text() + "\n")
    elif artifact == "plan":
        path = source.paths.run_dir / "rollout_plan.json"
        plan = load_json(path)
        plan["rollouts"][0]["round_orders"][0].reverse()
        write_json(path, plan)
    else:
        path = source.paths.calibration / "selection_manifest.json"
        selection = load_json(path)
        selection["samples"][0]["sha256"] = "changed"
        write_json(path, selection)
    with pytest.raises(ValueError):
        subset.import_repeat_zero(source.paths.run_dir, tmp_path / "once")
    assert not (tmp_path / "once").exists()


def test_destination_collision_and_changed_trace_never_overwrite(source_run, tmp_path):
    source, _ = source_run
    with pytest.raises(ValueError, match="independent"):
        subset.import_repeat_zero(source.paths.run_dir, source.paths.run_dir, resume=True)
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "keep.txt").write_text("user data")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        subset.import_repeat_zero(source.paths.run_dir, target, resume=True)
    assert (target / "keep.txt").read_text() == "user data"
    target = tmp_path / "once"
    subset.import_repeat_zero(source.paths.run_dir, target)
    path = next((target / "calibration/traces").glob("sample_*/trace.json"))
    trace = load_json(path)
    trace["events"][0]["output"] = "changed"
    write_json(path, trace)
    before = experiment.file_sha(path)
    with pytest.raises(ValueError, match="Destination trace differs"):
        subset.import_repeat_zero(source.paths.run_dir, target, resume=True)
    assert experiment.file_sha(path) == before


def test_stale_source_cache_is_skipped_and_destination_cache_is_never_replaced(source_run, tmp_path):
    source, _ = source_run
    path = next((source.paths.attribution / "sample_00000_repeat_00").glob("event_*.json"))
    cached = load_json(path)
    cached["configuration"]["event_sha256"] = "stale"
    write_json(path, cached)
    report = subset.import_repeat_zero(source.paths.run_dir, tmp_path / "once")
    assert report["num_reusable_events"] == 9
    assert report["num_complete_attribution_samples"] == 1
    assert len(report["skipped_caches"]) == 1
    assert not (tmp_path / "once/attribution" / path.parent.name / path.name).exists()
    target_cache = next((tmp_path / "once/attribution").glob("sample_*/event_*.json"))
    bad = load_json(target_cache)
    bad["configuration"]["target_recovery"] = 0.5
    write_json(target_cache, bad)
    checksum = experiment.file_sha(target_cache)
    with pytest.raises(ValueError, match="incompatible attribution configuration"):
        subset.import_repeat_zero(source.paths.run_dir, tmp_path / "once", resume=True)
    assert experiment.file_sha(target_cache) == checksum


def test_atomic_snapshot_detects_concurrent_source_replacement(tmp_path, monkeypatch):
    path = tmp_path / "source.json"
    write_json(path, {"version": 1})
    original = Path.read_bytes

    def changed_after_read(self):
        raw = original(self)
        if self == path:
            write_json(self, {"version": 2})
        return raw

    monkeypatch.setattr(Path, "read_bytes", changed_after_read)
    with pytest.raises(ValueError, match="changed during snapshot"):
        subset.stable_snapshot(path)


def test_resume_refreshes_sample_manifest_and_retains_new_cache_provenance_after_interruption(source_run, tmp_path, monkeypatch):
    source, _ = source_run
    source_cache = next((source.paths.attribution / "sample_00001_repeat_00").glob("event_*.json"))
    raw = source_cache.read_bytes()
    source_cache.unlink()
    target = tmp_path / "once"
    first = subset.import_repeat_zero(source.paths.run_dir, target)
    assert first["num_complete_attribution_samples"] == 1
    manifest_path = target / "attribution/manifest.json"
    assert load_json(manifest_path)["complete"] is False
    source_cache.write_bytes(raw)
    destination_cache = target / "attribution" / source_cache.parent.name / source_cache.name
    original_write = subset.write_json

    def write_then_interrupt(path, payload):
        result = original_write(path, payload)
        if Path(path) == destination_cache:
            raise RuntimeError("interrupted after a newly discovered cache was written")
        return result

    monkeypatch.setattr(subset, "write_json", write_then_interrupt)
    with pytest.raises(RuntimeError, match="newly discovered cache"):
        subset.import_repeat_zero(source.paths.run_dir, target, resume=True)
    monkeypatch.setattr(subset, "write_json", original_write)
    final = subset.import_repeat_zero(source.paths.run_dir, target, resume=True)
    assert final["num_complete_attribution_samples"] == 2
    manifest = load_json(manifest_path)
    assert manifest["complete"] is True and len(manifest["samples"]) == 2
    entry = next(e for e in final["event_provenance"] if e["destination_event_json"] == str(destination_cache))
    assert entry["source_event_json"] == str(source_cache)
    assert entry["snapshot_sha256"] == experiment.file_sha(source_cache)


def test_resume_preserves_valid_publication_and_existing_failures(source_run, tmp_path):
    source, _ = source_run
    target = tmp_path / "once"
    subset.import_repeat_zero(source.paths.run_dir, target)
    ctx = subset._context(source.config, target, 1)
    experiment.aggregate_run(ctx)
    path = ctx.paths.attribution / "manifest.json"
    manifest = load_json(path)
    subset.import_repeat_zero(source.paths.run_dir, target, resume=True)
    assert load_json(path) == manifest
    manifest["failures"] = [{"sample_id": "sample_00001_repeat_00", "error": "earlier runtime failure"}]
    write_json(path, manifest)
    subset.import_repeat_zero(source.paths.run_dir, target, resume=True)
    after = load_json(path)
    assert after["failures"] == manifest["failures"]
    assert after["complete"] is False and after["aggregation_complete"] is False
    assert "aggregate_sha256" not in after


def test_failed_rescue_cache_and_progress_are_relocated_without_changing_scores(small_context, tmp_path):
    source, tokenizer, _ = small_context
    asyncio.run(experiment.collect_run(source, llm=CompleteStub(tokenizer, answer="B")))
    experiment.attribute_run(source, scorer=FailingRescueScorer(tokenizer))
    report = subset.import_repeat_zero(source.paths.run_dir, tmp_path / "once")
    assert report["skipped_caches"] == []
    assert report["num_reusable_events"] == 10
    target = subset._context(source.config, tmp_path / "once", 1)
    for path in target.paths.attribution.glob("sample_*/event_*.json"):
        cached = load_json(path)
        if cached["status"] != "rescue_failed":
            continue
        old = load_json(source.paths.attribution / path.parent.name / path.name)
        assert cached["attribution"] == old["attribution"]
        assert cached["rescue"] == old["rescue"]
        assert cached["edge_scores"] == old["edge_scores"]
        progress_path = Path(cached["artifacts"]["progress_json"])
        assert target.paths.attribution in progress_path.parents
        assert load_json(progress_path)["configuration"] == cached["configuration"]
        assert cached["configuration"]["event_sha256"] != old["configuration"]["event_sha256"]
    scorer = FailingRescueScorer(tokenizer)
    scorer.score_teacher_forced_targets = lambda *a, **kw: pytest.fail("cache recomputed")
    scorer.generate_from_rendered_chat = lambda *a, **kw: pytest.fail("verification rerun")
    assert experiment.attribute_run(target, scorer=scorer)["complete"] is True
    aggregate = experiment.aggregate_run(target)
    assert aggregate["num_correct_samples"] == 0
    assert aggregate["num_aggregated_samples"] == 2
