"""CPU coverage for independent fixed-order calibration resamples."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from AgentAttrPrune.attribution.calibrator import AttributionConfig
from experiments_attrprune import resampled_mmlu as run
from experiments_attrprune.common import benchmark_data_identity, build_run_config, write_json
from test_pipeline_stub import CharacterTokenizer, VisibleTokenScorer
from test_random_order_mmlu import CompleteStub, FailingRescueScorer


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(run, "NUM_QUESTIONS", 2)
    source, data = tmp_path / "baseline", tmp_path / "data"
    for split in ("dev", "val"):
        folder = data / split
        folder.mkdir(parents=True)
        (folder / f"test_{split}.csv").write_text("\n".join(
            f"Question {i},Correct,Wrong,Wrong,Wrong,A" for i in range(20)) + "\n")
    config = build_run_config(model_name="stub", num_workers=5, num_rounds=2,
        temperature=0, max_new_tokens=2048, seed=0,
        dataset_identity=benchmark_data_identity("mmlu", data, splits=("dev", "val")),
        reviewer_agent_ids={"knowledgeable_expert":"Ab12", "critic":"Cd34", "mathematician":"Ef56",
                            "psychologist":"Gh78", "historian":"Ij90"})
    write_json(source / "run_config.json", config)
    settings = AttributionConfig(attribution_mode="correctness-aware", aggregation_scope="all",
                                 visualizations=False).to_dict()
    aggregate = {"agents": [w["agent_id"] for w in config["workers"]],
                 "agent_roles": [w["role"] for w in config["workers"]],
                 "agent_types": [w["agent_type"] for w in config["workers"]],
                 "num_rounds": 2, "matrix_orientation": "[round][source][target]",
                 "metrics": {}, "run_fingerprint": config["fingerprint"],
                 "num_aggregated_samples": 2, "num_correct_samples": 2,
                 "aggregation_scope": "all", "attribution_configuration": settings}
    for metric in ("token_ratio", "word_ratio", "binary"):
        aggregate["metrics"][metric] = {}
        for edge in ("spatial", "temporal"):
            counts = np.zeros((2,5,5), dtype=int)
            if edge == "spatial": counts[:] = np.triu(np.ones((5,5),dtype=int),1)*2
            else: counts[1] = 2
            means = (np.arange(50).reshape(2,5,5) + 1) / 70
            means[counts == 0] = 0
            aggregate["metrics"][metric][edge] = {
                "sum": (means * counts).tolist(), "opportunity_count": counts.tolist(),
                "conditional_mean": means.tolist(), "candidate_count": counts.tolist(),
                "sample_mean": (means * counts / 2).tolist()}
    matrix_path = source / "attribution/aggregate_edge_matrices.json"
    write_json(matrix_path, aggregate)
    write_json(source / "attribution/manifest.json", {
        "complete":True,"aggregation_complete":True,"aggregate_sha256":run.file_sha(matrix_path),
        "failures":[],"skipped_incorrect":[],"configuration":settings,
        "samples":[{"sample_id":f"sample_{i:05d}"} for i in range(2)]})
    write_json(source / "calibration/selection_manifest.json",{
        "samples":[{"sample_index":i} for i in range(2)]})
    def args(seed=1, *extra):
        return run.parse_args(["--calibration-seed",str(seed),"--reference-run",str(source),
                               "--stage","prepare","--resume","--limit","3",*extra])
    return SimpleNamespace(source=source,data=data,config=config,aggregate=aggregate,args=args,
                           tokenizer=CharacterTokenizer())


def test_independent_sampling_and_only_selection_changes(setup, monkeypatch):
    # Full-size CPU plans establish the production sampling rule and group overlap.
    monkeypatch.setattr(run,"NUM_QUESTIONS",40)
    original_reference = run._reference
    monkeypatch.setattr(run,"NUM_QUESTIONS",2)
    base, aggregate, settings, hashes = original_reference(setup.source)
    monkeypatch.setattr(run,"NUM_QUESTIONS",40)
    for split in ("dev","val"):
        (setup.data/split/f"test_{split}.csv").write_text("\n".join(
            f"Question {i},Correct,Wrong,Wrong,Wrong,A" for i in range(285))+"\n")
    base["dataset"] = benchmark_data_identity("mmlu",setup.data,splits=("dev","val"))
    base["fingerprint"] = run.config_fingerprint(base)
    monkeypatch.setattr(run,"_reference",lambda path:(copy.deepcopy(base),aggregate,settings,hashes))
    first = run.prepare_context(setup.args(1));second = run.prepare_context(setup.args(2))
    indices = [c.plan["selected_dataset_indices"] for c in (first,second)]
    for seed,cohort in enumerate(indices,1):
        assert cohort == sorted(random.Random(seed).sample(range(285),40))
        assert len(set(cohort)) == 40
    assert indices[0] != indices[1]
    assert set(indices[0]) & set(indices[1])  # Independence permits overlap.
    assert run.prepare_context(setup.args(1)).plan == first.plan
    assert first.paths.run_dir != second.paths.run_dir
    for ctx in (first,second):
        inherited = {k:v for k,v in ctx.config.items() if k not in {"fingerprint","resampled_experiment"}}
        assert inherited == {k:v for k,v in base.items() if k != "fingerprint"}
        assert ctx.config["seed"] == 0
        assert run.attribution_config(ctx).to_dict() == settings
        assert ctx.round_orders == [[w["agent_id"] for w in base["workers"]]]*2
        for item in ctx.plan["questions"]:
            assert item["sample_sha256"] == run.digest(item["sample"])
    assert not first.paths.run_dir.exists() and not second.paths.run_dir.exists()


def test_prepare_resume_freezes_seed_data_content_and_reference(setup):
    before = {p:p.read_bytes() for p in setup.source.rglob("*") if p.is_file()}
    ctx=run.prepare_context(setup.args())
    run.prepare_run(ctx)
    frozen={p:p.read_bytes() for p in ctx.paths.run_dir.rglob("*") if p.is_file()}
    run.prepare_run(run.prepare_context(setup.args()))
    assert frozen == {p:p.read_bytes() for p in frozen}
    assert before == {p:p.read_bytes() for p in before}
    with pytest.raises(ValueError,match="configuration or plan"):
        run.prepare_context(setup.args(2,"--run-dir",str(ctx.paths.run_dir)))
    with pytest.raises(ValueError,match="configuration or plan"):
        run.prepare_context(setup.args(1,"--limit","2"))
    path=setup.data/"dev/test_dev.csv"
    path.write_text(path.read_text().replace("Question 0","Modified question 0"))
    with pytest.raises(ValueError,match="data fingerprint"):
        run.prepare_context(setup.args())


def test_collect_attribute_aggregate_and_resume_preserves_all_baseline_files(setup):
    before={p:p.read_bytes() for p in setup.source.rglob("*") if p.is_file()}
    ctx=run.prepare_context(setup.args());llm=CompleteStub(setup.tokenizer)
    summary=asyncio.run(run.collect_run(ctx,llm=llm))
    assert summary["num_rollouts"] == 2 and llm.calls == 22
    traces={p:p.read_bytes() for p in ctx.paths.traces.rglob("trace.json")}
    for rollout in ctx.plan["rollouts"]:
        trace=run.load_json(run.trace_path_for(ctx,rollout))
        assert trace["config"]["dataset_index"] == rollout["dataset_index"]
        assert trace["config"]["sample_index"] == rollout["sample_index"]
        assert trace["config"]["round_orders"] == ctx.round_orders
        assert trace["sample"] == ctx.samples[rollout["sample_index"]]
    run.attribute_run(ctx,scorer=VisibleTokenScorer(setup.tokenizer))
    aggregate=run.aggregate_run(ctx)
    assert aggregate["num_aggregated_samples"] == 2
    assert sum(aggregate["node_status_counts"].values()) == 22
    for edge in ("spatial","temporal"):
        expected=np.asarray(ctx.order_plan["candidate_masks"][edge])*2
        assert np.array_equal(aggregate["metrics"]["token_ratio"][edge]["opportunity_count"],expected)
    assert run.validate_ready_for_evaluation(ctx) == aggregate
    policies=run.load_policies(ctx)
    assert list(policies) == [25,50,60,70,80,90]
    assert [sum(np.asarray(p["masks"][edge]).sum() for edge in ("spatial","temporal"))
            for p in policies.values()] == [34,23,18,14,9,5]
    second=CompleteStub(setup.tokenizer);asyncio.run(run.collect_run(ctx,llm=second))
    assert second.calls == 0
    run.attribute_run(ctx,scorer=VisibleTokenScorer(setup.tokenizer))
    assert run.aggregate_run(ctx) == aggregate
    assert traces == {p:p.read_bytes() for p in traces}
    assert before == {p:p.read_bytes() for p in before}


def test_failed_rescues_are_observed_zero_and_trace_is_immutable(setup):
    ctx=run.prepare_context(setup.args())
    asyncio.run(run.collect_run(ctx,llm=CompleteStub(setup.tokenizer,answer="B")))
    before={p:p.read_bytes() for p in ctx.paths.traces.rglob("trace.json")}
    run.attribute_run(ctx,scorer=FailingRescueScorer(setup.tokenizer))
    aggregate=run.aggregate_run(ctx)
    assert aggregate["num_correct_samples"] == 0
    assert aggregate["node_status_counts"]["rescue_failed"] > 0
    for edge in ("spatial","temporal"):
        assert not np.asarray(aggregate["metrics"]["token_ratio"][edge]["sum"]).any()
        assert np.array_equal(aggregate["metrics"]["token_ratio"][edge]["opportunity_count"],
                              np.asarray(ctx.order_plan["candidate_masks"][edge])*2)
    assert before == {p:p.read_bytes() for p in before}


def test_failed_collection_resumes_only_missing_sample(setup):
    ctx=run.prepare_context(setup.args())
    with pytest.raises(RuntimeError):
        asyncio.run(run.collect_run(ctx,llm=CompleteStub(setup.tokenizer,fail_after=11)))
    state_path=ctx.paths.calibration/"collection_state.json"
    state=run.load_json(state_path)
    assert len(state["completed"]) == 1 and state["failures"] and not state["complete"]
    with pytest.raises((ValueError,FileNotFoundError)):
        run.attribute_run(ctx,scorer=VisibleTokenScorer(setup.tokenizer))
    # Recovery also handles an atomic trace whose checkpoint update was interrupted.
    trace_path=Path(state["completed"][0]["trace_json"]);before=trace_path.read_bytes()
    state["completed"]=[];write_json(state_path,state)
    llm=CompleteStub(setup.tokenizer);asyncio.run(run.collect_run(ctx,llm=llm))
    assert llm.calls == 11 and trace_path.read_bytes() == before
    assert not run.load_json(state_path)["failures"]


def test_truncation_never_enters_formal_collection(setup):
    ctx=run.prepare_context(setup.args())
    with pytest.raises(ValueError,match="incomplete"):
        asyncio.run(run.collect_run(ctx,llm=CompleteStub(setup.tokenizer,truncate=True)))
    assert not list(ctx.paths.traces.glob("sample_*/trace.json"))
    assert not (ctx.paths.calibration/"selection_manifest.json").exists()


@pytest.mark.parametrize("target",["plan","trace","duplicate_selection","missing_trace","foreign_trace"])
def test_changed_collection_identity_blocks_resume_without_inference(setup,target):
    ctx=run.prepare_context(setup.args());asyncio.run(run.collect_run(ctx,llm=CompleteStub(setup.tokenizer)))
    path=next(ctx.paths.traces.glob("sample_*/trace.json"))
    if target=="plan":
        plan_path=ctx.paths.run_dir/"calibration_plan.json";plan=run.load_json(plan_path)
        plan["selected_dataset_indices"].reverse();write_json(plan_path,plan)
        with pytest.raises(ValueError,match="configuration or plan"):
            run.prepare_context(setup.args())
        return
    if target=="trace":path.write_text(path.read_text()+"\n")
    elif target=="missing_trace":path.unlink()
    elif target=="foreign_trace":write_json(ctx.paths.traces/"sample_foreign/trace.json",run.load_json(path))
    else:
        manifest_path=ctx.paths.calibration/"selection_manifest.json";value=run.load_json(manifest_path)
        value["samples"][-1]=value["samples"][0];write_json(manifest_path,value)
    llm=CompleteStub(setup.tokenizer)
    with pytest.raises((ValueError,FileNotFoundError)):
        asyncio.run(run.collect_run(ctx,llm=llm))
    assert llm.calls == 0


@pytest.mark.parametrize("target",["missing_node","incomplete_index","changed_matrix","changed_event_settings","duplicate_manifest","failed_manifest"])
def test_partial_attribution_blocks_aggregate_and_policies(setup,target):
    ctx=run.prepare_context(setup.args());asyncio.run(run.collect_run(ctx,llm=CompleteStub(setup.tokenizer)))
    run.attribute_run(ctx,scorer=VisibleTokenScorer(setup.tokenizer))
    manifest_path=ctx.paths.attribution/"manifest.json";manifest=run.load_json(manifest_path)
    entry=manifest["samples"][0]
    if target=="missing_node":
        index=run.load_json(Path(entry["node_attribution_index"]));Path(index["events"][0]["json"]).unlink()
    elif target=="incomplete_index":
        path=Path(entry["node_attribution_index"]);value=run.load_json(path);value["complete"]=False;write_json(path,value)
    elif target=="changed_matrix":
        path=Path(entry["sample_matrix"]);value=run.load_json(path)
        value["matrices"]["token_ratio"]["temporal"][1][0][0]=0.987;write_json(path,value)
    elif target=="changed_event_settings":
        index=run.load_json(Path(entry["node_attribution_index"]))
        path=Path(index["events"][0]["json"]);value=run.load_json(path)
        value["configuration"]["patience"]=999;write_json(path,value)
    elif target=="duplicate_manifest":
        manifest["samples"].append(manifest["samples"][0]);write_json(manifest_path,manifest)
    else:manifest["failures"]=[{"error":"OOM"}];write_json(manifest_path,manifest)
    with pytest.raises((ValueError,FileNotFoundError)):
        run.aggregate_run(ctx)
    assert run.load_json(manifest_path)["aggregation_complete"] is False
    assert not (ctx.paths.run_dir/"policies").exists()


def test_frozen_reference_changes_rejected(setup):
    ctx=run.prepare_context(setup.args());run.prepare_run(ctx)
    path=setup.source/"run_config.json";path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError,match="reference artifact changed"):
        run.prepare_run(ctx)
    with pytest.raises(ValueError,match="configuration or plan"):
        run.prepare_context(setup.args())


def test_cpu_dry_run_gpu_guard_and_output_isolation(setup,monkeypatch,capsys):
    import AgentAttrPrune.llm as llms
    monkeypatch.setattr(llms,"HFChat",lambda *a,**kw:pytest.fail("CPU check loaded GPU model"))
    command=["--calibration-seed","1","--reference-run",str(setup.source),"--limit","3"]
    run.main([*command,"--dry-run"]);value=json.loads(capsys.readouterr().out)
    assert value["percentages"] == [25,50,60,70,80,90] and value["generation_seed"] == 0
    assert not Path(value["run_dir"]).exists()
    with pytest.raises(ValueError,match="CUDA_VISIBLE_DEVICES"):
        run.main(command)
    assert not Path(value["run_dir"]).exists()
    with pytest.raises(ValueError,match="independent"):
        run.prepare_context(setup.args(1,"--run-dir",str(setup.source)))


def test_all_releases_model_memory_between_stages_and_rates(setup,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","6,7")
    calls=[];monkeypatch.setattr(run.subprocess,"run",lambda command,**kwargs:calls.append((command,kwargs)))
    run.main(["--calibration-seed","1","--reference-run",str(setup.source),"--limit","3","--resume"])
    assert [cmd[cmd.index("--stage")+1] for cmd,_ in calls] == [
        "prepare","collect","attribute","aggregate",*(["eval"]*6),"summarize"]
    assert [cmd[cmd.index("--pruning-percent")+1] for cmd,_ in calls if "--pruning-percent" in cmd] == [
        "25","50","60","70","80","90"]
    assert all(options["check"] and "--resume" in cmd for cmd,options in calls)
