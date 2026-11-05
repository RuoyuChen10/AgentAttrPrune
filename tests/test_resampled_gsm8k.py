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
from experiments_attrprune import resampled_gsm8k as run
from experiments_attrprune.common import benchmark_data_identity, build_run_config, write_json
from test_pipeline_stub import CharacterTokenizer, VisibleTokenScorer
from test_random_order_mmlu import CompleteStub as BaseCompleteStub, FailingRescueScorer as BaseFailingRescueScorer
from AgentAttrPrune.llm import generation_observer
import torch


class CompleteStub(BaseCompleteStub):
    def __init__(self, tokenizer, **kwargs):
        kwargs.setdefault("answer", "12.5")
        super().__init__(tokenizer, **kwargs)

    async def agen(self, messages, *, max_tokens=512, temperature=0, num_comps=1):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("simulated inference failure")
        self.calls += 1
        output = f"Reasoning: Useful evidence verifies this arithmetic.\nThe answer is {self.answer}"
        rendered = self.tokenizer.apply_chat_template(messages)
        ids = self.tokenizer(rendered)["input_ids"]
        observer = generation_observer.get()
        if observer:
            observer({"model_name": "stub", "messages": messages, "rendered_prompt": rendered,
                      "input_ids": [ids], "attention_mask": [[1] * len(ids)],
                      "generated_ids": [self.tokenizer(output)["input_ids"]], "output_text": [output],
                      "finish_reasons": ["length" if self.truncate else "eos"],
                      "hit_max_new_tokens": [self.truncate],
                      "generation_kwargs": {"max_tokens": max_tokens, "temperature": temperature,
                                            "num_completions": num_comps}})
        return output


class FailingRescueScorer(BaseFailingRescueScorer):
    def generate_from_rendered_chat(self, rendered, *, max_new_tokens, do_sample):
        assert do_sample is False
        text = "Reasoning: Incorrect arithmetic.\nThe answer is 99"
        return {"input_ids": torch.tensor(self.tokenizer(rendered)["input_ids"]),
                "generated_answer_ids": torch.tensor(self.tokenizer(text)["input_ids"] + [0]),
                "output_text": text}


def write_dataset(data, count):
    data.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        (data / f"{split}.jsonl").write_text("".join(
            json.dumps({"question": f"Question {i}: calculate 25 divided by 2.",
                        "answer": "We divide the number. #### 12.5"}) + "\n"
            for i in range(count)))



@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(run, "NUM_QUESTIONS", 2)
    source, data = tmp_path / "baseline", tmp_path / "data"
    write_dataset(data,20)
    config = build_run_config(benchmark="gsm8k", model_name="stub", num_workers=4, num_rounds=2,
        temperature=0, max_new_tokens=2048, seed=0,
        calibration_split="train", evaluation_split="test",
        dataset_identity=benchmark_data_identity("gsm8k", data, splits=("train", "test")),
        reviewer_agent_ids={"math_solver":"Ab12", "mathematical_analyst":"Cd34",
                            "programming_expert":"Ef56", "inspector":"Gh78"})
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
            counts = np.zeros((2,4,4), dtype=int)
            if edge == "spatial": counts[:] = np.triu(np.ones((4,4),dtype=int),1)*2
            else: counts[1] = 2
            means = (np.arange(32).reshape(2,4,4) + 1) / 70
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
    write_dataset(setup.data,7473)
    base["dataset"] = benchmark_data_identity("gsm8k",setup.data,splits=("train","test"))
    base["fingerprint"] = run.config_fingerprint(base)
    monkeypatch.setattr(run,"_reference",lambda path:(copy.deepcopy(base),aggregate,settings,hashes))
    first = run.prepare_context(setup.args(1));second = run.prepare_context(setup.args(2))
    indices = [c.plan["selected_dataset_indices"] for c in (first,second)]
    for seed,cohort in enumerate(indices,1):
        assert cohort == sorted(random.Random(seed).sample(range(7473),40))
        assert len(set(cohort)) == 40
    assert indices[0] != indices[1]
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
    path=setup.data/"train.jsonl"
    path.write_text(path.read_text().replace("Question 0","Modified question 0"))
    with pytest.raises(ValueError,match="data fingerprint"):
        run.prepare_context(setup.args())


def test_collect_attribute_aggregate_and_resume_preserves_all_baseline_files(setup):
    before={p:p.read_bytes() for p in setup.source.rglob("*") if p.is_file()}
    ctx=run.prepare_context(setup.args());llm=CompleteStub(setup.tokenizer)
    summary=asyncio.run(run.collect_run(ctx,llm=llm))
    assert summary["num_rollouts"] == 2 and llm.calls == 18
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
    assert sum(aggregate["node_status_counts"].values()) == 18
    for edge in ("spatial","temporal"):
        expected=np.asarray(ctx.order_plan["candidate_masks"][edge])*2
        assert np.array_equal(aggregate["metrics"]["token_ratio"][edge]["opportunity_count"],expected)
    assert run.validate_ready_for_evaluation(ctx) == aggregate
    policies=run.load_policies(ctx)
    assert list(policies) == [25,50,60,70,80,90]
    assert [sum(np.asarray(p["masks"][edge]).sum() for edge in ("spatial","temporal"))
            for p in policies.values()] == [21,14,11,8,6,3]
    second=CompleteStub(setup.tokenizer);asyncio.run(run.collect_run(ctx,llm=second))
    assert second.calls == 0
    run.attribute_run(ctx,scorer=VisibleTokenScorer(setup.tokenizer))
    assert run.aggregate_run(ctx) == aggregate
    assert traces == {p:p.read_bytes() for p in traces}
    assert before == {p:p.read_bytes() for p in before}


def test_failed_rescues_are_observed_zero_and_trace_is_immutable(setup):
    ctx=run.prepare_context(setup.args())
    asyncio.run(run.collect_run(ctx,llm=CompleteStub(setup.tokenizer,answer="99")))
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


def test_fixed_order_dense_prompts_and_sources_match_original_gsm8k(setup):
    from AgentAttrPrune.graph import MultiAgentGraph
    from AgentAttrPrune.prompt import GSM8KPromptSet
    from AgentAttrPrune.tracing import GenerationTraceRecorder
    from experiments_attrprune.common import worker_specs_from_config, reviewer_agent_ids_from_config
    ctx = run.prepare_context(setup.args())
    asyncio.run(run.collect_run(ctx, llm=CompleteStub(setup.tokenizer)))
    rollout = ctx.plan["rollouts"][0]
    actual = run.load_json(run.trace_path_for(ctx, rollout))
    recorder = GenerationTraceRecorder(config=ctx.config, sample=ctx.samples[0])
    graph = MultiAgentGraph(CompleteStub(setup.tokenizer), worker_specs_from_config(ctx.config),
        num_rounds=2, prompt_set=GSM8KPromptSet, temperature=0, max_new_tokens=2048,
        reviewer_agent_ids=reviewer_agent_ids_from_config(ctx.config))
    asyncio.run(graph.run_sample(ctx.samples[0], recorder=recorder, phase="calibration", sample_index=0))
    assert len(actual["events"]) == len(recorder.events) == 9
    for fixed, original in zip(actual["events"], recorder.events):
        for key in ("agent_id", "round", "input_messages", "input_ids", "generated_ids", "output", "incoming"):
            assert fixed[key] == original[key]
    assert "The answer is <number>" in actual["events"][0]["input_messages"][0]["content"]


def test_failed_collection_resumes_only_missing_sample(setup):
    ctx=run.prepare_context(setup.args())
    with pytest.raises(RuntimeError):
        asyncio.run(run.collect_run(ctx,llm=CompleteStub(setup.tokenizer,fail_after=9)))
    state_path=ctx.paths.calibration/"collection_state.json"
    state=run.load_json(state_path)
    assert len(state["completed"]) == 1 and state["failures"] and not state["complete"]
    with pytest.raises((ValueError,FileNotFoundError)):
        run.attribute_run(ctx,scorer=VisibleTokenScorer(setup.tokenizer))
    # Recovery also handles an atomic trace whose checkpoint update was interrupted.
    trace_path=Path(state["completed"][0]["trace_json"]);before=trace_path.read_bytes()
    state["completed"]=[];write_json(state_path,state)
    llm=CompleteStub(setup.tokenizer);asyncio.run(run.collect_run(ctx,llm=llm))
    assert llm.calls == 9 and trace_path.read_bytes() == before
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
