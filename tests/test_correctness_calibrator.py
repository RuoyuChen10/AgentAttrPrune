import copy
import json

import pytest
import torch

from AgentAttrPrune.attribution import AttributionConfig, EdgeAttributionCalibrator
from AgentAttrPrune.prompt.mmlu import MMLUPromptSet
from test_correctness_attribution import RescueScorer, worker_event


class CalibrationScorer(RescueScorer):
    model_name = "stub"

    def score_teacher_forced_targets(self, prompt_input_ids, generated_ids,
                                    selected_output_token_indices, attention_mask, **kwargs):
        if prompt_input_ids.shape[1] == len(self.weights):
            return super().score_teacher_forced_targets(
                prompt_input_ids, generated_ids, selected_output_token_indices, attention_mask, **kwargs
            )
        score = attention_mask.float().sum(dim=1)
        return {"score": score, "logit_sum": score, "logit_mean": score,
                "logprob_sum": -score, "logprob_mean": -score}


def setup_run(tmp_path, *, correct=True, empty_peak=False, worker_correct=False):
    worker = worker_event()
    if worker_correct:
        worker["output"] = "Reasoning: useful analysis.\nDecision: B"
    scorer = CalibrationScorer(worker, correct=correct, empty_peak=empty_peak)
    tokenizer = scorer.tokenizer
    source_output = worker["incoming"][0]["output"]
    expert = {
        "event_id": 0, "agent_id": "expert", "role": "Knowlegable Expert", "round": 0,
        "incoming": [], "output": source_output,
        "input_messages": [{"role": "system", "content": MMLUPromptSet.get_analyze_constraint("Knowlegable Expert")},
                           {"role": "user", "content": "Question with Option A, Option B, Option C, Option D."}],
    }
    reviewer = {
        "event_id": 2, "agent_id": "dense_reviewer", "role": "Reviewer", "round": 1,
        "output": "D", "incoming": [{"agent_id": "critic", "round": 0, "edge_type": "spatial", "output": worker["output"]}],
        "input_messages": [{"role": "system", "content": "Choose a letter."},
                           {"role": "user", "content": "Anonymous z4Ax: " + worker["output"]}],
    }
    for event in (expert, worker, reviewer):
        event.update(
            model_name="stub", agent_type="AnalyzeAgent", finish_reason="eos",
            hit_max_new_tokens=False, llm_call={"max_tokens": 256},
            generated_ids=tokenizer(event["output"])["input_ids"] + [0],
        )
        rendered = tokenizer.apply_chat_template(event["input_messages"])
        event.update(rendered_prompt=rendered, input_ids=tokenizer(rendered)["input_ids"])
    trace = {
        "schema_version": 1,
        "config": {"benchmark": "mmlu", "sample_index": 0, "model_name": "stub",
                   "fingerprint": "frozen-test-run", "num_rounds": 1, "decoding": {"max_new_tokens": 256}},
        "topology": {"workers": [{"agent_id": "expert", "role": "Knowlegable Expert"},
                                  {"agent_id": "critic", "role": "Critic"}]},
        "sample": {"task": "Question with Option A, Option B, Option C, Option D.", "answer": "B"},
        "final": {"parsed_answer": "D", "gold_answer": "B"},
        "events": [expert, worker, reviewer],
    }
    path = tmp_path / "traces" / "sample_00000" / "trace.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(trace))
    config = AttributionConfig(attribution_mode="correctness-aware", aggregation_scope="all", batch_size=2)
    calibrator = EdgeAttributionCalibrator(tmp_path / "traces", tmp_path / "attribution",
                                          model_name="stub", config=config, scorer=scorer)
    return calibrator, scorer, trace, path


@pytest.mark.parametrize("correct,empty_peak", [(True, False), (False, False), (True, True)])
def test_frozen_trace_to_all_sample_policy_with_rescue_and_observed_zero(tmp_path, correct, empty_peak):
    calibrator, scorer, trace, source_path = setup_run(tmp_path, correct=correct, empty_peak=empty_peak)
    before = source_path.read_bytes()
    output = calibrator.run()
    aggregate = output["aggregate"]
    assert aggregate["num_aggregated_samples"] == 1
    assert aggregate["num_correct_samples"] == 0
    assert source_path.read_bytes() == before
    event_path = tmp_path / "attribution/sample_00000/event_001_r00_critic.json"
    event = json.loads(event_path.read_text())
    assert event["status"] == ("rescued" if correct else "rescue_failed")
    assert event["output"]["text"] == trace["events"][1]["output"]
    assert event["input"]["messages"] == trace["events"][1]["input_messages"]
    assert "Decision: D" in event["scoring_input"]["messages"][1]["content"]
    assert "Decision: B" not in event["rescue"]["verification"]["rendered_prompt"]
    edge = event["edge_scores"][0]
    assert edge["opportunity"] == 1
    assert (edge["token_ratio"] > 0) is (correct and not empty_peak)
    metric = aggregate["metrics"]["token_ratio"]["spatial"]
    assert metric["opportunity_count"][0][0][1] == 1
    assert metric["conditional_mean"][0][0][1] == edge["token_ratio"]
    assert event["rescue"]["edge_region_indices"] == (event["rescue"]["peak_region_indices"] if correct else [])
    assert len(scorer.calls) == 2
    assert "Gold-prefix Insertion / Deletion" in event_path.with_suffix(".insertion_curve.svg").read_text()
    assert "Actual verification input" in event_path.with_suffix(".html").read_text()
    reviewer = json.loads((event_path.parent / "event_002_r01_dense_reviewer.json").read_text())
    assert reviewer["correctness"]["route"] == "reviewer_diagnostic"
    assert reviewer["input"]["messages"] == trace["events"][2]["input_messages"]
    # Cached reruns regenerate reports, but do not replay successful/failed rescues.
    calibrator.run()
    assert len(scorer.calls) == 2
    assert "progress_json" in json.loads(event_path.read_text())["artifacts"]


def test_correct_worker_uses_original_attribution_even_when_reviewer_wrong(tmp_path):
    calibrator, scorer, _, _ = setup_run(tmp_path, worker_correct=True)
    result = calibrator.run()
    assert result["aggregate"]["num_correct_samples"] == 0
    assert scorer.calls == []
    event = json.loads((tmp_path / "attribution/sample_00000/event_001_r00_critic.json").read_text())
    assert event["correctness"]["route"] == "original"
    assert event["target"]["mode"] == "all"
    assert "rescue" not in event


def test_cache_identity_changes_with_gold_and_verification_settings(tmp_path):
    calibrator, _, trace, _ = setup_run(tmp_path)
    event = trace["events"][1]
    original = calibrator._event_configuration(event, trace)
    modified = copy.deepcopy(trace)
    modified["sample"]["answer"] = "C"
    assert calibrator._event_configuration(event, modified) != original
    modified = copy.deepcopy(trace)
    modified["config"]["decoding"]["max_new_tokens"] = 512
    assert calibrator._event_configuration(event, modified) != original


def test_rescue_execution_error_keeps_diagnostics_and_blocks_aggregate(tmp_path):
    calibrator, _, trace, path = setup_run(tmp_path)
    trace["config"]["decoding"]["max_new_tokens"] = 3
    path.write_text(json.dumps(trace))
    with pytest.raises(RuntimeError, match="truncated"):
        calibrator.run()
    assert not (tmp_path / "attribution/aggregate_edge_matrices.json").exists()
    manifest = json.loads((tmp_path / "attribution/manifest.json").read_text())
    assert manifest["complete"] is False
    assert len(manifest["failures"]) == 1
    diagnostic = tmp_path / "attribution/sample_00000/diagnostics/event_001_r00_critic"
    assert (diagnostic / "error.json").is_file()
    assert (diagnostic / "progress.json").is_file()
    with pytest.raises(ValueError, match="partial formal aggregate"):
        calibrator.run(aggregate_only=True)
