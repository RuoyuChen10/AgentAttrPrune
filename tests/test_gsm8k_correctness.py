import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from AgentAttrPrune.attribution import AttributionConfig, EdgeAttributionCalibrator
from AgentAttrPrune.attribution.correctness import (
    answer_first_system_prompt,
    gold_prefix_target,
    normalize_gold_answer,
    parse_worker_answer,
    run_gold_prefix_rescue,
)
from AgentAttrPrune.attribution.edge import build_incoming_sentence_masks
from AgentAttrPrune.prompt.gsm8k import GSM8KPromptSet
from test_correctness_attribution import CharacterTokenizer
from test_correctness_calibrator import CalibrationScorer


@pytest.mark.parametrize("text, expected", [
    ("Reasoning mentions 99.\nThe answer is 12.500", "12.5"),
    ("**The answer is 1,200**", "1200"),
    ("**The answer is** **1,200**", "1200"),
    ("`The answer is -3/2`", "-1.5"),
    ("The answer is `12.5`", "12.5"),
    ("__The answer is__ _12.5_", "12.5"),
    ("`**The answer is 12.5**`", "12.5"),
    ("the ANSWER is -3/2", "-1.5"),
    ("The answer is +.50\nThe answer is 1/2", "0.5"),
    ("The answer is -0.000", "0"),
    ("The answer is 2/6", "1/3"),
    ("The answer is 123456789012345678901234567890.25", "123456789012345678901234567890.25"),
    ("The answer is 12.", "12"),
    ("The answer is 12.5.", "12.5"),
    ("Reasoning: The answer is 12.5", None),
    ("We calculate 12.5.", None),
    ("12.5", None),
    ("Answer: 12.5", None),
    ("The answer is 12.5 units", None),
    ("The answer is 12.5 or 13", None),
    ("The answer is 12.5\nThe answer is 13", None),
    ("The answer is 12.5\nThe answer is unknown", None),
    ("The answer is NaN", None),
    ("The answer is inf", None),
    ("The answer is 1/0", None),
    ("The answer is 12,34", None),
    ("The answer is 1_000", None),
    ("The answer is 2*3", None),
    ("The answer is 1**2", None),
    ("The answer is 1`2", None),
    ("The answer is `2*3`", None),
    ("**The answer is** **2*3**", None),
    ("The answer is 12\nThe answer is 1**2", None),
])
def test_gsm8k_strict_answer_parser(text, expected):
    assert parse_worker_answer(text, benchmark="gsm8k") == expected


@pytest.mark.parametrize("gold", ["", "unknown", "NaN", "Infinity", "1/0", "12,34", "The answer is 12"])
def test_invalid_numeric_gold_is_rejected(gold):
    with pytest.raises(ValueError, match="finite numeric literal"):
        normalize_gold_answer(gold, benchmark="gsm8k")


def test_numeric_prefix_scores_all_answer_tokens_and_preserves_worker_prompt():
    target = gold_prefix_target(CharacterTokenizer(), "-1,234.50", benchmark="gsm8k")
    assert target["text"] == "The answer is -1234.5"
    assert len(target["answer_token_indices"]) == len("-1234.5")
    assert target["selected_indices"] == list(range(len(target["text"])))
    assert target["answer_token_indices"] == list(range(len("The answer is "), len(target["text"])))
    original = GSM8KPromptSet.get_analyze_constraint("Inspector")
    transformed = answer_first_system_prompt(original, benchmark="gsm8k")
    assert "First write the result" in transformed
    assert "Then write concise reasoning" in transformed
    assert "End with the result" not in transformed
    assert original == GSM8KPromptSet.get_analyze_constraint("Inspector")
    assert "End with the result" in original


def worker_event(*, correct=False):
    source = "Useful evidence identifies the required operation.\nMisleading claim.\nThe answer is 99"
    incoming = [{"agent_id": "math_solver", "role": "Math Solver", "round": 0,
                 "edge_type": "spatial", "output": source}]
    return {
        "event_id": 1, "agent_id": "inspector", "role": "Inspector", "round": 0,
        "incoming": incoming,
        "output": "Reasoning from the original trace.\nThe answer is " + ("12.500" if correct else "100"),
        "input_messages": GSM8KPromptSet.worker_messages(
            role="Inspector", task="Consider the quantities 4 and 5.",
            spatial_incoming=incoming, temporal_incoming=[],
        ),
    }


class NumericScorer(CalibrationScorer):
    def __init__(self, event, *, correct=True, empty_peak=False):
        self.tokenizer = CharacterTokenizer()
        self.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[0, 999]))
        changed = copy.deepcopy(event)
        changed["input_messages"][0]["content"] = answer_first_system_prompt(
            changed["input_messages"][0]["content"], benchmark="gsm8k"
        )
        _, ids, _, spans, _ = build_incoming_sentence_masks(self.tokenizer, changed)
        self.weights = torch.zeros(len(ids))
        for span in spans:
            self.weights[span.token_indices[0]] = 3 if "Useful" in span.text and not empty_peak else -1
        self.correct = correct
        self.calls = []

    def generate_from_rendered_chat(self, rendered, *, max_new_tokens, do_sample):
        self.calls.append((rendered, max_new_tokens, do_sample))
        text = (
            "The answer is 100\nDiagnostic reasoning."
            if "First write the result" in rendered
            else "Original-format reasoning.\nThe answer is " + ("25/2" if self.correct else "99")
        )
        ids = (self.tokenizer(text)["input_ids"] + [0])[:max_new_tokens]
        return {"input_ids": torch.tensor(self.tokenizer(rendered)["input_ids"]),
                "generated_answer_ids": torch.tensor(ids), "output_text": self.tokenizer.decode(ids)}


def setup_run(tmp_path, *, correct=True, empty_peak=False, worker_correct=False):
    worker = worker_event(correct=worker_correct)
    scorer = NumericScorer(worker, correct=correct, empty_peak=empty_peak)
    source = {
        "event_id": 0, "agent_id": "math_solver", "role": "Math Solver", "round": 0,
        "incoming": [], "output": worker["incoming"][0]["output"],
        "input_messages": GSM8KPromptSet.worker_messages(
            role="Math Solver", task="Consider the quantities 4 and 5.",
            spatial_incoming=[], temporal_incoming=[],
        ),
    }
    reviewer = {
        "event_id": 2, "agent_id": "dense_reviewer", "role": "Reviewer", "round": 1,
        "output": "The answer is 99",
        "incoming": [{"agent_id": "inspector", "round": 0, "edge_type": "spatial", "output": worker["output"]}],
        "input_messages": [{"role": "system", "content": "Verify the number."},
                           {"role": "user", "content": "Anonymous z4Ax: " + worker["output"]}],
    }
    for event in (source, worker, reviewer):
        event.update(model_name="stub", agent_type="AnalyzeAgent", finish_reason="eos",
                     hit_max_new_tokens=False, llm_call={"max_tokens": 256},
                     generated_ids=scorer.tokenizer(event["output"])["input_ids"] + [0])
        rendered = scorer.tokenizer.apply_chat_template(event["input_messages"])
        event.update(rendered_prompt=rendered, input_ids=scorer.tokenizer(rendered)["input_ids"])
    trace = {
        "schema_version": 1,
        "config": {"benchmark": "gsm8k", "sample_index": 0, "model_name": "stub",
                   "fingerprint": "frozen-numeric-run", "num_rounds": 1, "decoding": {"max_new_tokens": 256},
                   "prompt": {"version": "gsm8k-test-contract"}},
        "topology": {"workers": [{"agent_id": "math_solver", "role": "Math Solver"},
                                  {"agent_id": "inspector", "role": "Inspector"}]},
        "sample": {"benchmark": "gsm8k", "task": "Consider the quantities 4 and 5.", "answer": "12.5"},
        "final": {"parsed_answer": "99", "gold_answer": "12.5"},
        "events": [source, worker, reviewer],
    }
    path = tmp_path / "traces/sample_00000/trace.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(trace))
    calibrator = EdgeAttributionCalibrator(
        tmp_path / "traces", tmp_path / "attribution", model_name="stub", scorer=scorer,
        config=AttributionConfig(attribution_mode="correctness-aware", aggregation_scope="all", batch_size=2),
    )
    return calibrator, scorer, trace, path


@pytest.mark.parametrize("correct,empty_peak", [(True, False), (False, False), (True, True)])
def test_numeric_rescue_frozen_trace_zero_denominator_and_cache(tmp_path, correct, empty_peak):
    calibrator, scorer, trace, trace_path = setup_run(tmp_path, correct=correct, empty_peak=empty_peak)
    before = trace_path.read_bytes()
    aggregate = calibrator.run()["aggregate"]
    assert aggregate["num_aggregated_samples"] == 1
    assert aggregate["num_correct_samples"] == 0
    assert trace_path.read_bytes() == before
    event_dir = tmp_path / "attribution/sample_00000"
    event = json.loads((event_dir / "event_001_r00_inspector.json").read_text())
    assert event["status"] == ("rescued" if correct else "rescue_failed")
    assert event["correctness"]["original_answer"] == "100"
    assert event["correctness"]["answer_field"] == "The answer is <number>"
    assert "original_decision" not in event["correctness"]
    assert event["output"]["text"] == trace["events"][1]["output"]
    assert event["input"]["messages"] == trace["events"][1]["input_messages"]
    assert "The answer is 99" in event["scoring_input"]["messages"][1]["content"]
    assert event["target"]["text"] == "The answer is 12.5"
    verification = event["rescue"]["verification"]
    assert verification["messages"][0] == trace["events"][1]["input_messages"][0]
    assert "12.5" not in verification["rendered_prompt"]
    assert "25/2" not in verification["rendered_prompt"]
    assert verification["parsed_answer"] == ("12.5" if correct else "99")
    edge = event["edge_scores"][0]
    assert edge["opportunity"] == 1
    assert (edge["token_ratio"] > 0) is (correct and not empty_peak)
    metric = aggregate["metrics"]["token_ratio"]["spatial"]
    assert metric["opportunity_count"][0][0][1] == 1
    assert metric["conditional_mean"][0][0][1] == edge["token_ratio"]
    assert event["rescue"]["edge_region_indices"] == (event["rescue"]["peak_region_indices"] if correct else [])
    assert len(event["attribution"]["curve"]) == len(event["sentences"])
    diagnostic = event["attribution"]["curve"][0]["insertion_diagnostics"]
    assert diagnostic["prefix_joint_probability"] < diagnostic["gold_answer_conditional_probability"]
    assert diagnostic["prefix_joint_probability"] == pytest.approx(math.exp(diagnostic["logprob_sum"]))
    reviewer = json.loads((event_dir / "event_002_r01_dense_reviewer.json").read_text())
    assert reviewer["correctness"]["route"] == "reviewer_diagnostic"
    assert reviewer["input"]["messages"] == trace["events"][2]["input_messages"]
    assert len(scorer.calls) == 2
    calibrator.run()
    assert len(scorer.calls) == 2


def test_correct_numeric_worker_uses_original_attribution(tmp_path):
    calibrator, scorer, _, _ = setup_run(tmp_path, worker_correct=True)
    aggregate = calibrator.run()["aggregate"]
    assert aggregate["num_correct_samples"] == 0
    assert scorer.calls == []
    event = json.loads((tmp_path / "attribution/sample_00000/event_001_r00_inspector.json").read_text())
    assert event["correctness"]["original_answer"] == "12.5"
    assert event["correctness"]["route"] == "original"
    assert event["target"]["mode"] == "all"


def test_numeric_cache_tracks_gold_prompt_contract_and_verification(tmp_path):
    calibrator, _, trace, _ = setup_run(tmp_path)
    event = trace["events"][1]
    original = calibrator._event_configuration(event, trace)
    assert original["correctness_benchmark"] == "gsm8k"
    assert original["answer_parser"] == "explicit_gsm8k_answer_v1"
    assert original["rescue_prompt_version"] == "gsm8k_answer_first_v1"
    assert "decision_parser" not in original
    for key, value in [("gold", "13"), ("cap", 512), ("prompt", "changed")]:
        modified = copy.deepcopy(trace)
        if key == "gold":
            modified["sample"]["answer"] = value
        elif key == "cap":
            modified["config"]["decoding"]["max_new_tokens"] = value
        else:
            modified["config"]["prompt"]["version"] = value
        assert calibrator._event_configuration(event, modified) != original


def test_numeric_truncation_is_failure_not_zero(tmp_path):
    calibrator, _, trace, path = setup_run(tmp_path)
    trace["config"]["decoding"]["max_new_tokens"] = 3
    path.write_text(json.dumps(trace))
    with pytest.raises(RuntimeError, match="truncated"):
        calibrator.run()
    assert not (tmp_path / "attribution/aggregate_edge_matrices.json").exists()
    manifest = json.loads((tmp_path / "attribution/manifest.json").read_text())
    assert manifest["complete"] is False
    assert len(manifest["failures"]) == 1
    assert (tmp_path / "attribution/sample_00000/diagnostics/event_001_r00_inspector/error.json").is_file()


def test_unsupported_correctness_benchmark_is_rejected(tmp_path):
    calibrator, _, trace, path = setup_run(tmp_path)
    trace["config"]["benchmark"] = "humaneval"
    with pytest.raises(ValueError, match="MMLU and GSM8K only"):
        calibrator._validate_trace(trace, path)
    with pytest.raises(ValueError, match="Unsupported correctness-aware benchmark"):
        parse_worker_answer("The answer is 12", benchmark="humaneval")
