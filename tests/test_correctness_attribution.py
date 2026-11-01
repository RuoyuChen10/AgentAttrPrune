import copy
import math
from types import SimpleNamespace

import pytest
import torch

from AgentAttrPrune.attribution.correctness import (
    RescueGenerationError,
    answer_first_system_prompt,
    gold_prefix_target,
    parse_worker_decision,
    prune_input_messages,
    run_gold_prefix_rescue,
)
from AgentAttrPrune.attribution.edge import build_incoming_sentence_masks
from AgentAttrPrune.prompt.mmlu import MMLUPromptSet


class CharacterTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return "".join(f"<{item['role']}>{item['content']}</{item['role']}>" for item in messages) + "<assistant>"

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        payload = {"input_ids": [ord(char) + 1 for char in text]}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return payload

    def decode(self, ids, **kwargs):
        return "".join(chr(int(value) - 1) for value in ids if int(value))


def worker_event():
    output = "Useful evidence.\nMisleading claim.\nDecision: D"
    return {
        "event_id": 1,
        "round": 0,
        "agent_id": "critic",
        "role": "Critic",
        "output": "Reasoning: original response.\nDecision: D",
        "input_messages": [
            {"role": "system", "content": MMLUPromptSet.get_analyze_constraint("Critic")},
            {"role": "user", "content": "Question with Option A, Option B, Option C, Option D.\nAgent expert output:\n" + output},
        ],
        "incoming": [{"agent_id": "expert", "round": 0, "edge_type": "spatial", "output": output}],
    }


class RescueScorer:
    def __init__(self, event, *, empty_peak=False, correct=True):
        self.tokenizer = CharacterTokenizer()
        self.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[0, 999]))
        changed = copy.deepcopy(event)
        changed["input_messages"][0]["content"] = answer_first_system_prompt(changed["input_messages"][0]["content"])
        _, ids, _, spans, _ = build_incoming_sentence_masks(self.tokenizer, changed)
        self.weights = torch.zeros(len(ids))
        for span in spans:
            self.weights[span.token_indices[0]] = 3 if "Useful" in span.text and not empty_peak else -1
        self.correct = correct
        self.calls = []

    def score_teacher_forced_targets(self, prompt_input_ids, generated_ids, selected_output_token_indices, attention_mask, **kwargs):
        score = (attention_mask.float() * self.weights).sum(dim=1)
        logprobs = torch.full((len(score), len(selected_output_token_indices)), -0.05)
        logprobs[:, -1] = torch.nn.functional.logsigmoid(score)
        return {
            "score": score,
            "logit_sum": score,
            "logit_mean": score / len(selected_output_token_indices),
            "logprob_sum": logprobs.sum(dim=1),
            "logprob_mean": logprobs.mean(dim=1),
            "target_logprobs": logprobs,
        }

    def generate_from_rendered_chat(self, rendered, *, max_new_tokens, do_sample):
        self.calls.append((rendered, max_new_tokens, do_sample))
        answer_first = "First write your final choice. Then" in rendered
        text = (
            "Decision: D\nReasoning: diagnostic."
            if answer_first
            else f"Reasoning: local replay.\nDecision: {'B' if self.correct else 'D'}"
        )
        ids = (self.tokenizer(text)["input_ids"] + [0])[:max_new_tokens]
        return {
            "input_ids": torch.tensor(self.tokenizer(rendered)["input_ids"]),
            "generated_answer_ids": torch.tensor(ids),
            "output_text": self.tokenizer.decode(ids),
        }


@pytest.mark.parametrize("text, expected", [
    ("Reasoning: Option C seems likely.\nDecision: B", "B"),
    ("Reasoning: A is mentioned.", None),
    ("A", None),
    ("Reasoning: Decision: B", None),
    ("**Decision:** **b**", "B"),
    ("Decision: B\nDecision: C", None),
    ("Decision: B\nDecision: maybe C", None),
    ("Decision: B\nDecision: B", "B"),
    ("Decision: B or C", None),
])
def test_explicit_decision_parser(text, expected):
    assert parse_worker_decision(text) == expected


def test_rescue_preserves_frozen_upstream_and_uses_unforced_original_prompt():
    event = worker_event()
    original = copy.deepcopy(event)
    scorer = RescueScorer(event)
    progress = []
    result = run_gold_prefix_rescue(scorer, event, "B", batch_size=2, max_new_tokens=256, progress_callback=progress.append)

    assert event == original
    assert result["success"] is True
    assert any(span.text == "Decision: D" for span in result["spans"])
    assert result["event"]["incoming"] == original["incoming"]
    assert result["verification"]["messages"][0] == original["input_messages"][0]
    assert "Useful evidence." in result["verification"]["rendered_prompt"]
    assert "Misleading claim." not in result["verification"]["rendered_prompt"]
    assert "Decision: B" not in result["verification"]["rendered_prompt"]
    assert "Question with Option A, Option B, Option C, Option D." in result["verification"]["rendered_prompt"]
    assert result["full_generation"]["parsed_answer"] == "D"
    assert len(scorer.calls) == 2
    assert all(do_sample is False for _, _, do_sample in scorer.calls)
    assert len(result["attribution"]["curve"]) == len(result["spans"])
    assert result["attribution"]["best_step"] == 1
    diagnostic = result["attribution"]["curve"][0]["insertion_diagnostics"]
    assert 0 < diagnostic["prefix_joint_probability"] < diagnostic["gold_answer_conditional_probability"] < 1
    assert diagnostic["prefix_joint_probability"] == pytest.approx(math.exp(diagnostic["logprob_sum"]))
    assert [item["stage"] for item in progress] == [
        "rescue_input", "full_answer_first_generation", "ranking_endpoints",
        *(["ranking_step"] * len(result["spans"])), "peak_verification",
    ]
    assert all("curve" not in item for item in progress)


@pytest.mark.parametrize("correct", [True, False])
def test_empty_peak_and_failed_verification_are_reported_without_replacing_peak(correct):
    event = worker_event()
    result = run_gold_prefix_rescue(RescueScorer(event, empty_peak=True, correct=correct), event, "B", batch_size=2, max_new_tokens=256)
    assert result["attribution"]["best_step"] == 0
    assert result["attribution"]["selected_region_indices"] == []
    assert result["success"] is correct
    assert "Useful evidence." not in result["verification"]["rendered_prompt"]
    assert "Agent expert output:" in result["verification"]["rendered_prompt"]


def test_failed_verification_keeps_peak_for_diagnostics():
    event = worker_event()
    result = run_gold_prefix_rescue(RescueScorer(event, correct=False), event, "B", batch_size=2, max_new_tokens=256)
    assert result["success"] is False
    assert result["attribution"]["best_region_indices"] == [0]


def test_truncation_is_exception_with_saved_diagnostics_and_progress():
    event = worker_event()
    progress = []
    with pytest.raises(RescueGenerationError, match="truncated") as exc:
        run_gold_prefix_rescue(RescueScorer(event), event, "B", batch_size=2, max_new_tokens=3, progress_callback=progress.append)
    assert exc.value.diagnostics["finish_reason"] == "length"
    assert exc.value.diagnostics["generated_token_count"] == 3
    assert progress[-1]["stage"] == "full_answer_first_generation"


def test_eos_at_generation_ceiling_is_not_truncation():
    event = worker_event()
    scorer = RescueScorer(event)
    generate = scorer.generate_from_rendered_chat

    def cap_with_eos(rendered, *, max_new_tokens, do_sample):
        result = generate(rendered, max_new_tokens=max_new_tokens, do_sample=do_sample)
        result["generated_answer_ids"][-1] = 0
        return result

    scorer.generate_from_rendered_chat = cap_with_eos
    result = run_gold_prefix_rescue(scorer, event, "B", batch_size=2, max_new_tokens=3)
    assert result["full_generation"]["finish_reason"] == "eos"
    assert result["verification"]["hit_max_new_tokens"] is False


def test_pruning_preserves_quoted_task_text_and_repeated_edge_occurrences():
    messages = [{"role": "system", "content": "Keep system."}, {"role": "user", "content": "Question quotes X.\nAgent a: X.\nHistory a: X."}]
    edges = [SimpleNamespace(incoming_index=i, source_output="X.") for i in range(2)]
    pruned = prune_input_messages(messages, edges, {0: "", 1: "X."})
    assert pruned[1]["content"] == "Question quotes X.\nAgent a: \nHistory a: X."
    assert messages[1]["content"].count("X.") == 3


def test_target_probability_indices_follow_actual_token_offsets():
    class MergedTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [10, 11], "offset_mapping": [(0, 9), (9, len(text))]}
    target = gold_prefix_target(MergedTokenizer(), "B")
    assert target["selected_indices"] == [0, 1]
    assert target["answer_token_indices"] == [1]
