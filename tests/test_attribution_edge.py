from types import SimpleNamespace

import numpy as np
import pytest
import torch

from AgentAttrPrune.attribution.edge import (
    GreedyFullScoreAttributor,
    IncomingEdge,
    IncomingSentenceSpan,
    aggregate_sample_matrices,
    apply_early_stop_sentence_saliency,
    build_incoming_sentence_masks,
    build_pruned_masks,
    build_sample_matrices,
    compute_event_edge_scores,
    select_target_token_indices,
    trace_is_correct,
)
from AgentAttrPrune.attribution.visualization import (
    write_event_insertion_curve_svg,
    write_event_svg,
)


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        del tokenize, kwargs
        rendered = "".join(f"<{item['role']}>{item['content']}</{item['role']}>" for item in messages)
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        del add_special_tokens, kwargs
        values = [ord(char) % 251 + 1 for char in text]
        result = {"input_ids": values}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


def test_incoming_builder_excludes_system_question_headers_and_disambiguates_duplicate_sources():
    repeated = "Useful first sentence. Useful second sentence."
    user = (
        "Question that must stay.\n"
        "Agent a output:\n" + repeated + "\n"
        "Previous Agent a output:\n" + repeated
    )
    event = {
        "event_id": 3,
        "round": 1,
        "agent_id": "b",
        "input_messages": [
            {"role": "system", "content": "System must stay."},
            {"role": "user", "content": user},
        ],
        "incoming": [
            {"agent_id": "a", "round": 1, "edge_type": "spatial", "output": repeated},
            {"agent_id": "a", "round": 0, "edge_type": "temporal", "output": repeated},
        ],
    }

    rendered, _, masks, spans, edges = build_incoming_sentence_masks(CharacterTokenizer(), event)

    assert len(edges) == 2
    assert [edge.edge_type for edge in edges] == ["spatial", "temporal"]
    assert len(spans) == 4
    assert masks.shape[0] == 4
    assert np.all(masks.sum(axis=0) <= 1)
    assert {span.incoming_index for span in spans} == {0, 1}
    selected_chars = {index for span in spans for index in span.token_indices}
    for fixed_text in ("System must stay.", "Question that must stay.", "Agent a output:"):
        start = rendered.index(fixed_text)
        assert not selected_chars.intersection(range(start, start + len(fixed_text)))


def test_incoming_builder_uses_agent_occurrence_when_short_text_collides_with_fixed_prompt():
    event = {
        "event_id": 1,
        "round": 0,
        "agent_id": "b",
        "input_messages": [
            {"role": "system", "content": "System example A. must remain."},
            {
                "role": "user",
                "content": "Question also mentions A.\nAgent a output:\nA.",
            },
        ],
        "incoming": [
            {"agent_id": "a", "round": 0, "edge_type": "spatial", "output": "A."}
        ],
    }

    rendered, _, _, spans, _ = build_incoming_sentence_masks(CharacterTokenizer(), event)

    assert len(spans) == 1
    assert spans[0].char_start == rendered.rfind("A.")
    assert spans[0].text == "A."


class SumVisibleScorer:
    """A deterministic stand-in for Qwen teacher-forced scoring."""

    def score_teacher_forced_targets(
        self,
        prompt_input_ids,
        generated_ids=None,
        selected_output_token_indices=None,
        attention_mask=None,
        score_type="logit",
        reduction="sum",
        **kwargs,
    ):
        del generated_ids, selected_output_token_indices, score_type, reduction, kwargs
        ids = torch.as_tensor(prompt_input_ids)
        masks = torch.as_tensor(attention_mask)
        score = (ids * masks).sum(dim=1).float()
        return {
            "score": score,
            "logit_sum": score,
            "logit_mean": score,
            "logprob_sum": score - 1,
            "logprob_mean": score - 1,
        }


def test_greedy_attribution_stops_before_ranking_every_sentence():
    # Token 1 is fixed. Candidate regions expose tokens 2, 3, 4, and 5.
    masks = np.zeros((4, 5), dtype=np.float32)
    for region, token in enumerate(range(1, 5)):
        masks[region, token] = 1
    attributor = GreedyFullScoreAttributor(
        SumVisibleScorer(), target_recovery=0.80, batch_size=2
    )

    result = attributor(
        [1, 2, 3, 4, 5],
        masks,
        generated_ids=[9],
        selected_target_indices=[0],
    )

    assert result["baseline_score"] == 1
    assert result["full_score"] == 15
    assert result["selected_region_indices"] == [3, 2, 1]
    assert result["stop_reason"] == "near_full_score"
    assert result["stop_step"] == 3
    assert len(result["curve"]) == 3


def _saliency_result(scores, *, selected_count=None):
    selected_count = len(scores) if selected_count is None else selected_count
    return {
        "attribution": {
            "baseline_score": 0.0,
            "selected_region_indices": list(range(selected_count)),
            "curve": [
                {
                    "step": step,
                    "selected_region_index": index,
                    "score": score,
                }
                for step, (index, score) in enumerate(
                    zip(range(len(scores)), scores),
                    start=1,
                )
            ],
        },
        "sentences": [
            {"index": index, "selected": index < selected_count}
            for index in range(len(scores) + 1)
        ],
    }


def test_early_stop_saliency_uses_cumulative_absolute_score_differences():
    result = _saliency_result([4.0, 6.0, 7.0])

    values = apply_early_stop_sentence_saliency(result, saliency_floor=0.4)

    assert list(values.values()) == pytest.approx([1.0, 0.6, 0.4])
    assert [item["absolute_score_delta"] for item in result["attribution"]["curve"]] == [
        4.0,
        2.0,
        1.0,
    ]
    assert [
        item["cumulative_absolute_score_delta"]
        for item in result["attribution"]["curve"]
    ] == [4.0, 6.0, 7.0]
    assert [item["selection_step"] for item in result["sentences"]] == [
        1,
        2,
        3,
        None,
    ]
    assert result["sentences"][-1]["saliency"] is None


def test_early_stop_saliency_handles_score_reversal_and_final_prefix_only():
    result = _saliency_result([4.0, 3.0, 5.0, 9.0], selected_count=3)

    apply_early_stop_sentence_saliency(result, saliency_floor=0.4)

    curve = result["attribution"]["curve"]
    assert [item["absolute_score_delta"] for item in curve] == [
        4.0,
        1.0,
        2.0,
        None,
    ]
    assert [item["saliency"] for item in curve[:3]] == pytest.approx(
        [1.0, 0.8, 0.4]
    )
    assert curve[-1]["saliency"] is None
    assert curve[-1]["cumulative_absolute_score_delta"] is None


def test_early_stop_saliency_single_and_degenerate_rank_fallback():
    single = _saliency_result([2.0])
    apply_early_stop_sentence_saliency(single)
    assert single["sentences"][0]["saliency"] == 1.0
    assert (
        single["attribution"]["saliency_configuration"]["normalization"]
        == "single_selected"
    )

    degenerate = _saliency_result([2.0, 2.0, 2.0])
    apply_early_stop_sentence_saliency(degenerate, saliency_floor=0.4)
    assert [item["saliency"] for item in degenerate["sentences"][:3]] == pytest.approx(
        [1.0, 0.7, 0.4]
    )
    configuration = degenerate["attribution"]["saliency_configuration"]
    assert configuration["normalization"] == "rank_linear_fallback"
    assert configuration["degenerate_fallback_used"] is True


def test_early_stop_saliency_empty_selection_is_explicit():
    result = {
        "attribution": {"selected_region_indices": [], "curve": []},
        "sentences": [{"index": 0, "selected": False}],
    }

    assert apply_early_stop_sentence_saliency(result) == {}
    assert result["sentences"][0]["saliency"] is None
    assert result["attribution"]["saliency_configuration"]["normalization"] == "empty"


def test_sentence_svg_uses_multiple_saliency_colors_and_labels(tmp_path):
    result = _saliency_result([4.0, 6.0, 7.0])
    result["attribution"].update(
        {
            "full_score": 7.1,
            "target_recovery": 0.95,
            "selected_recovery": 0.97,
            "score_type": "logit",
            "reduction": "sum",
        }
    )
    for step, record in enumerate(result["attribution"]["curve"], start=1):
        record["selected_sentence_fraction"] = step / 3
        record["selected_token_fraction"] = step / 4
    for index, sentence in enumerate(result["sentences"][:3]):
        sentence.update(
            {
                "incoming_index": 0,
                "text": f"Sentence {index}.",
                "source_token_indices": [index],
            }
        )
    result["sentences"] = result["sentences"][:3]
    result.update(
        {
            "sample_id": "sample_00000",
            "sample": {"task": "Question?"},
            "event": {"round": 0, "agent_id": "target", "role": "Critic"},
            "input": {"messages": []},
            "output": {"text": "Answer."},
            "incoming_edges": [
                {
                    "incoming_index": 0,
                    "source_agent": "source",
                    "source_round": 0,
                    "edge_type": "spatial",
                    "source_output": "Sentence 0. Sentence 1. Sentence 2.",
                }
            ],
            "edge_scores": [
                {
                    "incoming_index": 0,
                    "selected_token_count": 3,
                    "source_token_count": 3,
                    "token_ratio": 1.0,
                }
            ],
        }
    )
    apply_early_stop_sentence_saliency(result, saliency_floor=0.4)

    path = write_event_svg(result, tmp_path / "event.svg")
    svg = path.read_text(encoding="utf-8")

    assert 'id="saliency-gradient"' in svg
    assert 'fill="#f97316"' in svg
    assert 'fill="#fed7aa"' in svg
    assert "saliency 1.00 · step 1" in svg
    assert "saliency 0.60 · step 2" in svg
    assert "saliency 0.40 · step 3" in svg

    curve_path = write_event_insertion_curve_svg(
        result,
        tmp_path / "event.insertion_curve.svg",
    )
    curve_svg = curve_path.read_text(encoding="utf-8")
    assert "Early-stop Insertion Score Curve" in curve_svg
    assert "Retained Significant Sentences (%)" in curve_svg
    assert "Insertion Score (logit · sum)" in curve_svg
    assert "stop: 100.0% sentences (3/3)" in curve_svg
    assert "75.0% tokens · recovery 97.0%" in curve_svg
    assert "95% recovery target" in curve_svg
    assert "full input · 7.1" in curve_svg


def _span(index, incoming_index, source, words, text="sentence"):
    return IncomingSentenceSpan(
        index=index,
        incoming_index=incoming_index,
        source_agent=source,
        target_agent="dst",
        edge_type="spatial",
        source_round=0,
        target_round=0,
        source_event_id=None,
        text=text,
        char_start=index,
        char_end=index + 1,
        token_indices=(index,),
        word_count=words,
        source_word_count=20,
        source_char_start=index,
        source_char_end=index + 1,
        source_token_indices=(index,),
        source_token_count=4 if source == "a" else 2,
    )


def test_token_binary_and_word_ratios_follow_requested_edge_rules():
    edges = [
        IncomingEdge(
            0, "a", "dst", "spatial", 0, 0, None, "a", 20,
            source_token_count=4,
        ),
        IncomingEdge(
            1, "b", "dst", "spatial", 0, 0, None, "b", 10,
            source_token_count=2,
        ),
    ]
    spans = [
        _span(0, 0, "a", 5),
        _span(1, 0, "a", 7),
        _span(2, 0, "a", 8),
        _span(3, 1, "b", 2),
        _span(4, 1, "b", 8),
    ]

    scores = compute_event_edge_scores(
        spans, edges, [0, 1, 3], min_selected_sentences=2, min_sentence_words=4
    )

    assert scores[0]["binary"] == 1
    assert scores[0]["word_ratio"] == pytest.approx(0.6)
    assert scores[0]["token_ratio"] == pytest.approx(0.5)
    assert scores[0]["selected_token_count"] == 2
    assert scores[1]["binary"] == 0
    # Short selected sentences still count for the user's word-ratio metric.
    assert scores[1]["word_ratio"] == pytest.approx(0.2)
    assert scores[1]["token_ratio"] == pytest.approx(0.5)


def _trace(correct=True):
    return {
        "config": {"sample_index": 7, "rounds": 2},
        "topology": {
            "workers": [
                {"agent_id": "a", "role": "Critic", "agent_type": "AnalyzeAgent"},
                {
                    "agent_id": "dst",
                    "role": "Mathematician",
                    "agent_type": "AnalyzeAgent",
                },
            ]
        },
        "final": {"parsed_answer": "A" if correct else "B", "gold_answer": "A"},
        "events": [
            {
                "round": 0,
                "agent_id": "dst",
                "incoming": [
                    {"agent_id": "a", "round": 0, "edge_type": "spatial", "output": "x"}
                ],
            }
        ],
    }


def _event_result(binary=1, ratio=0.6):
    return {
        "edge_scores": [
            {
                "edge_type": "spatial",
                "source_agent": "a",
                "target_agent": "dst",
                "target_round": 0,
                "binary": binary,
                "word_ratio": ratio,
            }
        ]
    }


def test_round_source_target_matrix_aggregation_and_pruned_policy():
    first = build_sample_matrices(_trace(), [_event_result(binary=1, ratio=0.6)])
    second_trace = _trace()
    second_trace["config"]["sample_index"] = 8
    second = build_sample_matrices(second_trace, [_event_result(binary=0, ratio=0.2)])

    assert first["matrices"]["binary"]["spatial"][0][0][1] == 1
    assert first["matrices"]["binary"]["spatial"][0][1][0] == 0
    aggregate = aggregate_sample_matrices([first, second])
    assert aggregate["agent_roles"] == ["Critic", "Mathematician"]
    assert aggregate["agent_types"] == ["AnalyzeAgent", "AnalyzeAgent"]
    cell = aggregate["metrics"]["word_ratio"]["spatial"]
    assert cell["conditional_mean"][0][0][1] == pytest.approx(0.4)
    assert cell["opportunity_count"][0][0][1] == 2
    token_cell = aggregate["metrics"]["token_ratio"]["spatial"]
    # Legacy fixtures without token_ratio fall back to word_ratio.
    assert token_cell["conditional_mean"][0][0][1] == pytest.approx(0.4)

    policy = build_pruned_masks(
        aggregate,
        metric="binary",
        statistic="conditional_mean",
        threshold=0.6,
        pruning_rate=None,
        unobserved="keep",
    )
    assert policy["masks"]["spatial"][0][0][1] == 0
    # Non-candidate reverse edges are never invented by unobserved=keep.
    assert policy["masks"]["spatial"][0][1][0] == 0
    assert policy["masks"]["spatial"][0][0][0] == 0
    assert policy["agent_roles"] == ["Critic", "Mathematician"]
    assert policy["agent_types"] == ["AnalyzeAgent", "AnalyzeAgent"]
    assert not any(any(row) for row in policy["masks"]["temporal"][0])


def test_incorrect_trace_is_not_a_verified_calibration_sample():
    assert trace_is_correct(_trace())
    assert not trace_is_correct(_trace(correct=False))
    incorrect = build_sample_matrices(_trace(correct=False), [_event_result()])
    with pytest.raises(ValueError, match="verified-correct"):
        aggregate_sample_matrices([incorrect])


def test_all_sample_aggregation_counts_failed_rescue_as_observed_zero():
    correct = build_sample_matrices(_trace(), [_event_result(ratio=0.8)])
    failed = build_sample_matrices(
        _trace(correct=False), [_event_result(binary=0, ratio=0.0)]
    )

    aggregate = aggregate_sample_matrices([correct, failed], aggregation_scope="all")

    assert aggregate["aggregation_scope"] == "all"
    assert aggregate["num_aggregated_samples"] == 2
    assert aggregate["num_correct_samples"] == 1
    spatial = aggregate["metrics"]["token_ratio"]["spatial"]
    assert spatial["opportunity_count"][0][0][1] == 2
    assert spatial["conditional_mean"][0][0][1] == pytest.approx(0.4)
    assert failed["correct_verified"] is False

    legacy = aggregate_sample_matrices([correct, failed])
    assert legacy["aggregation_scope"] == "correct"
    assert legacy["num_aggregated_samples"] == legacy["num_correct_samples"] == 1
    assert legacy["metrics"]["token_ratio"]["spatial"]["conditional_mean"][0][0][1] == 0.8


def test_all_sample_aggregation_accepts_all_incorrect_and_preserves_metadata():
    matrix = build_sample_matrices(
        _trace(correct=False), [_event_result(binary=0, ratio=0.0)]
    )
    matrix["attribution_configuration"] = {"attribution_mode": "correctness-aware"}
    aggregate = aggregate_sample_matrices([matrix], aggregation_scope="all")
    policy = build_pruned_masks(aggregate, metric="token_ratio", pruning_rate=0)

    assert aggregate["num_aggregated_samples"] == 1
    assert aggregate["num_correct_samples"] == 0
    assert policy["calibration_num_aggregated_samples"] == 1
    assert policy["calibration_num_correct_samples"] == 0
    assert policy["calibration_aggregation_scope"] == "all"
    assert policy["calibration_attribution_mode"] == "correctness-aware"
    assert policy["masks"]["spatial"][0][0][1] == 1
    with pytest.raises(ValueError, match="aggregation_scope"):
        aggregate_sample_matrices([matrix], aggregation_scope="wrong")


def test_all_sample_aggregation_rejects_mixed_mode_in_incorrect_sample():
    correct = build_sample_matrices(_trace(), [_event_result()])
    incorrect = build_sample_matrices(_trace(correct=False), [_event_result()])
    correct["attribution_configuration"] = {"attribution_mode": "correctness-aware"}
    incorrect["attribution_configuration"] = {"attribution_mode": "original"}

    with pytest.raises(ValueError, match="attribution configuration"):
        aggregate_sample_matrices([correct, incorrect], aggregation_scope="all")


def test_spatial_policy_projects_cyclic_candidate_union_by_importance():
    zeros = [[[0.0, 0.0], [0.0, 0.0]]]
    aggregate = {
        "agents": ["a", "b"],
        "agent_roles": ["Critic", "Mathematician"],
        "num_rounds": 1,
        "metrics": {
            "binary": {
                "spatial": {
                    "conditional_mean": [[[0.0, 0.2], [0.9, 0.0]]],
                    "sample_mean": [[[0.0, 0.2], [0.9, 0.0]]],
                    "opportunity_count": [[[0.0, 1.0], [1.0, 0.0]]],
                    "candidate_count": [[[0.0, 1.0], [1.0, 0.0]]],
                },
                "temporal": {
                    "conditional_mean": zeros,
                    "sample_mean": zeros,
                    "opportunity_count": zeros,
                    "candidate_count": zeros,
                },
            }
        },
    }

    policy = build_pruned_masks(
        aggregate,
        metric="binary",
        statistic="conditional_mean",
        threshold=0.0,
        pruning_rate=None,
    )

    assert policy["masks"]["spatial"][0] == [[0, 0], [1, 0]]
    assert policy["retention"]["spatial"]["dag_projection_dropped_edges"] == 1
    assert policy["retention"]["spatial"]["effective_candidate_pruning_rate"] == 0.5


def test_round_zero_temporal_edges_are_excluded_from_sample_statistics():
    trace = _trace()
    trace["events"][0]["incoming"][0]["edge_type"] = "temporal"
    result = _event_result()
    result["edge_scores"][0]["edge_type"] = "temporal"

    sample = build_sample_matrices(trace, [result])

    assert not np.asarray(sample["matrices"]["candidate"]["temporal"]).any()
    assert not np.asarray(sample["matrices"]["opportunity"]["temporal"]).any()


def test_invalid_attribution_limits_are_rejected_instead_of_slicing_or_empty_selection():
    with pytest.raises(ValueError, match="target_token_limit"):
        select_target_token_indices(
            CharacterTokenizer(), [1, 2], mode="all", target_token_limit=-1
        )
    with pytest.raises(ValueError, match="max_selected_sentences"):
        GreedyFullScoreAttributor(SumVisibleScorer(), max_selected_sentences=0)
    with pytest.raises(ValueError, match="target_recovery"):
        GreedyFullScoreAttributor(SumVisibleScorer(), target_recovery=0)
    with pytest.raises(ValueError, match="finite"):
        GreedyFullScoreAttributor(SumVisibleScorer(), absolute_atol=float("nan"))
    with pytest.raises(ValueError, match="must be positive"):
        compute_event_edge_scores([], [], [], min_sentence_words=0)


def test_aggregate_rejects_mixed_attribution_configurations():
    first = build_sample_matrices(_trace(), [_event_result()])
    second = build_sample_matrices(_trace(), [_event_result()])
    first["attribution_configuration"] = {"score_type": "logit"}
    second["attribution_configuration"] = {"score_type": "logprob"}

    with pytest.raises(ValueError, match="attribution configuration"):
        aggregate_sample_matrices([first, second])

    first["attribution_configuration"] = {}
    second["attribution_configuration"] = {}
    first["selection_configuration"] = {"min_incoming": 2}
    second["selection_configuration"] = {"min_incoming": 3}
    with pytest.raises(ValueError, match="event selection configuration"):
        aggregate_sample_matrices([first, second])


def test_aggregate_rejects_sample_matrices_from_different_runs():
    first = build_sample_matrices(_trace(), [_event_result()])
    second = build_sample_matrices(_trace(), [_event_result()])
    first["run_fingerprint"] = "run-a"
    second["run_fingerprint"] = "run-b"

    with pytest.raises(ValueError, match="same run fingerprint"):
        aggregate_sample_matrices([first, second])
