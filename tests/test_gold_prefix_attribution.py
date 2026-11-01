import numpy as np
import pytest
import torch

from AgentAttrPrune.attribution.gold_prefix import GreedyPrefixLogitAttributor


class WeightedVisibilityScorer:
    """Score the three attributable token positions with fixed weights."""

    weights = torch.tensor([2.0, 1.0, -3.0, 0.0])

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
        del (
            prompt_input_ids,
            generated_ids,
            selected_output_token_indices,
            score_type,
            reduction,
            kwargs,
        )
        masks = torch.as_tensor(attention_mask, dtype=torch.float32)
        score = (masks * self.weights).sum(dim=1)
        return {
            "score": score,
            "logit_sum": score,
            "logit_mean": score,
            "logprob_sum": score - 10.0,
            "logprob_mean": score - 10.0,
        }


def test_gold_prefix_search_ranks_all_sentences_and_keeps_global_peak():
    masks = np.zeros((3, 4), dtype=np.float32)
    masks[0, 0] = 1
    masks[1, 1] = 1
    masks[2, 2] = 1
    attributor = GreedyPrefixLogitAttributor(
        WeightedVisibilityScorer(), batch_size=2
    )

    result = attributor(
        [10, 11, 12, 13],
        masks,
        generated_ids=[20, 21],
        selected_target_indices=[0, 1],
    )

    assert result["early_stop"] is False
    assert result["stop_reason"] == "all_sentences_ranked"
    assert result["ordered_region_indices"] == [0, 1, 2]
    assert len(result["curve"]) == 3
    assert [record["insertion_score"] for record in result["curve"]] == pytest.approx(
        [2.0, 3.0, 0.0]
    )
    assert [record["deletion_score"] for record in result["curve"]] == pytest.approx(
        [-2.0, -3.0, 0.0]
    )
    assert result["best_step"] == 2
    assert result["best_region_indices"] == [0, 1]
    assert result["best_score"] == pytest.approx(3.0)
    assert result["full_score"] == pytest.approx(0.0)
    assert all(record["candidate_evaluations"] for record in result["curve"])


def test_gold_prefix_search_validates_inputs():
    with pytest.raises(ValueError, match="batch_size"):
        GreedyPrefixLogitAttributor(WeightedVisibilityScorer(), batch_size=0)

    attributor = GreedyPrefixLogitAttributor(WeightedVisibilityScorer())
    with pytest.raises(ValueError, match="At least one sentence"):
        attributor(
            [1, 2],
            np.zeros((0, 2), dtype=np.float32),
            generated_ids=[3],
            selected_target_indices=[0],
        )


def test_gold_prefix_equal_maxima_use_earliest_prefix():
    scorer = WeightedVisibilityScorer()
    scorer.weights = torch.tensor([2.0, 0.0, -3.0, 0.0])
    masks = np.eye(4, dtype=np.float32)[:3]
    result = GreedyPrefixLogitAttributor(scorer)(
        [10, 11, 12, 13], masks, generated_ids=[20], selected_target_indices=[0]
    )
    assert [point["insertion_score"] for point in result["curve"]] == [2, 2, -1]
    assert result["best_step"] == 1
    assert result["best_region_indices"] == [0]
