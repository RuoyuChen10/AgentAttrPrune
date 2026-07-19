from types import SimpleNamespace

import pytest
import torch
from torch import nn

from AgentAttrPrune.attribution.scoring import TeacherForcedTextScorer


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9


class FakeCausalLM(nn.Module):
    def __init__(self, vocab_size=7, output_dtype=torch.bfloat16):
        super().__init__()
        self.device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.output_dtype = output_dtype
        self.last_call = None
        self.last_full_logits = None

    def _compute_logits(self, input_ids, attention_mask, position_ids):
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
        context = torch.cumsum(input_ids.float() * attention_mask.float(), dim=1)
        vocabulary = torch.arange(self.vocab_size, dtype=torch.float32).view(1, 1, -1)
        logits = (
            context.unsqueeze(-1) * (vocabulary + 1.0) * 0.07
            + position_ids.float().unsqueeze(-1) * 0.03
            + vocabulary * 0.11
        )
        return logits.to(self.output_dtype)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        return_dict=True,
        use_cache=None,
        logits_to_keep=0,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        full_logits = self._compute_logits(input_ids, attention_mask, position_ids)
        self.last_full_logits = full_logits
        self.last_call = {
            "input_ids": input_ids.detach().clone(),
            "attention_mask": attention_mask.detach().clone(),
            "position_ids": None if position_ids is None else position_ids.detach().clone(),
            "use_cache": use_cache,
            "logits_to_keep": (
                logits_to_keep.detach().clone()
                if isinstance(logits_to_keep, torch.Tensor)
                else logits_to_keep
            ),
        }
        if isinstance(logits_to_keep, torch.Tensor):
            logits = full_logits[:, logits_to_keep]
        elif logits_to_keep:
            logits = full_logits[:, -int(logits_to_keep) :]
        else:
            logits = full_logits
        return SimpleNamespace(logits=logits)


class FakeLegacyCausalLM(FakeCausalLM):
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        return_dict=True,
        use_cache=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        logits = self._compute_logits(input_ids, attention_mask, position_ids)
        self.last_full_logits = logits
        self.last_call = {
            "input_ids": input_ids.detach().clone(),
            "attention_mask": attention_mask.detach().clone(),
            "position_ids": None if position_ids is None else position_ids.detach().clone(),
            "use_cache": use_cache,
        }
        return SimpleNamespace(logits=logits)


class FakeShardedCausalLM(FakeCausalLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Only the distinct values matter to the adaptor; computation remains
        # on CPU so this test does not require GPUs.
        self.hf_device_map = {"model.layers.0": "cuda:0", "lm_head": "cuda:1"}


def make_adaptor(model):
    return TeacherForcedTextScorer(
        model=model,
        tokenizer=FakeTokenizer(),
        model_name="fake",
        device=torch.device("cpu"),
    )


def test_teacher_forced_scoring_supports_deleted_variable_length_prompts_and_logits_to_keep():
    model = FakeCausalLM()
    adaptor = make_adaptor(model)

    result = adaptor.score_teacher_forced_targets(
        prompt_input_ids=[[1, 99, 3], [4, 5, 6]],
        attention_mask=[[1, 0, 1], [1, 1, 1]],
        generated_ids=[2, 1, 4],
        selected_output_token_indices=[0, 2],
        score_type="logprob",
        reduction="mean",
    )

    assert result["used_logits_to_keep"] is True
    assert model.last_call["use_cache"] is False
    assert model.last_call["logits_to_keep"].tolist() == [2, 4]
    assert model.last_call["input_ids"].tolist() == [
        [0, 1, 3, 2, 1],
        [4, 5, 6, 2, 1],
    ]
    assert model.last_call["attention_mask"].tolist() == [
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    assert model.last_call["position_ids"].tolist() == [
        [0, 0, 1, 2, 3],
        [0, 1, 2, 3, 4],
    ]
    assert result["prompt_lengths"].tolist() == [2, 3]
    assert result["target_token_indices"].tolist() == [0, 2]
    assert result["target_token_ids"].tolist() == [2, 4]

    position_logits = model.last_full_logits[:, torch.tensor([2, 4])].float()
    gather = torch.tensor([2, 4]).view(1, 2, 1).expand(2, -1, -1)
    expected_logits = position_logits.gather(2, gather).squeeze(-1)
    expected_logprobs = torch.log_softmax(position_logits, dim=-1).gather(2, gather).squeeze(-1)
    expected_probabilities = expected_logprobs.exp()

    assert result["target_logits"].dtype == torch.float32
    assert result["target_logprobs"].dtype == torch.float32
    assert result["target_probabilities"].dtype == torch.float32
    torch.testing.assert_close(result["target_logits"], expected_logits)
    torch.testing.assert_close(result["target_logprobs"], expected_logprobs)
    torch.testing.assert_close(result["target_probabilities"], expected_probabilities)
    torch.testing.assert_close(result["logit_sum"], expected_logits.sum(dim=-1))
    torch.testing.assert_close(result["logit_mean"], expected_logits.mean(dim=-1))
    torch.testing.assert_close(result["logprob_sum"], expected_logprobs.sum(dim=-1))
    torch.testing.assert_close(result["probability_mean"], expected_probabilities.mean(dim=-1))
    torch.testing.assert_close(result["score"], expected_logprobs.mean(dim=-1))

    assert adaptor.prompt_length is None
    assert adaptor.generated_answer_ids is None


def test_teacher_forced_scoring_falls_back_to_full_logits_for_legacy_models():
    model = FakeLegacyCausalLM(output_dtype=torch.float32)
    adaptor = make_adaptor(model)

    result = adaptor.score_teacher_forced_targets(
        prompt_input_ids=torch.tensor([1, 2, 3]),
        generated_ids=torch.tensor([4, 5]),
        selected_output_token_indices=[1],
        score_type="probability",
        reduction="none",
    )

    assert result["used_logits_to_keep"] is False
    assert result["score"].shape == (1, 1)
    expected_position_logits = model.last_full_logits[:, 3].float()
    expected_probability = torch.softmax(expected_position_logits, dim=-1)[:, 5:6]
    torch.testing.assert_close(result["target_probabilities"], expected_probability)
    torch.testing.assert_close(result["score"], expected_probability)


def test_sharded_model_uses_integer_trailing_keep_and_falls_back_for_sparse_positions():
    model = FakeShardedCausalLM(output_dtype=torch.float32)
    adaptor = make_adaptor(model)

    trailing = adaptor.score_teacher_forced_targets(
        prompt_input_ids=[1, 2, 3],
        generated_ids=[4, 5, 6],
        selected_output_token_indices=[0, 1, 2],
    )
    assert trailing["used_logits_to_keep"] is True
    assert model.last_call["logits_to_keep"] == 3

    sparse = adaptor.score_teacher_forced_targets(
        prompt_input_ids=[1, 2, 3],
        generated_ids=[4, 5, 6],
        selected_output_token_indices=[0, 2],
    )
    assert sparse["used_logits_to_keep"] is False
    assert model.last_call["logits_to_keep"] == 0


def test_existing_forward_probability_behavior_is_unchanged():
    model = FakeCausalLM(output_dtype=torch.float32)
    adaptor = make_adaptor(model)
    adaptor.prompt_length = 3
    adaptor.set_targets(torch.tensor([2, 4]), selected_output_token_indices=[0, 1])

    scores = adaptor(torch.tensor([1, 2, 3]))

    expected_logits = model.last_full_logits[:, torch.tensor([2, 3])]
    gather = torch.tensor([2, 4]).view(1, 2, 1)
    expected = torch.softmax(expected_logits, dim=-1).gather(2, gather).squeeze(0).squeeze(-1)
    assert scores.shape == (2,)
    torch.testing.assert_close(scores, expected)


@pytest.mark.parametrize("score_type", ["bad", "raw"])
def test_teacher_forced_scoring_rejects_unknown_score_type(score_type):
    adaptor = make_adaptor(FakeCausalLM())
    with pytest.raises(ValueError, match="Unsupported score_type"):
        adaptor.score_teacher_forced_targets(
            prompt_input_ids=[1, 2],
            generated_ids=[3],
            score_type=score_type,
        )
