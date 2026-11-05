"""Offline parity checks for query-row chunking of Qwen3 eager attention."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3 import modeling_qwen3 as qwen3
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from experiments_attrprune.qwen3_chunked_prefill import (
    chunked_prefill_enabled,
    make_chunked_attention,
)


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(dtype=torch.float32, *, q_length=11, k_length=17, mask_kind="causal"):
    generator = torch.Generator().manual_seed(41)
    query = torch.randn(2, 4, q_length, 8, generator=generator).to(dtype)
    key = torch.randn(2, 2, k_length, 8, generator=generator).to(dtype)
    value = torch.randn(2, 2, k_length, 8, generator=generator).to(dtype)
    module = SimpleNamespace(num_key_value_groups=2, training=False)
    if mask_kind == "none":
        mask = None
    elif mask_kind == "broadcast":
        # A broadcast query dimension and extra unused key columns both occur
        # in compatible attention masks; neither should be sliced incorrectly.
        mask = torch.zeros(2, 1, 1, k_length + 3, dtype=dtype)
        mask[0, :, :, :2] = torch.finfo(dtype).min
    else:
        mask = torch.zeros(2, 1, q_length, k_length + 3, dtype=dtype)
        positions = torch.arange(q_length) + k_length - q_length
        blocked = torch.arange(k_length + 3)[None, :] > positions[:, None]
        mask.masked_fill_(blocked[None, None], torch.finfo(dtype).min)
        mask[0, :, :, :2] = torch.finfo(dtype).min  # left padding
    return module, query, key, value, mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_kind", ["causal", "broadcast", "none"])
def test_chunking_matches_eager_with_gqa_padding_cached_keys_and_short_last_chunk(dtype, mask_kind):
    args = inputs(dtype, mask_kind=mask_kind)
    expected, _ = qwen3.eager_attention_forward(*args, scaling=8**-0.5)
    events = []
    chunked = make_chunked_attention(qwen3.eager_attention_forward,
                                     chunk_size=4, min_query_length=5, emit=events.append)
    with torch.inference_mode(), chunked_prefill_enabled():
        actual, weights = chunked(*args, scaling=8**-0.5, dropout=0.0,
                                  output_attentions=False)
    assert actual.shape == (2, 11, 4, 8)
    assert weights is None  # retaining all weights would recreate the OOM
    assert events
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_softmax_never_materializes_all_query_rows_and_always_sees_all_keys(monkeypatch):
    args = inputs(q_length=11, k_length=17)
    expected, _ = qwen3.eager_attention_forward(*args, scaling=0.25)
    real_softmax = torch.nn.functional.softmax
    shapes = []

    def observed_softmax(tensor, *args, **kwargs):
        shapes.append(tuple(tensor.shape))
        assert tensor.shape[-2] <= 4
        assert tensor.shape[-1] == 17
        assert kwargs.get("dtype") == torch.float32
        return real_softmax(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "softmax", observed_softmax)
    chunked = make_chunked_attention(qwen3.eager_attention_forward, chunk_size=4,
                                     min_query_length=5, emit=lambda event: None)
    with torch.inference_mode(), chunked_prefill_enabled():
        actual, weights = chunked(*args, scaling=0.25)
    assert [shape[-2] for shape in shapes] == [4, 4, 3]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert weights is None


@pytest.mark.parametrize("reason", ["disabled", "short", "decode", "training", "dropout", "attentions"])
def test_unsupported_or_ordinary_calls_delegate_to_the_original_unchanged(reason):
    args = list(inputs(q_length=1 if reason == "decode" else 3 if reason == "short" else 11))
    if reason == "training":
        args[0].training = True
    kwargs = {"scaling": 0.25, "dropout": 0.1 if reason == "dropout" else 0.0,
              "output_attentions": reason == "attentions", "sliding_window": None}
    sentinel = object()
    calls = []

    def original(*received, **received_kwargs):
        calls.append((received, received_kwargs))
        return sentinel

    chunked = make_chunked_attention(original, chunk_size=4, min_query_length=5,
                                     emit=lambda event: pytest.fail("Unexpected chunking"))
    with nullcontext() if reason == "disabled" else chunked_prefill_enabled():
        assert chunked(*args, **kwargs) is sentinel
    assert len(calls) == 1
    assert all(actual is expected for actual, expected in zip(calls[0][0], args))
    assert calls[0][1] == kwargs


def test_chunked_context_is_scoped_and_restored_after_exception():
    args = inputs()
    original = qwen3.eager_attention_forward
    events = []
    chunked = make_chunked_attention(original, chunk_size=4, min_query_length=5, emit=events.append)
    with pytest.raises(RuntimeError, match="leave scope"):
        with torch.inference_mode(), chunked_prefill_enabled():
            assert chunked(*args, scaling=0.25)[1] is None
            raise RuntimeError("leave scope")
    count = len(events)
    assert count > 0
    assert chunked(*args, scaling=0.25)[1] is not None
    assert len(events) == count


def test_small_qwen3_prefill_cached_decode_and_greedy_generation_match(monkeypatch):
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=64,
                        attention_dropout=0.0, bos_token_id=1, eos_token_id=None,
                        pad_token_id=0)
    config._attn_implementation = "eager"
    model = qwen3.Qwen3ForCausalLM(config).eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    input_ids = torch.tensor([[0, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                              [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
    mask = (input_ids != 0).long()
    next_ids = torch.tensor([[16], [17]])
    next_mask = torch.cat([mask, torch.ones(2, 1, dtype=mask.dtype)], dim=1)

    def run():
        prefill = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
        decoded = model(input_ids=next_ids, attention_mask=next_mask,
                        past_key_values=prefill.past_key_values, use_cache=True)
        generated = model.generate(input_ids=input_ids, attention_mask=mask,
                                   max_new_tokens=4, do_sample=False, use_cache=True)
        return prefill.logits, decoded.logits, generated

    with torch.inference_mode():
        expected = run()
    events = []
    chunked = make_chunked_attention(qwen3.eager_attention_forward, chunk_size=4,
                                     min_query_length=5, emit=events.append)
    monkeypatch.setattr(qwen3, "eager_attention_forward", chunked)
    with torch.inference_mode(), chunked_prefill_enabled():
        actual = run()
    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=1e-5)
    assert torch.equal(actual[2], expected[2])
    assert events
