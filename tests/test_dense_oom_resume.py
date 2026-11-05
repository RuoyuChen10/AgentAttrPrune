"""Generation retry and saved-result preservation for Dense OOM recovery."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

from experiments_attrprune.qwen3_chunked_prefill import (
    chunked_prefill_enabled,
    make_chunked_attention,
    make_retrying_generate,
)
from experiments_attrprune.resume_dense_gsm8k import verify_preserved


def recovery_is_enabled():
    attention = make_chunked_attention(lambda *args, **kwargs: (None, "original"),
                                      chunk_size=2, min_query_length=4)
    tensor = torch.zeros(1, 1, 5, 2)
    module = SimpleNamespace(num_key_value_groups=1, training=False)
    return attention(module, tensor, tensor, tensor, None, scaling=1.0)[1] is None


def settings():
    return {"input_ids": torch.arange(8)[None], "attention_mask": torch.ones(1, 8),
            "max_new_tokens": 2048, "do_sample": False,
            "return_dict_in_generate": True, "output_scores": False}


def test_first_success_preserves_inputs_and_does_not_activate_recovery():
    kwargs = settings()
    calls, releases, events = [], [], []
    result = object()

    def original(*args, **received):
        calls.append((args, received))
        assert not recovery_is_enabled()
        assert received.keys() == kwargs.keys()
        assert all(received[key] is value for key, value in kwargs.items())
        return result

    generate = make_retrying_generate(original, lambda: releases.append(True), events.append,
                                     min_query_length=4)
    assert generate(**kwargs) is result
    assert len(calls) == 1 and not releases and not events


def test_oom_cleans_failed_frames_then_retries_same_inputs_once_outside_handler():
    kwargs = settings()
    attempts, releases, events, references = [], [], [], []
    result = object()

    class Allocation:
        pass

    def original(*args, **received):
        attempts.append(recovery_is_enabled())
        assert received.keys() == kwargs.keys()
        assert all(received[key] is value for key, value in kwargs.items())
        if len(attempts) == 1:
            allocation = Allocation()
            references.append(weakref.ref(allocation))
            raise torch.cuda.OutOfMemoryError("simulated long prefill")
        assert releases == [True]
        return result

    def release():
        assert sys.exc_info() == (None, None, None)
        gc.collect()
        assert references[0]() is None
        releases.append(True)

    generate = make_retrying_generate(original, release, events.append, min_query_length=4)
    assert generate(**kwargs) is result
    assert attempts == [False, True]
    assert [event["event"] for event in events] == ["prefill_oom_retry", "prefill_retry_completed"]
    assert not recovery_is_enabled()


def test_second_oom_is_terminal_and_context_is_restored():
    calls, releases, events = [], [], []

    def original(**kwargs):
        calls.append(recovery_is_enabled())
        raise torch.cuda.OutOfMemoryError("still insufficient")

    generate = make_retrying_generate(original, lambda: releases.append(True), events.append,
                                     min_query_length=4)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="still insufficient"):
        generate(**settings())
    assert calls == [False, True]
    assert releases == [True]
    assert [event["event"] for event in events] == ["prefill_oom_retry"]
    assert not recovery_is_enabled()


def test_non_oom_generation_error_is_not_retried():
    calls, releases, events = [], [], []

    def original(**kwargs):
        calls.append(True)
        raise ValueError("invalid generation inputs")

    generate = make_retrying_generate(original, lambda: releases.append(True), events.append,
                                     min_query_length=4)
    with pytest.raises(ValueError, match="invalid generation inputs"):
        generate(**settings())
    assert calls == [True] and not releases and not events


@pytest.mark.parametrize("reason", ["sampling", "past_key_values", "short", "already_recovering"])
def test_ineligible_generation_is_never_retried(reason):
    kwargs = settings()
    if reason == "sampling":
        kwargs["do_sample"] = True
    elif reason == "past_key_values":
        kwargs["past_key_values"] = object()
    elif reason == "short":
        kwargs["input_ids"] = kwargs["input_ids"][:, :2]
        kwargs["attention_mask"] = kwargs["attention_mask"][:, :2]
    calls, releases, events = [], [], []

    def original(**received):
        calls.append(received)
        raise torch.cuda.OutOfMemoryError("not eligible")

    generate = make_retrying_generate(original, lambda: releases.append(True), events.append,
                                     min_query_length=4)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="not eligible"):
        if reason == "already_recovering":
            with chunked_prefill_enabled():
                generate(**kwargs)
        else:
            generate(**kwargs)
    assert len(calls) == 1 and not releases and not events
    assert not recovery_is_enabled()


@pytest.fixture
def saved_results(tmp_path):
    prediction = tmp_path / "predictions.jsonl"
    records = [{"sample_index": index, "correct": index % 2 == 0, "output": "text\u0085separator"}
               for index in range(704)]
    prefix = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode()
    prediction.write_bytes(prefix)
    protected = {}
    for name in ("run_config.json", "completed_trace.json", "trained_policy.pt"):
        path = tmp_path / name
        path.write_bytes(("original " + name).encode())
        protected[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    runtime = tmp_path / "runtime.py"
    runtime.write_bytes(b"original runtime source")
    manifest = {"protected_files_sha256": protected,
                "runtime_source_hashes": {str(runtime): hashlib.sha256(runtime.read_bytes()).hexdigest()},
                "predictions_prefix_bytes": len(prefix),
                "predictions_prefix_sha256": hashlib.sha256(prefix).hexdigest()}
    return tmp_path, manifest, prefix


def test_704_saved_records_allow_append_without_rewriting_prefix(saved_results):
    directory, manifest, prefix = saved_results
    verify_preserved(directory, manifest)
    with (directory / "predictions.jsonl").open("ab") as handle:
        handle.write(b'{"sample_index":704,"correct":true}\n')
    verify_preserved(directory, manifest)
    assert (directory / "predictions.jsonl").read_bytes().startswith(prefix)


@pytest.mark.parametrize("change", ["edit", "delete", "truncate"])
def test_saved_prediction_edits_or_deletions_are_rejected(saved_results, change):
    directory, manifest, prefix = saved_results
    if change == "edit":
        updated = prefix.replace(b'"sample_index": 0', b'"sample_index": 9', 1)
    elif change == "delete":
        lines = prefix.split(b"\n")
        del lines[123]
        updated = b"\n".join(lines)
    else:
        updated = prefix[:100]
    (directory / "predictions.jsonl").write_bytes(updated)
    with pytest.raises(ValueError, match="Previously completed predictions changed"):
        verify_preserved(directory, manifest)


@pytest.mark.parametrize("name", ["run_config.json", "completed_trace.json", "trained_policy.pt", "runtime.py"])
def test_config_training_trace_and_runtime_source_changes_are_rejected(saved_results, name):
    directory, manifest, _ = saved_results
    (directory / name).write_bytes(b"changed artifact")
    with pytest.raises(ValueError, match="Protected source, configuration, or completed trace changed"):
        verify_preserved(directory, manifest)
