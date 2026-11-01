from __future__ import annotations

import asyncio
import contextvars
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch

from AgentAttrPrune.llm.hf_chat import HFChat, generation_observer
from AgentAttrPrune.tracing.recorder import to_jsonable


class Encoding(dict):
    def to(self, device):
        return Encoding({key: value.to(device) for key, value in self.items()})


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 0
    chat_template = "fake"
    padding_side = "right"

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

    def __call__(self, prompt, **kwargs):
        ids = torch.tensor([[int(value) for value in prompt.split()]])
        return Encoding(input_ids=ids, attention_mask=torch.ones_like(ids))

    def batch_decode(self, ids, **kwargs):
        return [" ".join(str(value) for value in row.tolist() if value) for row in ids]


class Model:
    device = torch.device("cpu")
    generation_config = SimpleNamespace(num_beams=1, min_length=0)

    def __init__(self):
        self.calls = []
        self.fail_next = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append((input_ids.clone(), attention_mask.clone(), dict(kwargs)))
        self.started.set()
        assert self.release.wait(timeout=5)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("fake generation failed")
        rows = []
        for row in input_ids:
            last = int(row[-1])
            output = [last + 10, 0] if last != 3 else [13, 14, 15, 0]
            rows.append(output[:kwargs["max_new_tokens"]])
        width = max(map(len, rows))
        output = torch.tensor([row + [0] * (width - len(row)) for row in rows])
        return SimpleNamespace(sequences=torch.cat((input_ids, output), dim=1))


def make_llm():
    llm = object.__new__(HFChat)
    llm.model_name = "hf:fake"
    llm.model_path = "fake"
    llm.tokenizer = Tokenizer()
    llm.model = Model()
    llm.logits_processors = []
    llm._generation_lock = threading.Lock()
    return llm


class Allocation:
    pass


class MemoryLimitedModel(Model):
    """CPU fake that keeps an allocation alive only through failed frames."""

    def __init__(self, max_batch_size=1, failing_tokens=()):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.failing_tokens = set(failing_tokens)
        self.attempts = []
        self.failed_allocations = []

    def _raise_oom(self):
        allocation = Allocation()
        self.failed_allocations.append(weakref.ref(allocation))
        try:
            raise RuntimeError("inner allocation frame")
        except RuntimeError as error:
            raise torch.cuda.OutOfMemoryError("fake CUDA allocation failed") from error

    def generate(self, input_ids, attention_mask, **kwargs):
        # A previous failed attempt must be fully released before any retry,
        # including another failed attempt or an unrelated request.
        assert all(reference() is None for reference in self.failed_allocations)
        self.attempts.append(input_ids.clone())
        if (input_ids.shape[0] > self.max_batch_size
                or any(int(row[-1]) in self.failing_tokens for row in input_ids)):
            self._raise_oom()
        return super().generate(input_ids, attention_mask, **kwargs)


def messages(prompt):
    return [{"role": "user", "content": prompt}]


async def observed_call(llm, prompt, events, **kwargs):
    token = generation_observer.set(lambda event: events.append(to_jsonable(event)))
    try:
        return await llm.agen(messages(prompt), **kwargs)
    finally:
        generation_observer.reset(token)


def test_batches_preserve_outputs_and_unpadded_trace_payloads():
    llm = make_llm()

    async def run():
        serial_events = [[], []]
        serial = [
            await observed_call(llm, prompt, events, max_tokens=3)
            for prompt, events in zip(("1", "0 2 3"), serial_events)
        ]
        llm.model.calls.clear()
        batch_events = [[], []]
        async with llm.async_batching(4):
            batched = await asyncio.gather(*[
                observed_call(llm, prompt, events, max_tokens=3)
                for prompt, events in zip(("1", "0 2 3"), batch_events)
            ])
        assert batched == serial == ["11", "13 14 15"]
        for events in serial_events:
            events[0]["generation_kwargs"].update({
                "requested_batch_size": 4,
                "effective_batch_size": 2,
                "oom_split_depth": 0,
                "oom_fallback": False,
            })
        assert batch_events == serial_events
        assert len(llm.model.calls) == 1
        ids, mask, _ = llm.model.calls[0]
        assert ids.tolist() == [[0, 0, 1], [0, 2, 3]]
        assert mask.tolist() == [[0, 0, 1], [1, 1, 1]]
        assert batch_events[0][0]["generated_ids"] == [[11, 0]]
        assert batch_events[0][0]["finish_reasons"] == ["eos"]
        assert batch_events[1][0]["finish_reasons"] == ["length"]
        assert llm.tokenizer.padding_side == "right"
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(run())


def test_singleton_and_disabled_batching_keep_original_path():
    llm = make_llm()

    async def run():
        expected = await observed_call(llm, "1", [], max_tokens=3)
        for size in (1, 4):
            async with llm.async_batching(size):
                assert await observed_call(llm, "1", [], max_tokens=3) == expected
        assert [call[0].shape for call in llm.model.calls] == [(1, 1)] * 3
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(run())


def test_batch_sizes_and_generation_limits_are_respected():
    llm = make_llm()

    async def run():
        async with llm.async_batching(2):
            results = await asyncio.gather(*[
                llm.agen(messages("3"), max_tokens=limit)
                for limit in (1, 1, 3, 3, 3)
            ])
        assert results == ["13", "13", "13 14 15", "13 14 15", "13 14 15"]
        assert [call[0].shape[0] for call in llm.model.calls] == [2, 2, 1]
        assert [call[2]["max_new_tokens"] for call in llm.model.calls] == [1, 3, 3]

    asyncio.run(run())


def test_generation_failure_reaches_all_requests_and_worker_recovers():
    llm = make_llm()
    llm.model.fail_next = True

    async def run():
        async with llm.async_batching(2):
            results = await asyncio.gather(
                llm.agen(messages("1")), llm.agen(messages("2")),
                return_exceptions=True,
            )
            assert all(isinstance(result, RuntimeError) for result in results)
            assert len(llm.model.calls) == 1  # Non-OOM failures are never split.
            assert await llm.agen(messages("1")) == "11"
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(run())


def test_cancelling_inflight_request_keeps_other_request_and_worker_valid():
    llm = make_llm()
    llm.model.release.clear()

    async def run():
        events = [[], []]
        async with llm.async_batching(2):
            first = asyncio.create_task(observed_call(llm, "1", events[0]))
            second = asyncio.create_task(observed_call(llm, "2", events[1]))
            assert await asyncio.to_thread(llm.model.started.wait, 5)
            first.cancel()
            llm.model.release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert await second == "12"
            assert events[0] == []
            assert events[1][0]["messages"] == messages("2")
            assert await llm.agen(messages("1")) == "11"
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(run())


@pytest.mark.parametrize("error_type", [RuntimeError, torch.cuda.OutOfMemoryError])
def test_observer_runs_in_request_context_and_failure_is_isolated(error_type):
    llm = make_llm()
    marker = contextvars.ContextVar("marker", default="unset")
    seen = []

    async def run_one(prompt, fail):
        marker_token = marker.set(prompt)

        def observe(event):
            seen.append((marker.get(), event["rendered_prompt"]))
            if fail:
                raise error_type("observer failed")

        observer_token = generation_observer.set(observe)
        try:
            return await llm.agen(messages(prompt))
        finally:
            generation_observer.reset(observer_token)
            marker.reset(marker_token)

    async def run():
        async with llm.async_batching(2):
            results = await asyncio.gather(
                run_one("1", True), run_one("2", False), return_exceptions=True
            )
        assert isinstance(results[0], error_type)
        assert results[1] == "12"
        assert seen == [("1", "1"), ("2", "2")]
        assert len(llm.model.calls) == 1

    asyncio.run(run())


def test_recursive_oom_splitting_releases_frames_and_preserves_exact_traces(monkeypatch):
    llm = make_llm()
    model = MemoryLimitedModel()
    cleanups = []

    def empty_cache():
        assert all(reference() is None for reference in model.failed_allocations)
        cleanups.append(True)

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    prompts = ("1", "0 2 3", "4", "0 5")

    async def run():
        expected_events = [[] for _ in prompts]
        expected = [
            await observed_call(llm, prompt, events, max_tokens=3)
            for prompt, events in zip(prompts, expected_events)
        ]
        llm.model = model
        events = [[] for _ in prompts]
        async with llm.async_batching(4):
            results = await asyncio.gather(*[
                observed_call(llm, prompt, captured, max_tokens=3)
                for prompt, captured in zip(prompts, events)
            ])
        assert results == expected == ["11", "13 14 15", "14", "15"]
        assert [list(ids.shape) for ids in model.attempts] == [
            [4, 3], [2, 3], [1, 1], [1, 3], [2, 2], [1, 1], [1, 2],
        ]
        assert len(cleanups) == 3
        for captured in expected_events:
            captured[0]["generation_kwargs"].update({
                "requested_batch_size": 4,
                "effective_batch_size": 1,
                "oom_split_depth": 2,
                "oom_fallback": True,
            })
        # Equality also proves one event per request, original unpadded inputs,
        # EOS trimming, finish reasons, and no phantom failed-attempt events.
        assert events == expected_events
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(run())


def test_successful_subbatch_delivered_before_later_irrecoverable_singleton(monkeypatch):
    llm = make_llm()
    delivered = []

    class PartiallyFailingModel(MemoryLimitedModel):
        def generate(self, input_ids, attention_mask, **kwargs):
            if input_ids.shape[0] == 2 and int(input_ids[0, -1]) == 3:
                assert delivered == ["1", "2"]
            return super().generate(input_ids, attention_mask, **kwargs)

    llm.model = PartiallyFailingModel(max_batch_size=2, failing_tokens={3})
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    events = [[] for _ in range(4)]

    async def run_one(index):
        def observe(event):
            delivered.append(event["rendered_prompt"])
            events[index].append(to_jsonable(event))

        token = generation_observer.set(observe)
        try:
            return await llm.agen(messages(str(index + 1)))
        finally:
            generation_observer.reset(token)

    async def run():
        async with llm.async_batching(4):
            results = await asyncio.gather(*[run_one(i) for i in range(4)], return_exceptions=True)
            assert results[:2] == ["11", "12"]
            assert isinstance(results[2], torch.cuda.OutOfMemoryError)
            assert "fake CUDA allocation failed" in str(results[2])
            assert results[3] == "14"
            assert await llm.agen(messages("5")) == "15"
        assert [ids.shape[0] for ids in llm.model.attempts] == [4, 2, 2, 1, 1, 1]
        assert [len(captured) for captured in events] == [1, 1, 0, 1]
        assert [events[i][0]["generation_kwargs"]["effective_batch_size"] for i in (0, 1, 3)] == [2, 2, 1]
        assert [events[i][0]["generation_kwargs"]["oom_split_depth"] for i in (0, 1, 3)] == [1, 1, 2]
        assert all(reference() is None for reference in llm.model.failed_allocations)

    asyncio.run(run())


def test_fallback_singleton_observer_oom_is_not_retried(monkeypatch):
    llm = make_llm()
    llm.model = MemoryLimitedModel()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    seen = []

    async def run_one(prompt):
        def observe(event):
            seen.append(event["rendered_prompt"])
            if prompt == "1":
                raise torch.cuda.OutOfMemoryError("observer OOM")

        token = generation_observer.set(observe)
        try:
            return await llm.agen(messages(prompt))
        finally:
            generation_observer.reset(token)

    async def run():
        async with llm.async_batching(2):
            results = await asyncio.gather(run_one("1"), run_one("2"), return_exceptions=True)
        assert isinstance(results[0], torch.cuda.OutOfMemoryError)
        assert results[1] == "12"
        assert seen == ["1", "2"]
        assert [ids.shape[0] for ids in llm.model.attempts] == [2, 1, 1]

    asyncio.run(run())


def test_cancelled_inflight_fallback_singleton_is_not_traced(monkeypatch):
    llm = make_llm()
    started = threading.Event()
    release = threading.Event()

    class BlockingSingletonModel(MemoryLimitedModel):
        def generate(self, input_ids, attention_mask, **kwargs):
            if input_ids.shape[0] == 1 and int(input_ids[0, -1]) == 1:
                started.set()
                assert release.wait(5)
            return super().generate(input_ids, attention_mask, **kwargs)

    llm.model = BlockingSingletonModel()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    async def run():
        events = [[], []]
        async with llm.async_batching(2):
            first = asyncio.create_task(observed_call(llm, "1", events[0]))
            second = asyncio.create_task(observed_call(llm, "2", events[1]))
            assert await asyncio.to_thread(started.wait, 5)
            first.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert await second == "12"
        assert events[0] == []
        assert len(events[1]) == 1
        assert events[1][0]["generation_kwargs"]["effective_batch_size"] == 1
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(run())


@pytest.mark.parametrize("kwargs", [{"temperature": 0.1}, {"num_comps": 2}, {"max_tokens": 0}])
def test_unsupported_decoding_is_rejected(kwargs):
    llm = make_llm()

    async def run():
        async with llm.async_batching(2):
            with pytest.raises(ValueError):
                await llm.agen(messages("1"), **kwargs)
        assert not llm.model.calls

    asyncio.run(run())


def test_custom_logits_processors_and_unsupported_tokenizer_fields_rejected():
    llm = make_llm()

    class ExtraTokenizer(Tokenizer):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            result["token_type_ids"] = torch.zeros_like(result["input_ids"])
            return result

    async def run():
        llm.logits_processors.append(object())
        with pytest.raises(ValueError, match="custom logits"):
            async with llm.async_batching(2):
                pass
        llm.logits_processors.clear()
        llm.tokenizer = ExtraTokenizer()
        async with llm.async_batching(2):
            results = await asyncio.gather(
                llm.agen(messages("1")), llm.agen(messages("2")), return_exceptions=True
            )
        assert all(isinstance(result, ValueError) for result in results)
        assert not llm.model.calls

    asyncio.run(run())


@pytest.mark.parametrize(
    ("option", "value"),
    [("num_beams", 2), ("min_length", 10), ("max_time", 1.0), ("stop_strings", ["STOP"])],
)
def test_unsupported_model_generation_settings_rejected(option, value):
    llm = make_llm()
    llm.model.generation_config = SimpleNamespace(**{option: value})

    async def run():
        with pytest.raises(ValueError, match=option):
            async with llm.async_batching(2):
                pass
        assert not llm.model.calls
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(run())


def test_context_exit_waits_for_inflight_generation_after_cancellation():
    llm = make_llm()
    llm.model.release.clear()

    async def owner():
        async with llm.async_batching(2):
            await llm.agen(messages("1"))

    async def run():
        task = asyncio.create_task(owner())
        assert await asyncio.to_thread(llm.model.started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        llm.model.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(asyncio.all_tasks()) == 1
        assert await llm.agen(messages("2")) == "12"

    asyncio.run(run())
