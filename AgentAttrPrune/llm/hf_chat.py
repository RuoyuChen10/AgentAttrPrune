from __future__ import annotations

import asyncio
import contextvars
import copy
import gc
import logging
import os
import threading
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

from .base import LLM, Message


GenerationObserver = Callable[[dict[str, Any]], None]
generation_observer: contextvars.ContextVar[GenerationObserver | None] = (
    contextvars.ContextVar("agent_attr_prune_generation_observer", default=None)
)
_batch_sessions: contextvars.ContextVar[Mapping[int, "_AsyncBatchSession"]] = (
    contextvars.ContextVar("agent_attr_prune_batch_sessions", default={})
)
_logger = logging.getLogger(__name__)


def _clear_exception_tracebacks(error: BaseException) -> None:
    """Release failed generation frames before attempting another CUDA call."""
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        pending.extend(
            linked
            for linked in (current.__cause__, current.__context__)
            if linked is not None
        )
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None


def _release_cuda_memory() -> None:
    import torch

    # Run after leaving the exception handler, when its references are gone.
    gc.collect()
    torch.cuda.empty_cache()


@dataclass
class _BatchRequest:
    messages: list[dict[str, str]]
    max_tokens: int
    context: contextvars.Context
    future: asyncio.Future


class _AsyncBatchSession:
    """One model worker, scoped to a caller's async context and event loop."""

    def __init__(self, llm: "HFChat", batch_size: int) -> None:
        self.llm = llm
        self.batch_size = batch_size
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue[_BatchRequest | None] = asyncio.Queue()
        self.closed = False
        self.worker = asyncio.create_task(self._run())

    async def submit(self, messages: Sequence[Message], max_tokens: int) -> str:
        if self.closed or asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("HFChat batching session is closed or belongs to another loop")
        future = self.loop.create_future()
        self.queue.put_nowait(
            _BatchRequest(
                [self.llm._message_to_dict(message) for message in messages],
                max_tokens,
                contextvars.copy_context(),
                future,
            )
        )
        return await future

    async def close(self) -> None:
        self.closed = True
        self.queue.put_nowait(None)
        # Cancelling a to_thread await cannot stop model.generate. Finish the
        # worker before releasing the session, including on caller cancellation.
        try:
            await asyncio.shield(self.worker)
        except asyncio.CancelledError:
            await asyncio.shield(self.worker)
            raise

    def _generate_single(
        self, request: _BatchRequest
    ) -> tuple[str, dict[str, Any] | None]:
        # Keep the original unpadded generation path, but defer observer delivery
        # until the future is still live. Observer errors must not trigger OOM
        # recovery or duplicate a generation.
        events: list[dict[str, Any]] = []
        token = generation_observer.set(events.append)
        try:
            response = self.llm._generate(
                request.messages,
                max_tokens=request.max_tokens,
                temperature=0.0,
                num_comps=1,
            )
        finally:
            generation_observer.reset(token)
        return response, events[0] if events else None

    def _deliver(
        self,
        requests: Sequence[_BatchRequest],
        results: Sequence[tuple[str, dict[str, Any] | None]],
        *,
        split_depth: int,
    ) -> None:
        for request, (response, event) in zip(requests, results):
            if request.future.done():
                continue
            try:
                observer = request.context.get(generation_observer)
                if observer is not None and event is not None:
                    event = {
                        **event,
                        "generation_kwargs": {
                            **event.get("generation_kwargs", {}),
                            "requested_batch_size": self.batch_size,
                            "effective_batch_size": len(requests),
                            "oom_split_depth": split_depth,
                            "oom_fallback": split_depth > 0,
                        },
                    }
                    request.context.run(observer, event)
            except Exception as error:
                request.future.set_exception(error)
            else:
                request.future.set_result(response)

    async def _process_batch(
        self, requests: Sequence[_BatchRequest], *, split_depth: int = 0
    ) -> None:
        import torch

        batch = [request for request in requests if not request.future.done()]
        if not batch:
            return
        try:
            if len(batch) == 1:
                first = batch[0]
                results = [await asyncio.to_thread(
                    first.context.run, self._generate_single, first
                )]
            else:
                results = await asyncio.to_thread(self.llm._generate_batch, batch)
        except torch.cuda.OutOfMemoryError as error:
            oom_message = str(error)
            _clear_exception_tracebacks(error)
        except Exception as error:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(error)
            return
        else:
            self._deliver(batch, results, split_depth=split_depth)
            return

        await asyncio.to_thread(_release_cuda_memory)
        if len(batch) == 1:
            # Surface a fresh exception without the CUDA allocation frames. A
            # terminal failure for one request must not poison its sibling.
            if not batch[0].future.done():
                batch[0].future.set_exception(torch.cuda.OutOfMemoryError(oom_message))
            return

        midpoint = len(batch) // 2
        _logger.warning(
            "CUDA OOM for generation batch of %d requests; retrying sequential "
            "batches of %d and %d (split depth %d)",
            len(batch), midpoint, len(batch) - midpoint, split_depth + 1,
        )
        # Deliver each successful sub-batch before starting the next. Recheck
        # cancellation at every recursive call and preserve request order.
        await self._process_batch(batch[:midpoint], split_depth=split_depth + 1)
        await self._process_batch(batch[midpoint:], split_depth=split_depth + 1)

    async def _run(self) -> None:
        carry: _BatchRequest | None = None
        stopping = False
        while not stopping:
            first = carry if carry is not None else await self.queue.get()
            carry = None
            if first is None:
                break
            if first.future.cancelled():
                continue
            # Let other ready sample tasks enqueue their next graph invocation.
            await asyncio.sleep(0)
            batch = [first]
            while len(batch) < self.batch_size:
                try:
                    candidate = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if candidate is None:
                    stopping = True
                    break
                if candidate.future.cancelled():
                    continue
                if candidate.max_tokens != first.max_tokens:
                    carry = candidate
                    break
                batch.append(candidate)
            batch = [request for request in batch if not request.future.cancelled()]
            if not batch:
                continue
            await self._process_batch(batch)


def _truthy_environment(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _generation_finish_reasons(
    generated_ids: Any,
    *,
    eos_token_id: Any,
    max_tokens: int,
) -> list[str]:
    if eos_token_id is None:
        eos_token_ids: set[int] = set()
    elif isinstance(eos_token_id, (list, tuple, set)):
        eos_token_ids = {int(value) for value in eos_token_id}
    else:
        eos_token_ids = {int(eos_token_id)}
    reasons = []
    for sequence in generated_ids:
        if hasattr(sequence, "detach"):
            sequence = sequence.detach().cpu().tolist()
        token_ids = [int(value) for value in sequence]
        if token_ids and token_ids[-1] in eos_token_ids:
            reasons.append("eos")
        elif len(token_ids) >= max_tokens:
            reasons.append("length")
        else:
            reasons.append("stopped")
    return reasons


class HFChat(LLM):
    """Local Hugging Face causal LM backend.

    Model names are intentionally explicit: only ``local:/path`` and
    ``hf:/path-or-hub-id`` are accepted. Instances are cached so all agents in
    a graph share one loaded model.
    """

    _instances: dict[str, "HFChat"] = {}
    _instances_lock = threading.Lock()

    def __new__(cls, model_name: str):
        model_path = cls.resolve_model_name(model_name)
        with cls._instances_lock:
            instance = cls._instances.get(model_path)
            if instance is None:
                instance = super().__new__(cls)
                instance._initialization_lock = threading.Lock()
                cls._instances[model_path] = instance
            return instance

    def __init__(self, model_name: str) -> None:
        with self._initialization_lock:
            if getattr(self, "_initialized", False):
                return
            model_path = self.resolve_model_name(model_name)
            try:
                import torch
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    LogitsProcessorList,
                )
            except ImportError as error:  # pragma: no cover - environment failure
                raise ImportError(
                    "HFChat requires torch and transformers; "
                    "install AgentAttrPrune first"
                ) from error

            self.model_name = model_name
            self.model_path = model_path
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="balanced",
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logits_processors = LogitsProcessorList()
            self._generation_lock = threading.Lock()
            self._initialized = True

    @staticmethod
    def resolve_model_name(model_name: str) -> str:
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string")
        for prefix in ("local:", "hf:"):
            if model_name.startswith(prefix):
                resolved = model_name[len(prefix) :]
                if not resolved:
                    raise ValueError(f"{prefix} model name cannot be empty")
                return resolved
        raise ValueError("HFChat model_name must start with 'local:' or 'hf:'")

    def __deepcopy__(self, memo: dict[int, Any]) -> "HFChat":
        return self

    @classmethod
    def clear_instance_cache(cls) -> None:
        """Drop cached references. Intended for tests, not live model unloading."""
        with cls._instances_lock:
            cls._instances.clear()

    def set_logits_processors(self, processors: Any) -> None:
        self.logits_processors = processors

    def add_logits_processor(self, processor: Any) -> None:
        self.logits_processors.append(processor)

    @staticmethod
    def _message_to_dict(message: Message) -> dict[str, str]:
        if isinstance(message, Mapping):
            return {
                "role": str(message["role"]),
                "content": str(message["content"]),
            }
        return {
            "role": str(getattr(message, "role")),
            "content": str(getattr(message, "content")),
        }

    def render_messages(self, messages: str | Sequence[Message]) -> str:
        if isinstance(messages, str):
            return messages
        rendered = [self._message_to_dict(message) for message in messages]
        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template
        ):
            template_kwargs: dict[str, Any] = {}
            if "HF_CHAT_ENABLE_THINKING" in os.environ:
                template_kwargs["enable_thinking"] = _truthy_environment(
                    "HF_CHAT_ENABLE_THINKING", False
                )
            return self.tokenizer.apply_chat_template(
                rendered,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        return "\n".join(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in rendered
        ) + "\nAssistant:"

    def _generate(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float,
        num_comps: int,
    ) -> str | list[str]:
        import torch

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if num_comps <= 0:
            raise ValueError("num_comps must be positive")

        message_dicts = [self._message_to_dict(message) for message in messages]
        prompt = self.render_messages(message_dicts)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_comps,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "logits_processor": copy.copy(self.logits_processors),
            "return_dict_in_generate": True,
            "output_scores": _truthy_environment("HF_CHAT_CAPTURE_SCORES", False),
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            if os.environ.get("HF_CHAT_TOP_P"):
                generate_kwargs["top_p"] = float(os.environ["HF_CHAT_TOP_P"])
            if os.environ.get("HF_CHAT_TOP_K"):
                generate_kwargs["top_k"] = int(os.environ["HF_CHAT_TOP_K"])

        with self._generation_lock, torch.inference_mode():
            output = self.model.generate(**inputs, **generate_kwargs)

        prompt_length = int(inputs["input_ids"].shape[-1])
        generated_ids = output.sequences[:, prompt_length:]
        responses = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        finish_reasons = _generation_finish_reasons(
            generated_ids,
            eos_token_id=generate_kwargs["eos_token_id"],
            max_tokens=max_tokens,
        )
        observer = generation_observer.get()
        if observer is not None:
            observer(
                {
                    "model_name": self.model_name,
                    "model_path": self.model_path,
                    "messages": message_dicts,
                    "rendered_prompt": prompt,
                    "input_ids": inputs["input_ids"].detach().cpu(),
                    "attention_mask": (
                        inputs["attention_mask"].detach().cpu()
                        if "attention_mask" in inputs
                        else None
                    ),
                    "generated_ids": generated_ids.detach().cpu(),
                    "output_text": responses,
                    "finish_reasons": finish_reasons,
                    "hit_max_new_tokens": [
                        reason == "length" for reason in finish_reasons
                    ],
                    "generation_kwargs": {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "num_completions": num_comps,
                        "do_sample": do_sample,
                        **{
                            key: generate_kwargs[key]
                            for key in ("top_p", "top_k")
                            if key in generate_kwargs
                        },
                    },
                }
            )
        return responses[0] if num_comps == 1 else responses

    def gen(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        num_comps: int = 1,
    ) -> str | list[str]:
        return self._generate(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            num_comps=num_comps,
        )

    def _validate_batching(self) -> None:
        if self.logits_processors:
            raise ValueError("HFChat batching does not support custom logits processors")
        config = getattr(self.model, "generation_config", None)
        if getattr(config, "num_beams", 1) != 1:
            raise ValueError("HFChat batching requires greedy decoding (num_beams=1)")
        if getattr(config, "min_length", 0):
            raise ValueError("HFChat batching does not support absolute min_length")
        for option in ("max_time", "stop_strings"):
            if getattr(config, option, None) is not None:
                raise ValueError(f"HFChat batching does not support {option}")

    @asynccontextmanager
    async def async_batching(self, batch_size: int = 1) -> AsyncIterator[None]:
        """Batch concurrent ``agen`` calls made by tasks in this context.

        Size one retains the original path. Larger sizes require greedy,
        single-completion decoding. Graph dependencies remain the caller's
        responsibility; batching can introduce GPU floating-point differences.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        sessions = dict(_batch_sessions.get())
        if id(self) in sessions:
            raise RuntimeError("HFChat batching contexts for the same model cannot nest")
        if batch_size == 1:
            yield
            return
        self._validate_batching()
        session = _AsyncBatchSession(self, batch_size)
        sessions[id(self)] = session
        token = _batch_sessions.set(sessions)
        try:
            yield
        finally:
            _batch_sessions.reset(token)
            await session.close()

    def _generate_batch(
        self, requests: Sequence[_BatchRequest]
    ) -> list[tuple[str, dict[str, Any]]]:
        """Generate compatible requests and return unpadded per-request traces."""
        import torch

        self._validate_batching()
        prompts = [self.render_messages(request.messages) for request in requests]
        # Tokenize exactly as the single-request path does. Padding is local to
        # this call; the shared tokenizer's padding_side is never mutated.
        tokenized = [self.tokenizer(prompt, return_tensors="pt") for prompt in prompts]
        for inputs in tokenized:
            unexpected = set(inputs) - {"input_ids", "attention_mask"}
            if unexpected:
                raise ValueError(
                    "HFChat batching cannot safely pad tokenizer fields: "
                    + ", ".join(sorted(unexpected))
                )
        lengths = [int(inputs["input_ids"].shape[-1]) for inputs in tokenized]
        width = max(lengths)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("HFChat batching requires a pad_token_id")
        input_ids = tokenized[0]["input_ids"].new_full((len(requests), width), pad_id)
        attention_mask = torch.zeros_like(input_ids)
        for index, (inputs, length) in enumerate(zip(tokenized, lengths)):
            input_ids[index, width - length:] = inputs["input_ids"][0]
            attention_mask[index, width - length:] = inputs.get(
                "attention_mask", torch.ones_like(inputs["input_ids"])
            )[0]
        max_tokens = requests[0].max_tokens
        if any(request.max_tokens != max_tokens for request in requests):
            raise ValueError("Batched generation requests must share max_tokens")
        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "num_return_sequences": 1,
            "pad_token_id": pad_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "logits_processor": copy.copy(self.logits_processors),
            "return_dict_in_generate": True,
            "output_scores": _truthy_environment("HF_CHAT_CAPTURE_SCORES", False),
        }
        with self._generation_lock, torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device),
                **generate_kwargs,
            )
        generated = output.sequences[:, width:].detach().cpu()
        eos_id = generate_kwargs["eos_token_id"]
        eos_ids = set(eos_id) if isinstance(eos_id, (list, tuple, set)) else {eos_id}
        results = []
        for index, request in enumerate(requests):
            token_ids = generated[index].tolist()
            # HF pads finished rows until the longest row stops. Preserve the
            # first EOS (including when EOS == PAD), but discard all later pads.
            stop = next(
                (offset + 1 for offset, token in enumerate(token_ids) if token in eos_ids),
                len(token_ids),
            )
            generated_ids = generated[index:index + 1, :stop]
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            finish_reasons = _generation_finish_reasons(
                generated_ids, eos_token_id=eos_id, max_tokens=max_tokens
            )
            inputs = tokenized[index]
            event = {
                "model_name": self.model_name,
                "model_path": self.model_path,
                "messages": request.messages,
                "rendered_prompt": prompts[index],
                "input_ids": inputs["input_ids"].detach().cpu(),
                "attention_mask": (
                    inputs["attention_mask"].detach().cpu()
                    if "attention_mask" in inputs else None
                ),
                "generated_ids": generated_ids,
                "output_text": responses,
                "finish_reasons": finish_reasons,
                "hit_max_new_tokens": [reason == "length" for reason in finish_reasons],
                "generation_kwargs": {
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "num_completions": 1,
                    "do_sample": False,
                },
            }
            results.append((responses[0], event))
        return results

    async def agen(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        num_comps: int = 1,
    ) -> str | list[str]:
        session = _batch_sessions.get().get(id(self))
        if session is not None:
            if temperature != 0 or num_comps != 1:
                raise ValueError(
                    "HFChat batching requires temperature=0 and num_comps=1"
                )
            if max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
            self._validate_batching()
            return await session.submit(messages, max_tokens)
        return await asyncio.to_thread(
            self._generate,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            num_comps=num_comps,
        )

    def next_token_logits(self, messages: Sequence[Message]):
        import torch

        prompt = self.render_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self._generation_lock, torch.inference_mode():
            output = self.model(**inputs)
        return output.logits[:, -1, :].detach().cpu()


def load_llm(model_name: str) -> HFChat:
    """Create the only production backend supported by this repository."""
    return HFChat(model_name)
