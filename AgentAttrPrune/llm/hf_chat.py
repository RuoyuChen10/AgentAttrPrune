from __future__ import annotations

import asyncio
import contextvars
import copy
import os
import threading
from typing import Any, Callable, Mapping, Sequence

from .base import LLM, Message


GenerationObserver = Callable[[dict[str, Any]], None]
generation_observer: contextvars.ContextVar[GenerationObserver | None] = (
    contextvars.ContextVar("agent_attr_prune_generation_observer", default=None)
)


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
                device_map="auto",
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

    async def agen(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        num_comps: int = 1,
    ) -> str | list[str]:
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
