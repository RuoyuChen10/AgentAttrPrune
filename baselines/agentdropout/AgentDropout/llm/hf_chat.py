"""Local Hugging Face backend with optional one-model-per-GPU replicas."""

from __future__ import annotations

import asyncio
import os
import queue
import threading
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from AgentDropout.llm.format import Message
from AgentDropout.llm.llm import LLM
from AgentDropout.llm.llm_registry import LLMRegistry


class _GenerationRequest:
    def __init__(self, prompt: str, max_tokens: int, temperature: float):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.event = threading.Event()
        self.result: Optional[str] = None
        self.error: Optional[BaseException] = None


@LLMRegistry.register("HFChat")
class HFChat(LLM):
    """Share a local model pool across every node and graph deepcopy.

    ``HF_CHAT_NUM_REPLICAS`` defaults to one.  When several GPUs are visible,
    setting it to (for example) four places one complete model replica on each
    of cuda:0..cuda:3 and schedules concurrent requests across the replicas.
    """

    _instances: Dict[str, "HFChat"] = {}
    _instances_lock = threading.Lock()

    def __new__(cls, model_name: str):
        with cls._instances_lock:
            if model_name not in cls._instances:
                cls._instances[model_name] = super().__new__(cls)
            return cls._instances[model_name]

    def __init__(self, model_name: str):
        if getattr(self, "_initialized", False):
            return
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        requested_replicas = int(os.environ.get("HF_CHAT_NUM_REPLICAS", "1"))
        if requested_replicas < 1:
            raise ValueError("HF_CHAT_NUM_REPLICAS must be positive")
        if torch.cuda.is_available():
            replica_count = min(requested_replicas, torch.cuda.device_count())
            devices = [torch.device(f"cuda:{index}") for index in range(replica_count)]
        else:
            if requested_replicas != 1:
                raise RuntimeError("Multiple HF chat replicas require CUDA")
            devices = [torch.device("cpu")]

        self.models = []
        for device in devices:
            print(f"Loading {model_name} on {device}", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                trust_remote_code=True,
                local_files_only=True,
            ).to(device)
            model.eval()
            # Qwen's shipped generation defaults are sampling-oriented.  Greedy
            # runs ignore them, but clearing them avoids misleading warnings and
            # makes the effective decoding configuration explicit.
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
            self.models.append(model)
        self._requests: queue.Queue[_GenerationRequest] = queue.Queue()
        self._batch_size = int(os.environ.get("HF_CHAT_BATCH_SIZE", "4"))
        self._batch_wait_seconds = float(os.environ.get("HF_CHAT_BATCH_WAIT", "0.02"))
        if self._batch_size < 1:
            raise ValueError("HF_CHAT_BATCH_SIZE must be positive")
        self._workers = []
        for index in range(len(self.models)):
            worker = threading.Thread(
                target=self._batch_worker,
                args=(index,),
                name=f"hf-chat-cuda-worker-{index}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
        self._initialized = True

    def __deepcopy__(self, memo):
        return self

    @staticmethod
    def _message_to_dict(message: Union[Message, dict]) -> dict:
        if isinstance(message, dict):
            return message
        return {"role": message.role, "content": message.content}

    def _render_messages(self, messages: Union[str, List[Union[Message, dict]]]) -> str:
        if isinstance(messages, str):
            return messages
        rendered = [self._message_to_dict(message) for message in messages]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            template_kwargs = {}
            enable_thinking = os.environ.get("HF_CHAT_ENABLE_THINKING", "0")
            template_kwargs["enable_thinking"] = enable_thinking not in {
                "0", "false", "False"
            }
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

    def _batch_worker(self, replica_index: int) -> None:
        model = self.models[replica_index]
        while True:
            first = self._requests.get()
            batch = [first]
            signature = (first.max_tokens, first.temperature)
            while len(batch) < self._batch_size:
                try:
                    candidate = self._requests.get(timeout=self._batch_wait_seconds)
                except queue.Empty:
                    break
                if (candidate.max_tokens, candidate.temperature) != signature:
                    self._requests.put(candidate)
                    break
                batch.append(candidate)
            try:
                prompts = [request.prompt for request in batch]
                inputs = self.tokenizer(
                    prompts, return_tensors="pt", padding=True
                ).to(model.device)
                do_sample = first.temperature > 0
                generation_kwargs = {}
                if do_sample:
                    generation_kwargs["temperature"] = first.temperature
                    if os.environ.get("HF_CHAT_TOP_P"):
                        generation_kwargs["top_p"] = float(os.environ["HF_CHAT_TOP_P"])
                    if os.environ.get("HF_CHAT_TOP_K"):
                        generation_kwargs["top_k"] = int(os.environ["HF_CHAT_TOP_K"])
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=first.max_tokens,
                        do_sample=do_sample,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        **generation_kwargs,
                    )
                prompt_length = inputs["input_ids"].shape[-1]
                responses = self.tokenizer.batch_decode(
                    output[:, prompt_length:], skip_special_tokens=True
                )
                for request, response in zip(batch, responses):
                    request.result = response
            except BaseException as error:
                for request in batch:
                    request.error = error
            finally:
                for request in batch:
                    request.event.set()

    def _generate(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        max_tokens = self.DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        temperature = self.DEFAULT_TEMPERATURE if temperature is None else temperature
        num_comps = self.DEFUALT_NUM_COMPLETIONS if num_comps is None else num_comps
        if num_comps != 1:
            raise ValueError("The batched local backend supports one completion per request")
        request = _GenerationRequest(
            self._render_messages(messages), max_tokens, temperature
        )
        self._requests.put(request)
        request.event.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RuntimeError("local generation completed without a result")
        return request.result

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        return self._generate(messages, max_tokens, temperature, num_comps)

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        return await asyncio.to_thread(
            self._generate, messages, max_tokens, temperature, num_comps
        )
