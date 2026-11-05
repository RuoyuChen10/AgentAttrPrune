"""Process-local, OOM-only Qwen3 eager attention recovery for evaluation.

Split query rows, never key/value context. Each row still uses the original
scaling, mask, FP32 softmax, and value projection. Short prefill and cached
decode use the original implementation. GPU GEMM shapes may affect rounding.
No installed library or shared AgentAttrPrune source is edited.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import gc

import torch

_enabled = ContextVar("qwen3_chunked_prefill_recovery", default=False)


@contextmanager
def chunked_prefill_enabled():
    token = _enabled.set(True)
    try:
        yield
    finally:
        _enabled.reset(token)


def make_chunked_attention(original, *, chunk_size=256, min_query_length=1024,
                           emit=lambda event: None):
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    if chunk_size < 1 or min_query_length < 1:
        raise ValueError("Chunk size and threshold must be positive")

    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        length = query.shape[-2]
        if (not _enabled.get() or length < min_query_length or length <= chunk_size
                or module.training or dropout != 0 or kwargs.get("output_attentions", False)
                or (torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)))):
            return original(module, query, key, value, attention_mask,
                            scaling=scaling, dropout=dropout, **kwargs)

        if attention_mask is not None and attention_mask.shape[-2] not in (1, length):
            raise ValueError("Unexpected attention mask query dimension")
        keys = repeat_kv(key, module.num_key_value_groups)
        values = repeat_kv(value, module.num_key_value_groups)
        result = query.new_empty((query.shape[0], length, query.shape[1], values.shape[-1]))
        key_length = keys.shape[-2]
        for start in range(0, length, chunk_size):
            stop = min(start + chunk_size, length)
            weights = torch.matmul(query[:, :, start:stop, :], keys.transpose(2, 3)) * scaling
            if attention_mask is not None:
                mask = attention_mask[..., :key_length]
                if mask.shape[-2] != 1:
                    mask = mask[:, :, start:stop, :]
                weights = weights + mask
            weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
            # Dropout is exactly zero on this guarded inference-only path.
            output = torch.matmul(weights, values)
            result[:, start:stop, :, :] = output.transpose(1, 2)
            del weights, output
        emit({"event": "chunked_attention", "layer": getattr(module, "layer_idx", None),
              "batch_size": query.shape[0], "query_length": length, "key_length": key_length,
              "query_chunk_size": chunk_size, "device": str(query.device)})
        return result, None

    return attention


def make_retrying_generate(original, release_memory, emit, *, min_query_length=1024):
    """Retry only model.generate, before trace observers or graph code run."""
    from AgentAttrPrune.llm.hf_chat import _clear_exception_tracebacks

    def generate(*args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        eligible = (isinstance(ids, torch.Tensor) and ids.ndim == 2
                    and ids.shape[-1] >= min_query_length and kwargs.get("do_sample") is False
                    and kwargs.get("past_key_values") is None and not _enabled.get())
        try:
            return original(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as error:
            if not eligible:
                raise
            message = str(error)
            _clear_exception_tracebacks(error)
        # Run outside the exception handler after failed generation/KV frames
        # have been freed. The caller-owned input tensors are reused verbatim.
        release_memory()
        emit({"event": "prefill_oom_retry", "batch_size": int(ids.shape[0]),
              "input_length": int(ids.shape[-1]), "max_new_tokens": kwargs.get("max_new_tokens"),
              "do_sample": False, "error": message})
        with chunked_prefill_enabled():
            result = original(*args, **kwargs)
        emit({"event": "prefill_retry_completed", "batch_size": int(ids.shape[0]),
              "input_length": int(ids.shape[-1])})
        return result

    return generate


def install_recovery(emit):
    """Install before the evaluator constructs its sole shared model instance."""
    from AgentAttrPrune.llm.hf_chat import HFChat
    from transformers.models.qwen3 import modeling_qwen3

    if HFChat._instances:
        raise RuntimeError("Install recovery before loading the evaluation model")
    original_attention = modeling_qwen3.eager_attention_forward
    modeling_qwen3.eager_attention_forward = make_chunked_attention(original_attention, emit=emit)
    original_init = HFChat.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if getattr(self, "_chunked_prefill_recovery_installed", False):
            return
        if self.model.config.model_type != "qwen3" or self.model.config._attn_implementation != "eager":
            raise ValueError("This recovery is restricted to the original Qwen3 eager backend")
        if self.model.training or getattr(self.model.generation_config, "cache_implementation", None) is not None:
            raise ValueError("Recovery requires eval mode and the original dynamic KV cache")

        def release_memory():
            gc.collect()
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    torch.cuda.empty_cache()

        self.model.generate = make_retrying_generate(self.model.generate, release_memory, emit)
        self._chunked_prefill_recovery_installed = True
        emit({"event": "recovery_installed", "attention_backend": "eager",
              "device_map": {k: str(v) for k, v in getattr(self.model, "hf_device_map", {}).items()},
              "query_chunk_size": 256, "min_query_length": 1024,
              "activation": "retry the same greedy generation only after CUDA OOM"})

    HFChat.__init__ = initialize
