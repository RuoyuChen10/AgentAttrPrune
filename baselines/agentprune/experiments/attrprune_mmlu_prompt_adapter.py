"""New AgentPrune nodes sharing AgentAttrPrune's exact MMLU prompt builders."""
from __future__ import annotations

import contextvars
import os
from pathlib import Path
import sys

ATTRPRUNE_ROOT = Path(os.environ.get("BASELINES_ATTRPRUNE_ROOT",
    Path(__file__).resolve().parents[3])).expanduser().resolve()
if str(ATTRPRUNE_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTRPRUNE_ROOT))

from AgentAttrPrune.llm.hf_chat import HFChat, generation_observer
from AgentAttrPrune.prompt import MMLUPromptSet
from AgentPrune.agents.agent_registry import AgentRegistry
from AgentPrune.graph.node import Node

WORKER_NAME = "AttrPruneAlignedMMLUWorker"
REVIEWER_NAME = "AttrPruneAlignedMMLUReviewer"
invocation_recorder = contextvars.ContextVar("agentprune_aligned_mmlu_recorder", default=None)
_settings = None
_backend = None


def configure(alignment, *, model_name, max_new_tokens=2048, backend=None):
    global _settings, _backend
    workers = [(row["agent_id"], row["role"]) for row in alignment["workers"]]
    aliases = MMLUPromptSet.prepare_reviewer_agent_ids(
        [worker_id for worker_id, _ in workers], alignment["reviewer_agent_ids"]
    )
    settings = {"workers": workers, "aliases": aliases,
                "model_name": model_name, "max_new_tokens": max_new_tokens}
    if _settings is not None and _settings != settings:
        raise ValueError("MMLU prompt configuration is already frozen in this process")
    _settings = settings
    if backend is not None:
        if _backend is not None and _backend is not backend:
            raise ValueError("MMLU backend is already configured")
        _backend = backend


def backend():
    global _backend
    if _settings is None:
        raise ValueError("Configure the aligned MMLU prompt before constructing nodes")
    if _backend is None:
        _backend = HFChat(_settings["model_name"])
    return _backend


def incoming_items(info, *, canonical=False):
    roles = dict(_settings["workers"])
    if set(info) - set(roles):
        raise ValueError("Unknown canonical worker ID in MMLU input")
    ids = [agent_id for agent_id in roles if agent_id in info] if canonical else list(info)
    items = []
    for agent_id in ids:
        if info[agent_id]["role"] != roles[agent_id]:
            raise ValueError("MMLU source ID/role mismatch")
        items.append({"agent_id": agent_id, "role": roles[agent_id], "output": info[agent_id]["output"]})
    return items


class _TextNode(Node):
    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        messages = self.build_messages(raw_inputs, spatial_info, temporal_info)
        return messages[0]["content"], messages[1]["content"]

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        return self.llm.gen(self.build_messages(input, spatial_info, temporal_info),
                            temperature=0, max_tokens=_settings["max_new_tokens"])

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        messages = self.build_messages(input, spatial_info, temporal_info)
        recorder = invocation_recorder.get()
        invocation = recorder.begin(self, spatial_info, temporal_info, messages) if recorder else None
        token = generation_observer.set(invocation.capture) if invocation else None
        try:
            return await self.llm.agen(messages, temperature=0, max_tokens=_settings["max_new_tokens"])
        except Exception as error:
            if invocation:
                invocation.fail(error)
            raise
        finally:
            if token is not None:
                generation_observer.reset(token)


@AgentRegistry.register(WORKER_NAME)
class AttrPruneAlignedMMLUWorker(_TextNode):
    def __init__(self, id=None, role=None, domain="mmlu", llm_name=""):
        if _settings is None or dict(_settings["workers"]).get(id) != role:
            raise ValueError("Use an explicitly configured canonical MMLU worker ID and role")
        super().__init__(id, WORKER_NAME, domain, llm_name)
        self.role, self.llm = role, backend()

    def build_messages(self, raw_inputs, spatial_info, temporal_info):
        return MMLUPromptSet.worker_messages(
            role=self.role, task=raw_inputs["task"],
            spatial_incoming=incoming_items(spatial_info),
            temporal_incoming=incoming_items(temporal_info, canonical=True),
        )


@AgentRegistry.register(REVIEWER_NAME)
class AttrPruneAlignedMMLUReviewer(_TextNode):
    def __init__(self, id=None, domain="mmlu", llm_name=""):
        super().__init__(id or "aligned_mmlu_reviewer", REVIEWER_NAME, domain, llm_name)
        self.role, self.llm = MMLUPromptSet.get_decision_role(), backend()

    def build_messages(self, raw_inputs, spatial_info, temporal_info):
        if temporal_info:
            raise ValueError("MMLU reviewer must only receive the final-round worker outputs")
        outputs = incoming_items(spatial_info, canonical=True)
        for item in outputs:
            item["reviewer_agent_id"] = _settings["aliases"][item["agent_id"]]
        return MMLUPromptSet.reviewer_messages(task=raw_inputs["task"], worker_outputs=outputs)
