"""Additive SafeSieve nodes with the exact frozen AttrPrune MMLU prompts."""
from __future__ import annotations

import contextvars
import os
from pathlib import Path
import sys

ATTRPRUNE_ROOT = Path(os.environ.get("BASELINES_ATTRPRUNE_ROOT", Path(__file__).resolve().parents[3])).expanduser().resolve()
if str(ATTRPRUNE_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTRPRUNE_ROOT))

from AgentAttrPrune.llm import HFChat
from AgentAttrPrune.prompt import MMLUPromptSet
from SafeSieve.agents.agent_registry import AgentRegistry
from SafeSieve.graph.node import Node


WORKER_NAME = "AttrPruneMMLUWorker"
REVIEWER_NAME = "AttrPruneMMLUReviewer"
node_context = contextvars.ContextVar("safesieve_aligned_node", default=None)
node_failure = contextvars.ContextVar("safesieve_aligned_failures", default=None)
_alignment = None
_llm_factory = HFChat


def configure(alignment: dict, *, llm_factory=None):
    global _alignment, _llm_factory
    workers = alignment["workers"]
    ids = [worker["agent_id"] for worker in workers]
    aliases = MMLUPromptSet.prepare_reviewer_agent_ids(ids, alignment["reviewer_agent_ids"])
    contract = {"workers": workers, "reviewer_agent_ids": aliases,
                "decoding": alignment["reference_run_config"]["decoding"]}
    if _alignment is not None and _alignment != contract:
        raise ValueError("Cannot change a frozen prompt contract within one process")
    _alignment = contract
    _llm_factory = llm_factory or HFChat


def _items(info, *, canonical=False):
    workers = {worker["agent_id"]: worker for worker in _alignment["workers"]}
    if set(info) - set(workers):
        raise ValueError("Incoming source is not a canonical AttrPrune worker")
    order = [agent_id for agent_id in workers if agent_id in info] if canonical else list(info)
    result = []
    for agent_id in order:
        value = info[agent_id]
        if value["role"] != workers[agent_id]["role"]:
            raise ValueError("Incoming source role differs from its canonical identity")
        result.append({"agent_id": agent_id, "role": value["role"], "output": value["output"]})
    return result


class _TextNode(Node):
    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        messages = self.build_messages(raw_inputs, spatial_info, temporal_info)
        return tuple(message["content"] for message in messages)

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        return self.llm.gen(self.build_messages(input, spatial_info, temporal_info),
                            temperature=0, max_tokens=self.max_tokens)

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        token = node_context.set({"agent_id": self.id, "role": self.role})
        try:
            return await self.llm.agen(self.build_messages(input, spatial_info, temporal_info),
                                       temperature=0, max_tokens=self.max_tokens)
        except BaseException as error:
            errors = node_failure.get()
            if errors is not None:
                errors.append(f"{self.id}: {type(error).__name__}: {error}")
            raise
        finally:
            node_context.reset(token)


@AgentRegistry.register(WORKER_NAME)
class AttrPruneMMLUWorker(_TextNode):
    def __init__(self, id=None, role=None, domain="mmlu", llm_name=""):
        if _alignment is None:
            raise ValueError("Configure the frozen alignment before constructing a graph")
        worker = next((w for w in _alignment["workers"] if w["agent_id"] == id), None)
        if worker is None or worker["role"] != role:
            raise ValueError("Worker identity/role must match the canonical reference")
        super().__init__(id, WORKER_NAME, domain, llm_name)
        self.role = role
        self.max_tokens = _alignment["decoding"]["max_new_tokens"]
        self.llm = _llm_factory(llm_name)

    def build_messages(self, raw_inputs, spatial_info, temporal_info):
        return MMLUPromptSet.worker_messages(
            role=self.role, task=raw_inputs["task"],
            spatial_incoming=_items(spatial_info),
            temporal_incoming=_items(temporal_info, canonical=True))


@AgentRegistry.register(REVIEWER_NAME)
class AttrPruneMMLUReviewer(_TextNode):
    def __init__(self, id=None, domain="mmlu", llm_name=""):
        if _alignment is None:
            raise ValueError("Configure the frozen alignment before constructing a graph")
        super().__init__(id or "dense_reviewer", REVIEWER_NAME, domain, llm_name)
        self.role = MMLUPromptSet.get_decision_role()
        self.max_tokens = _alignment["decoding"]["max_new_tokens"]
        self.llm = _llm_factory(llm_name)
        self.reviewer_agent_ids = dict(_alignment["reviewer_agent_ids"])

    def build_messages(self, raw_inputs, spatial_info, temporal_info):
        if temporal_info:
            raise ValueError("The reviewer reads final-round worker outputs only")
        outputs = _items(spatial_info, canonical=True)
        for item in outputs:
            item["reviewer_agent_id"] = self.reviewer_agent_ids[item["agent_id"]]
        return MMLUPromptSet.reviewer_messages(task=raw_inputs["task"], worker_outputs=outputs)
