"""AgentAttrPrune MMLU messages on the original AgentDropout graph/optimizer.

Only the message contract and stable source identities change. Node dropout,
Bernoulli edges, cycle filters, FIFO execution and round handling are inherited.
"""
from __future__ import annotations

from AgentAttrPrune.prompt import MMLUPromptSet
from AgentDropout.agents.agent_registry import AgentRegistry
from AgentDropout.graph.graph import Graph
from AgentDropout.graph.node import Node
from AgentDropout.llm.llm_registry import LLMRegistry


def incoming_records(info: dict, order=None) -> list[dict]:
    # Match AnalyzeAgent's treatment of the released dropout sentinel.
    if order is not None:
        if set(info) - set(order):
            raise ValueError("Unknown temporal source identity")
        info = {identity: info[identity] for identity in order if identity in info}
    return [
        {"agent_id": agent_id, "role": value["role"], "output": value["output"]}
        for agent_id, value in info.items()
        if "None." not in (value["output"] if isinstance(value["output"], list)
                           else [value["output"]])
    ]


class _AlignedNode(Node):
    def __init__(self, id=None, role="", domain="mmlu", llm_name=""):
        super().__init__(id, self.__class__.__name__, domain, llm_name)
        self.role = role
        self.llm = LLMRegistry.get(llm_name)
        self.generation_errors = []
        self.worker_ids = []

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        try:
            messages = self._process_inputs(input, spatial_info, temporal_info)
            return await self.llm.agen(messages, max_tokens=2048, temperature=0.0)
        except Exception as error:
            self.generation_errors.append(repr(error))
            raise

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        return self.llm.gen(self._process_inputs(input, spatial_info, temporal_info),
                            max_tokens=2048, temperature=0.0)


@AgentRegistry.register("AttrPruneDev40MMLUWorker")
class AttrPruneDev40MMLUWorker(_AlignedNode):
    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        return MMLUPromptSet.worker_messages(
            role=self.role, task=raw_inputs["task"],
            spatial_incoming=incoming_records(spatial_info),
            temporal_incoming=incoming_records(temporal_info, self.worker_ids),
        )


@AgentRegistry.register("AttrPruneDev40MMLUReviewer")
class AttrPruneDev40MMLUReviewer(_AlignedNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = "dense_reviewer"
        self.role = MMLUPromptSet.get_decision_role()
        self.worker_ids = []
        self.reviewer_agent_ids = {}

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        # FinalRefer includes dropped workers as "None."; preserve that behavior.
        if set(spatial_info) != set(self.worker_ids):
            raise ValueError("Reviewer must receive every canonical worker, including dropout sentinels")
        outputs = [
            {"agent_id": agent_id, "reviewer_agent_id": self.reviewer_agent_ids[agent_id],
             "role": spatial_info[agent_id]["role"], "output": spatial_info[agent_id]["output"]}
            for agent_id in self.worker_ids
        ]
        return MMLUPromptSet.reviewer_messages(task=raw_inputs["task"], worker_outputs=outputs)


class AlignedGraph(Graph):
    """Original execution, but failed generation cannot silently score as zero."""
    async def arun(self, *args, **kwargs):
        for node in [*self.nodes.values(), self.decision_node]:
            node.generation_errors = []
        try:
            return await super().arun(*args, **kwargs)
        finally:
            errors = {node.id: node.generation_errors for node in [*self.nodes.values(), self.decision_node]
                      if node.generation_errors}
            if errors:
                raise RuntimeError(f"AgentDropout generation failed; refusing to record an answer: {errors}")


def build_graph(alignment: dict, model_path: str):
    workers = alignment["workers"]
    size = len(workers)
    graph = AlignedGraph(
        domain="mmlu", llm_name=f"local:{model_path}",
        agent_names=["AttrPruneDev40MMLUWorker"] * size,
        decision_method="AttrPruneDev40MMLUReviewer",
        optimized_spatial=True, initial_spatial_probability=0.5,
        fixed_spatial_masks=[[int(i != j) for j in range(size)] for i in range(size)],
        optimized_temporal=True, initial_temporal_probability=0.5,
        fixed_temporal_masks=[[1] * size for _ in range(size)],
        diff=True, dec=True, rounds=2,
        node_kwargs=[{"id": w["agent_id"], "role": w["role"]} for w in workers],
    )
    graph.decision_node.worker_ids = [w["agent_id"] for w in workers]
    graph.decision_node.reviewer_agent_ids = dict(alignment["reviewer_agent_ids"])
    for node in graph.nodes.values():
        node.worker_ids = list(graph.decision_node.worker_ids)
    return graph
