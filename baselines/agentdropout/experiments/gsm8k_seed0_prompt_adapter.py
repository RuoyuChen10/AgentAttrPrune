"""AgentAttrPrune GSM8K messages on AgentDropout's unchanged stochastic graph."""
from __future__ import annotations

from AgentAttrPrune.prompt import GSM8KPromptSet
from AgentDropout.agents.agent_registry import AgentRegistry
from experiments.mmlu_dev40_prompt_adapter import _AlignedNode, AlignedGraph


def incoming_records(info: dict, order=None) -> list[dict]:
    # Original GSM8K MathSolver includes the dropout sentinel. Do not inherit
    # the MMLU worker's sentinel filtering or execute generated Python code.
    if order is not None:
        if set(info) - set(order):
            raise ValueError("Unknown source identity")
        info = {identity: info[identity] for identity in order if identity in info}
    return [{"agent_id": identity, "role": item["role"], "output": item["output"]}
            for identity, item in info.items()]


@AgentRegistry.register("AttrPruneSeed0GSM8KWorker")
class AttrPruneSeed0GSM8KWorker(_AlignedNode):
    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        return GSM8KPromptSet.worker_messages(
            role=self.role, task=raw_inputs["task"],
            spatial_incoming=incoming_records(spatial_info),
            temporal_incoming=incoming_records(temporal_info, self.worker_ids),
        )


@AgentRegistry.register("AttrPruneSeed0GSM8KReviewer")
class AttrPruneSeed0GSM8KReviewer(_AlignedNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = "dense_reviewer"
        self.role = GSM8KPromptSet.get_decision_role()
        self.reviewer_agent_ids = {}

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        if set(spatial_info) != set(self.worker_ids):
            raise ValueError("Reviewer must receive every worker, including dropout sentinels")
        outputs = [{"agent_id": identity, "reviewer_agent_id": self.reviewer_agent_ids[identity],
                    "role": spatial_info[identity]["role"], "output": spatial_info[identity]["output"]}
                   for identity in self.worker_ids]
        return GSM8KPromptSet.reviewer_messages(task=raw_inputs["task"], worker_outputs=outputs)


def build_graph(alignment: dict, model_path: str):
    workers = alignment["workers"]
    size = len(workers)
    graph = AlignedGraph(
        domain="gsm8k", llm_name=f"local:{model_path}",
        agent_names=["AttrPruneSeed0GSM8KWorker"] * size,
        decision_method="AttrPruneSeed0GSM8KReviewer",
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
