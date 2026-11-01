from __future__ import annotations

from copy import deepcopy

from AgentAttrPrune.prompt import MMLUPromptSet


def test_reviewer_uses_saved_random_ids_without_mutating_provenance():
    worker_outputs = [
        {
            "agent_id": "knowledgeable_expert",
            "reviewer_agent_id": "7vWW",
            "role": "Knowlegable Expert",
            "output": "Reasoning: first view\nDecision: D",
        },
        {
            "agent_id": "critic",
            "reviewer_agent_id": "5h3K",
            "role": "Critic",
            "output": "Reasoning: second view\nDecision: A",
        },
    ]
    original = deepcopy(worker_outputs)

    messages = MMLUPromptSet.reviewer_messages(
        task="Question\nA. one\nB. two\nC. three\nD. four",
        worker_outputs=worker_outputs,
    )
    reviewer_prompt = messages[-1]["content"]

    assert "7vWW: Reasoning: first view\nDecision: D" in reviewer_prompt
    assert "5h3K: Reasoning: second view\nDecision: A" in reviewer_prompt
    assert "knowledgeable_expert" not in reviewer_prompt
    assert "Knowlegable Expert" not in reviewer_prompt
    assert "critic" not in reviewer_prompt
    assert "Critic" not in reviewer_prompt
    assert worker_outputs == original
