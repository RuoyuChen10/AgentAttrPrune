from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


WORKER_ROLES = (
    ("project_manager", "Project Manager"),
    ("algorithm_designer", "Algorithm Designer"),
    ("programming_expert", "Programming Expert"),
    ("test_analyst", "Test Analyst"),
    ("bug_fixer", "Bug Fixer"),
)

ROLE_DESCRIPTIONS = {
    "Project Manager": (
        "You are a project manager. Identify the required behavior and propose "
        "a concise implementation structure without over-engineering."
    ),
    "Algorithm Designer": (
        "You are an algorithm designer. Derive a correct algorithm, including "
        "edge cases and complexity when relevant."
    ),
    "Programming Expert": (
        "You are a Python programming expert. Produce a complete, concise, "
        "correct implementation of the requested function."
    ),
    "Test Analyst": (
        "You are a test analyst. Find likely defects and enumerate boundary "
        "conditions that a correct implementation must handle."
    ),
    "Bug Fixer": (
        "You are a bug fixer. Reconcile the proposed designs and code, repair "
        "any issues, and provide a complete corrected implementation."
    ),
}


def extract_python_code(value: Any) -> str:
    """Extract a model's Python implementation without altering its contents."""

    text = str(value).strip()
    if not text:
        return ""
    fenced = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return (fenced[-1] if fenced else text).strip()


def _incoming_text(items: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        f"Agent {item['agent_id']}, role is {item['role']}, output is:\n\n"
        f"{item['output']}\n\n"
        for item in items
    )


class HumanEvalPromptSet:
    """Prompts shared by dense and attribution-pruned HumanEval runs."""

    @staticmethod
    def get_analyze_constraint(role: str) -> str:
        return ROLE_DESCRIPTIONS.get(role, "") + """
The task contains a Python function signature and docstring.
Use other agents' outputs as advice, but verify them independently.
If you provide code, restate the complete function and preserve its name and signature.
Keep non-code discussion concise.
"""

    @staticmethod
    def get_decision_role() -> str:
        return (
            "You are the top decision-maker and senior Python implementer. "
            "Select or repair the best proposed solution."
        )

    @staticmethod
    def get_decision_constraint() -> str:
        return """
Return a complete implementation of the function in the task.
Preserve the function name and signature.
Respond with exactly one Python code block and no text outside that block.
"""

    @classmethod
    def worker_messages(
        cls,
        *,
        role: str,
        task: str,
        spatial_incoming: Sequence[Mapping[str, Any]],
        temporal_incoming: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        user_prompt = f"The task is:\n```python\n{task}\n```\n"
        if spatial_incoming:
            user_prompt += (
                "\nCurrent-round proposals from earlier agents:\n\n"
                f"{_incoming_text(spatial_incoming)}"
            )
        if temporal_incoming:
            user_prompt += (
                "\nProposals from the previous round:\n\n"
                f"{_incoming_text(temporal_incoming)}"
            )
        return [
            {"role": "system", "content": cls.get_analyze_constraint(role)},
            {"role": "user", "content": user_prompt},
        ]

    @classmethod
    def reviewer_messages(
        cls,
        *,
        task: str,
        worker_outputs: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        outputs = "\n\n".join(
            f"{item['agent_id']}:\n{item['output']}" for item in worker_outputs
        )
        return [
            {
                "role": "system",
                "content": (
                    f"{cls.get_decision_role()}\n"
                    f"{cls.get_decision_constraint()}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The task is:\n```python\n{task}\n```\n\n"
                    f"The agents' proposals are:\n{outputs}"
                ),
            },
        ]

    @staticmethod
    def normalize_answer(value: Any) -> str:
        return extract_python_code(value)
