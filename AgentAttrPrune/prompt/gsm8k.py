from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence


WORKER_ROLES = (
    ("math_solver", "Math Solver"),
    ("mathematical_analyst", "Mathematical Analyst"),
    ("programming_expert", "Programming Expert"),
    ("inspector", "Inspector"),
)

ROLE_DESCRIPTIONS = {
    "Math Solver": (
        "You are a math expert. Solve the word problem step by step and check "
        "that every operation follows from the quantities in the question."
    ),
    "Mathematical Analyst": (
        "You are a mathematical analyst. Translate the problem into variables "
        "and equations before substituting the numerical values."
    ),
    "Programming Expert": (
        "You are a programming expert. Independently verify the arithmetic, "
        "using short Python-like calculations when they make the result clearer."
    ),
    "Inspector": (
        "You are an exacting inspector. Look for unit mistakes, ambiguous "
        "interpretations, missing operations, and arithmetic errors."
    ),
}


def _incoming_text(items: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        f"Agent {item['agent_id']}, role is {item['role']}, output is:\n\n"
        f"{item['output']}\n\n"
        for item in items
    )


class GSM8KPromptSet:
    """Prompts shared by dense and attribution-pruned GSM8K runs."""

    @staticmethod
    def get_analyze_constraint(role: str) -> str:
        description = ROLE_DESCRIPTIONS.get(role, "")
        return description + """
Use other agents' work only as advice and verify it independently.
Keep the reasoning concise. End with the result in exactly this form:
The answer is <number>
Do not put units or explanatory text after the final number.
"""

    @staticmethod
    def get_decision_role() -> str:
        return (
            "You are the top decision-maker for grade-school mathematics. "
            "Reconcile the proposed solutions and return the verified result."
        )

    @staticmethod
    def get_decision_constraint() -> str:
        return """
Solve the given math word problem using the other agents' analyses as evidence.
Check the arithmetic yourself. Your final line must be exactly:
The answer is <number>
Do not put units or other text after the number.
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
        user_prompt = f"The task is:\n{task}\n"
        if spatial_incoming:
            user_prompt += (
                "\nCurrent-round analyses from earlier agents:\n\n"
                f"{_incoming_text(spatial_incoming)}"
            )
        if temporal_incoming:
            user_prompt += (
                "\nAnalyses from the previous round:\n\n"
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
            f"{item['agent_id']}: {item['output']}" for item in worker_outputs
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
                    f"The task is:\n{task}\n\n"
                    f"The agents' proposed solutions are:\n{outputs}"
                ),
            },
        ]

    @staticmethod
    def normalize_answer(value: Any) -> str:
        text = str(value).strip()
        if not text:
            return ""
        boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
        if boxed:
            candidate = boxed[-1]
        else:
            answer_lines = re.findall(
                r"(?:the\s+answer\s+is|answer\s*:)\s*"
                r"([-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:/[+-]?\d+)?)",
                text,
                flags=re.IGNORECASE,
            )
            if answer_lines:
                candidate = answer_lines[-1]
            else:
                numbers = re.findall(
                    r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:/[+-]?\d+)?",
                    text,
                )
                candidate = numbers[-1] if numbers else ""
        candidate = candidate.replace(",", "").replace("$", "").strip()
        if not candidate:
            return ""
        try:
            if "/" in candidate:
                numerator, denominator = candidate.split("/", 1)
                number = Decimal(numerator) / Decimal(denominator)
            else:
                number = Decimal(candidate)
        except (InvalidOperation, ZeroDivisionError, ValueError):
            return candidate
        normalized = format(number.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        return "0" if normalized in {"-0", ""} else normalized
