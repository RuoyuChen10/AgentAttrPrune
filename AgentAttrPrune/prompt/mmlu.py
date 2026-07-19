from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


WORKER_ROLES = (
    ("knowledgeable_expert", "Knowlegable Expert"),
    ("critic", "Critic"),
    ("mathematician", "Mathematician"),
    ("psychologist", "Psychologist"),
    ("historian", "Historian"),
    ("doctor", "Doctor"),
    ("lawyer", "Lawyer"),
    ("economist", "Economist"),
    ("programmer", "Programmer"),
)

ROLE_DESCRIPTIONS = {
    "Knowlegable Expert": """
You are a knowlegable expert in question answering.
Please give several key entities that need to be searched in wikipedia to solve the problem.
Key entities that need to be searched are included between two '@' when output, for example: @catfish effect@, @broken window effect@, @Shakespeare@.
If there is no entity in the question that needs to be searched in Wikipedia, you don't have to provide it
""",
    "Critic": """
You are an excellent critic.
Please point out potential issues in other agent's analysis point by point.
""",
    "Mathematician": """
You are a mathematician who is good at math games, arithmetic calculation, and long-term planning.
""",
    "Psychologist": """
You are a psychologist.
You are good at psychology, sociology, and philosophy.
You give people scientific suggestions that will make them feel better.
""",
    "Historian": """
You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.
""",
    "Doctor": """
You are a doctor and come up with creative treatments for illnesses or diseases.
You are able to recommend conventional medicines, herbal remedies and other natural alternatives.
You also consider the patient's age, lifestyle and medical history when providing your recommendations.
""",
    "Lawyer": """
You are good at law, politics, and history.
""",
    "Economist": """
You are good at economics, finance, and business.
You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.
""",
    "Programmer": """
You are good at computer science, engineering, and physics.
You have experience in designing and developing computer software and hardware.
""",
}


class MMLUPromptSet:
    """Prompts shared by dense and attribution-pruned MMLU runs."""

    @staticmethod
    def get_analyze_constraint(role: str) -> str:
        description = ROLE_DESCRIPTIONS.get(role, "")
        return description + """
I will ask you a question and 4 answers enumerated as A, B, C and D.
Only one answer out of the offered 4 is correct.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents
First write a concise role-specific reasoning analysis. Then write your final choice.
Your reply must be less than 180 words.
Use exactly this format:
Reasoning: <your brief analysis>
Decision: <one letter: A, B, C, or D>
"""

    @staticmethod
    def get_decision_role() -> str:
        return (
            "You are the top decision-maker and are good at analyzing and "
            "summarizing other people's opinions, finding errors and giving "
            "final answers."
        )

    @staticmethod
    def get_decision_constraint() -> str:
        return """
        I will ask you a question.
        I will also give you 4 answers enumerated as A, B, C and D.
        Only one answer out of the offered 4 is correct.
        You must choose the correct answer to the question.
        Your response must be one of the 4 letters: A, B, C or D,
        corresponding to the correct answer.
        I will give you some other people's answers and analysis.
        Your reply must only contain one letter and cannot have any other characters.
        For example, your reply can be A.
"""

    @staticmethod
    def get_decision_few_shot() -> str:
        return ""

    @classmethod
    def worker_messages(
        cls,
        *,
        role: str,
        task: str,
        spatial_incoming: Sequence[Mapping[str, Any]],
        temporal_incoming: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        user_prompt = f"The task is: {task}\n"
        if spatial_incoming:
            spatial = "".join(
                f"Agent {item['agent_id']}, role is {item['role']}, output is:\n\n "
                f"{item['output']}\n\n"
                for item in spatial_incoming
            )
            user_prompt += (
                "At the same time, the outputs of other agents are as follows:\n\n"
                f"{spatial} \n\n"
            )
        if temporal_incoming:
            temporal = "".join(
                f"Agent {item['agent_id']}, role is {item['role']}, output is:\n\n "
                f"{item['output']}\n\n"
                for item in temporal_incoming
            )
            user_prompt += (
                "In the last round of dialogue, the outputs of other agents were: "
                f"\n\n{temporal}"
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
        system_prompt = (
            f"{cls.get_decision_role()}.\n {cls.get_decision_constraint()}"
        )
        outputs = "\n\n".join(
            f"{item['agent_id']}: {item['output']}" for item in worker_outputs
        )
        user_prompt = (
            f"{cls.get_decision_few_shot()} The task is:\n\n {task}.\n "
            "At the same time, the output of other agents is as follows:\n\n"
            f"{outputs}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def normalize_answer(value: Any) -> str:
        text = str(value).strip().upper()
        if not text:
            return ""
        first_line = text.splitlines()[0].strip()
        if first_line in {"A", "B", "C", "D"}:
            return first_line
        for pattern in (
            r"\bDECISION\s*:?\s*(?:OPTION\s+)?([ABCD])\b",
            r"\bANSWER\s+IS\s*:?\s*(?:OPTION\s+)?([ABCD])\b",
            r"\bOPTION\s+([ABCD])\b",
            r"\b([ABCD])\b",
        ):
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return text[:1]
