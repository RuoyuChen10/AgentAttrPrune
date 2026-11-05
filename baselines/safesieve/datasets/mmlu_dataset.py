"""Original MMLU answer parser; data loading is shared with AgentAttrPrune."""
import re
from typing import Union, List

class MMLUDataset:
    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
            if isinstance(answer, list):
                if len(answer) > 0:
                    answer = answer[0]
                else:
                    answer = ""
            if not isinstance(answer, str):
                raise Exception("Expected string")
            answer = answer.strip()
            if not answer:
                return ""

            # AgentPrune's Qwen prompt emits "Decision: <A-D>" after its reasoning.
            decision_match = re.search(
                r"\bDecision\s*:\s*(?:Option\s*)?([ABCD])\b",
                answer,
                flags=re.IGNORECASE,
            )
            if decision_match:
                return decision_match.group(1).upper()

            answer_match = re.search(
                r"\banswer\s+is\s*:?[ ]*(?:Option\s*)?([ABCD])\b",
                answer,
                flags=re.IGNORECASE,
            )
            if answer_match:
                return answer_match.group(1).upper()

            # FinalRefer is instructed to return only one option letter.
            leading_option = re.match(r"(?:Option\s*)?([ABCD])\b", answer, re.IGNORECASE)
            return leading_option.group(1).upper() if leading_option else ""
