"""Original answer parser; data loading is shared with AgentAttrPrune."""
from typing import List, Union

class MMLUDataset:
    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) > 0:
            ans_pos = answer.find("answer is")
            if ans_pos != -1:
                answer = answer[ans_pos+len("answer is"):].strip(":").strip().strip("Option").strip()
            answer = answer[0] # Try to format the answer by taking the first letter
        return answer

