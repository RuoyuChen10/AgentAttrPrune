"""Original expert-scoring prompt, used to validate the bundled similarity cache."""

def _build_evaluation_prompt(prompt_a: str, prompt_b: str, task_type: str, max_score: float) -> str:
    """
    Build prompt for LLM evaluation.
    """
    prompt_template = f"""You are a task evaluation expert, now need to arrange multiple agents to collaborate on {task_type} dataset tasks.

Please evaluate the collaboration complementarity of the following two agent role descriptions in solving {task_type} problems:

<Role A Description>
{prompt_a}
</Role A Description>

<Role B Description>
{prompt_b}
</Role B Description>

Scoring Criteria:
- 0-{max_score*0.3:.1f} points: Completely irrelevant, two roles have no collaboration value in solving {task_type} problems
- {max_score*0.3:.1f}-{max_score*0.5:.1f} points: Generally related, some collaboration potential but weak complementarity
- {max_score*0.5:.1f}-{max_score*0.7:.1f} points: Good complementarity, two roles can play different roles
- {max_score*0.7:.1f}-{max_score*0.9:.1f} points: Very good complementarity, two roles can form effective collaboration in problem solving
- {max_score*0.9:.1f}-{max_score} points: Perfect complementarity, two roles can perfectly cooperate to solve complex problems

Please carefully analyze the capability characteristics, problem-solving methods, and professional domains of the two roles, then give a numerical score between 0 and {max_score}.

Important: You must return a precise numerical value between 0 and {max_score}, format like: "Score: X.X" (where X.X is the specific value)."""

    return prompt_template
