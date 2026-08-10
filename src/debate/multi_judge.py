"""
Multi-Judge Faithfulness Debate — Feature 8b of ApexRAG Blueprint.

Employs two independent local LLM judges (llama3.1:8b and phi3:mini) to arbitrate answer faithfulness and reduce shared model bias.
"""

from src.generation.llm_client import chat_with_ollama
from src.config import PRIMARY_MODEL, SECONDARY_MODEL


def debate_faithfulness(question: str, context: str, answer: str) -> dict:
    """
    Two-judge debate protocol to determine whether an answer is strictly grounded in retrieved context.
    """
    judge_prompt = """Context: {context}
Answer: {answer}
Question: {question}
Is every claim in the answer directly supported by the context? Answer YES or NO and explain in one sentence."""

    prompt_str = judge_prompt.format(context=context, answer=answer, question=question)

    response_a = chat_with_ollama(model=PRIMARY_MODEL, messages=[{"role": "user", "content": prompt_str}])
    response_b = chat_with_ollama(model=SECONDARY_MODEL, messages=[{"role": "user", "content": prompt_str}])

    verdict_a = "YES" in response_a.upper()
    verdict_b = "YES" in response_b.upper()

    if verdict_a == verdict_b:
        return {"faithful": verdict_a, "agreement": True, "judge_a": response_a, "judge_b": response_b}

    # Disagreement -> Arbitrate with third pass
    arbitration_prompt = f"""Two reviewers disagree on whether this answer is grounded.
Reviewer A ({PRIMARY_MODEL}): {response_a}
Reviewer B ({SECONDARY_MODEL}): {response_b}
Give the final YES/NO verdict and concise justification."""

    final_resp = chat_with_ollama(model=PRIMARY_MODEL, messages=[{"role": "user", "content": arbitration_prompt}])
    final_verdict = "YES" in final_resp.upper()

    return {
        "faithful": final_verdict,
        "agreement": False,
        "arbitration": final_resp,
        "judge_a": response_a,
        "judge_b": response_b,
    }
