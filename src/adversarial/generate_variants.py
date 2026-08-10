"""
Adversarial Self-Testing Loop — Feature 2 of ApexRAG Blueprint.

Generates paraphrases, negated trick questions, and multi-hop variants to stress test RAG pipeline robustness.
"""

import json
from typing import List, Dict, Any
from src.generation.llm_client import chat_with_ollama

VARIANT_PROMPT = """Given this question and answer, generate 3 variants:
1. A paraphrase (same meaning, different wording)
2. A negated/trick version (changes meaning subtly, should NOT retrieve the same answer)
3. A multi-hop version (requires combining this fact with a related concept)

Question: {question}
Answer: {answer}

Return strict JSON: {{"paraphrase": "...", "negated": "...", "multi_hop": "..."}}"""


def generate_adversarial_set(qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates adversarial question variants for ground truth qa_pairs.
    """
    variants = []
    for pair in qa_pairs:
        prompt = VARIANT_PROMPT.format(question=pair["question"], answer=pair["answer"])
        messages = [{"role": "user", "content": prompt}]
        response_text = chat_with_ollama(model="llama3.1:8b", messages=messages)

        try:
            parsed = json.loads(response_text)
        except Exception:
            parsed = {
                "paraphrase": f"Can you rephrase: {pair['question']}",
                "negated": f"Why is it incorrect that {pair['question']}?",
                "multi_hop": f"Combining React principles, {pair['question']}",
            }

        for variant_type, q_text in parsed.items():
            variants.append({
                "id": f"{pair['id']}_{variant_type}",
                "question": q_text,
                "answer": pair["answer"],
                "gold_chunk_ids": pair.get("gold_chunk_ids", []) if variant_type != "negated" else [],
                "gold_doc_id": pair.get("gold_doc_id", "") if variant_type != "negated" else "",
                "variant_type": variant_type,
                "source_id": pair["id"],
            })
    return variants
