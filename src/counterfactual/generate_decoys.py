"""
Counterfactual Retrieval Testing — Feature 7 of ApexRAG Blueprint.

Generates near-duplicate decoy distractors with altered key facts to test whether retrieval is fooled.
"""

from typing import List, Dict, Any
from src.generation.llm_client import chat_with_ollama

DECOY_PROMPT = """Take this factual chunk and create a near-duplicate version
with ONE key fact altered (a number, date, method name, or boolean value changed), keeping
everything else nearly identical in wording and structure.

Original: {chunk_text}

Return only the altered chunk text."""


def generate_decoy(chunk_text: str) -> str:
    messages = [{"role": "user", "content": DECOY_PROMPT.format(chunk_text=chunk_text)}]
    return chat_with_ollama(model="llama3.1:8b", messages=messages)


def build_counterfactual_test_set(qa_pairs: List[Dict[str, Any]], chunks_by_id: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Synthesizes a counterfactual benchmark suite from verified ground-truth QA pairs.

    Injects minimal semantic perturbations into gold chunks (e.g. flipping boolean guards,
    altering hook signatures, or mutating return types) to test whether vector similarity
    and cross-encoder rerankers discriminate real technical specifications from plausible decoys.

    Args:
        qa_pairs: List of verified QA pair dictionaries.
        chunks_by_id: Mapping from chunk ID to original raw document text.

    Returns:
        List of counterfactual test cases containing question, real gold ID, and decoy text.
    """
    test_set = []
    for pair in qa_pairs:
        gold_id = pair.get("gold_chunk_ids", [None])[0]
        if gold_id and gold_id in chunks_by_id:
            gold_text = chunks_by_id[gold_id]
            decoy_text = generate_decoy(gold_text)
            test_set.append({
                "question": pair["question"],
                "real_gold_id": gold_id,
                "decoy_chunk_text": decoy_text,
                "decoy_chunk_id": f"decoy_{pair['id']}",
            })
    return test_set
