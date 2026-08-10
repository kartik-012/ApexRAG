"""
Unit tests for adversarial variant generator, counterfactual decoy generator,
active learning flags, and multi-judge debate module.
"""

from src.adversarial.generate_variants import generate_adversarial_set
from src.counterfactual.generate_decoys import generate_decoy, build_counterfactual_test_set
from src.active_learning.flag_for_review import flag_disagreements
from src.debate.multi_judge import debate_faithfulness


def test_adversarial_generation(sample_qa_pairs):
    """Adversarial generator must return 3 variants per input pair."""
    variants = generate_adversarial_set(sample_qa_pairs[:1])
    # Each pair generates 3 variants (paraphrase, negated, multi_hop)
    assert len(variants) == 3
    variant_types = {v["variant_type"] for v in variants}
    assert "paraphrase" in variant_types
    assert "negated" in variant_types
    assert "multi_hop" in variant_types


def test_adversarial_negated_has_empty_gold(sample_qa_pairs):
    """Negated variants must have empty gold_chunk_ids (different answer expected)."""
    variants = generate_adversarial_set(sample_qa_pairs[:1])
    negated = next(v for v in variants if v["variant_type"] == "negated")
    assert negated["gold_chunk_ids"] == []
    assert negated["gold_doc_id"] == ""


def test_generate_decoy():
    """Decoy generator must return non-empty altered chunk text."""
    chunk = "React.Component is the base class for ES6 class components."
    decoy = generate_decoy(chunk)
    assert isinstance(decoy, str)
    assert len(decoy) > 0


def test_build_counterfactual_test_set(sample_qa_pairs):
    """Counterfactual test set builder produces one entry per QA pair with gold chunk."""
    chunks_by_id = {
        "legacy_state-and-lifecycle_chunk0": "State is initialized in constructor with this.state.",
        "current_useRef_chunk0": "useRef holds a mutable reference that doesn't trigger re-render.",
    }
    test_set = build_counterfactual_test_set(sample_qa_pairs, chunks_by_id)
    assert len(test_set) == 2
    assert "decoy_chunk_text" in test_set[0]
    assert test_set[0]["real_gold_id"] == "legacy_state-and-lifecycle_chunk0"


def test_flag_disagreements_high_conf_wrong():
    """Flag when system confidence is HIGH but answer was marked incorrect."""
    eval_results = [
        {"qid": "q1", "confidence": "HIGH", "correct": False},
        {"qid": "q2", "confidence": "LOW", "correct": False},  # should NOT be flagged
        {"qid": "q3", "confidence": "HIGH", "correct": True},   # should NOT be flagged
    ]
    flagged = flag_disagreements(eval_results, confidence_threshold=0.3)
    assert len(flagged) == 1
    assert flagged[0]["qid"] == "q1"


def test_debate_faithfulness():
    """Multi-judge debate must return a verdict dict with 'faithful' key."""
    result = debate_faithfulness(
        question="What is useState?",
        context="useState is a React Hook that lets you add state to function components.",
        answer="useState is a Hook that adds state to components.",
    )
    assert "faithful" in result
    assert isinstance(result["faithful"], bool)
    assert "agreement" in result
