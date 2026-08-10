"""Counterfactual testing package: decoy distractor generation and evaluation."""
from .generate_decoys import generate_decoy, build_counterfactual_test_set

__all__ = ["generate_decoy", "build_counterfactual_test_set"]
