"""Embedding drift detection package: centroid shift tracking for corpus staleness."""
from .check_drift import compute_drift

__all__ = ["compute_drift"]
