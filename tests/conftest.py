"""
Shared pytest fixtures and configuration for the entire test suite.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_qa_pairs():
    """Minimal QA pairs fixture for tests that need ground truth data."""
    return [
        {
            "id": "qa_test_01",
            "question": "How do you initialize state in class components?",
            "answer": "State is initialized in the constructor with this.state.",
            "gold_chunk_ids": ["legacy_state-and-lifecycle_chunk0"],
            "gold_doc_id": "legacy_state-and-lifecycle",
        },
        {
            "id": "qa_test_02",
            "question": "How does the useRef hook work?",
            "answer": "useRef holds a mutable reference that does not trigger re-renders.",
            "gold_chunk_ids": ["current_useRef_chunk0"],
            "gold_doc_id": "current_useRef",
        },
    ]


@pytest.fixture
def real_docs():
    """Loads the real React corpus for test isolation."""
    from src.ingestion.loader import load_legacy_docs, load_current_docs
    return load_legacy_docs("data/raw_docs/react-legacy") + load_current_docs("data/raw_docs/react-dev-current")
