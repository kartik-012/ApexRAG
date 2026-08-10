"""Unit tests for VectorStore, BM25Store, Hybrid RRF, Reranker, and StrategyRouter."""

import numpy as np
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker
from src.retrieval.router import StrategyRouter


def test_bm25_store():
    bm25 = BM25Store()
    chunk_ids = ["c1", "c2"]
    texts = ["React component state management", "DOM manipulation using refs"]
    bm25.build(chunk_ids, texts)

    res = bm25.query("state management", top_k=1)
    assert len(res) == 1
    assert res[0][0] == "c1"


def test_reciprocal_rank_fusion():
    bm25_ids = ["c1", "c2", "c3"]
    vector_ids = ["c2", "c1", "c4"]
    fused = reciprocal_rank_fusion(bm25_ids, vector_ids, k=60, top_k=2)
    assert len(fused) == 2
    # c1 and c2 appeared in both top ranks so they should top the fused list
    top_ids = [f.chunk_id for f in fused]
    assert "c1" in top_ids and "c2" in top_ids


def test_reranker_fallback():
    reranker = Reranker()
    candidates = [("c1", "useState hook state memory"), ("c2", "class component setState")]
    ranked = reranker.rerank("useRef hook", candidates, top_k=1)
    assert len(ranked) == 1


def test_strategy_router():
    router = StrategyRouter()
    strat = router.predict_strategy("How to use component state?")
    assert strat in ["simple", "semantic", "hybrid", "rerank"]
