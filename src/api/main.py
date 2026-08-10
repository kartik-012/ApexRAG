"""
FastAPI Production Server for ApexRAG Pipeline.

Exposes /, /query, /confidence, /eval, and /health endpoints.
"""

import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import os
from src.ingestion.loader import load_legacy_docs, load_current_docs
from src.ingestion.embedder import embed_texts
from src.ingestion.chunkers import simple_chunk
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker
from src.retrieval.router import StrategyRouter
from src.evaluation.uncertainty import embedding_spread_score, strategy_agreement_score, confidence_label
from src.evaluation.run_eval import evaluate_harness
from src.generation.llm_client import generate_answer

LEGACY_REPO = os.getenv("LEGACY_REPO", "data/raw_docs/react-legacy")
CURRENT_REPO = os.getenv("CURRENT_REPO", "data/raw_docs/react-dev-current")

# ── Global Pipeline State ────────────────────────────────────────
_vector_store = None
_bm25_store = None
_reranker = None
_router = None
_all_texts: list[str] = []
_all_ids: list[str] = []


def initialize_pipeline():
    """Lazily initializes the full RAG pipeline on first request."""
    global _vector_store, _bm25_store, _reranker, _router, _all_texts, _all_ids
    if _vector_store is not None:
        return

    _vector_store = VectorStore()
    _bm25_store = BM25Store()
    _reranker = Reranker()
    _router = StrategyRouter()

    docs = load_legacy_docs(LEGACY_REPO) + load_current_docs(CURRENT_REPO)
    _all_ids, _all_texts, metadatas = [], [], []

    for doc in docs:
        chunks = simple_chunk(doc.text, chunk_size=200, overlap=30)
        for idx, text in enumerate(chunks):
            cid = f"{doc.doc_id}_chunk{idx}"
            _all_ids.append(cid)
            _all_texts.append(text)
            metadatas.append({"source": doc.source, "doc_id": doc.doc_id, "title": doc.title})

    if _vector_store.count() == 0:
        embeddings = embed_texts(_all_texts)
        _vector_store.add_chunks(_all_ids, _all_texts, embeddings, metadatas)
    _bm25_store.build(_all_ids, _all_texts)


# ── Lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_pipeline()
    yield


# ── App Factory ──────────────────────────────────────────────────
app = FastAPI(
    title="ApexRAG Evaluation Harness & Production Pipeline API",
    description="Enterprise-Grade Zero-Cost RAG evaluation server with uncertainty quantification.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response Models ────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    strategy: Optional[str] = "auto"
    top_k: Optional[int] = 4


class QueryResponse(BaseModel):
    question: str
    answer: str
    strategy_used: str
    confidence: str
    spread_score: float
    agreement_score: float
    warning: Optional[str] = None
    context_chunks: List[str]
    latency_ms: float


# ── Routes ───────────────────────────────────────────────────────
@app.get("/")
def api_root():
    """ApexRAG Production API Root."""
    return {
        "name": "ApexRAG Evaluation Harness & Production Pipeline API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "status": "online",
    }


@app.get("/health")
def health_check():
    """Returns server health status and indexed chunk count."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "indexed_chunks": _vector_store.count() if _vector_store else 0,
        "strategies": ["simple", "semantic", "hybrid", "rerank", "auto"],
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    """Execute the full RAG pipeline: Retrieve → Rank → Generate → Quantify Uncertainty."""
    t_start = time.perf_counter()
    initialize_pipeline()

    question = req.question
    top_k = req.top_k or 4

    # Strategy selection
    if req.strategy == "auto" or not req.strategy:
        strategy = _router.predict_strategy(question)
    else:
        strategy = req.strategy

    # Embedding + Retrieval
    q_emb = embed_texts([question])[0]
    vec_results = _vector_store.query(q_emb, top_k=10)
    vec_cids = [r.chunk_id for r in vec_results]

    bm25_results = _bm25_store.query(question, top_k=10)
    bm25_cids = [cid for cid, _ in bm25_results]

    fused = reciprocal_rank_fusion(bm25_cids, vec_cids, k=60, top_k=10)
    hybrid_cids = [f.chunk_id for f in fused]

    candidates = [(cid, _all_texts[_all_ids.index(cid)]) for cid in hybrid_cids[:10] if cid in _all_ids]
    reranked = _reranker.rerank(question, candidates, top_k=10)
    rerank_cids = [cid for cid, _, _ in reranked]

    strat_map = {"simple": bm25_cids, "semantic": vec_cids, "hybrid": hybrid_cids, "rerank": rerank_cids}
    selected_cids = strat_map.get(strategy, hybrid_cids)[:top_k]

    # Feature 1: Uncertainty Quantification
    top_vecs = _vector_store.get_all_embeddings_for_ids(vec_cids[:top_k])
    spread = embedding_spread_score(top_vecs)
    agreement = strategy_agreement_score(strat_map)
    conf = confidence_label(spread, agreement)

    # Generation
    retrieved_texts = [_all_texts[_all_ids.index(cid)] for cid in selected_cids if cid in _all_ids]
    answer = generate_answer(question, retrieved_texts)

    warning_msg = None
    if conf == "LOW":
        warning_msg = "⚠ Low retrieval confidence — answer may be unreliable, consider rephrasing."

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return QueryResponse(
        question=question,
        answer=answer,
        strategy_used=strategy,
        confidence=conf,
        spread_score=round(spread, 4),
        agreement_score=round(agreement, 4),
        warning=warning_msg,
        context_chunks=retrieved_texts,
        latency_ms=latency_ms,
    )


@app.get("/confidence")
def check_confidence(question: str):
    """Standalone uncertainty check without LLM generation."""
    initialize_pipeline()
    q_emb = embed_texts([question])[0]
    vec_results = _vector_store.query(q_emb, top_k=4)
    vec_cids = [r.chunk_id for r in vec_results]
    bm25_results = _bm25_store.query(question, top_k=4)
    bm25_cids = [cid for cid, _ in bm25_results]
    hybrid_cids = [f.chunk_id for f in reciprocal_rank_fusion(bm25_cids, vec_cids, k=60, top_k=4)]

    strat_map = {"simple": bm25_cids, "semantic": vec_cids, "hybrid": hybrid_cids}
    top_vecs = _vector_store.get_all_embeddings_for_ids(vec_cids[:4])
    spread = embedding_spread_score(top_vecs)
    agreement = strategy_agreement_score(strat_map)
    conf = confidence_label(spread, agreement)

    return {
        "question": question,
        "confidence": conf,
        "spread_score": round(spread, 4),
        "agreement_score": round(agreement, 4),
    }


@app.post("/eval")
def trigger_eval():
    """Trigger full evaluation harness and return structured results."""
    res = evaluate_harness()
    return {"status": "completed", "results": res}
