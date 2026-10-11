"""
FastAPI Production Server & Interactive Observability Studio for ApexRAG Pipeline.

Exposes:
  - GET  /            (JSON API root for clients/tests; HTML Studio for browsers)
  - GET  /dashboard   (Interactive ApexRAG Enterprise Studio Web UI)
  - GET  /health      (Health check, corpus counts, model status, semantic cache telemetry)
  - POST /query       (Full RAG pipeline with uncertainty quantification, caching, era filtering)
  - GET  /confidence  (Fast pre-generation uncertainty & strategy agreement check)
  - POST /confidence  (POST variant matching README specification)
  - POST /compare     (Live 5-strategy side-by-side retrieval matrix & rank comparison)
  - POST /debate      (Multi-judge faithfulness debate & arbitration)
  - POST /adversarial (Adversarial variant + counterfactual decoy generator)
  - GET  /benchmark   (Load latest 100-question evaluation benchmark & drift telemetry)
  - POST /eval        (Trigger live evaluation harness run)
"""

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from pydantic import BaseModel

from src.observability import (
    trace_span,
    get_current_trace_id,
    get_recent_traces,
    get_trace_by_id,
    metrics_registry,
    render_prometheus_metrics,
    record_request_metric,
    record_stage_duration,
    record_router_decision,
    record_cache_event,
    record_tokens_generated,
    record_evaluation_scores,
)

from src.active_learning.flag_for_review import flag_disagreements
from src.adversarial.generate_variants import generate_adversarial_set
from src.config import (
    DRIFT_LOGS_DIR,
    EMBEDDING_MODEL_NAME,
    EVAL_RUNS_DIR,
    GROUND_TRUTH_DIR,
    PRIMARY_MODEL,
    RERANKER_MODEL_NAME,
    SECONDARY_MODEL,
)
from src.counterfactual.generate_decoys import generate_decoy
from src.debate.multi_judge import debate_faithfulness
from src.evaluation.cost_latency_tracker import estimate_cost_attribution
from src.evaluation.run_eval import evaluate_harness
from src.evaluation.uncertainty import (
    confidence_label,
    embedding_spread_score,
    strategy_agreement_score,
)
from src.generation.llm_client import generate_answer, generate_answer_stream
from src.guardrails.security import apply_input_guardrails, apply_output_guardrails
from src.guardrails.rbac import UserContext, ClearanceLevel, can_access_chunk, filter_authorized_chunks
from src.ingestion.ast_chunker import hierarchical_markdown_chunk
from src.ingestion.chunkers import simple_chunk
from src.ingestion.embedder import embed_texts, ensure_embedder_fitted
from src.ingestion.loader import load_current_docs, load_legacy_docs
from src.retrieval.bm25_store import BM25Store
from src.retrieval.crag import execute_crag_pipeline, CRAGResult
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.query_intelligence import SemanticQueryCache, analyze_and_expand_query
from src.retrieval.reranker import Reranker
from src.retrieval.router import StrategyRouter
from src.retrieval.vector_store import VectorStore
from src.api.concurrency import concurrency_mgr

LEGACY_REPO = os.getenv("LEGACY_REPO", "data/raw_docs/react-legacy")
CURRENT_REPO = os.getenv("CURRENT_REPO", "data/raw_docs/react-dev-current")
STATIC_DIR = Path(__file__).resolve().parent / "static"
FEEDBACK_DIR = Path("data/feedback")
FEEDBACK_FILE = FEEDBACK_DIR / "user_feedback.jsonl"

# ── Global Pipeline State ────────────────────────────────────────
_vector_store: Optional[VectorStore] = None
_bm25_store: Optional[BM25Store] = None
_reranker: Optional[Reranker] = None
_router: Optional[StrategyRouter] = None
_semantic_cache = SemanticQueryCache()
_all_texts: list[str] = []
_all_ids: list[str] = []
_id_to_text: dict[str, str] = {}
_id_to_meta: dict[str, dict] = {}
_parent_id_to_text: dict[str, str] = {}
_era_counts: dict[str, int] = {"legacy": 0, "current": 0}


def initialize_pipeline():
    """Lazily initializes the full RAG pipeline on first request using AST Parent-Document Chunking."""
    global _vector_store, _bm25_store, _reranker, _router, _all_texts, _all_ids, _id_to_text, _id_to_meta, _parent_id_to_text, _era_counts
    if _vector_store is not None:
        return

    _vector_store = VectorStore()
    _bm25_store = BM25Store()
    _reranker = Reranker()
    _router = StrategyRouter()

    docs = load_legacy_docs(LEGACY_REPO) + load_current_docs(CURRENT_REPO)
    _all_ids, _all_texts, metadatas = [], [], []
    _id_to_text, _id_to_meta, _parent_id_to_text = {}, {}, {}
    _era_counts = {"legacy": 0, "current": 0}

    for doc in docs:
        h_chunks = hierarchical_markdown_chunk(doc.doc_id, doc.title, doc.text, source=doc.source)
        if h_chunks:
            for hc in h_chunks:
                _all_ids.append(hc.child_id)
                _all_texts.append(hc.child_text)
                meta = dict(hc.metadata)
                meta["parent_id"] = hc.parent_id
                metadatas.append(meta)
                _id_to_text[hc.child_id] = hc.child_text
                _id_to_meta[hc.child_id] = meta
                _parent_id_to_text[hc.parent_id] = hc.parent_text
                _era_counts[doc.source] = _era_counts.get(doc.source, 0) + 1
        else:
            chunks = simple_chunk(doc.text, chunk_size=200, overlap=30)
            for idx, text in enumerate(chunks):
                cid = f"{doc.doc_id}_chunk{idx}"
                meta = {"source": doc.source, "doc_id": doc.doc_id, "title": doc.title, "parent_id": cid}
                _all_ids.append(cid)
                _all_texts.append(text)
                metadatas.append(meta)
                _id_to_text[cid] = text
                _id_to_meta[cid] = meta
                _parent_id_to_text[cid] = text
                _era_counts[doc.source] = _era_counts.get(doc.source, 0) + 1

    if _vector_store.count() == 0:
        embeddings = embed_texts(_all_texts)
        _vector_store.add_chunks(_all_ids, _all_texts, embeddings, metadatas)
    else:
        ensure_embedder_fitted(_all_texts)
    _bm25_store.build(_all_ids, _all_texts, metadatas)


# ── Lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await concurrency_mgr.run_in_thread(initialize_pipeline)
    yield


# ── App Factory ──────────────────────────────────────────────────
app = FastAPI(
    title="ApexRAG Evaluation Harness & Production Pipeline API",
    description="Enterprise-Grade Zero-Cost RAG evaluation server with uncertainty quantification and interactive Studio.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def trace_and_metrics_middleware(request: Request, call_next):
    """
    OpenTelemetry Request-level Tracing & Prometheus Metrics Middleware.
    Propagates W3C Trace Context, attaches X-Trace-Id, and records request latency histograms.
    """
    t0 = time.perf_counter()
    metrics_registry.active_requests.inc()
    path = request.url.path
    method = request.method

    with trace_span(f"{method} {path}", {"http.method": method, "http.target": path}):
        trace_id = get_current_trace_id()
        request.state.trace_id = trace_id
        try:
            response = await call_next(request)
            duration_sec = time.perf_counter() - t0
            duration_ms = round(duration_sec * 1000.0, 2)
            response.headers["X-Trace-Id"] = trace_id
            response.headers["X-Response-Time-Ms"] = str(duration_ms)
            record_request_metric(path, method, response.status_code, duration_sec)
            return response
        except Exception as exc:
            duration_sec = time.perf_counter() - t0
            record_request_metric(path, method, 500, duration_sec)
            raise exc
        finally:
            metrics_registry.active_requests.dec()


# ── Request / Response Models ────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    strategy: Optional[str] = "auto"
    top_k: Optional[int] = 4
    era_filter: Optional[str] = "all"
    enable_expansion: Optional[bool] = True
    use_cache: Optional[bool] = True
    use_bandit: Optional[bool] = False
    user_id: Optional[str] = "anonymous"
    tenant_id: Optional[str] = "public"
    roles: Optional[List[str]] = ["public"]
    clearance_level: Optional[str] = "public"
    enable_crag: Optional[bool] = True


class QueryResponse(BaseModel):
    question: str
    answer: str
    strategy_used: str
    confidence: str
    spread_score: float
    agreement_score: float
    warning: Optional[str] = None
    context_chunks: List[str]
    retrieved_chunks: List[str] = []
    retrieved_metadatas: List[Dict[str, Any]] = []
    router_probabilities: Dict[str, float] = {}
    router_decision: Dict[str, Any] = {}
    query_analysis: Dict[str, Any] = {}
    cache_hit: bool = False
    stage_latencies_ms: Dict[str, float] = {}
    guardrail_telemetry: Optional[Dict[str, Any]] = None
    cost_attribution: Optional[Dict[str, Any]] = None
    rbac_telemetry: Optional[Dict[str, Any]] = None
    crag_telemetry: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    latency_ms: float



class FeedbackRequest(BaseModel):
    trace_id: Optional[str] = None
    rating: int
    category: Optional[str] = None
    comment: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConfidenceRequest(BaseModel):
    question: str


class DebateRequest(BaseModel):
    question: str
    context: str
    answer: str


class AdversarialRequest(BaseModel):
    question: str
    answer: Optional[str] = "State is managed via useState in function components."
    chunk_text: Optional[str] = None


# ── Routes ───────────────────────────────────────────────────────
@app.get("/")
def api_root(request: Request):
    """ApexRAG Production API Root (serves interactive Studio to browsers, JSON to API clients)."""
    accept = request.headers.get("accept", "")
    user_agent = request.headers.get("user-agent", "").lower()
    if "text/html" in accept and "testclient" not in user_agent:
        html_file = STATIC_DIR / "index.html"
        if html_file.exists():
            return HTMLResponse(content=html_file.read_text(encoding="utf-8"))

    return {
        "name": "ApexRAG Evaluation Harness & Production Pipeline API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "dashboard_url": "/dashboard",
        "health_url": "/health",
        "status": "online",
    }


@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def dashboard_ui():
    """Serves the interactive ApexRAG Enterprise Studio Web UI."""
    html_file = STATIC_DIR / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI asset not found.")
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.get("/health")
async def health_check():
    """Returns server health status, indexed chunk count, model status, cache telemetry, and concurrency metrics."""
    if _vector_store is None:
        await concurrency_mgr.run_in_thread(initialize_pipeline)
    return {
        "status": "ok",
        "version": "1.0.0",
        "indexed_chunks": _vector_store.count() if _vector_store else 0,
        "chunks_loaded": len(_all_ids),
        "era_breakdown": _era_counts,
        "vector_backend": _vector_store.backend if _vector_store else "unknown",
        "is_pgvector": bool(_vector_store and _vector_store.is_pgvector),
        "strategies": ["simple", "semantic", "hybrid", "rerank", "auto"],
        "models": {
            "embedding": EMBEDDING_MODEL_NAME,
            "reranker": RERANKER_MODEL_NAME,
            "primary_llm": PRIMARY_MODEL,
            "secondary_llm": SECONDARY_MODEL,
            "router_loaded": bool(_router and _router.is_loaded),
        },
        "cache": _semantic_cache.stats(),
        "concurrency": concurrency_mgr.get_telemetry(),
        "observability": {
            "metrics_snapshot": metrics_registry.snapshot(),
            "recent_traces_count": len(get_recent_traces(limit=10)),
        },
    }


@app.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus 0.0.4 text format exposition endpoint.
    Scraped by Prometheus, Grafana Agent, Datadog OpenMetrics, or VictoriaMetrics.
    """
    telemetry = concurrency_mgr.get_telemetry()
    metrics_registry.active_semaphore_permits.set(float(telemetry.get("active_permits", 0)))
    content = render_prometheus_metrics()
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/traces")
async def list_recent_traces(limit: int = Query(20, ge=1, le=100)):
    """Returns recent distributed traces recorded in the in-memory ring buffer."""
    traces = get_recent_traces(limit=limit)
    return {
        "count": len(traces),
        "traces": traces,
    }


@app.get("/traces/{trace_id}")
async def get_trace_detail(trace_id: str):
    """Returns detailed span breakdown and flamegraph waterfall timeline for a specific trace_id."""
    trace_data = get_trace_by_id(trace_id)
    if not trace_data:
        raise HTTPException(status_code=404, detail=f"Trace with ID '{trace_id}' not found in buffer.")
    return trace_data


@app.post("/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    """Execute the full RAG pipeline non-blocking: Analyze → Embed → Parallel Retrieve → Rank → Quantify Uncertainty → Generate."""
    t_start = time.perf_counter()
    concurrency_mgr.track_request_start()
    try:
        if _vector_store is None:
            await concurrency_mgr.run_in_thread(initialize_pipeline)

        question = req.question.strip()
        guard_res = apply_input_guardrails(question)
        if not guard_res.is_safe:
            raise HTTPException(
                status_code=400,
                detail=f"Query rejected by Enterprise Security Guardrails: prompt injection signature detected ({', '.join(guard_res.violations)})."
            )
        question = guard_res.sanitized_text

        top_k = max(1, min(req.top_k or 4, 10))
        era_filter = (req.era_filter or "all").lower()
        where_clause = {"source": era_filter} if era_filter in ("legacy", "current") else None

        # Step 0: Query Intelligence, HyDE & Multi-Part Decomposition
        q_analysis = analyze_and_expand_query(question, enable_expansion=bool(req.enable_expansion))
        retrieval_query = q_analysis.expanded_query

        # Step 1: Embed query with HyDE alignment (guarded by concurrency semaphore)
        with trace_span("retrieval.embed_query", {"query_len": len(retrieval_query), "is_compound": q_analysis.is_compound}):
            t_emb0 = time.perf_counter()
            texts_to_embed = [retrieval_query]
            if q_analysis.hyde_passage:
                texts_to_embed.append(q_analysis.hyde_passage)
            q_embs = await concurrency_mgr.embed_texts_async(embed_texts, texts_to_embed)
            if len(q_embs) > 1:
                q_vec = np.array(q_embs[0], dtype=np.float32)
                h_vec = np.array(q_embs[1], dtype=np.float32)
                blended = 0.7 * q_vec + 0.3 * h_vec
                norm = np.linalg.norm(blended)
                q_emb = (blended / norm) if norm > 0 else q_vec
            else:
                q_emb = q_embs[0]
            emb_ms = round((time.perf_counter() - t_emb0) * 1000.0, 2)
            record_stage_duration("embed", emb_ms / 1000.0)

        # Dense Strategy selection + Shannon Entropy Gating & Thompson Sampling
        with trace_span("router.predict_strategy"):
            routing_decision = await concurrency_mgr.run_in_thread(
                _router.predict_routing_decision, question, q_emb, bool(req.use_bandit)
            )
            record_router_decision(
                routing_decision.chosen_strategy,
                entropy=routing_decision.entropy,
                is_fallback=routing_decision.is_entropy_gated,
            )
        router_strat = routing_decision.chosen_strategy
        router_probs = routing_decision.probabilities
        if req.strategy == "auto" or not req.strategy:
            strategy = router_strat
        else:
            strategy = req.strategy

        if req.use_cache:
            cached = _semantic_cache.get(question, strategy, top_k, era_filter=era_filter, query_embedding=q_emb)
            if cached is not None:
                record_cache_event(True)
                latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                return QueryResponse(
                    **{
                        **cached,
                        "cache_hit": True,
                        "trace_id": get_current_trace_id(),
                        "latency_ms": latency_ms,
                    }
                )
            record_cache_event(False)

        # Step 2: Multi-strategy parallel retrieval (Dense Vector + Sparse BM25 via asyncio.gather)
        with trace_span("retrieval.parallel_dual_branch", {"top_k": 10}):
            t_dual0 = time.perf_counter()
            vec_results, bm25_results = await concurrency_mgr.retrieve_dual_branch_parallel(
                _vector_store.query,
                _bm25_store.query,
                q_emb,
                retrieval_query,
                top_k=10,
                where_clause=where_clause,
            )
            dual_ms = round((time.perf_counter() - t_dual0) * 1000.0, 2)
            record_stage_duration("dense_search", dual_ms / 1000.0)
            record_stage_duration("bm25_search", dual_ms / 1000.0)

        vec_cids = [r.chunk_id for r in vec_results]
        bm25_cids = [cid for cid, _ in bm25_results]

        with trace_span("retrieval.hybrid_rrf", {"k": 60}):
            t_hyb0 = time.perf_counter()
            fused = reciprocal_rank_fusion(bm25_cids, vec_cids, k=60, top_k=10)
            hybrid_cids = [f.chunk_id for f in fused]
            hyb_ms = round((time.perf_counter() - t_hyb0) * 1000.0, 2)
            record_stage_duration("hybrid_rrf", hyb_ms / 1000.0)

        with trace_span("retrieval.cross_encoder_rerank", {"candidates": min(10, len(hybrid_cids))}):
            t_rerank0 = time.perf_counter()
            candidates = [(cid, _id_to_text[cid]) for cid in hybrid_cids[:10] if cid in _id_to_text]
            reranked = await concurrency_mgr.rerank_async(_reranker.rerank, question, candidates, top_k=10)
            rerank_cids = [cid for cid, _, _ in reranked]
            rerank_ms = round((time.perf_counter() - t_rerank0) * 1000.0, 2)
            record_stage_duration("rerank", rerank_ms / 1000.0)

        strat_map = {"simple": bm25_cids, "semantic": vec_cids, "hybrid": hybrid_cids, "rerank": rerank_cids}
        raw_selected_cids = strat_map.get(strategy, hybrid_cids)

        # Multi-Tenant Role-Based Access Control (RBAC) Pre-Filtering
        user_ctx = UserContext.from_dict({
            "user_id": req.user_id,
            "tenant_id": req.tenant_id,
            "roles": req.roles,
            "clearance_level": req.clearance_level,
        })
        authorized_cids = [cid for cid in raw_selected_cids if can_access_chunk(_id_to_meta.get(cid, {}), user_ctx)]
        rbac_telemetry = {
            "user_id": user_ctx.user_id,
            "tenant_id": user_ctx.tenant_id,
            "clearance_level": user_ctx.clearance_level.value,
            "pre_filter_count": len(raw_selected_cids),
            "authorized_count": len(authorized_cids),
            "denied_count": len(raw_selected_cids) - len(authorized_cids),
        }
        selected_cids = authorized_cids[:top_k]

        # Step 3: Uncertainty Quantification
        with trace_span("evaluation.uncertainty_quantification"):
            top_vecs = await concurrency_mgr.run_in_thread(_vector_store.get_all_embeddings_for_ids, vec_cids[:top_k])
            spread = embedding_spread_score(top_vecs)
            agreement = strategy_agreement_score(strat_map)
            conf = confidence_label(spread, agreement)

        # Step 4: Generation with Parent-Document Context Expansion
        retrieved_texts = []
        seen_parents = set()
        for cid in selected_cids:
            pid = _id_to_meta.get(cid, {}).get("parent_id", cid)
            if pid in _parent_id_to_text and pid not in seen_parents:
                seen_parents.add(pid)
                retrieved_texts.append(_parent_id_to_text[pid])
            elif cid in _id_to_text and cid not in seen_parents:
                seen_parents.add(cid)
                retrieved_texts.append(_id_to_text[cid])
        if not retrieved_texts:
            retrieved_texts = [_id_to_text[cid] for cid in selected_cids if cid in _id_to_text]

        # Step 4.5: Corrective RAG (CRAG) Retrieval Self-Grading & Knowledge Refinement
        crag_telemetry = None
        if req.enable_crag and retrieved_texts:
            with trace_span("retrieval.crag_evaluation"):
                crag_res = execute_crag_pipeline(question, retrieved_texts)
                crag_telemetry = crag_res.to_dict()
                retrieved_texts = crag_res.refined_chunks

        retrieved_metas = [
            {"chunk_id": cid, **_id_to_meta.get(cid, {})}
            for cid in selected_cids
            if cid in _id_to_text
        ]
        with trace_span("generation.llm_generate", {"model": PRIMARY_MODEL}):
            t_gen0 = time.perf_counter()
            answer = await concurrency_mgr.run_in_thread(generate_answer, question, retrieved_texts)
            gen_ms = round((time.perf_counter() - t_gen0) * 1000.0, 2)
            record_stage_duration("llm_generate", gen_ms / 1000.0)

        # Output Guardrails verification
        out_guard = apply_output_guardrails(answer, "\n\n".join(retrieved_texts))
        answer = out_guard.sanitized_text

        warning_msg = None
        if conf == "LOW":
            warning_msg = "⚠ Low retrieval confidence — answer may be unreliable, consider rephrasing."
        elif not out_guard.is_safe:
            warning_msg = "⚠ Notice: Generated answer contains claims not fully grounded in retrieved documentation."

        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        stage_latencies = {
            "embedding_ms": emb_ms,
            "vector_ms": dual_ms,
            "bm25_ms": dual_ms,
            "parallel_retrieval_ms": dual_ms,
            "hybrid_rrf_ms": hyb_ms,
            "rerank_ms": rerank_ms,
            "generation_ms": gen_ms,
        }

        cost_attr = estimate_cost_attribution(question, retrieved_texts, answer, PRIMARY_MODEL)

        response_payload = {
            "question": question,
            "answer": answer,
            "strategy_used": strategy,
            "confidence": conf,
            "spread_score": round(spread, 4),
            "agreement_score": round(agreement, 4),
            "warning": warning_msg,
            "context_chunks": retrieved_texts,
            "retrieved_chunks": selected_cids,
            "retrieved_metadatas": retrieved_metas,
            "router_probabilities": router_probs,
            "router_decision": routing_decision.to_dict(),
            "query_analysis": {
                "expanded_query": q_analysis.expanded_query,
                "code_symbols": q_analysis.code_symbols,
                "detected_era": q_analysis.detected_era,
                "complexity": q_analysis.query_complexity,
                "is_compound": q_analysis.is_compound,
                "sub_queries": q_analysis.sub_queries,
                "has_hyde": bool(q_analysis.hyde_passage),
            },
            "guardrail_telemetry": guard_res.to_dict(),
            "cost_attribution": cost_attr,
            "rbac_telemetry": rbac_telemetry,
            "crag_telemetry": crag_telemetry,
            "cache_hit": False,
            "stage_latencies_ms": stage_latencies,
            "trace_id": get_current_trace_id(),
            "latency_ms": latency_ms,
        }

        if req.use_cache:
            _semantic_cache.put(
                question,
                strategy,
                top_k,
                payload=response_payload,
                original_latency_ms=latency_ms,
                era_filter=era_filter,
                query_embedding=q_emb,
            )

        return QueryResponse(**response_payload)
    finally:
        concurrency_mgr.track_request_end()


async def stream_rag_events(
    question: str,
    strategy: str = "auto",
    top_k: int = 4,
    era_filter: str = "all",
    enable_expansion: bool = True,
    use_cache: bool = True,
):
    """
    Asynchronous SSE event generator emitting:
      - event: meta  (retrieved context, strategy, uncertainty scores, stage latencies)
      - event: token (incremental token deltas as generated by LLM)
      - event: done  (final token counts, generation timing, total latency)
      - event: error (error telemetry if mid-stream failure occurs)
    """
    t_start = time.perf_counter()
    concurrency_mgr.track_request_start()
    try:
        if _vector_store is None:
            await concurrency_mgr.run_in_thread(initialize_pipeline)

        top_k = max(1, min(top_k or 4, 10))
        where_clause = {"source": era_filter} if era_filter in ("legacy", "current") else None

        # Enterprise Input Guardrails
        guard_res = apply_input_guardrails(question)
        if not guard_res.is_safe:
            yield f"event: error\ndata: {json.dumps({'error': f'Query rejected by Enterprise Guardrails: {', '.join(guard_res.violations)}'})}\n\n"
            return
        question = guard_res.sanitized_text

        # Step 0: Query Intelligence, HyDE & Multi-Part Decomposition
        q_analysis = analyze_and_expand_query(question, enable_expansion=bool(enable_expansion))
        retrieval_query = q_analysis.expanded_query

        # Step 1: Embed query + Semantic Cache check
        t_emb0 = time.perf_counter()
        texts_to_embed = [retrieval_query]
        if q_analysis.hyde_passage:
            texts_to_embed.append(q_analysis.hyde_passage)
        q_embs = await concurrency_mgr.embed_texts_async(embed_texts, texts_to_embed)
        if len(q_embs) > 1:
            q_vec = np.array(q_embs[0], dtype=np.float32)
            h_vec = np.array(q_embs[1], dtype=np.float32)
            blended = 0.7 * q_vec + 0.3 * h_vec
            norm = np.linalg.norm(blended)
            q_emb = (blended / norm) if norm > 0 else q_vec
        else:
            q_emb = q_embs[0]
        emb_ms = round((time.perf_counter() - t_emb0) * 1000.0, 2)

        # Dense Strategy selection + Shannon Entropy Gating
        routing_decision = await concurrency_mgr.run_in_thread(
            _router.predict_routing_decision, question, q_emb
        )
        strategy_to_use = routing_decision.chosen_strategy if (strategy == "auto" or not strategy) else strategy
        router_probs = routing_decision.probabilities

        if use_cache:
            cached = _semantic_cache.get(
                question, strategy_to_use, top_k, era_filter=era_filter, query_embedding=q_emb
            )
            if cached is not None:
                cached_meta = {
                    "question": question,
                    "strategy_used": cached.get("strategy_used", strategy_to_use),
                    "confidence": cached.get("confidence", "HIGH"),
                    "spread_score": cached.get("spread_score", 0.0),
                    "agreement_score": cached.get("agreement_score", 1.0),
                    "warning": cached.get("warning"),
                    "context_chunks": cached.get("context_chunks", []),
                    "retrieved_chunks": cached.get("retrieved_chunks", []),
                    "retrieved_metadatas": cached.get("retrieved_metadatas", []),
                    "router_probabilities": cached.get("router_probabilities", router_probs),
                    "router_decision": cached.get("router_decision", routing_decision.to_dict()),
                    "cache_hit": True,
                    "latency_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
                }
                yield f"event: meta\ndata: {json.dumps(cached_meta)}\n\n"
                words = re.findall(r"\S+\s*", cached.get("answer", "")) or [cached.get("answer", "")]
                for idx, w in enumerate(words):
                    yield f"event: token\ndata: {json.dumps({'token': w, 'index': idx + 1})}\n\n"
                    await asyncio.sleep(0.005)
                done_meta = {
                    "question": question,
                    "tokens_generated": len(words),
                    "total_latency_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
                    "finish_reason": "cache_hit",
                }
                yield f"event: done\ndata: {json.dumps(done_meta)}\n\n"
                return

        # Step 2: Multi-strategy parallel retrieval (Dense Vector + Sparse BM25 via asyncio.gather)
        t_dual0 = time.perf_counter()
        vec_results, bm25_results = await concurrency_mgr.retrieve_dual_branch_parallel(
            _vector_store.query,
            _bm25_store.query,
            q_emb,
            retrieval_query,
            top_k=10,
            where_clause=where_clause,
        )
        dual_ms = round((time.perf_counter() - t_dual0) * 1000.0, 2)
        vec_cids = [r.chunk_id for r in vec_results]
        bm25_cids = [cid for cid, _ in bm25_results]

        t_hyb0 = time.perf_counter()
        fused = reciprocal_rank_fusion(bm25_cids, vec_cids, k=60, top_k=10)
        hybrid_cids = [f.chunk_id for f in fused]
        hyb_ms = round((time.perf_counter() - t_hyb0) * 1000.0, 2)

        t_rerank0 = time.perf_counter()
        candidates = [(cid, _id_to_text[cid]) for cid in hybrid_cids[:10] if cid in _id_to_text]
        reranked = await concurrency_mgr.rerank_async(_reranker.rerank, question, candidates, top_k=10)
        rerank_cids = [cid for cid, _, _ in reranked]
        rerank_ms = round((time.perf_counter() - t_rerank0) * 1000.0, 2)

        strat_map = {"simple": bm25_cids, "semantic": vec_cids, "hybrid": hybrid_cids, "rerank": rerank_cids}
        selected_cids = strat_map.get(strategy_to_use, hybrid_cids)[:top_k]

        # Step 3: Uncertainty Quantification
        top_vecs = await concurrency_mgr.run_in_thread(_vector_store.get_all_embeddings_for_ids, vec_cids[:top_k])
        spread = embedding_spread_score(top_vecs)
        agreement = strategy_agreement_score(strat_map)
        conf = confidence_label(spread, agreement)

        warning_msg = None
        if conf == "LOW":
            warning_msg = "⚠ Low retrieval confidence — answer may be unreliable, consider rephrasing."

        # Parent-Document Context Expansion for streaming generation
        retrieved_texts = []
        seen_parents = set()
        for cid in selected_cids:
            pid = _id_to_meta.get(cid, {}).get("parent_id", cid)
            if pid in _parent_id_to_text and pid not in seen_parents:
                seen_parents.add(pid)
                retrieved_texts.append(_parent_id_to_text[pid])
            elif cid in _id_to_text and cid not in seen_parents:
                seen_parents.add(cid)
                retrieved_texts.append(_id_to_text[cid])
        if not retrieved_texts:
            retrieved_texts = [_id_to_text[cid] for cid in selected_cids if cid in _id_to_text]

        retrieved_metas = [
            {"chunk_id": cid, **_id_to_meta.get(cid, {})}
            for cid in selected_cids
            if cid in _id_to_text
        ]

        stage_latencies = {
            "embedding_ms": emb_ms,
            "parallel_retrieval_ms": dual_ms,
            "hybrid_rrf_ms": hyb_ms,
            "rerank_ms": rerank_ms,
        }

        # ── PHASE 1: Emit 'meta' SSE event ───────────────────────
        current_tid = get_current_trace_id()
        meta_event = {
            "question": question,
            "strategy_used": strategy_to_use,
            "confidence": conf,
            "spread_score": round(spread, 4),
            "agreement_score": round(agreement, 4),
            "warning": warning_msg,
            "context_chunks": retrieved_texts,
            "retrieved_chunks": selected_cids,
            "retrieved_metadatas": retrieved_metas,
            "router_probabilities": router_probs,
            "router_decision": routing_decision.to_dict(),
            "query_analysis": {
                "expanded_query": q_analysis.expanded_query,
                "code_symbols": q_analysis.code_symbols,
                "detected_era": q_analysis.detected_era,
                "complexity": q_analysis.query_complexity,
            },
            "cache_hit": False,
            "stage_latencies_ms": stage_latencies,
            "trace_id": current_tid,
        }
        yield f"event: meta\ndata: {json.dumps(meta_event)}\n\n"

        # ── PHASE 2: Emit 'token' SSE events as LLM tokens are produced ──
        t_gen0 = time.perf_counter()
        token_count = 0
        accumulated_tokens = []

        gen_stream = generate_answer_stream(question, retrieved_texts)
        for token in gen_stream:
            token_count += 1
            accumulated_tokens.append(token)
            token_payload = {"token": token, "index": token_count}
            yield f"event: token\ndata: {json.dumps(token_payload)}\n\n"
            await asyncio.sleep(0.005)

        gen_ms = round((time.perf_counter() - t_gen0) * 1000.0, 2)
        full_answer = "".join(accumulated_tokens)
        total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        record_tokens_generated("ollama", PRIMARY_MODEL, token_count)
        record_stage_duration("llm_stream", gen_ms / 1000.0)

        if use_cache:
            final_payload = {
                **meta_event,
                "answer": full_answer,
                "latency_ms": total_latency_ms,
            }
            _semantic_cache.put(
                question,
                strategy_to_use,
                top_k,
                payload=final_payload,
                original_latency_ms=total_latency_ms,
                era_filter=era_filter,
                query_embedding=q_emb,
            )

        # ── PHASE 3: Emit 'done' SSE event with final token and latency telemetry ──
        done_payload = {
            "question": question,
            "full_answer": full_answer,
            "tokens_generated": token_count,
            "generation_ms": gen_ms,
            "total_latency_ms": total_latency_ms,
            "finish_reason": "stop",
            "trace_id": current_tid,
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    except Exception as err:
        error_payload = {"error": str(err)}
        yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
    finally:
        concurrency_mgr.track_request_end()


@app.post("/query/stream")
async def stream_query_post(req: QueryRequest):
    """
    Server-Sent Events (SSE) token streaming endpoint (POST).
    Emits event: meta (retrieval + uncertainty), event: token (tokens), event: done (metrics).
    """
    return StreamingResponse(
        stream_rag_events(
            question=req.question.strip(),
            strategy=req.strategy or "auto",
            top_k=req.top_k or 4,
            era_filter=(req.era_filter or "all").lower(),
            enable_expansion=bool(req.enable_expansion),
            use_cache=bool(req.use_cache),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream; charset=utf-8",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/query/stream")
async def stream_query_get(
    question: str = Query(..., description="Query to stream"),
    strategy: str = Query("auto", description="Retrieval strategy"),
    top_k: int = Query(4, description="Top K context chunks"),
    era_filter: str = Query("all", description="Era filter: legacy, current, all"),
):
    """
    Server-Sent Events (SSE) token streaming endpoint (GET for browser EventSource clients).
    """
    return StreamingResponse(
        stream_rag_events(
            question=question.strip(),
            strategy=strategy,
            top_k=top_k,
            era_filter=era_filter,
            enable_expansion=True,
            use_cache=True,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream; charset=utf-8",
            "X-Accel-Buffering": "no",
        },
    )


async def _compute_confidence_payload(question: str) -> Dict[str, Any]:
    if _vector_store is None:
        await concurrency_mgr.run_in_thread(initialize_pipeline)
    q_embs = await concurrency_mgr.embed_texts_async(embed_texts, [question])
    q_emb = q_embs[0]

    vec_results, bm25_results = await concurrency_mgr.retrieve_dual_branch_parallel(
        _vector_store.query,
        _bm25_store.query,
        q_emb,
        question,
        top_k=4,
    )
    vec_cids = [r.chunk_id for r in vec_results]
    bm25_cids = [cid for cid, _ in bm25_results]
    hybrid_cids = [f.chunk_id for f in reciprocal_rank_fusion(bm25_cids, vec_cids, k=60, top_k=4)]

    strat_map = {"simple": bm25_cids, "semantic": vec_cids, "hybrid": hybrid_cids}
    top_vecs = await concurrency_mgr.run_in_thread(_vector_store.get_all_embeddings_for_ids, vec_cids[:4])
    spread = round(embedding_spread_score(top_vecs), 4)
    agreement = round(strategy_agreement_score(strat_map), 4)
    conf = confidence_label(spread, agreement)

    return {
        "question": question,
        "confidence": conf,
        "spread_score": spread,
        "agreement_score": agreement,
        "embedding_spread": spread,
        "strategy_agreement": agreement,
    }


@app.get("/confidence")
async def check_confidence_get(question: str = Query(..., description="Question to evaluate retrieval uncertainty for")):
    """Standalone uncertainty check without LLM generation (GET)."""
    return await _compute_confidence_payload(question)


@app.post("/confidence")
async def check_confidence_post(req: ConfidenceRequest):
    """Standalone uncertainty check without LLM generation (POST)."""
    return await _compute_confidence_payload(req.question)


@app.post("/compare")
async def compare_strategies(req: QueryRequest):
    """
    Executes all 5 retrieval strategies side-by-side with non-blocking concurrency
    and returns detailed rank, score, source era, and latency telemetry for interactive visualization.
    """
    concurrency_mgr.track_request_start()
    try:
        if _vector_store is None:
            await concurrency_mgr.run_in_thread(initialize_pipeline)
        question = req.question.strip()
        top_k = max(1, min(req.top_k or 4, 10))
        era_filter = (req.era_filter or "all").lower()
        where_clause = {"source": era_filter} if era_filter in ("legacy", "current") else None

        q_analysis = analyze_and_expand_query(question, enable_expansion=bool(req.enable_expansion))
        retrieval_query = q_analysis.expanded_query

        t0 = time.perf_counter()
        q_embs = await concurrency_mgr.embed_texts_async(embed_texts, [retrieval_query])
        q_emb = q_embs[0]
        emb_ms = (time.perf_counter() - t0) * 1000.0

        # Parallel Dual-Branch Retrieval (Vector + BM25)
        t_dual = time.perf_counter()
        vec_res, bm25_res = await concurrency_mgr.retrieve_dual_branch_parallel(
            _vector_store.query,
            _bm25_store.query,
            q_emb,
            retrieval_query,
            top_k=10,
            where_clause=where_clause,
        )
        dual_ms = (time.perf_counter() - t_dual) * 1000.0

        # 1. BM25 (Simple)
        bm25_ms = round(dual_ms, 2)
        bm25_cids = [cid for cid, _ in bm25_res]
        bm25_scores = {cid: round(sc, 4) for cid, sc in bm25_res}

        # 2. Vector (Semantic)
        vec_ms = round(emb_ms + dual_ms, 2)
        vec_cids = [r.chunk_id for r in vec_res]
        vec_scores = {r.chunk_id: round(1.0 - r.distance, 4) for r in vec_res}

        # 3. Hybrid RRF
        t_hyb = time.perf_counter()
        fused = reciprocal_rank_fusion(bm25_cids, vec_cids, k=60, top_k=10)
        hyb_ms = round(dual_ms + (time.perf_counter() - t_hyb) * 1000.0, 2)
        hybrid_cids = [f.chunk_id for f in fused]
        hybrid_meta = {
            f.chunk_id: {
                "score": round(f.fused_score, 5),
                "bm25_rank": f.bm25_rank,
                "vector_rank": f.vector_rank,
            }
            for f in fused
        }

        # 4. Cross-Encoder Rerank (semaphore guarded)
        t_rr = time.perf_counter()
        candidates = [(cid, _id_to_text[cid]) for cid in hybrid_cids[:10] if cid in _id_to_text]
        reranked = await concurrency_mgr.rerank_async(_reranker.rerank, question, candidates, top_k=10)
        rerank_ms = round(hyb_ms + (time.perf_counter() - t_rr) * 1000.0, 2)
        rerank_cids = [cid for cid, _, _ in reranked]
        rerank_scores = {cid: round(sc, 4) for cid, _, sc in reranked}

        # 5. Router (Dense + Shannon Entropy Gated)
        t_rt = time.perf_counter()
        routing_decision = await concurrency_mgr.run_in_thread(
            _router.predict_routing_decision, question, q_emb
        )
        chosen_strat = routing_decision.chosen_strategy
        router_probs = routing_decision.probabilities
        router_overhead = (time.perf_counter() - t_rt) * 1000.0
        strat_cids = {"simple": bm25_cids, "semantic": vec_cids, "hybrid": hybrid_cids, "rerank": rerank_cids}
        strat_lats = {"simple": bm25_ms, "semantic": vec_ms, "hybrid": hyb_ms, "rerank": rerank_ms}
        router_cids = strat_cids.get(chosen_strat, hybrid_cids)
        router_ms = round(router_overhead + strat_lats.get(chosen_strat, hyb_ms), 2)

        def _format_chunks(cids: list[str], score_lookup: dict, extra_lookup: dict | None = None):
            items = []
            for rank, cid in enumerate(cids[:top_k], start=1):
                meta = _id_to_meta.get(cid, {})
                text = _id_to_text.get(cid, "")
                entry = {
                    "rank": rank,
                    "chunk_id": cid,
                    "title": meta.get("title", cid),
                    "source": meta.get("source", "unknown"),
                    "doc_id": meta.get("doc_id", ""),
                    "score": score_lookup.get(cid, 0.0),
                    "snippet": text[:260] + ("..." if len(text) > 260 else ""),
                    "text": text,
                }
                if extra_lookup and cid in extra_lookup:
                    entry.update(extra_lookup[cid])
                items.append(entry)
            return items

        top_vecs = await concurrency_mgr.run_in_thread(_vector_store.get_all_embeddings_for_ids, vec_cids[:top_k])
        spread = round(embedding_spread_score(top_vecs), 4)
        agreement = round(strategy_agreement_score(strat_cids), 4)
        conf = confidence_label(spread, agreement)

        return {
            "question": question,
            "query_analysis": {
                "expanded_query": q_analysis.expanded_query,
                "code_symbols": q_analysis.code_symbols,
                "detected_era": q_analysis.detected_era,
                "complexity": q_analysis.query_complexity,
            },
            "router_decision": routing_decision.to_dict(),
            "uncertainty": {
                "confidence": conf,
                "spread_score": spread,
                "agreement_score": agreement,
            },
            "strategies": {
                "simple": {
                    "label": "BM25 Keyword",
                    "latency_ms": bm25_ms,
                    "chunks": _format_chunks(bm25_cids, bm25_scores),
                },
                "semantic": {
                    "label": "Dense Vector (MiniLM/BGE)",
                    "latency_ms": vec_ms,
                    "chunks": _format_chunks(vec_cids, vec_scores),
                },
                "hybrid": {
                    "label": "Hybrid RRF (k=60)",
                    "latency_ms": hyb_ms,
                    "chunks": _format_chunks(
                        hybrid_cids,
                        {k: v["score"] for k, v in hybrid_meta.items()},
                        extra_lookup=hybrid_meta,
                    ),
                },
                "rerank": {
                    "label": "Cross-Encoder Re-Ranker",
                    "latency_ms": rerank_ms,
                    "chunks": _format_chunks(rerank_cids, rerank_scores),
                },
                "router": {
                    "label": f"ML Router → {chosen_strat.upper()}",
                    "latency_ms": router_ms,
                    "routed_to": chosen_strat,
                    "chunks": _format_chunks(router_cids, rerank_scores if chosen_strat == "rerank" else bm25_scores),
                },
            },
        }
    finally:
        concurrency_mgr.track_request_end()


@app.post("/debate")
async def run_faithfulness_debate(req: DebateRequest):
    """Executes the Dual-LLM Multi-Judge Faithfulness Debate protocol non-blocking in worker thread."""
    verdict = await concurrency_mgr.run_in_thread(
        debate_faithfulness, req.question, req.context, req.answer
    )
    return {
        "question": req.question,
        "primary_judge_model": PRIMARY_MODEL,
        "secondary_judge_model": SECONDARY_MODEL,
        **verdict,
    }


@app.post("/adversarial")
async def run_adversarial_probe(req: AdversarialRequest):
    """Generates paraphrase, negated, and multi-hop adversarial variants plus a counterfactual decoy."""
    if _vector_store is None:
        await concurrency_mgr.run_in_thread(initialize_pipeline)
    variants = await concurrency_mgr.run_in_thread(
        generate_adversarial_set,
        [
            {
                "id": "live_probe",
                "question": req.question,
                "answer": req.answer or "",
                "gold_chunk_ids": [],
            }
        ],
    )
    source_chunk = req.chunk_text
    if not source_chunk:
        bm = await concurrency_mgr.run_in_thread(_bm25_store.query, req.question, 1)
        if bm and bm[0][0] in _id_to_text:
            source_chunk = _id_to_text[bm[0][0]]
        else:
            source_chunk = "React components use useState and useEffect hooks to manage state and side effects."
    decoy = await concurrency_mgr.run_in_thread(generate_decoy, source_chunk)
    return {
        "question": req.question,
        "variants": variants,
        "original_chunk": source_chunk[:400],
        "counterfactual_decoy": decoy,
    }


@app.get("/benchmark")
async def get_latest_benchmark():
    """Returns the most recent evaluation run JSON, drift logs, and sample ground truth QA pairs."""
    eval_files = sorted(EVAL_RUNS_DIR.glob("eval_run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest_eval = None
    if eval_files:
        try:
            latest_eval = json.loads(eval_files[0].read_text(encoding="utf-8"))
        except Exception:
            latest_eval = None

    drift_files = sorted(DRIFT_LOGS_DIR.glob("drift_check_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest_drift = None
    if drift_files:
        try:
            latest_drift = json.loads(drift_files[0].read_text(encoding="utf-8"))
        except Exception:
            latest_drift = None

    qa_sample = []
    qa_file = GROUND_TRUTH_DIR / "qa_pairs.json"
    if qa_file.exists():
        try:
            all_qa = json.loads(qa_file.read_text(encoding="utf-8"))
            qa_sample = all_qa[:20]
        except Exception:
            pass

    flagged = []
    if latest_eval and "per_question_results" in latest_eval:
        hyb_res = latest_eval["per_question_results"].get("hybrid", [])
        flagged = flag_disagreements(hyb_res, confidence_threshold=0.3)[:15]

    return {
        "latest_eval": latest_eval,
        "latest_drift": latest_drift,
        "active_learning_flags": flagged,
        "sample_questions": qa_sample,
    }


@app.post("/cache/clear")
async def clear_semantic_cache():
    """Clears the semantic query cache."""
    _semantic_cache.clear()
    return {"status": "cleared", "cache": _semantic_cache.stats()}


@app.post("/eval")
async def trigger_eval():
    """Trigger full evaluation harness non-blocking and return structured results."""
    res = await concurrency_mgr.run_in_thread(evaluate_harness)
    return {"status": "completed", "results": res}


class GenerationEvalRequest(BaseModel):
    question: str
    answer: str
    context: Union[List[str], str]
    ground_truth: Optional[str] = None


@app.post("/eval/generation")
async def evaluate_rag_generation(req: GenerationEvalRequest):
    """
    Ragas-Style End-to-End Generation Evaluation:
    Computes Faithfulness, Answer Relevance, Context Recall, and Root-Cause Failure Diagnosis.
    """
    from src.evaluation.generation_eval import evaluate_generation_sample
    res = await concurrency_mgr.run_in_thread(
        evaluate_generation_sample,
        question=req.question,
        answer=req.answer,
        context=req.context,
        ground_truth=req.ground_truth,
    )
    record_evaluation_scores(res.faithfulness, res.answer_relevance, res.context_recall)
    return res.to_dict()


@app.post("/feedback")
async def record_user_feedback(req: FeedbackRequest):
    """
    Records end-user feedback (ratings, corrections, error categories) for continuous RLHF,
    failure attribution, and active learning curation loops.
    """
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    feedback_entry = {
        "timestamp": time.time(),
        "trace_id": req.trace_id or get_current_trace_id(),
        "rating": req.rating,
        "category": req.category or ("positive" if req.rating > 0 else "negative"),
        "comment": req.comment,
        "question": req.question,
        "answer": req.answer,
        "metadata": req.metadata or {},
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_entry) + "\n")

    # Online Bayesian Bandit update from user rating
    if _router is not None and req.metadata and "strategy" in req.metadata:
        reward = 1.0 if req.rating >= 4 else (0.5 if req.rating == 3 else 0.0)
        _router.update_bandit_reward(str(req.metadata["strategy"]), reward)

    return {
        "status": "success",
        "message": "User feedback recorded successfully",
        "entry": feedback_entry,
    }


@app.get("/feedback")
async def get_user_feedback(limit: int = Query(50, ge=1, le=500)):
    """Retrieves recent user feedback entries for telemetry and analysis."""
    if not FEEDBACK_FILE.exists():
        return {"total": 0, "entries": []}
    entries = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return {"total": len(entries), "entries": entries[-limit:]}


class BanditRewardRequest(BaseModel):
    strategy: str
    reward: float


@app.get("/router/bandit")
async def get_router_bandit_telemetry():
    """Returns Thompson Sampling posterior distributions and exploration metrics."""
    if _router is None:
        return {"status": "uninitialized"}
    return _router.get_bandit_stats()


@app.post("/router/bandit/update")
async def update_router_bandit(req: BanditRewardRequest):
    """Updates Bayesian Thompson Sampling posterior with online reward signal."""
    if _router is not None:
        _router.update_bandit_reward(req.strategy, req.reward)
        return {"status": "updated", "bandit": _router.get_bandit_stats()}
    return {"status": "uninitialized"}


