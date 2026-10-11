"""
Non-blocking Asynchronous Concurrency & Thread Pool Management for ApexRAG.

Provides:
  - Non-blocking offloading of synchronous CPU-bound and I/O-bound operations
    (PyTorch embedding, CrossEncoder reranking, BM25 scoring, pgvector queries, LLM generation)
  - Concurrent dual-branch retrieval (Dense Vector + Sparse BM25 via asyncio.gather)
  - Bounded semaphores protecting neural embedding and reranking from thread thrashing/OOM
  - Real-time concurrency telemetry (active requests, peak concurrency, queue depth)
"""

import asyncio
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from src.config import (
    MAX_CONCURRENT_EMBEDS,
    MAX_CONCURRENT_RERANKS,
    PARALLEL_DUAL_RETRIEVAL,
)

T = TypeVar("T")


class ConcurrencyManager:
    """
    Manages non-blocking thread execution, parallel retrieval branches,
    and semaphore-guarded neural inferences for high-throughput FastAPI servers.
    """

    def __init__(
        self,
        max_concurrent_embeds: int = MAX_CONCURRENT_EMBEDS,
        max_concurrent_reranks: int = MAX_CONCURRENT_RERANKS,
        parallel_dual_retrieval: bool = PARALLEL_DUAL_RETRIEVAL,
    ):
        self.max_concurrent_embeds = max_concurrent_embeds
        self.max_concurrent_reranks = max_concurrent_reranks
        self.parallel_dual_retrieval = parallel_dual_retrieval

        # Semaphores are lazily bound to the running asyncio loop
        self._embed_sem: Optional[asyncio.Semaphore] = None
        self._rerank_sem: Optional[asyncio.Semaphore] = None

        # Telemetry
        self._active_requests: int = 0
        self._peak_concurrent_requests: int = 0
        self._total_requests_served: int = 0
        self._parallel_retrievals_executed: int = 0
        self._active_embed_jobs: int = 0
        self._active_rerank_jobs: int = 0

    def _get_embed_sem(self) -> asyncio.Semaphore:
        if self._embed_sem is None:
            self._embed_sem = asyncio.Semaphore(self.max_concurrent_embeds)
        return self._embed_sem

    def _get_rerank_sem(self) -> asyncio.Semaphore:
        if self._rerank_sem is None:
            self._rerank_sem = asyncio.Semaphore(self.max_concurrent_reranks)
        return self._rerank_sem

    def track_request_start(self):
        """Records an incoming request for concurrency tracking."""
        self._active_requests += 1
        if self._active_requests > self._peak_concurrent_requests:
            self._peak_concurrent_requests = self._active_requests

    def track_request_end(self):
        """Records a completed request."""
        self._active_requests = max(0, self._active_requests - 1)
        self._total_requests_served += 1

    async def run_in_thread(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Executes a blocking callable in a worker thread via asyncio.to_thread,
        preventing event loop head-of-line blocking.
        """
        if kwargs:
            call = partial(func, *args, **kwargs)
            return await asyncio.to_thread(call)
        return await asyncio.to_thread(func, *args)

    async def embed_texts_async(self, embed_fn: Callable[[List[str]], Any], texts: List[str]) -> Any:
        """
        Executes text embedding in a worker thread guarded by a bounded semaphore
        to prevent PyTorch memory spikes under heavy load.
        """
        sem = self._get_embed_sem()
        async with sem:
            self._active_embed_jobs += 1
            try:
                return await self.run_in_thread(embed_fn, texts)
            finally:
                self._active_embed_jobs = max(0, self._active_embed_jobs - 1)

    async def rerank_async(
        self,
        rerank_fn: Callable[..., Any],
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: int = 10,
    ) -> Any:
        """
        Executes cross-encoder reranking in a worker thread guarded by a bounded semaphore
        to avoid CPU cache thrashing.
        """
        sem = self._get_rerank_sem()
        async with sem:
            self._active_rerank_jobs += 1
            try:
                return await self.run_in_thread(rerank_fn, query, candidates, top_k)
            finally:
                self._active_rerank_jobs = max(0, self._active_rerank_jobs - 1)

    async def retrieve_dual_branch_parallel(
        self,
        vec_query_fn: Callable[..., Any],
        bm25_query_fn: Callable[..., Any],
        query_embedding: Any,
        retrieval_query: str,
        top_k: int = 10,
        where_clause: Optional[dict] = None,
    ) -> Tuple[Any, Any]:
        """
        Executes Dense Vector search and Sparse BM25 search in parallel across
        separate OS worker threads via asyncio.gather().

        Reduces retrieval phase latency from t_vec + t_bm25 down to max(t_vec, t_bm25).
        """
        if not self.parallel_dual_retrieval:
            # Fallback to sequential execution if disabled
            vec_res = await self.run_in_thread(vec_query_fn, query_embedding, top_k, where_clause)
            bm25_res = await self.run_in_thread(bm25_query_fn, retrieval_query, top_k, where_clause)
            return vec_res, bm25_res

        self._parallel_retrievals_executed += 1
        vec_task = self.run_in_thread(vec_query_fn, query_embedding, top_k, where_clause)
        bm25_task = self.run_in_thread(bm25_query_fn, retrieval_query, top_k, where_clause)

        vec_res, bm25_res = await asyncio.gather(vec_task, bm25_task)
        return vec_res, bm25_res

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns runtime concurrency metrics for /health and observability dashboards."""
        return {
            "active_requests": self._active_requests,
            "peak_concurrent_requests": self._peak_concurrent_requests,
            "total_requests_served": self._total_requests_served,
            "parallel_retrievals_executed": self._parallel_retrievals_executed,
            "parallel_retrieval_enabled": self.parallel_dual_retrieval,
            "active_embed_jobs": self._active_embed_jobs,
            "active_rerank_jobs": self._active_rerank_jobs,
            "max_concurrent_embeds": self.max_concurrent_embeds,
            "max_concurrent_reranks": self.max_concurrent_reranks,
        }


# Singleton concurrency manager
concurrency_mgr = ConcurrencyManager()
