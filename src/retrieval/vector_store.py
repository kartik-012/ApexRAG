"""
Vector store wrapper — Dual Backend: Production PostgreSQL/pgvector with HNSW index
and ChromaDB local persistent store fallback.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from src.config import (
    CHROMA_PERSIST_PATH,
    CHROMA_COLLECTION,
    VECTOR_STORE_BACKEND,
)
from src.retrieval.pgvector_store import PgVectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    distance: float   # lower = more similar


class VectorStore:
    def __init__(
        self,
        persist_path: str = CHROMA_PERSIST_PATH,
        collection_name: str = CHROMA_COLLECTION,
        backend_preference: str = VECTOR_STORE_BACKEND,
    ):
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.backend = "in_memory"
        self._fallback_store = {}
        self.pg_store: Optional[PgVectorStore] = None

        # 1. Attempt PostgreSQL + pgvector if preferred or auto
        if backend_preference in ("pgvector", "auto"):
            try:
                pg = PgVectorStore()
                if pg.is_healthy():
                    self.pg_store = pg
                    self.backend = "pgvector"
            except Exception:
                self.pg_store = None

        # 2. Fall back to ChromaDB if pgvector is not available or explicitly requested
        if self.backend != "pgvector":
            try:
                import chromadb
                self.client = chromadb.PersistentClient(path=persist_path)
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                self.backend = "chroma"
            except Exception as e:
                print(f"[Warning] ChromaDB PersistentClient init failed ({e}). Using in-memory vector store.")
                self.backend = "in_memory"

    @property
    def is_pgvector(self) -> bool:
        return self.backend == "pgvector"

    def add_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings,
        metadatas: list[dict],
    ) -> None:
        if not (len(chunk_ids) == len(texts) == len(metadatas)):
            raise ValueError("chunk_ids, texts, and metadatas must be the same length")

        if self.backend == "pgvector" and self.pg_store:
            self.pg_store.add_chunks(chunk_ids, texts, embeddings, metadatas)
            return

        if self.backend == "in_memory":
            for cid, text, emb, meta in zip(chunk_ids, texts, embeddings, metadatas):
                self._fallback_store[cid] = {
                    "text": text,
                    "embedding": np.array(emb),
                    "metadata": meta,
                }
            return

        emb_list = embeddings if isinstance(embeddings, list) else embeddings.tolist()
        batch_size = 500
        for start in range(0, len(chunk_ids), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=chunk_ids[start:end],
                embeddings=emb_list[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

    def query(
        self,
        query_embedding,
        top_k: int = 4,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        if self.backend == "pgvector" and self.pg_store:
            res = self.pg_store.query(query_embedding, top_k=top_k, where=where)
            return [
                RetrievedChunk(
                    chunk_id=r["chunk_id"],
                    text=r["text"],
                    metadata=r["metadata"],
                    distance=r["distance"],
                )
                for r in res
            ]

        query_vec = np.array(query_embedding).squeeze()

        if self.backend == "in_memory":
            results = []
            for cid, item in self._fallback_store.items():
                if where:
                    match = all(item["metadata"].get(k) == v for k, v in where.items())
                    if not match:
                        continue
                emb = item["embedding"].squeeze()
                norm_q = np.linalg.norm(query_vec)
                norm_e = np.linalg.norm(emb)
                cos_sim = np.dot(query_vec, emb) / (norm_q * norm_e + 1e-9)
                dist = max(0.0, 1.0 - float(cos_sim))
                results.append(RetrievedChunk(
                    chunk_id=cid,
                    text=item["text"],
                    metadata=item["metadata"],
                    distance=dist,
                ))
            results.sort(key=lambda r: r.distance)
            return results[:top_k]

        q_list = query_vec.tolist()
        query_res = self.collection.query(
            query_embeddings=[q_list],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        if query_res["ids"] and query_res["ids"][0]:
            for i in range(len(query_res["ids"][0])):
                chunks.append(RetrievedChunk(
                    chunk_id=query_res["ids"][0][i],
                    text=query_res["documents"][0][i],
                    metadata=query_res["metadatas"][0][i],
                    distance=query_res["distances"][0][i],
                ))
        return chunks

    def count(self) -> int:
        if self.backend == "pgvector" and self.pg_store:
            return self.pg_store.count()
        if self.backend == "in_memory":
            return len(self._fallback_store)
        return self.collection.count()

    def get_all_embeddings_for_ids(self, chunk_ids: list[str]) -> list[np.ndarray]:
        if self.backend == "pgvector" and self.pg_store:
            return self.pg_store.get_all_embeddings_for_ids(chunk_ids)
        if self.backend == "in_memory":
            return [self._fallback_store[cid]["embedding"] for cid in chunk_ids if cid in self._fallback_store]

        res = self.collection.get(ids=chunk_ids, include=["embeddings"])
        return [np.array(e) for e in res.get("embeddings", [])]

