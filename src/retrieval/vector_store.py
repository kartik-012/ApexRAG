"""
Vector store wrapper — ChromaDB local persistent store with cosine distance.
"""

from dataclasses import dataclass
import numpy as np
from src.config import CHROMA_PERSIST_PATH, CHROMA_COLLECTION


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    distance: float   # lower = more similar


class VectorStore:
    def __init__(self, persist_path: str = CHROMA_PERSIST_PATH, collection_name: str = CHROMA_COLLECTION):
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.use_fallback = False
        self._fallback_store = {}

        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=persist_path)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            print(f"[Warning] ChromaDB PersistentClient init failed ({e}). Using in-memory vector store.")
            self.use_fallback = True

    def add_chunks(self, chunk_ids: list[str], texts: list[str],
                    embeddings, metadatas: list[dict]) -> None:
        if not (len(chunk_ids) == len(texts) == len(metadatas)):
            raise ValueError("chunk_ids, texts, and metadatas must be the same length")
        
        if self.use_fallback:
            for cid, text, emb, meta in zip(chunk_ids, texts, embeddings, metadatas):
                self._fallback_store[cid] = {
                    "text": text,
                    "embedding": np.array(emb),
                    "metadata": meta,
                }
            return

        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings if isinstance(embeddings, list) else embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, query_embedding, top_k: int = 4, where: dict | None = None) -> list[RetrievedChunk]:
        query_vec = np.array(query_embedding).squeeze()

        if self.use_fallback:
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
        if self.use_fallback:
            return len(self._fallback_store)
        return self.collection.count()

    def get_all_embeddings_for_ids(self, chunk_ids: list[str]) -> list[np.ndarray]:
        if self.use_fallback:
            return [self._fallback_store[cid]["embedding"] for cid in chunk_ids if cid in self._fallback_store]

        res = self.collection.get(ids=chunk_ids, include=["embeddings"])
        return [np.array(e) for e in res.get("embeddings", [])]
