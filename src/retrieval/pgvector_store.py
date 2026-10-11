"""
PostgreSQL + pgvector Production Vector Store (Solution 2).

Features:
- Enterprise HNSW index (Hierarchical Navigable Small World) for sub-10ms approximate nearest neighbor search.
- Thread-safe connection pooling via psycopg2.pool.ThreadedConnectionPool.
- Native vector registration via pgvector.psycopg2.register_vector.
- Idempotent schema initialization & batched upserts.
- JSONB metadata filtering and cosine distance scoring.
"""

import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from src.config import (
    PG_HOST,
    PG_PORT,
    PG_DATABASE,
    PG_USER,
    PG_PASSWORD,
    PG_DSN,
)

logger = logging.getLogger(__name__)


class PgVectorStore:
    def __init__(
        self,
        dsn: str = PG_DSN,
        min_conn: int = 1,
        max_conn: int = 10,
        dimension: int = 384,
    ):
        self.dsn = dsn
        self.min_conn = min_conn
        self.max_conn = max_conn
        self.dimension = dimension
        self.pool = None
        self._is_ready = False
        self._init_pool()

    def _init_pool(self) -> bool:
        try:
            import psycopg2
            from psycopg2 import pool
            from pgvector.psycopg2 import register_vector

            self.pool = pool.ThreadedConnectionPool(
                minconn=self.min_conn,
                maxconn=self.max_conn,
                dsn=self.dsn,
                connect_timeout=3,
            )
            # Verify and register vector type
            conn = self.pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                register_vector(conn)
                self.init_schema(conn)
                conn.commit()
                self._is_ready = True
                logger.info(f"Connected to pgvector at {PG_HOST}:{PG_PORT}/{PG_DATABASE}")
            finally:
                self.pool.putconn(conn)
            return True
        except Exception as e:
            logger.warning(f"pgvector pool initialization failed ({e}). Database might be offline.")
            self.pool = None
            self._is_ready = False
            return False

    def is_healthy(self) -> bool:
        if not self._is_ready or not self.pool:
            return False
        try:
            conn = self.pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                return True
            finally:
                self.pool.putconn(conn)
        except Exception:
            return False

    def init_schema(self, conn) -> None:
        """Initializes pgvector extension, chunks table, and HNSW index idempotently."""
        from pgvector.psycopg2 import register_vector
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding vector({self.dimension}) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Create HNSW index for high-scale cosine nearest neighbor search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_cosine
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            # GIN metadata index for fast source/doc_id filtering
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_metadata_gin
                ON document_chunks
                USING gin (metadata);
            """)

    def add_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        batch_size: int = 250,
    ) -> None:
        if not (len(chunk_ids) == len(texts) == len(metadatas)):
            raise ValueError("chunk_ids, texts, and metadatas must have equal length")

        if not self.is_healthy():
            raise RuntimeError("pgvector database connection is not available.")

        from psycopg2.extras import execute_batch

        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                emb_list = embeddings if isinstance(embeddings, list) else embeddings.tolist()
                params = [
                    (
                        cid,
                        txt,
                        np.array(emb_list[i]).tolist(),
                        json.dumps(metadatas[i]),
                    )
                    for i, (cid, txt) in enumerate(zip(chunk_ids, texts))
                ]
                sql = """
                    INSERT INTO document_chunks (id, text, embedding, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata;
                """
                execute_batch(cur, sql, params, page_size=batch_size)
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.is_healthy():
            raise RuntimeError("pgvector database connection is not available.")

        q_vec = np.array(query_embedding).squeeze().tolist()
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                where_clauses = []
                params = [q_vec]

                if where:
                    for k, v in where.items():
                        where_clauses.append("(metadata->>%s) = %s")
                        params.extend([k, str(v)])

                where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
                params.extend([q_vec, top_k])

                sql = f"""
                    SELECT id, text, metadata, (embedding <=> %s::vector) AS distance
                    FROM document_chunks
                    {where_sql}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """
                cur.execute(sql, params)
                rows = cur.fetchall()

                results = []
                for r in rows:
                    cid, txt, meta, dist = r
                    meta_dict = meta if isinstance(meta, dict) else json.loads(meta or "{}")
                    results.append({
                        "chunk_id": cid,
                        "text": txt,
                        "metadata": meta_dict,
                        "distance": float(dist),
                    })
                return results
        finally:
            self.pool.putconn(conn)

    def count(self) -> int:
        if not self.is_healthy():
            return 0
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM document_chunks;")
                return cur.fetchone()[0]
        finally:
            self.pool.putconn(conn)

    def get_all_embeddings_for_ids(self, chunk_ids: List[str]) -> List[np.ndarray]:
        if not self.is_healthy() or not chunk_ids:
            return []
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, embedding FROM document_chunks WHERE id = ANY(%s);",
                    (chunk_ids,),
                )
                lookup = {r[0]: np.array(r[1]) for r in cur.fetchall()}
                return [lookup[cid] for cid in chunk_ids if cid in lookup]
        finally:
            self.pool.putconn(conn)

    def close(self) -> None:
        if self.pool:
            self.pool.closeall()
            self.pool = None
            self._is_ready = False
