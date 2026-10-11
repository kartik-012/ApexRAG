-- ApexRAG PostgreSQL + pgvector Initialization Script
-- Executed on container startup to initialize vector extension and HNSW indexes

CREATE EXTENSION IF NOT EXISTS vector;

-- Production document chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Production HNSW index for sub-10ms approximate nearest neighbor search (Cosine Distance)
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_cosine
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Metadata filtering GIN index for source ('legacy' vs 'current') and doc_id
CREATE INDEX IF NOT EXISTS idx_chunks_metadata_gin
ON document_chunks
USING gin (metadata);

CREATE INDEX IF NOT EXISTS idx_chunks_metadata_source
ON document_chunks ((metadata->>'source'));
