"""
Full-corpus vector store build script for local development and production.

Builds ChromaDB vector store and BM25 index from the React documentation corpus.
Uses synthetic corpus by default; pass --legacy-dir and --current-dir to use real clones.

Usage:
    python scripts/build_index.py                           # Use synthetic corpus
    python scripts/build_index.py --legacy-dir=data/raw_docs/legacy --current-dir=data/raw_docs/current
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from src.ingestion.loader import load_legacy_docs, load_current_docs
from src.ingestion.chunkers import simple_chunk
from src.ingestion.embedder import embed_texts
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.config import CHROMA_PERSIST_PATH, CHROMA_COLLECTION


def main():
    parser = argparse.ArgumentParser(description="ApexRAG — Build Vector + BM25 Index")
    parser.add_argument("--legacy-dir", type=str, default=os.getenv("LEGACY_REPO", "data/raw_docs/react-legacy"),
                        help="Path to reactjs.org v17 clone root")
    parser.add_argument("--current-dir", type=str, default=os.getenv("CURRENT_REPO", "data/raw_docs/react-dev-current"),
                        help="Path to react.dev clone root")
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="Words per chunk (default: 200)")
    parser.add_argument("--overlap", type=int, default=30,
                        help="Word overlap between chunks (default: 30)")
    args = parser.parse_args()

    print("=" * 60)
    print(" ApexRAG — Index Builder")
    print("=" * 60)

    # Load corpus
    docs = []
    if Path(args.legacy_dir).exists():
        legacy_docs = load_legacy_docs(args.legacy_dir)
        print(f"Loaded {len(legacy_docs)} legacy docs from {args.legacy_dir}")
        docs.extend(legacy_docs)
    if Path(args.current_dir).exists():
        current_docs = load_current_docs(args.current_dir)
        print(f"Loaded {len(current_docs)} current docs from {args.current_dir}")
        docs.extend(current_docs)

    # Build chunks
    vector_store = VectorStore(persist_path=CHROMA_PERSIST_PATH, collection_name=CHROMA_COLLECTION)
    bm25_store = BM25Store()

    chunk_ids, texts, metadatas = [], [], []
    for doc in docs:
        chunks = simple_chunk(doc.text, chunk_size=args.chunk_size, overlap=args.overlap)
        for idx, text in enumerate(chunks):
            cid = f"{doc.doc_id}_chunk{idx}"
            chunk_ids.append(cid)
            texts.append(text)
            metadatas.append({"source": doc.source, "doc_id": doc.doc_id, "title": doc.title})

    print(f"\nGenerated {len(chunk_ids)} chunks from {len(docs)} documents.")
    print("Embedding chunks (this may take a minute on first run)...")

    embeddings = embed_texts(texts)
    vector_store.add_chunks(chunk_ids, texts, embeddings, metadatas)
    bm25_store.build(chunk_ids, texts)

    print(f"\n✓ VectorStore: {vector_store.count()} chunks indexed at {CHROMA_PERSIST_PATH}")
    print(f"✓ BM25Store: {len(chunk_ids)} chunks indexed (in-memory)")
    print("\nRun `uvicorn src.api.main:app --reload` to start the production API.")


if __name__ == "__main__":
    main()
