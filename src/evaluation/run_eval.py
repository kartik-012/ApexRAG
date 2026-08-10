"""
Evaluation Harness Runner — ties together ingestion, retrieval, metrics, uncertainty, and attribution.
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from src.config import GROUND_TRUTH_DIR, EVAL_RUNS_DIR
from src.ingestion.loader import load_legacy_docs, load_current_docs, DocFile
from src.ingestion.chunkers import simple_chunk, semantic_chunk

LEGACY_REPO = os.getenv("LEGACY_REPO", "data/raw_docs/react-legacy")
CURRENT_REPO = os.getenv("CURRENT_REPO", "data/raw_docs/react-dev-current")
from src.ingestion.embedder import embed_texts
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker
from src.retrieval.router import StrategyRouter
from src.evaluation.metrics import compute_recall_at_k, compute_mrr
from src.evaluation.uncertainty import embedding_spread_score, strategy_agreement_score, confidence_label
from src.evaluation.failure_attribution import attribute_failure, FailureType


def prepare_benchmark_data(docs: List[DocFile]):
    """Chunks docs and builds VectorStore + BM25 index in memory for evaluation."""
    vector_store = VectorStore(persist_path="./data/chroma_eval_db", collection_name="eval_docs")
    bm25_store = BM25Store()

    all_ids, all_texts, all_metas = [], [], []
    chunk_to_doc = {}

    for doc in docs:
        chunks = simple_chunk(doc.text, chunk_size=200, overlap=30)
        for idx, text in enumerate(chunks):
            cid = f"{doc.doc_id}_chunk{idx}"
            all_ids.append(cid)
            all_texts.append(text)
            all_metas.append({"source": doc.source, "doc_id": doc.doc_id, "title": doc.title})
            chunk_to_doc[cid] = doc.doc_id

    embeddings = embed_texts(all_texts)
    vector_store.add_chunks(all_ids, all_texts, embeddings, all_metas)
    bm25_store.build(all_ids, all_texts)

    return vector_store, bm25_store, chunk_to_doc, all_ids, all_texts


def evaluate_harness(qa_pairs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Executes full evaluation harness across all retrieval strategies.
    """
    if qa_pairs is None:
        qa_file = GROUND_TRUTH_DIR / "qa_pairs.json"
        if qa_file.exists():
            with open(qa_file, "r", encoding="utf-8") as f:
                qa_pairs = json.load(f)
        else:
            qa_pairs = [
                {
                    "id": "qa_1",
                    "question": "How do you initialize and update state in class components?",
                    "answer": "State is initialized in constructor and updated using setState.",
                    "gold_chunk_ids": ["legacy_state-and-lifecycle_chunk0"],
                    "gold_doc_id": "legacy_state-and-lifecycle",
                },
                {
                    "id": "qa_2",
                    "question": "How does the useRef hook work in modern React?",
                    "answer": "useRef lets you reference values without re-rendering.",
                    "gold_chunk_ids": ["current_useRef_chunk0"],
                    "gold_doc_id": "current_useRef",
                },
                {
                    "id": "qa_3",
                    "question": "How to pass data deeply using Context API?",
                    "answer": "Context API passes data deeply without prop drilling using useContext or Consumer.",
                    "gold_chunk_ids": ["current_passing-data-deeply-with-context_chunk0"],
                    "gold_doc_id": "current_passing-data-deeply-with-context",
                },
            ]

    docs = load_legacy_docs(LEGACY_REPO) + load_current_docs(CURRENT_REPO)
    vector_store, bm25_store, chunk_to_doc, all_ids, all_texts = prepare_benchmark_data(docs)
    reranker = Reranker()
    router = StrategyRouter()

    strategies = ["simple", "semantic", "hybrid", "rerank", "router"]
    results_by_strategy: Dict[str, List[Dict[str, Any]]] = {s: [] for s in strategies}
    failure_counts: Dict[str, Dict[str, int]] = {s: {ft.name: 0 for ft in FailureType} for s in strategies}

    for item in qa_pairs:
        qid = item["id"]
        question = item["question"]
        gold_doc = item.get("gold_doc_id", "")
        gold_chunks = set(item.get("gold_chunk_ids", []))

        # Embed query
        q_emb = embed_texts([question])[0]

        # 1. Vector Search (Semantic)
        vec_retrieved = vector_store.query(q_emb, top_k=10)
        vec_cids = [r.chunk_id for r in vec_retrieved]

        # 2. BM25 Search (Simple Keyword)
        bm25_retrieved = bm25_store.query(question, top_k=10)
        bm25_cids = [cid for cid, _ in bm25_retrieved]

        # 3. Hybrid RRF Search
        fused = reciprocal_rank_fusion(bm25_cids, vec_cids, k=60, top_k=10)
        hybrid_cids = [f.chunk_id for f in fused]

        # 4. Re-ranker Search
        candidates = [(cid, all_texts[all_ids.index(cid)]) for cid in hybrid_cids[:10] if cid in all_ids]
        reranked = reranker.rerank(question, candidates, top_k=10)
        rerank_cids = [cid for cid, _, _ in reranked]

        # 5. Router Search
        chosen_strat = router.predict_strategy(question)
        strat_map = {"simple": bm25_cids, "semantic": vec_cids, "hybrid": hybrid_cids, "rerank": rerank_cids}
        router_cids = strat_map.get(chosen_strat, hybrid_cids)

        strat_cids_map = {
            "simple": bm25_cids,
            "semantic": vec_cids,
            "hybrid": hybrid_cids,
            "rerank": rerank_cids,
            "router": router_cids,
        }

        # Calculate agreement & spread
        strat_results_for_agree = {s: cids for s, cids in strat_cids_map.items() if s != "router"}
        top_k_vecs = vector_store.get_all_embeddings_for_ids(vec_cids[:4])
        spread = embedding_spread_score(top_k_vecs)
        agreement = strategy_agreement_score(strat_results_for_agree)
        conf_label = confidence_label(spread, agreement)

        for strat, cids in strat_cids_map.items():
            top4_cids = cids[:4]
            # Match either exact chunk ID or matching document ID
            top4_docs = {chunk_to_doc.get(c, "") for c in top4_cids}
            doc_hit = gold_doc in top4_docs if gold_doc else False
            chunk_hit = bool(set(top4_cids) & gold_chunks) if gold_chunks else False
            is_correct = doc_hit or chunk_hit

            recall = compute_recall_at_k(top4_cids, gold_chunks if gold_chunks else {gold_doc}, k=4)
            mrr = compute_mrr(cids, gold_chunks if gold_chunks else {gold_doc})

            failure = attribute_failure(qid, top4_cids, gold_chunks if gold_chunks else {gold_doc}, cids, 0.9, is_correct)
            if failure:
                failure_counts[strat][failure.name] += 1

            results_by_strategy[strat].append({
                "qid": qid,
                "recall": recall,
                "mrr": mrr,
                "correct": is_correct,
                "spread": spread,
                "agreement": agreement,
                "confidence": conf_label,
            })

    # Summary metrics per strategy
    summary = {}
    for strat, res_list in results_by_strategy.items():
        n = len(res_list)
        avg_recall = sum(r["recall"] for r in res_list) / float(n) if n else 0.0
        avg_mrr = sum(r["mrr"] for r in res_list) / float(n) if n else 0.0
        acc = sum(1 for r in res_list if r["correct"]) / float(n) if n else 0.0

        summary[strat] = {
            "avg_recall_at_4": round(avg_recall, 4),
            "avg_mrr": round(avg_mrr, 4),
            "accuracy": round(acc, 4),
            "failure_attributions": failure_counts[strat],
        }

    eval_result = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(qa_pairs),
        "summary": summary,
    }

    # Save timestamped JSON run
    output_file = EVAL_RUNS_DIR / f"eval_run_{int(time.time())}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2)

    return eval_result
