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
from src.ingestion.embedder import embed_texts, ensure_embedder_fitted
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.query_intelligence import analyze_and_expand_query
from src.retrieval.reranker import Reranker
from src.retrieval.router import StrategyRouter
from src.evaluation.metrics import compute_recall_at_k, compute_mrr, compute_ndcg_at_k
from src.evaluation.uncertainty import embedding_spread_score, strategy_agreement_score, confidence_label
from src.evaluation.failure_attribution import attribute_failure, FailureType
from src.evaluation.cost_latency_tracker import select_pareto_optimal


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

    if vector_store.count() != len(all_ids):
        embeddings = embed_texts(all_texts)
        vector_store.add_chunks(all_ids, all_texts, embeddings, all_metas)
    else:
        ensure_embedder_fitted(all_texts)
    bm25_store.build(all_ids, all_texts, all_metas)

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
    id_to_text = dict(zip(all_ids, all_texts))
    reranker = Reranker()
    router = StrategyRouter()

    strategies = ["simple", "semantic", "hybrid", "rerank", "router"]
    results_by_strategy: Dict[str, List[Dict[str, Any]]] = {s: [] for s in strategies}
    failure_counts: Dict[str, Dict[str, int]] = {s: {ft.name: 0 for ft in FailureType} for s in strategies}

    for item in qa_pairs:
        qid = item["id"]
        question = item["question"]
        retrieval_query = analyze_and_expand_query(question, enable_expansion=True).expanded_query
        gold_chunks = set(item.get("gold_chunk_ids", []))
        gold_doc = item.get("gold_doc_id", "")
        if not gold_doc and gold_chunks:
            first_gold = next(iter(gold_chunks))
            if "_chunk" in first_gold:
                gold_doc = first_gold.rsplit("_chunk", 1)[0]

        # Embed query & measure per-stage latencies
        t0 = time.perf_counter()
        q_emb = embed_texts([retrieval_query])[0]
        emb_ms = (time.perf_counter() - t0) * 1000.0

        # 1. Vector Search (Semantic)
        t_vec = time.perf_counter()
        vec_retrieved = vector_store.query(q_emb, top_k=10)
        vec_cids = [r.chunk_id for r in vec_retrieved]
        vec_ms = emb_ms + (time.perf_counter() - t_vec) * 1000.0

        # 2. BM25 Search (Simple Keyword)
        t_bm25 = time.perf_counter()
        bm25_retrieved = bm25_store.query(retrieval_query, top_k=10)
        bm25_cids = [cid for cid, _ in bm25_retrieved]
        bm25_ms = (time.perf_counter() - t_bm25) * 1000.0

        # 3. Hybrid RRF Search
        t_hyb = time.perf_counter()
        fused = reciprocal_rank_fusion(bm25_cids, vec_cids, k=60, top_k=10)
        hybrid_cids = [f.chunk_id for f in fused]
        hybrid_ms = vec_ms + bm25_ms + (time.perf_counter() - t_hyb) * 1000.0

        # 4. Re-ranker Search
        t_rerank = time.perf_counter()
        candidates = [(cid, id_to_text[cid]) for cid in hybrid_cids[:10] if cid in id_to_text]
        reranked = reranker.rerank(question, candidates, top_k=10)
        rerank_cids = [cid for cid, _, _ in reranked]
        rerank_ms = hybrid_ms + (time.perf_counter() - t_rerank) * 1000.0

        # 5. Router Search
        t_router = time.perf_counter()
        chosen_strat = router.predict_strategy(question)
        router_overhead_ms = (time.perf_counter() - t_router) * 1000.0
        strat_map = {"simple": bm25_cids, "semantic": vec_cids, "hybrid": hybrid_cids, "rerank": rerank_cids}
        latency_map = {"simple": bm25_ms, "semantic": vec_ms, "hybrid": hybrid_ms, "rerank": rerank_ms}
        router_cids = strat_map.get(chosen_strat, hybrid_cids)
        router_ms = router_overhead_ms + latency_map.get(chosen_strat, hybrid_ms)

        strat_cids_map = {
            "simple": bm25_cids,
            "semantic": vec_cids,
            "hybrid": hybrid_cids,
            "rerank": rerank_cids,
            "router": router_cids,
        }
        strat_latency_map = {
            "simple": bm25_ms,
            "semantic": vec_ms,
            "hybrid": hybrid_ms,
            "rerank": rerank_ms,
            "router": router_ms,
        }

        # Calculate agreement & spread
        strat_results_for_agree = {s: cids for s, cids in strat_cids_map.items() if s != "router"}
        top_k_vecs = vector_store.get_all_embeddings_for_ids(vec_cids[:4])
        spread = embedding_spread_score(top_k_vecs)
        agreement = strategy_agreement_score(strat_results_for_agree)
        conf_label = confidence_label(spread, agreement)

        for strat, cids in strat_cids_map.items():
            top4_cids = cids[:4]
            top4_docs = {chunk_to_doc.get(c, c.rsplit("_chunk", 1)[0] if "_chunk" in c else "") for c in top4_cids}
            doc_hit = gold_doc in top4_docs if gold_doc else False
            chunk_hit = bool(set(top4_cids) & gold_chunks) if gold_chunks else False
            is_correct = doc_hit or chunk_hit

            # Normalize target set so document-level sibling match is scored on a [0, 1] scale
            effective_gold = set(gold_chunks) if gold_chunks else ({gold_doc} if gold_doc else set())
            if doc_hit and not chunk_hit and gold_doc:
                for c in cids:
                    if chunk_to_doc.get(c, c.rsplit("_chunk", 1)[0] if "_chunk" in c else "") == gold_doc:
                        effective_gold = {c}
                        break

            recall = compute_recall_at_k(top4_cids, effective_gold, k=4)
            mrr = compute_mrr(cids, effective_gold)
            ndcg = compute_ndcg_at_k(top4_cids, effective_gold, k=4)

            failure = attribute_failure(qid, top4_cids, effective_gold, cids, 0.9, is_correct)
            if failure:
                failure_counts[strat][failure.name] += 1

            results_by_strategy[strat].append({
                "qid": qid,
                "question": question,
                "recall": recall,
                "mrr": mrr,
                "ndcg": ndcg,
                "correct": is_correct,
                "spread": round(spread, 4),
                "agreement": round(agreement, 4),
                "confidence": conf_label,
                "latency_ms": round(strat_latency_map[strat], 2),
            })

    # Summary metrics per strategy
    summary = {}
    pareto_candidates = []
    for strat, res_list in results_by_strategy.items():
        n = len(res_list)
        avg_recall = sum(r["recall"] for r in res_list) / float(n) if n else 0.0
        avg_mrr = sum(r["mrr"] for r in res_list) / float(n) if n else 0.0
        avg_ndcg = sum(r["ndcg"] for r in res_list) / float(n) if n else 0.0
        avg_latency = sum(r["latency_ms"] for r in res_list) / float(n) if n else 0.0
        acc = sum(1 for r in res_list if r["correct"]) / float(n) if n else 0.0

        summary[strat] = {
            "avg_recall_at_4": round(avg_recall, 4),
            "avg_mrr": round(avg_mrr, 4),
            "avg_ndcg_at_4": round(avg_ndcg, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "accuracy": round(acc, 4),
            "failure_attributions": failure_counts[strat],
        }
        pareto_candidates.append({
            "strategy": strat,
            "recall_at_k": round(avg_recall, 4),
            "latency_ms": round(avg_latency, 2),
        })

    pareto_optimal = select_pareto_optimal(pareto_candidates, max_latency_ms=300.0)

    # End-to-End Generation Metrics (Ragas-Style)
    from src.evaluation.generation_eval import evaluate_generation_sample
    from src.generation.llm_client import generate_answer

    gen_samples = []
    for item in qa_pairs[:10]:
        q_text = item["question"]
        gold_ans = item.get("answer", "")
        chosen_cids = strat_cids_map.get("router", strat_cids_map.get("hybrid", []))
        c_texts = [id_to_text[cid] for cid in chosen_cids[:4] if cid in id_to_text]
        gen_ans = generate_answer(q_text, c_texts)
        gen_res = evaluate_generation_sample(
            question=q_text,
            answer=gen_ans,
            context=c_texts,
            ground_truth=gold_ans,
        )
        gen_samples.append(gen_res)

    mean_faith = sum(g.faithfulness for g in gen_samples) / float(len(gen_samples)) if gen_samples else 1.0
    mean_rel = sum(g.answer_relevance for g in gen_samples) / float(len(gen_samples)) if gen_samples else 1.0
    recall_vals = [g.context_recall for g in gen_samples if g.context_recall is not None]
    mean_rec = (sum(recall_vals) / float(len(recall_vals))) if recall_vals else 1.0

    eval_result = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(qa_pairs),
        "summary": summary,
        "pareto_optimal": pareto_optimal,
        "generation_metrics": {
            "mean_faithfulness": round(mean_faith, 4),
            "mean_answer_relevance": round(mean_rel, 4),
            "mean_context_recall": round(mean_rec, 4),
            "samples_evaluated": len(gen_samples),
            "diagnoses": {
                diag: sum(1 for g in gen_samples if g.diagnosis == diag)
                for diag in ["SUCCESS", "RETRIEVAL_FAILURE", "GENERATION_HALLUCINATION", "COMPOUND_FAILURE"]
            },
        },
        "per_question_results": results_by_strategy,
    }

    # Save timestamped JSON run
    output_file = EVAL_RUNS_DIR / f"eval_run_{int(time.time())}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2)

    return eval_result

