import os
import json
from flask import Blueprint, request, jsonify
from app.database.models import get_documents, get_queries_from_qrels, get_qrels
from app.evaluation.metrics import (
    mean_average_precision,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k
)
from app.services.bm25_service.offline.build_bm25 import load_inverted_index, compute_bm25

bm25_eval_bp = Blueprint("bm25_eval", __name__)

@bm25_eval_bp.route("/bm25_eval/", methods=["POST"])
def bm25_eval():
    try:
        data = request.get_json()
        dataset_id = data.get("dataset_id")
        k1 = data.get("k1", 1.5)
        b = data.get("b", 0.75)

        if dataset_id is None:
            return jsonify({"error": "يرجى تمرير dataset_id"}), 400

        metrics, results_file, metrics_file = evaluate_bm25(
            dataset_id=int(dataset_id),
            k1=float(k1),
            b=float(b)
        )

        return jsonify({
            "message": "✅ تم تنفيذ تقييم BM25 بنجاح على جميع النتائج",
            "metrics": metrics,
            "results_file": results_file,
            "metrics_file": metrics_file
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def evaluate_bm25(dataset_id, k1=1.5, b=0.75):
    docs = get_documents(dataset_id)
    queries = get_queries_from_qrels(dataset_id)
    qrels_db = get_qrels(dataset_id)

    qrels = {}
    for qid, doc_id, rel in qrels_db:
        if rel > 0:
            qrels.setdefault(str(qid), {})[str(doc_id)] = rel

    index_data = load_inverted_index(dataset_id)
    if not index_data:
        raise ValueError("⚠️ الفهرس غير موجود لمجموعة البيانات المحددة")

    predictions = {}
    scores = {}
    results_out = []

    for idx, (qid, query_text) in enumerate(queries):
        qid = str(qid)
        hits = compute_bm25(query_text, index_data, k1=k1, b=b)

        predictions[qid] = [h["doc_id"] for h in hits]
        scores[qid] = [h["score"] for h in hits]

        results_out.append({
            "query_index": idx,
            "query_id": qid,
            "top_matches": hits
        })

    metrics = {
        "MAP": round(mean_average_precision(qrels, predictions, scores), 4),
        "MRR": round(mean_reciprocal_rank(qrels, predictions), 4),
        "P@10": round(precision_at_k(qrels, predictions, 10), 4),
        "R@100": round(recall_at_k(qrels, predictions, 100), 4)
    }

    out_dir = "evaluation_results"
    os.makedirs(out_dir, exist_ok=True)

    result_file = os.path.join(out_dir, f"bm25_full_results_{dataset_id}.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        for r in results_out:
            f.write(f"\n🔍 Query {r['query_index']} (id={r['query_id']})\n")
            for m in r["top_matches"]:
                f.write(f"   → {m['doc_id']}: {m['score']:.4f}\n")

    metrics_file = os.path.join(out_dir, f"bm25_full_metrics_{dataset_id}.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics, result_file, metrics_file
