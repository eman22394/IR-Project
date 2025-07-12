from flask import Blueprint, request, jsonify
from app.database.models import get_documents, get_queries_from_qrels, get_qrels
from app.evaluation.metrics import mean_average_precision, mean_reciprocal_rank, precision_at_k, recall_at_k
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np
import os
import json

bp = Blueprint("tfidf_eval", __name__, url_prefix="/tfidf_eval")

@bp.route("/offline", methods=["POST"])
def tfidf_offline_eval():
    try:
        data = request.json
        dataset_id = data.get("dataset_id")

        if not dataset_id:
            return jsonify({"error": "Missing dataset_id"}), 400

        print(f"🚀 Starting TF-IDF evaluation for dataset {dataset_id}")

        # جلب البيانات
        docs = get_documents(dataset_id)
        queries = get_queries_from_qrels(dataset_id)
        qrels_raw = get_qrels(dataset_id)

        doc_ids = [str(doc[0]) for doc in docs]
        query_ids = [str(q[0]) for q in queries]

        qrels = {}
        for qid, doc_id, rel in qrels_raw:
            if rel > 0:
                qrels.setdefault(str(qid), {})[str(doc_id)] = rel

        tfidf_docs_path = f"data/tfidf/documents_{dataset_id}/tfidf_matrix.pkl"
        tfidf_queries_path = f"data/tfidf/queries{dataset_id}/tfidf_matrix.pkl"

        if not os.path.exists(tfidf_docs_path) or not os.path.exists(tfidf_queries_path):
            return jsonify({"error": "TF-IDF files not found"}), 404

        docs_tfidf = joblib.load(tfidf_docs_path).astype(np.float32)
        queries_tfidf = joblib.load(tfidf_queries_path).astype(np.float32)

        results = []
        predictions = {}
        scores = {}

        for i in range(queries_tfidf.shape[0]):
            query_id = query_ids[i]
            query_vector = queries_tfidf[i]
            sim_row = cosine_similarity(docs_tfidf, query_vector.reshape(1, -1)).flatten()
            top_indices = sim_row.argsort()[::-1][:10]

            top_matches = [
                {
                    "doc_index": int(idx),
                    "doc_id": doc_ids[idx],
                    "score": float(sim_row[idx])
                }
                for idx in top_indices
            ]

            predictions[query_id] = [m["doc_id"] for m in top_matches]
            scores[query_id] = [m["score"] for m in top_matches]

            results.append({
                "query_index": i,
                "query_id": query_id,
                "top_matches": top_matches
            })

        for qid in qrels:
            if qid not in predictions:
                print(f"⚠️ الاستعلام {qid} غير موجود في التوقعات!")

            if not qrels[qid]:
                print(f"⚠️ الاستعلام {qid} لا يحتوي على مستندات ذات صلة!")

        print("📊 Calculating evaluation metrics...")
        metrics = {
            "MAP": round(mean_average_precision(qrels, predictions, scores), 4),
            "MRR": round(mean_reciprocal_rank(qrels, predictions), 4),
            "P@10": round(precision_at_k(qrels, predictions, 10), 4),
            "R@100": round(recall_at_k(qrels, predictions, 100), 4)
        }

        print("✅ Evaluation complete. Saving results...")

        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)

        result_txt_path = os.path.join(results_dir, f"tfidf_results_{dataset_id}.txt")
        with open(result_txt_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"\n🔍 Query {r['query_index']} (query_id={r['query_id']}) top matches:\n")
                for match in r["top_matches"]:
                    f.write(f"   → Doc {match['doc_index']} (doc_id={match['doc_id']}): {match['score']:.3f}\n")

        metrics_path = os.path.join(results_dir, f"tfidf_metrics_{dataset_id}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        return jsonify({
            "message": "TF-IDF evaluation complete",
            "metrics": metrics,
            "results_file": result_txt_path,
            "metrics_file": metrics_path
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500
