from flask import Blueprint, request, jsonify
from app.database.models import get_documents, get_queries, get_qrels , get_queries_from_qrels
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

        doc_ids = [doc[0] for doc in docs]
        query_ids = [q[0] for q in queries]

        # qrels بشكل dict من dict { query_id: {doc_id: relevance, ...}, ... }# بعد جلب qrels
        qrels = {}
        for qid, doc_id, rel in qrels_raw:
            if rel > 0:
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = rel

        # لا تتحقق من 'predictions' هنا لأن المتغير غير معرف بعد!

        # تحميل ملفات TF-IDF
        tfidf_docs_path = f"data/tfidf/documents_{dataset_id}/tfidf_matrix.pkl"
        tfidf_queries_path = f"data/tfidf/queries{dataset_id}/tfidf_matrix.pkl"

        if not os.path.exists(tfidf_docs_path) or not os.path.exists(tfidf_queries_path):
            return jsonify({"error": "TF-IDF files not found"}), 404

        docs_tfidf = joblib.load(tfidf_docs_path).astype(np.float32)
        queries_tfidf = joblib.load(tfidf_queries_path).astype(np.float32)

        def match_all_queries():
            results = []
            for i in range(queries_tfidf.shape[0]):
                query_vector = queries_tfidf[i]
                similarities = cosine_similarity(docs_tfidf, query_vector).flatten()
                top_indices = similarities.argsort()[::-1][:10]
                result = {
                    "query_index": i,
                    "query_id": query_ids[i],
                    "top_matches": [
                        {"doc_index": int(idx), "doc_id": doc_ids[idx], "score": float(similarities[idx])}
                        for idx in top_indices
                    ]
                }
                results.append(result)
            return results

        results = match_all_queries()

        predictions = {
            r['query_id']: [m['doc_id'] for m in r['top_matches']]
            for r in results
        }

        # تحقق الآن من وجود التوقعات لجميع الاستعلامات في qrels
        for qid in qrels:
            if qid not in predictions:
                print(f"الاستعلام {qid} غير موجود في التوقعات!")

        # تحقق من أن جميع الاستعلامات في qrels لها مستندات ذات صلة
        for qid in qrels:
            if not qrels[qid]:
                print(f"الاستعلام {qid} لا يحتوي على مستندات ذات صلة!")

        metrics = {
            "MAP": round(mean_average_precision(qrels, predictions), 4),
            "MRR": round(mean_reciprocal_rank(qrels, predictions), 4),
            "P@10": round(precision_at_k(qrels, predictions, 10), 4),
            "R@100": round(recall_at_k(qrels, predictions, 100), 4)
        }


        print("✅ Evaluation complete. Saving results...")

        # حفظ النتائج النصية
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        result_txt_path = os.path.join(results_dir, f"tfidf_results_{dataset_id}.txt")
        with open(result_txt_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"\n🔍 Query {r['query_index']} (query_id={r['query_id']}) top matches:\n")
                for match in r["top_matches"]:
                    f.write(f"   → Doc {match['doc_index']} (doc_id={match['doc_id']}): {match['score']:.3f}\n")

        # حفظ المقاييس كـ JSON
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
