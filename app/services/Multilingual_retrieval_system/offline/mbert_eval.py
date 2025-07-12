from flask import Blueprint, request, jsonify
import os
import joblib
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from app.database.models import get_documents, get_queries_from_qrels, get_qrels
from app.evaluation.metrics import mean_average_precision, mean_reciprocal_rank, precision_at_k, recall_at_k

bp = Blueprint("mbert_eval", __name__, url_prefix="/mbert_eval")

@bp.route("/offline", methods=["POST"])
def bert_offline_eval():
    try:
        data = request.json
        dataset_id = data.get("dataset_id")

        if not dataset_id:
            return jsonify({"error": "Missing dataset_id"}), 400

        print(f"🚀 Starting BERT evaluation for dataset {dataset_id}")

        docs = get_documents(dataset_id)
        queries = get_queries_from_qrels(dataset_id)
        qrels_raw = get_qrels(dataset_id)

        doc_ids = [str(doc[0]) for doc in docs]
        query_ids_all = [str(q[0]) for q in queries]

        qrels = {}
        for qid, doc_id, rel in qrels_raw:
            if rel > 0:
                qrels.setdefault(str(qid), {})[str(doc_id)] = rel

        doc_vecs_path = f"data/mbert/documents_{dataset_id}/doc_vectors.pkl"
        query_vecs_path = f"data/mbert/queries_{dataset_id}/doc_vectors.pkl"

        if not os.path.exists(doc_vecs_path) or not os.path.exists(query_vecs_path):
            return jsonify({"error": "BERT vectors not found"}), 404

        doc_vectors = joblib.load(doc_vecs_path)
        query_vectors = joblib.load(query_vecs_path)

        valid_queries = [
            (qid, query_vectors[qid])
            for qid in query_ids_all
            if qid in query_vectors and query_vectors[qid] is not None
        ]
        print(f"✅ Valid queries: {len(valid_queries)} / {len(queries)}")

        if not valid_queries:
            return jsonify({"error": "No valid queries with vectors found"}), 400

        # مصفوفة وثائق كاملة
        doc_matrix = np.array([doc_vectors[doc_id] for doc_id in doc_ids], dtype=np.float32)
        print(f"📐 Document matrix shape          : {doc_matrix.shape}")

        # 🔍 2) التشابه والترتيب Query‑by‑Query  ----------
        results = []
        predictions = {}
        scores = {}

        total_q = len(valid_queries)
        for idx, (query_id, query_vec) in enumerate(valid_queries, 1):

            # 1) احسب التشابه مع كل الوثائق
            sim_row = cosine_similarity(
                query_vec.reshape(1, -1),
                doc_matrix
            ).flatten()

            # 2) رتّب نزولياً وخُذ أفضل 10
            top_indices = np.argsort(sim_row)[::-1][:10]

            top_matches = [
                {
                    "doc_index": int(doc_idx),
                    "doc_id": doc_ids[doc_idx],
                    "score": float(sim_row[doc_idx])
                }
                for doc_idx in top_indices
            ]

            # 3) خزن للقيم المتريّة
            predictions[str(query_id)] = [m["doc_id"] for m in top_matches]
            scores[str(query_id)] = [m["score"] for m in top_matches]

            # 4) خزن للملف التفصيلي
            results.append({
                "query_index": idx,
                "query_id": query_id,
                "top_matches": top_matches
            })

            # 5) طباعة تقدُّم كل 500 استعلام
            if idx % 500 == 0 or idx == total_q:
                print(f"   • processed {idx}/{total_q} queries")

            # 6) مثال تتبّع لِـ query محدَّد إن أحببت
            if query_id == "10024":
                print("📌 Relevant docs for 10024:", qrels.get("10024"))
                print("📌 Predicted docs for 10024:", predictions[str(query_id)])

        # حساب المقاييس
        print("📊 Calculating evaluation metrics...")
        metrics = {
            "MAP": round(mean_average_precision(qrels, predictions, scores), 4),
            "MRR": round(mean_reciprocal_rank(qrels, predictions), 4),
            "P@10": round(precision_at_k(qrels, predictions, 10), 4),
            "R@100": round(recall_at_k(qrels, predictions, 100), 4)
        }

        print("✅ Evaluation complete. Saving results...")

        # حفظ النتائج
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)

        results_path = os.path.join(results_dir, f"mbert_results_{dataset_id}.txt")
        with open(results_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"\n🔍 Query {r['query_index']} (query_id={r['query_id']}) top matches:\n")
                for match in r["top_matches"]:
                    f.write(f"   → Doc {match['doc_index']} (doc_id={match['doc_id']}): {match['score']:.3f}\n")

        metrics_path = os.path.join(results_dir, f"mbert_metrics_{dataset_id}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        return jsonify({
            "message": "mBERT evaluation complete",
            "metrics": metrics,
            "results_file": results_path,
            "metrics_file": metrics_path
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500
