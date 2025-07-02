# file: app/routes/bert_eval.py

from flask import Blueprint, request, jsonify
import os
import joblib
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from app.database.models import get_documents, get_queries_from_qrels, get_qrels
from app.evaluation.metrics import mean_average_precision, mean_reciprocal_rank, precision_at_k, recall_at_k

bp = Blueprint("bert_eval", __name__, url_prefix="/bert_eval")

@bp.route("/offline", methods=["POST"])
def bert_offline_eval():
    try:
        data = request.json
        dataset_id = data.get("dataset_id")

        if not dataset_id:
            return jsonify({"error": "Missing dataset_id"}), 400

        print(f"🚀 Starting BERT evaluation for dataset {dataset_id}")

        # تحميل الوثائق والاستعلامات والـ qrels
        docs = get_documents(dataset_id)
        queries = get_queries_from_qrels(dataset_id)
        qrels_raw = get_qrels(dataset_id)

        doc_ids = [str(doc[0]) for doc in docs]
        query_ids_all = [str(q[0]) for q in queries]

        # تجهيز qrels: {query_id: {doc_id: relevance}} مع تحويل المفاتيح إلى str
        qrels = {}
        for qid, doc_id, rel in qrels_raw:
            if rel > 0:
                qrels.setdefault(str(qid), {})[str(doc_id)] = rel

        # تحميل التمثيلات الشعاعية
        doc_vecs_path = f"data/bert/documents_{dataset_id}/doc_vectors.pkl"
        query_vecs_path = f"data/bert/queries_{dataset_id}/doc_vectors.pkl"

        if not os.path.exists(doc_vecs_path) or not os.path.exists(query_vecs_path):
            return jsonify({"error": "BERT vectors not found"}), 404

        doc_vectors = joblib.load(doc_vecs_path)
        query_vectors = joblib.load(query_vecs_path)

        # تصفية الاستعلامات الصالحة
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

        # حساب التشابه استعلام باستعلام (لتقليل استهلاك الذاكرة)
        results = []
        predictions = {}
        scores = {}

        for i, (query_id, query_vec) in enumerate(valid_queries):
            sim_row = cosine_similarity([query_vec], doc_matrix).flatten()
            top_indices = sim_row.argsort()[::-1][:10]

            top_matches = [
                {
                    "doc_index": int(idx),
                    "doc_id": doc_ids[idx],
                    "score": float(sim_row[idx])
                }
                for idx in top_indices
            ]

            predictions[str(query_id)] = [str(m["doc_id"]) for m in top_matches]
            scores[str(query_id)] = [m["score"] for m in top_matches]

            results.append({
                "query_index": i,
                "query_id": query_id,
                "top_matches": top_matches
            })

            if query_id == "10024":
                print("📌 Relevant docs for query 10024:", qrels.get("10024"))
                print("📌 Predicted docs for query 10024:", predictions.get("10024"))

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
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        results_path = os.path.join(results_dir, f"bert_results_{dataset_id}.txt")
        with open(results_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"\n🔍 Query {r['query_index']} (query_id={r['query_id']}) top matches:\n")
                for match in r["top_matches"]:
                    f.write(f"   → Doc {match['doc_index']} (doc_id={match['doc_id']}): {match['score']:.3f}\n")

        metrics_path = os.path.join(results_dir, f"bert_metrics_{dataset_id}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        return jsonify({
            "message": "BERT evaluation complete",
            "metrics": metrics,
            "results_file": results_path,
            "metrics_file": metrics_path
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

# # file: app/routes/word2vec_eval.py

# from flask import Blueprint, request, jsonify
# import os
# import joblib
# import numpy as np
# import json
# from sklearn.metrics.pairwise import cosine_similarity
# from app.database.models import get_documents, get_queries_from_qrels, get_qrels
# from app.evaluation.metrics import mean_average_precision, mean_reciprocal_rank, precision_at_k, recall_at_k

# bp = Blueprint("word2vec_eval", __name__, url_prefix="/word2vec_eval")


# @bp.route("/offline", methods=["POST"])
# def word2vec_offline_eval():
#     try:
#         data = request.json
#         dataset_id = data.get("dataset_id")

#         if not dataset_id:
#             return jsonify({"error": "Missing dataset_id"}), 400

#         print(f"🚀 Starting Word2Vec evaluation for dataset {dataset_id}")

#         # تحميل البيانات
#         docs = get_documents(dataset_id)
#         queries = get_queries_from_qrels(dataset_id)
#         qrels_raw = get_qrels(dataset_id)

#         doc_ids = [str(doc[0]) for doc in docs]
#         query_ids_all = [str(q[0]) for q in queries]

#         # تجهيز qrels على شكل {query_id: {doc_id: relevance}} (كل القيم نصوص)
#         qrels = {}
#         for qid, doc_id, rel in qrels_raw:
#             if rel > 0:
#                 qrels.setdefault(str(qid), {})[str(doc_id)] = rel

#         # تحميل التمثيلات الشعاعية
#         doc_vecs_path = f"data/word2vec/documents_{dataset_id}/doc_vectors.pkl"
#         query_vecs_path = f"data/word2vec/queries_{dataset_id}/doc_vectors.pkl"

#         if not os.path.exists(doc_vecs_path) or not os.path.exists(query_vecs_path):
#             return jsonify({"error": "Word2Vec vectors not found"}), 404

#         doc_vectors = joblib.load(doc_vecs_path)
#         query_vectors = joblib.load(query_vecs_path)

#         # 🔍 تصفية الاستعلامات الصالحة فقط (التي تملك تمثيل شعاعي)
#         valid_queries = [
#             (qid, query_vectors[qid])
#             for qid in query_ids_all
#             if qid in query_vectors and query_vectors[qid] is not None
#         ]
#         print(f"✅ Valid queries: {len(valid_queries)} / {len(queries)}")
#         if not valid_queries:
#             return jsonify({"error": "No valid queries with vectors found"}), 400

#         query_ids = [qid for qid, _ in valid_queries]
#         query_matrix = np.array([vec for _, vec in valid_queries], dtype=np.float32)
#         doc_matrix = np.array([doc_vectors[doc_id] for doc_id in doc_ids], dtype=np.float32)

#         # ⚡ حساب التشابه دفعة واحدة
#         print("⚡ Computing cosine similarities in batch...")
#         similarity_matrix = cosine_similarity(query_matrix, doc_matrix)
#         for qid, vec in valid_queries[:3]:
#             print(f"Query {qid} → vector sample: {vec[:5]}")

#         results = []
#         predictions = {}

#         for i, query_id in enumerate(query_ids):
#             sim_row = similarity_matrix[i]
#             top_indices = sim_row.argsort()[::-1][:10]

#             top_matches = [
#                 {
#                     "doc_index": int(idx),
#                     "doc_id": doc_ids[idx],
#                     "score": float(sim_row[idx])
#                 }
#                 for idx in top_indices
#             ]

#             predictions[query_id] = [match["doc_id"] for match in top_matches]

#             results.append({
#                 "query_index": i,
#                 "query_id": query_id,
#                 "top_matches": top_matches
#             })

#             # للتأكد من وجود تطابق
#             if query_id == "10024":
#                 print("📌 Relevant docs for query 10024:", qrels.get("10024"))
#                 print("📌 Predicted docs for query 10024:", predictions.get("10024"))

#         # 🧮 حساب المقاييس
#         print("📊 Calculating evaluation metrics...")
#         metrics = {
#             "MAP": round(mean_average_precision(qrels, predictions), 4),
#             "MRR": round(mean_reciprocal_rank(qrels, predictions), 4),
#             "P@10": round(precision_at_k(qrels, predictions, 10), 4),
#             "R@100": round(recall_at_k(qrels, predictions, 100), 4)
#         }

#         print("✅ Evaluation complete. Saving results...")

#         # 💾 حفظ النتائج
#         results_dir = "results"
#         os.makedirs(results_dir, exist_ok=True)

#         results_path = os.path.join(results_dir, f"word2vec_results_{dataset_id}.txt")
#         with open(results_path, "w", encoding="utf-8") as f:
#             for r in results:
#                 f.write(f"\n🔍 Query {r['query_index']} (query_id={r['query_id']}) top matches:\n")
#                 for match in r["top_matches"]:
#                     f.write(f"   → Doc {match['doc_index']} (doc_id={match['doc_id']}): {match['score']:.3f}\n")

#         metrics_path = os.path.join(results_dir, f"word2vec_metrics_{dataset_id}.json")
#         with open(metrics_path, "w", encoding="utf-8") as f:
#             json.dump(metrics, f, indent=2)

#         return jsonify({
#             "message": "Word2Vec evaluation complete",
#             "metrics": metrics,
#             "results_file": results_path,
#             "metrics_file": metrics_path
#         })

#     except Exception as e:
#         print("❌ Error:", e)
#         return jsonify({"error": str(e)}), 500
