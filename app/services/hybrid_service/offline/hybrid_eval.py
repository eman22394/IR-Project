from flask import Blueprint, request, jsonify
import os, json, joblib, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.database.models import get_documents, get_queries_from_qrels, get_qrels
from app.evaluation.metrics import mean_average_precision, mean_reciprocal_rank, precision_at_k, recall_at_k

bp = Blueprint("hybrid_eval", __name__, url_prefix="/hybrid_eval")

ALPHA      = 0.4     # وزن TF‑IDF مقابل BERT
K_RECALL   = 1000    # عدد وثائق TF‑IDF الأوّلية (للتسريع)

@bp.route("/offline", methods=["POST"])
def fusion_offline_eval():
    data = request.json
    dataset_id = data.get("dataset_id")
    if not dataset_id:
        return jsonify({"error": "Missing dataset_id"}), 400

    # 1) تحميل البيانات
    docs      = get_documents(dataset_id)
    queries   = get_queries_from_qrels(dataset_id)
    qrels_raw = get_qrels(dataset_id)

    doc_ids   = [str(d[0]) for d in docs]
    query_ids = [str(q[0]) for q in queries]

    qrels = {}
    for qid, doc_id, rel in qrels_raw:
        if rel > 0:
            qrels.setdefault(str(qid), {})[str(doc_id)] = rel

    # 2) تحميل المتجهات
    tfidf_docs  = joblib.load(f"data/tfidf/documents_{dataset_id}/tfidf_matrix.pkl").astype(np.float32)
    tfidf_qvecs = joblib.load(f"data/tfidf/queries{dataset_id}/tfidf_matrix.pkl").astype(np.float32)
    bert_docs   = joblib.load(f"data/mbert/documents_{dataset_id}/doc_vectors.pkl")
    bert_qvecs  = joblib.load(f"data/mbert/queries_{dataset_id}/doc_vectors.pkl")

    # مصفوفة الوثائق لـ BERT
    bert_matrix = np.array([bert_docs[did] for did in doc_ids], dtype=np.float32)

    predictions, scores = {}, {}
    for i, qid in enumerate(query_ids):
        # --- 2‑A. تشابه TF‑IDF ---
        tf_row = cosine_similarity(tfidf_docs, tfidf_qvecs[i].reshape(1,-1)).flatten()
        top_tf_idx = tf_row.argsort()[::-1][:K_RECALL]

        # --- 2‑B. تشابه BERT لنفس المرشَّحين ---
        qvec_bert  = bert_qvecs[qid]
        br_row     = cosine_similarity(bert_matrix[top_tf_idx], qvec_bert.reshape(1,-1)).flatten()

        # --- 3. الدمج (Weighted Sum) ---
        fused      = ALPHA * tf_row[top_tf_idx] + (1-ALPHA) * br_row

        order      = fused.argsort()[::-1][:10]
        preds      = [doc_ids[top_tf_idx[j]] for j in order]
        scs        = [float(fused[j]) for j in order]

        predictions[qid] = preds
        scores[qid]      = scs

    # 4) المقاييس
    metrics = {
        "MAP"  : round(mean_average_precision(qrels, predictions, scores), 4),
        "MRR"  : round(mean_reciprocal_rank(qrels, predictions), 4),
        "P@10" : round(precision_at_k(qrels, predictions, 10), 4),
        "R@100": round(recall_at_k(qrels, predictions, 100), 4)
    }

    return jsonify({"message":"Fusion evaluation complete","metrics":metrics})

# from flask import Blueprint, request, jsonify
# from app.database.models import get_documents, get_queries_from_qrels, get_qrels
# from app.evaluation.metrics import mean_average_precision, mean_reciprocal_rank, precision_at_k, recall_at_k
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# import numpy as np
# import os
# import json

# bp = Blueprint("hybrid_eval", __name__, url_prefix="/hybrid_eval")

# @bp.route("/offline", methods=["POST"])
# def hybrid_offline_eval():
#     try:
#         data = request.json
#         dataset_id = data.get("dataset_id")

#         if not dataset_id:
#             return jsonify({"error": "Missing dataset_id"}), 400

#         print(f"🚀 Starting Hybrid (BERT+TF-IDF) evaluation for dataset {dataset_id}")

#         # جلب البيانات من قاعدة البيانات
#         docs = get_documents(dataset_id)
#         queries = get_queries_from_qrels(dataset_id)
#         qrels_raw = get_qrels(dataset_id)

#         doc_ids = [str(doc[0]) for doc in docs]
#         query_ids = [str(q[0]) for q in queries]

#         # بناء qrels dict
#         qrels = {}
#         for qid, doc_id, rel in qrels_raw:
#             if rel > 0:
#                 qrels.setdefault(str(qid), {})[str(doc_id)] = rel

#         # تحميل الملفات
#         docs_path = f"data/hybrid_bert&tfidf/documents_{dataset_id}/hybrid_vectors.pkl"
#         queries_path = f"data/hybrid_bert&tfidf/queries_{dataset_id}/hybrid_vectors.pkl"

#         if not os.path.exists(docs_path) or not os.path.exists(queries_path):
#             return jsonify({"error": "Hybrid vectors not found"}), 404

#         docs_data = joblib.load(docs_path)
#         queries_data = joblib.load(queries_path)

#         docs_matrix = docs_data["hybrid"]
#         queries_matrix = queries_data["hybrid"]

#         results = []
#         predictions = {}
#         scores = {}

#         for i in range(queries_matrix.shape[0]):
#             query_id = query_ids[i]
#             query_vector = queries_matrix[i]

#             sim_row = cosine_similarity(docs_matrix, query_vector.reshape(1, -1)).flatten()
#             top_indices = sim_row.argsort()[::-1][:10]

#             top_matches = [
#                 {
#                     "doc_index": int(idx),
#                     "doc_id": doc_ids[idx],
#                     "score": float(sim_row[idx])
#                 }
#                 for idx in top_indices
#             ]

#             predictions[query_id] = [m["doc_id"] for m in top_matches]
#             scores[query_id] = [m["score"] for m in top_matches]

#             results.append({
#                 "query_index": i,
#                 "query_id": query_id,
#                 "top_matches": top_matches
#             })

#         # التحقق من qrels
#         for qid in qrels:
#             if qid not in predictions:
#                 print(f"⚠️ Query {qid} missing in predictions!")
#             if not qrels[qid]:
#                 print(f"⚠️ Query {qid} has no relevant documents!")

#         print("📊 Calculating evaluation metrics...")
#         metrics = {
#             "MAP": round(mean_average_precision(qrels, predictions, scores), 4),
#             "MRR": round(mean_reciprocal_rank(qrels, predictions), 4),
#             "P@10": round(precision_at_k(qrels, predictions, 10), 4),
#             "R@100": round(recall_at_k(qrels, predictions, 100), 4)
#         }

#         print("✅ Evaluation complete. Saving results...")

#         results_dir = "evaluation_results"
#         os.makedirs(results_dir, exist_ok=True)

#         result_txt_path = os.path.join(results_dir, f"hybrid_results_{dataset_id}.txt")
#         with open(result_txt_path, "w", encoding="utf-8") as f:
#             for r in results:
#                 f.write(f"\n🔍 Query {r['query_index']} (query_id={r['query_id']}) top matches:\n")
#                 for match in r["top_matches"]:
#                     f.write(f"   → Doc {match['doc_index']} (doc_id={match['doc_id']}): {match['score']:.3f}\n")

#         metrics_path = os.path.join(results_dir, f"hybrid_metrics_{dataset_id}.json")
#         with open(metrics_path, "w", encoding="utf-8") as f:
#             json.dump(metrics, f, indent=2)

#         return jsonify({
#             "message": "Hybrid evaluation complete",
#             "metrics": metrics,
#             "results_file": result_txt_path,
#             "metrics_file": metrics_path
#         })

#     except Exception as e:
#         print("❌ Error:", e)
#         return jsonify({"error": str(e)}), 500
