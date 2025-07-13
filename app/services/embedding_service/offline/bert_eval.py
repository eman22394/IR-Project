# file: app/routes/bert_eval.py
from flask import Blueprint, request, jsonify
import os, json, joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.database.models import (
    get_documents,
    get_queries_from_qrels,
    get_qrels,
)
from app.evaluation.metrics import (
    mean_average_precision,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)

bp = Blueprint("bert_eval", __name__, url_prefix="/bert_eval")

@bp.route("/offline", methods=["POST"])
def bert_offline_eval():
    try:
        data        = request.json
        dataset_id  = data.get("dataset_id")
        if not dataset_id:
            return jsonify({"error": "Missing dataset_id"}), 400

        print("📥 Loading documents, queries, qrels …")
        docs        = get_documents(dataset_id)
        queries     = get_queries_from_qrels(dataset_id)
        qrels_raw   = get_qrels(dataset_id)

        print(f"   • docs      : {len(docs):,}")
        print(f"   • queries(*) : {len(queries):,}  (*only those appearing in qrels)")

        doc_ids         = [str(d[0]) for d in docs]
        query_ids_all   = [str(q[0]) for q in queries]

        # qrels ➜ dict[query_id][doc_id] = rel
        qrels = {}
        for qid, doc_id, rel in qrels_raw:
            if rel > 0:
                qrels.setdefault(str(qid), {})[str(doc_id)] = rel
        print(f"   • qrels pairs: {len(qrels_raw):,}")

       
        vec_dir   = f"data/bert"
        doc_vecs_path   = f"{vec_dir}/documents_{dataset_id}/doc_vectors.pkl"
        query_vecs_path = f"{vec_dir}/queries_{dataset_id}/doc_vectors.pkl"

        if not (os.path.exists(doc_vecs_path) and os.path.exists(query_vecs_path)):
            return jsonify({"error": "BERT vectors not found"}), 404

        print("💾 Loading vector files …", end="", flush=True)
        doc_vectors   = joblib.load(doc_vecs_path)
        query_vectors = joblib.load(query_vecs_path)
        print(" done.")

        # ------------------------------------------------------------------ #
        # 3) فلترة الاستعلامات الصالحة
        # ------------------------------------------------------------------ #
        valid_queries = [
            (qid, query_vectors[qid])
            for qid in query_ids_all
            if qid in query_vectors and query_vectors[qid] is not None
        ]
        print(f"✅ Valid queries with vectors : {len(valid_queries):,}")

        if not valid_queries:
            return jsonify({"error": "No valid queries with vectors found"}), 400

        # ------------------------------------------------------------------ #
        # 4) إعداد مصفوفة الوثائق
        # ------------------------------------------------------------------ #
        doc_matrix = np.array([doc_vectors[d] for d in doc_ids], dtype=np.float32)
        print(f"📐 Document matrix shape          : {doc_matrix.shape}")

        # ------------------------------------------------------------------ #
        # 5) الحساب Query‑by‑Query  + تطبـيع تقدُّم كل 500 استعلام
        # ------------------------------------------------------------------ #
        results, predictions, scores = [], {}, {}
        print("🔄 Similarity & ranking …")
        for i, (qid, qvec) in enumerate(valid_queries, 1):
            sim_row    = cosine_similarity([qvec], doc_matrix).flatten()
            top_idx    = sim_row.argsort()[::-1][:10]

            top_matches = [
                {"doc_index": int(idx),
                 "doc_id"  : doc_ids[idx],
                 "score"   : float(sim_row[idx])}
                for idx in top_idx
            ]
            predictions[qid] = [m["doc_id"] for m in top_matches]
            scores[qid]      = [m["score"]   for m in top_matches]

            results.append({"query_index": i-1, "query_id": qid, "top_matches": top_matches})

            # طباعة تقدّم كل 500 استعلام
            if i % 500 == 0 or i == len(valid_queries):
                print(f"   • processed {i:,}/{len(valid_queries):,} queries")

        # مثال توضيحي لاستعلام محدّد
        sample_q = "10024"
        if sample_q in predictions:
            print(f"📌 qid {sample_q} | rel docs: {list(qrels.get(sample_q, {})[:5])} …")
            print(f"                     preds : {predictions[sample_q][:5]} …")

        # ------------------------------------------------------------------ #
        # 6) المقاييس
        # ------------------------------------------------------------------ #
        print("📊 Computing metrics …")
        metrics = {
            "MAP"  : round(mean_average_precision(qrels, predictions, scores), 4),
            "MRR"  : round(mean_reciprocal_rank(qrels, predictions), 4),
            "P@10" : round(precision_at_k(qrels, predictions, 10), 4),
            "R@100": round(recall_at_k(qrels, predictions, 100), 4),
        }
        print("✅ Metrics computed :", metrics)

        # ------------------------------------------------------------------ #
        # 7) حفظ النتائج
        # ------------------------------------------------------------------ #
        out_dir = "evaluation_results"
        os.makedirs(out_dir, exist_ok=True)

        res_path = f"{out_dir}/mbert_results_{dataset_id}.txt"
        with open(res_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"\n🔍 Query {r['query_index']} (qid={r['query_id']}) top matches:\n")
                for m in r["top_matches"]:
                    f.write(f"   → Doc{m['doc_index']:>7} | id={m['doc_id']} | score={m['score']:.3f}\n")

        met_path = f"{out_dir}/mbert_metrics_{dataset_id}.json"
        with open(met_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"💾 Files saved → {res_path} & {met_path}\n")

        return jsonify({
            "message"     : "mBERT evaluation complete",
            "metrics"     : metrics,
            "results_file": res_path,
            "metrics_file": met_path
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
