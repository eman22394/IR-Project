from flask import Blueprint, request, jsonify
from app.database.models import get_documents, get_qrels, get_queries_from_qrels
from app.evaluation.metrics import mean_average_precision, mean_reciprocal_rank, precision_at_k, recall_at_k
import joblib
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import json

bp = Blueprint('evaluation_offline', __name__, url_prefix='/evaluate')

@bp.route('/offline', methods=['POST'])
def evaluate_offline_models():
    try:
        data = request.json
        dataset_id = data.get("dataset_id")

        if not dataset_id:
            return jsonify({"error": "Missing dataset_id"}), 400

        print(f"🚀 Starting evaluation for dataset_id: {dataset_id}")

        # تحميل الاستعلامات والـ qrels
        print("📥 Loading queries and qrels...")
        queries = get_queries_from_qrels(dataset_id)
        qrels_raw = get_qrels(dataset_id)

        # تجهيز qrels كـ dict
        print("🔧 Preparing qrels dictionary...")
        qrels = {}
        for qid, doc_id, rel in qrels_raw:
            if rel > 0:
                qrels.setdefault(qid, []).append(doc_id)

        results_all = {}
        metrics_all = {}

        # تحميل الوثائق
        print("📄 Loading documents...")
        documents = get_documents(dataset_id)
        doc_ids = [doc[0] for doc in documents]

        # ================ ✅ تقييم TF-IDF =================
        tfidf_docs_path = f"data/tfidf/documents_{dataset_id}/tfidf_matrix.pkl"
        tfidf_queries_path = f"data/tfidf/queries_{dataset_id}/tfidf_matrix.pkl"

        if os.path.exists(tfidf_docs_path) and os.path.exists(tfidf_queries_path):
            print("🔍 Evaluating TF-IDF...")
            docs_tfidf = joblib.load(tfidf_docs_path).astype(np.float32)
            queries_tfidf = joblib.load(tfidf_queries_path).astype(np.float32)
            query_ids = [q[0] for q in queries]

            predictions = {}
            for i, (query_id, _) in enumerate(queries):
                if i >= queries_tfidf.shape[0]:
                    continue
                query_vec = queries_tfidf[i].reshape(1, -1)
                similarities = cosine_similarity(docs_tfidf, query_vec).flatten()
                top_k = similarities.argsort()[::-1][:10]
                predictions[query_id] = [doc_ids[idx] for idx in top_k]

            print("📊 Calculating TF-IDF metrics...")
            metrics_all["tfidf"] = {
                "MAP": round(mean_average_precision(qrels, predictions), 4),
                "MRR": round(mean_reciprocal_rank(qrels, predictions), 4),
                "P@10": round(precision_at_k(qrels, predictions, 10), 4),
                "R@100": round(recall_at_k(qrels, predictions, 100), 4)
            }

            results_all["tfidf"] = predictions
            print("✅ TF-IDF evaluation complete.")

        # ================ ✅ تقييم Word2Vec =================
        doc_vecs_path = f"data/word2vec/documents_{dataset_id}/doc_vectors.pkl"
        query_vecs_path = f"data/word2vec/queries_{dataset_id}/doc_vectors.pkl"

        if os.path.exists(doc_vecs_path) and os.path.exists(query_vecs_path):
            print("🔍 Evaluating Word2Vec...")
            doc_vectors = joblib.load(doc_vecs_path)
            query_vectors = joblib.load(query_vecs_path)

            docs_matrix = np.array(
                [doc_vectors[doc_id] for doc_id in doc_ids if doc_id in doc_vectors],
                dtype=np.float32
            )

            valid_queries = [
                (qid, query_vectors[qid]) for qid, _ in queries
                if qid in query_vectors and query_vectors[qid] is not None
            ]

            predictions = {}
            for qid, query_vec in valid_queries:
                query_vec = query_vec.reshape(1, -1)
                similarities = cosine_similarity(docs_matrix, query_vec).flatten()
                top_k = similarities.argsort()[::-1][:10]
                predictions[qid] = [doc_ids[idx] for idx in top_k]

            print("📊 Calculating Word2Vec metrics...")
            metrics_all["word2vec"] = {
                "MAP": round(mean_average_precision(qrels, predictions), 4),
                "MRR": round(mean_reciprocal_rank(qrels, predictions), 4),
                "P@10": round(precision_at_k(qrels, predictions, 10), 4),
                "R@100": round(recall_at_k(qrels, predictions, 100), 4)
            }

            results_all["word2vec"] = predictions
            print("✅ Word2Vec evaluation complete.")

        # 📦 حفظ النتائج
        print("💾 Saving evaluation results to file...")
        os.makedirs("evaluation_results", exist_ok=True)
        with open(f"evaluation_results/results_{dataset_id}.json", "w", encoding='utf-8') as f:
            json.dump(metrics_all, f, indent=2)

        print("🎉 Evaluation process complete.")
        return jsonify(metrics_all)

    except Exception as e:
        print("❌ Error occurred during evaluation:", e)
        return jsonify({"error": str(e)}), 500
