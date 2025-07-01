from flask import Blueprint, request, jsonify
import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.database.models import get_documents, get_queries

bp = Blueprint('match_queries_word2vec_fast', __name__, url_prefix='/word2vec')

@bp.route('/match_queries', methods=['POST'])
def match_queries_word2vec_fast():
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        top_k = data.get('top_k', 10)

        if not dataset_id:
            return jsonify({"error": "dataset_id is required"}), 400

        doc_vectors_path = f"data/word2vec/documents_{dataset_id}/doc_vectors.pkl"
        query_vectors_path = f"data/word2vec/queries_{dataset_id}/doc_vectors.pkl"

        if not os.path.exists(doc_vectors_path) or not os.path.exists(query_vectors_path):
            return jsonify({"error": "Vectors not found for this dataset"}), 404

        doc_vectors = joblib.load(doc_vectors_path)
        query_vectors = joblib.load(query_vectors_path)

        docs = get_documents(dataset_id)
        queries = get_queries(dataset_id)

        doc_ids = list(doc_vectors.keys())

        # ✨ تصفية الاستعلامات التي تملك تمثيل شعاعي صالح فقط
        valid_queries = [(qid, query_vectors[qid]) for qid in query_vectors if query_vectors[qid] is not None]
        query_ids = [qid for qid, _ in valid_queries]
        query_matrix = np.array([vec for _, vec in valid_queries])
        doc_matrix = np.array([doc_vectors[doc_id] for doc_id in doc_ids])

        # ⚡ حساب التشابه دفعة واحدة
        similarity_matrix = cosine_similarity(query_matrix, doc_matrix)  # [num_queries, num_docs]

        results = []

        for i, sim_row in enumerate(similarity_matrix):
            top_indices = sim_row.argsort()[::-1][:top_k]
            result = {
                "query_index": i,
                "query_id": query_ids[i],
                "top_matches": [
                    {
                        "doc_id": doc_ids[idx],
                        "score": float(sim_row[idx])
                    }
                    for idx in top_indices
                ]
            }
            results.append(result)

        # 💾 حفظ النتائج في ملف
        output_path = f"results/word2vec_results_{dataset_id}.txt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"\n🔍 Query {r['query_index']} (query_id={r['query_id']}) top matches:\n")
                for match in r["top_matches"]:
                    f.write(f"   → Doc ID: {match['doc_id']} | Score: {match['score']:.3f}\n")

        return jsonify({
            "results": results,
            "count": len(results),
            "file_path": output_path
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
