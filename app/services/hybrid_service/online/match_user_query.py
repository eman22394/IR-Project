# file: app/services/hybrid_service/endpoints.py
from flask import Blueprint, request, jsonify
import joblib
import os
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from app.services.embedding_service.utils import get_mean_vector, load_word2vec_model
from app.database.models import get_documents


bp = Blueprint('hybrid_query', __name__, url_prefix='/hybrid')


@bp.route('/match_query', methods=['POST'])
def match_query_hybrid():
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        query_text = data.get('text')
        alpha = float(data.get('alpha', 0.5))  # نسبة المزج بين TF-IDF و Word2Vec

        if not dataset_id or not query_text:
            return jsonify({"error": "Missing 'dataset_id' or 'text'"}), 400

        # ✅ تحميل تمثيلات الهجينة الجاهزة
        hybrid_path = f"data/hybrid/documents_{dataset_id}/hybrid_vectors.pkl"
        tfidf_model_path = f"data/tfidf/documents_{dataset_id}/vectorizer.pkl"

        if not os.path.exists(hybrid_path) or not os.path.exists(tfidf_model_path):
            return jsonify({"error": "Required hybrid or TF-IDF model not found"}), 404

        hybrid_data = joblib.load(hybrid_path)
        tfidf_doc_matrix = hybrid_data["tfidf"]
        w2v_doc_matrix = hybrid_data["w2v"]
        doc_ids = hybrid_data["doc_ids"]

        tfidf_vectorizer = joblib.load(tfidf_model_path)

        # 📄 جلب النصوص الأصلية للوثائق (من قاعدة البيانات)
        documents = get_documents(dataset_id)
        doc_text_map = {str(doc[0]): doc[1] for doc in documents}

        # 🔁 تنظيف الاستعلام باستخدام endpoint منفصل
        response = requests.post("http://127.0.0.1:5000/preprocess/query", json={"text": query_text})
        if response.status_code != 200:
            return jsonify({"error": "Failed to preprocess query"}), 500

        tokens = response.json().get("tokens", [])
        if not tokens:
            return jsonify({"error": "Query is empty after preprocessing"}), 400

        # 🧠 تمثيل الاستعلام
        query_tfidf = tfidf_vectorizer.transform([" ".join(tokens)])

        # استخدم نفس model.pkl الذي بُني به hybrid offline
        w2v_model_path = f"data/word2vec/documents_{dataset_id}/model.pkl"
        if not os.path.exists(w2v_model_path):
            return jsonify({"error": "Word2Vec model not found"}), 404

        w2v_model = load_word2vec_model(w2v_model_path)
        query_w2v = get_mean_vector(w2v_model, tokens).reshape(1, -1)

        # 🔍 حساب التشابه الهجين
        tfidf_scores = cosine_similarity(tfidf_doc_matrix, query_tfidf).flatten()
        w2v_scores = cosine_similarity(w2v_doc_matrix, query_w2v).flatten()
        final_scores = alpha * tfidf_scores + (1 - alpha) * w2v_scores

        # 🔝 أفضل النتائج
        top_k = int(data.get('top_k', 10))
        top_indices = final_scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "doc_id": doc_ids[idx],
                "score": float(final_scores[idx]),
                "text": doc_text_map.get(doc_ids[idx], "N/A")
            })

        return jsonify({
            "query_tokens": tokens,
            "top_matches": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

