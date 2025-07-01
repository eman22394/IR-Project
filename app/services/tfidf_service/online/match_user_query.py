from flask import Blueprint, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import requests
from app.database.models import get_documents
from app.services.tfidf_service.utils import calculate_query_tfidf

bp = Blueprint('tfidf_online', __name__, url_prefix='/tfidf')

@bp.route('/match_query', methods=['POST'])
def match_user_query():
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        query_text = data.get('text')

        if not dataset_id or not query_text:
            return jsonify({"error": "Missing 'dataset_id' or 'text'"}), 400

        # 1. إرسال نص الاستعلام إلى سيرفس المعالجة
        preprocess_url = "http://127.0.0.1:5000/preprocess/query"
        response = requests.post(preprocess_url, json={"text": query_text})
        if response.status_code != 200:
            return jsonify({"error": "Failed to preprocess query", "details": response.text}), 500

        tokens = response.json().get("tokens")
        if not tokens:
            return jsonify({"error": "No tokens returned from preprocess"}), 500

        # 2. تحميل TF-IDF ومصفوفة المستندات
        docs_tfidf_path = f"data/tfidf/documents_{dataset_id}/tfidf_matrix.pkl"
        vectorizer_path = f"data/tfidf/documents_{dataset_id}/vectorizer.pkl"

        if not os.path.exists(docs_tfidf_path) or not os.path.exists(vectorizer_path):
            return jsonify({"error": "TF-IDF model not found for this dataset_id"}), 404

        docs_tfidf = joblib.load(docs_tfidf_path)
        vectorizer = joblib.load(vectorizer_path)

        # 3. جلب المستندات من قاعدة البيانات مع النص
        documents = get_documents(dataset_id)  # [(doc_id, text), ...]
        doc_ids = [doc[0] for doc in documents]
        doc_texts = {doc[0]: doc[1] for doc in documents}

        # 4. تحويل التوكنز إلى الشكل المطلوب لـ TF-IDF
        query_dict = {0: tokens}
        query_vector = calculate_query_tfidf(query_dict, vectorizer)


        # 5. حساب التشابه
        similarities = cosine_similarity(docs_tfidf, query_vector).flatten()
        top_indices = similarities.argsort()[::-1][:10]

        # 6. إعداد النتائج مع نصوص المستندات
        results = [
            {
                "doc_id": int(doc_ids[idx]),
                "score": float(similarities[idx]),
                "text": doc_texts[doc_ids[idx]]
            }
            for idx in top_indices
        ]

        return jsonify({
            "query_tokens": tokens,
            "top_matches": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
