from flask import Blueprint, request, jsonify
import os
import joblib
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.database.models import get_documents
from nltk.corpus import stopwords

bp = Blueprint('bert_query', __name__, url_prefix='/bert')

@bp.route('/match_query', methods=['POST'])
def match_user_query():
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        query_text = data.get('text')

        if not dataset_id or not query_text:
            return jsonify({"error": "Missing 'dataset_id' or 'text'"}), 400

        # 🔁 معالجة الاستعلام
        preprocess_url = "http://127.0.0.1:5000/preprocess/query"
        response = requests.post(preprocess_url, json={"text": query_text})
        if response.status_code != 200:
            return jsonify({"error": "Failed to preprocess query"}), 500

        tokens = response.json().get("tokens")
        if not tokens:
            return jsonify({"error": "No tokens returned from preprocess"}), 500

        # 📦 تحميل التمثيلات
        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        model_dir = f"data/bert/documents_{dataset_id}"
        doc_vecs_path = os.path.join(model_dir, "doc_vectors.pkl")

        if not os.path.exists(doc_vecs_path):
            return jsonify({"error": "Document vectors not found"}), 404

        doc_vectors = joblib.load(doc_vecs_path)

        filtered_tokens = [t for t in tokens if t.lower() not in stopwords.words('english')]
        query_text = "query: " + " ".join(filtered_tokens)
        query_vector = model.encode(query_text, convert_to_numpy=True).reshape(1, -1)

        doc_ids = list(doc_vectors.keys())
        doc_matrix = [doc_vectors[doc_id] for doc_id in doc_ids]

        similarities = cosine_similarity(doc_matrix, query_vector).flatten()
        top_indices = similarities.argsort()[::-1][:10]

        all_docs = get_documents(dataset_id)
        doc_text_map = {str(doc[0]): doc[1] for doc in all_docs}

        results = [
            {
                "doc_id": doc_ids[i],
                "score": float(similarities[i]),
                "text": doc_text_map.get(doc_ids[i], "")
            }
            for i in top_indices
        ]

        return jsonify({
            "query_tokens": tokens,
            "top_matches": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# # file: word2vec_service/match_user_query.py

# from flask import Blueprint, request, jsonify
# import requests
# import joblib
# import os
# from sklearn.metrics.pairwise import cosine_similarity
# from app.services.embedding_service.utils import get_mean_vector, load_word2vec_model
# from app.database.models import get_documents

# bp = Blueprint('word2vec_query', __name__, url_prefix='/word2vec')

# @bp.route('/match_query', methods=['POST'])
# def match_user_query():
#     try:
#         data = request.json
#         dataset_id = data.get('dataset_id')
#         query_text = data.get('text')

#         if not dataset_id or not query_text:
#             return jsonify({"error": "Missing 'dataset_id' or 'text'"}), 400

#         # 🔁 معالجة الاستعلام من خلال خدمة التنظيف
#         preprocess_url = "http://127.0.0.1:5000/preprocess/query"
#         response = requests.post(preprocess_url, json={"text": query_text})
#         if response.status_code != 200:
#             return jsonify({"error": "Failed to preprocess query", "details": response.text}), 500

#         tokens = response.jsoِn().get("tokens")
#         if not tokens:
#             return jsonify({"error": "No tokens returned from preprocess"}), 500

#         # 🔁 تحميل النموذج وتمثيلات المستندات
#         model_dir = f"data/word2vec/documents_{dataset_id}"
#         model_path = os.path.join(model_dir, "model.pkl")
#         doc_vecs_path = os.path.join(model_dir, "doc_vectors.pkl")

#         if not os.path.exists(model_path) or not os.path.exists(doc_vecs_path):
#             return jsonify({"error": "Word2Vec model or document vectors not found"}), 404

#         model = load_word2vec_model(model_path)
#         doc_vectors = joblib.load(doc_vecs_path)  # {doc_id: vector}

#         # ✅ تمثيل الاستعلام
#         query_vector = get_mean_vector(model, tokens).reshape(1, -1)

#         # ✅ تحويل المستندات إلى مصفوفة
#         doc_ids = list(doc_vectors.keys())
#         doc_matrix = [doc_vectors[doc_id] for doc_id in doc_ids]

#         # ✅ حساب التشابه
#         similarities = cosine_similarity(doc_matrix, query_vector).flatten()
#         top_indices = similarities.argsort()[::-1][:10]

#         # ✅ جلب نصوص المستندات من قاعدة البيانات
#         all_docs = get_documents(dataset_id)  # [(doc_id, text), ...]
#         doc_text_map = {str(doc[0]): doc[1] for doc in all_docs}

#         results = [
#             {
#                 "doc_id": doc_ids[i],
#                 "score": float(similarities[i]),
#                 "text": doc_text_map.get(doc_ids[i], "")
#             }
#             for i in top_indices
#         ]

#         return jsonify({
#             "query_tokens": tokens,
#             "top_matches": results
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
