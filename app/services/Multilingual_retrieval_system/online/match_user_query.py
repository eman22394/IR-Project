from flask import Blueprint, request, jsonify
import os
import joblib
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.database.models import get_documents

bp = Blueprint('mbert_query', __name__, url_prefix='/mbert')

@bp.route('/match_query', methods=['POST'])
def match_multilingual_query():
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        query_text = data.get('text')

        if not dataset_id or not query_text:
            return jsonify({"error": "Missing 'dataset_id' or 'text'"}), 400

        preprocess_url = "http://127.0.0.1:5000/preprocess/query"
        response = requests.post(preprocess_url, json={
            "text": query_text,
            "options": {
                "normalize": True,
                "spell_correction": False,
                "process_dates": False,
                "tokenize": False,
                "remove_stopwords": False,
                "lemmatize": False,
                "stem": False
            }

            })
        if response.status_code != 200:
            return jsonify({"error": "Failed to preprocess query"}), 500

        tokens = response.json().get("tokens", [])
        if not tokens:
            return jsonify({"error": "No tokens returned from preprocessing"}), 500

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        model_dir = f"data/mbert/documents_{dataset_id}"
        doc_vecs_path = os.path.join(model_dir, "doc_vectors.pkl")

        if not os.path.exists(doc_vecs_path):
            return jsonify({"error": "Document vectors not found"}), 404

        doc_vectors = joblib.load(doc_vecs_path)

        query_vector = model.encode(" ".join(tokens), convert_to_numpy=True).reshape(1, -1)
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
