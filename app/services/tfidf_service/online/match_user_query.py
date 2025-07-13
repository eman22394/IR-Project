from flask import Blueprint, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import joblib, os, json, requests, logging
from app.services.tfidf_service.utils import calculate_query_tfidf
from app.services.bm25_service.offline.build_bm25 import load_inverted_index

bp = Blueprint("tfidf_online", __name__, url_prefix="/tfidf")

@bp.route("/match_query", methods=["POST"])
def match_user_query():
    try:
        data = request.get_json(silent=True) or {}
        dataset_id = int(data.get("dataset_id", -1))
        query_text = data.get("text", "").strip()

        if dataset_id < 0 or not query_text:
            return jsonify(error="Missing 'dataset_id' or 'text'"), 400

        # ------------------- 1. المعالجة المسبقة -------------------
        resp = requests.post("http://127.0.0.1:5000/preprocess/query",
                             json={"text": query_text}, timeout=10)
        if resp.status_code != 200:
            return jsonify(error="Preprocess service failed",
                           details=resp.text), 502

        tokens = resp.json().get("tokens") or []
        if not tokens:
            return jsonify(error="Preprocess returned empty tokens"), 500

        # ------------------- 2. تحميل بيانات الوثائق -------------------
        inverted_index = load_inverted_index(dataset_id)
        if not inverted_index or "documents" not in inverted_index:
            return jsonify(error="Inverted index not found or invalid"), 500

        doc_ids_path = f"data/tfidf/documents_{dataset_id}/doc_ids.json"
        if not os.path.isfile(doc_ids_path):
            return jsonify(error="doc_ids file not found"), 500

        with open(doc_ids_path, encoding="utf-8") as f:
            doc_ids = json.load(f)

        doc_texts = {int(i): inverted_index["documents"][i] for i in doc_ids}

        # ------------------- 3. تحميل ملفات TF-IDF -------------------
        tfidf_base_path = f"data/tfidf/documents_{dataset_id}"
        tfidf_matrix_path = os.path.join(tfidf_base_path, "tfidf_matrix.pkl")
        vectorizer_path = os.path.join(tfidf_base_path, "vectorizer.pkl")

        if not os.path.isfile(tfidf_matrix_path) or not os.path.isfile(vectorizer_path):
            return jsonify(error="TF-IDF model files not found"), 404

        tfidf_matrix = joblib.load(tfidf_matrix_path)
        vectorizer = joblib.load(vectorizer_path)

        # ------------------- 4. تحويل الاستعلام إلى TF-IDF -------------------
        query_vector = calculate_query_tfidf({0: tokens}, vectorizer)

        # ------------------- 5. حساب التشابه -------------------
        similarities = cosine_similarity(tfidf_matrix, query_vector).ravel()
        top_indices = similarities.argsort()[::-1][:10]

        results = []
        for idx in top_indices:
            doc_id = int(doc_ids[idx])
            results.append({
                "doc_id": doc_id,
                "score": float(similarities[idx]),
                "text": doc_texts.get(doc_id, "")
            })

        return jsonify(query_tokens=tokens, top_matches=results)

    except Exception as e:
        logging.exception("Error in /tfidf/match_query")
        return jsonify(error=str(e)), 500
