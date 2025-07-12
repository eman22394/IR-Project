from flask import Blueprint, request, jsonify
import joblib
import os
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from app.database.models import get_documents

bp = Blueprint('hybrid_query', __name__, url_prefix='/hybrid')
@bp.route("/match_query", methods=["POST"])
def fusion_match_query():
    try:
        data        = request.json
        dataset_id  = data.get("dataset_id")
        query_text  = data.get("text")
        alpha       = float(data.get("alpha", 0.4))
        k_recall    = int(data.get("k_recall", 1000))
        top_k       = int(data.get("top_k", 10))

        print(f"[DEBUG] dataset_id: {dataset_id}")
        print(f"[DEBUG] query_text: {query_text}")
        print(f"[DEBUG] alpha: {alpha}, k_recall: {k_recall}, top_k: {top_k}")

        if not dataset_id or not query_text:
            return jsonify({"error": "Missing 'dataset_id' or 'text'"}), 400

        # --- تحميل تمثيلات الوثائق ---
        tfidf_docs  = joblib.load(f"data/tfidf/documents_{dataset_id}/tfidf_matrix.pkl").astype(np.float32)
        tfidf_vec   = joblib.load(f"data/tfidf/documents_{dataset_id}/vectorizer.pkl")
        bert_docs   = joblib.load(f"data/bert/documents_{dataset_id}/doc_vectors.pkl")

        doc_ids     = list(bert_docs.keys())
        bert_matrix = np.array([bert_docs[d] for d in doc_ids], dtype=np.float32)

        print(f"[DEBUG] Loaded TF-IDF docs shape: {tfidf_docs.shape}")
        print(f"[DEBUG] Loaded BERT docs count: {len(doc_ids)}")
        print(f"[DEBUG] TF-IDF vocab size: {len(tfidf_vec.vocabulary_)}")

        # --- تنظيف الاستعلام & tokens ---
        prep = requests.post("http://127.0.0.1:5000/preprocess/query", json={
            "dataset_id" : dataset_id,
            "text": query_text })
        print(f"[DEBUG] Preprocess response status: {prep.status_code}")
        if prep.status_code != 200:
            return jsonify({"error": "Preprocess failed"}), 500

        tokens = prep.json().get("tokens", [])
        print(f"[DEBUG] Tokens after preprocessing: {tokens}")
        if not tokens:
            return jsonify({"error": "Empty query after preprocessing"}), 400

        # --- تمثيل TF‑IDF للاستعلام ---
        q_tfidf = tfidf_vec.transform([" ".join(tokens)]).astype(np.float32)
        print(f"[DEBUG] TF-IDF query vector shape: {q_tfidf.shape}, nnz: {q_tfidf.nnz}")
        if q_tfidf.nnz == 0:
            return jsonify({"error": "No TF‑IDF terms in vocabulary for query"}), 400

        # --- تمثيل BERT للاستعلام ---
        bert_resp = requests.post("http://127.0.0.1:5000/preprocess/query", json={
            "dataset_id" : dataset_id,
            "text": query_text})
        print(f"[DEBUG] BERT embed response status: {bert_resp.status_code}")
        if bert_resp.status_code != 200:
            return jsonify({"error": "BERT embed failed"}), 500

        q_bert = np.array(bert_resp.json().get("vector", []), dtype=np.float32).reshape(1, -1)
        print(f"[DEBUG] BERT query vector shape: {q_bert.shape}")

        # --- مرحلة الاسترجاع TF‑IDF ---
        tf_scores = cosine_similarity(tfidf_docs, q_tfidf).flatten()
        print(f"[DEBUG] TF-IDF scores length: {len(tf_scores)}")
        tf_top_idx = tf_scores.argsort()[::-1][:k_recall]

        # --- تشابه BERT على المرشَّحين ---
        bert_scores = cosine_similarity(bert_matrix[tf_top_idx], q_bert).flatten()
        print(f"[DEBUG] BERT scores length: {len(bert_scores)}")

        # --- الدمج ---
        fused = alpha * tf_scores[tf_top_idx] + (1 - alpha) * bert_scores

        # --- ترتيب نهائي ---
        order = fused.argsort()[::-1][:top_k]
        final_idx = [tf_top_idx[j] for j in order]

        # --- جلب نصوص الوثائق ---
        doc_text_map = {str(d[0]): d[1] for d in get_documents(dataset_id)}

        results = [{
            "doc_id": doc_ids[i],
            "score" : float(fused[order[idx]]),
            "text"  : doc_text_map.get(doc_ids[i], "")
        } for idx, i in enumerate(final_idx)]

        print(f"[DEBUG] Returning {len(results)} results")
        return jsonify({"query_tokens": tokens, "top_matches": results})

    except Exception as e:
        print(f"[ERROR] Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500

# from flask import Blueprint, request, jsonify
# import joblib
# import os
# import numpy as np
# import requests
# from sklearn.metrics.pairwise import cosine_similarity
# from app.services.embedding_service.utils import get_mean_vector, load_word2vec_model
# from app.database.models import get_documents


# bp = Blueprint('hybrid_query', __name__, url_prefix='/hybrid')


# @bp.route('/match_query', methods=['POST'])
# def match_query_hybrid():
#     try:
#         data = request.json
#         dataset_id = data.get('dataset_id')
#         query_text = data.get('text')
#         alpha = float(data.get('alpha', 0.5))  # نسبة المزج بين TF-IDF و Word2Vec

#         if not dataset_id or not query_text:
#             return jsonify({"error": "Missing 'dataset_id' or 'text'"}), 400

#         # ✅ تحميل تمثيلات الهجينة الجاهزة
#         hybrid_path = f"data/hybrid/documents_{dataset_id}/hybrid_vectors.pkl"
#         tfidf_model_path = f"data/tfidf/documents_{dataset_id}/vectorizer.pkl"

#         if not os.path.exists(hybrid_path) or not os.path.exists(tfidf_model_path):
#             return jsonify({"error": "Required hybrid or TF-IDF model not found"}), 404

#         hybrid_data = joblib.load(hybrid_path)
#         tfidf_doc_matrix = hybrid_data["tfidf"]
#         w2v_doc_matrix = hybrid_data["w2v"]
#         doc_ids = hybrid_data["doc_ids"]

#         tfidf_vectorizer = joblib.load(tfidf_model_path)

#         # 📄 جلب النصوص الأصلية للوثائق (من قاعدة البيانات)
#         documents = get_documents(dataset_id)
#         doc_text_map = {str(doc[0]): doc[1] for doc in documents}

#         # 🔁 تنظيف الاستعلام باستخدام endpoint منفصل
#         response = requests.post("http://127.0.0.1:5000/preprocess/query", json={"text": query_text})
#         if response.status_code != 200:
#             return jsonify({"error": "Failed to preprocess query"}), 500

#         tokens = response.json().get("tokens", [])
#         if not tokens:
#             return jsonify({"error": "Query is empty after preprocessing"}), 400

#         # 🧠 تمثيل الاستعلام
#         query_tfidf = tfidf_vectorizer.transform([" ".join(tokens)])

#         # استخدم نفس model.pkl الذي بُني به hybrid offline
#         w2v_model_path = f"data/word2vec/documents_{dataset_id}/model.pkl"
#         if not os.path.exists(w2v_model_path):
#             return jsonify({"error": "Word2Vec model not found"}), 404

#         w2v_model = load_word2vec_model(w2v_model_path)
#         query_w2v = get_mean_vector(w2v_model, tokens).reshape(1, -1)

#         # 🔍 حساب التشابه الهجين
#         tfidf_scores = cosine_similarity(tfidf_doc_matrix, query_tfidf).flatten()
#         w2v_scores = cosine_similarity(w2v_doc_matrix, query_w2v).flatten()
#         final_scores = alpha * tfidf_scores + (1 - alpha) * w2v_scores

#         # 🔝 أفضل النتائج
#         top_k = int(data.get('top_k', 10))
#         top_indices = final_scores.argsort()[::-1][:top_k]

#         results = []
#         for idx in top_indices:
#             results.append({
#                 "doc_id": doc_ids[idx],
#                 "score": float(final_scores[idx]),
#                 "text": doc_text_map.get(doc_ids[idx], "N/A")
#             })

#         return jsonify({
#             "query_tokens": tokens,
#             "top_matches": results
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

