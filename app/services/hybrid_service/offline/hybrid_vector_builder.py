# file: app/services/hybrid_service/build_hybrid_vectors_endpoint.py
from flask import Blueprint, request, jsonify
import os
import joblib
import numpy as np

from app.services.embedding_service.utils import load_word2vec_model
from app.database.models import get_documents

bp = Blueprint('hybrid_builder', __name__, url_prefix='/hybrid')

@bp.route('/build_vectors', methods=['POST'])
def build_hybrid_vectors():
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        if not dataset_id:
            return jsonify({"error": "Missing 'dataset_id'"}), 400

        # 🛠️ Paths
        w2v_model_path = f"data/word2vec/documents_{dataset_id}/model.pkl"
        w2v_doc_vecs_path = f"data/word2vec/documents_{dataset_id}/doc_vectors.pkl"
        tfidf_matrix_path = f"data/tfidf/documents_{dataset_id}/tfidf_matrix.pkl"
        output_path = f"data/hybrid/documents_{dataset_id}/hybrid_vectors.pkl"

        # ✅ Checks
        if not all(os.path.exists(p) for p in [w2v_doc_vecs_path, tfidf_matrix_path]):
            return jsonify({"error": "Required Word2Vec or TF-IDF files are missing"}), 404

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ✅ Load data
        tfidf_matrix = joblib.load(tfidf_matrix_path)
        w2v_doc_vectors = joblib.load(w2v_doc_vecs_path)

        documents = get_documents(dataset_id)
        doc_ids = [str(doc[0]) for doc in documents]
        w2v_matrix = np.array([w2v_doc_vectors[doc_id] for doc_id in doc_ids])

        # ✅ Dump hybrid
        joblib.dump({
            "tfidf": tfidf_matrix,
            "w2v": w2v_matrix,
            "doc_ids": doc_ids
        }, output_path)

        return jsonify({"message": f"✅ Hybrid vectors saved for dataset {dataset_id}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
