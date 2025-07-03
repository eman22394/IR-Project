from flask import Blueprint, request, jsonify
import os
import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from app.database.models import get_documents, get_queries_from_qrels

bp = Blueprint('bert_tfidf_hybrid', __name__, url_prefix='/hybrid')

@bp.route('/build_vectors', methods=['POST'])
def build_bert_tfidf_hybrid_vectors():
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        table_name = data.get('table_name', 'documents')

        if not dataset_id or table_name not in ['documents', 'queries']:
            return jsonify({"error": "Missing or invalid 'dataset_id' or 'table_name'"}), 400

        bert_path = f"data/bert/{table_name}_{dataset_id}/doc_vectors.pkl"
        tfidf_path = f"data/tfidf/{table_name}{dataset_id}/tfidf_matrix.pkl"
        output_dir = f"data/hybrid_bert&tfidf/{table_name}_{dataset_id}"
        output_path = os.path.join(output_dir, "hybrid_vectors.pkl")

        if not os.path.exists(bert_path):
            return jsonify({"error": "Missing BERT vectors file"}), 404
        if not os.path.exists(tfidf_path):
            return jsonify({"error": "Missing TF-IDF matrix file"}), 404
        os.makedirs(output_dir, exist_ok=True)

        # Load BERT vectors
        bert_vectors = joblib.load(bert_path)

        # Load TF-IDF matrix
        tfidf_matrix = joblib.load(tfidf_path)
        if not isinstance(tfidf_matrix, csr_matrix):
            tfidf_matrix = csr_matrix(tfidf_matrix)
        tfidf_matrix = tfidf_matrix.astype(np.float32)

        # Get entry IDs
        if table_name == 'documents':
            entries = get_documents(dataset_id)
        else:
            entries = get_queries_from_qrels(dataset_id)
        entry_ids = [str(e[0]) for e in entries]

        if len(entry_ids) != tfidf_matrix.shape[0]:
            return jsonify({"error": "Mismatch between TF-IDF rows and entries"}), 500

        # Build BERT matrix
        bert_matrix = np.array([bert_vectors[entry_id].astype(np.float32) for entry_id in entry_ids])
        bert_sparse = csr_matrix(bert_matrix)  # convert to sparse

        # Stack sparse BERT + TF-IDF
        hybrid_matrix = hstack([bert_sparse, tfidf_matrix], format='csr')

        # Save
        joblib.dump({
            "entry_ids": entry_ids,
            "hybrid": hybrid_matrix
        }, output_path)

        return jsonify({
            "message": f"✅ Hybrid BERT+TF-IDF vectors built for {table_name} in dataset {dataset_id}",
            "shape": hybrid_matrix.shape
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# # file: app/services/hybrid_service/build_hybrid_vectors_endpoint.py
# from flask import Blueprint, request, jsonify
# import os
# import joblib
# import numpy as np

# from app.services.embedding_service.utils import load_word2vec_model
# from app.database.models import get_documents

# bp = Blueprint('hybrid_builder', __name__, url_prefix='/hybrid')

# @bp.route('/build_vectors', methods=['POST'])
# def build_hybrid_vectors():
#     try:
#         data = request.json
#         dataset_id = data.get('dataset_id')
#         if not dataset_id:
#             return jsonify({"error": "Missing 'dataset_id'"}), 400

#         # 🛠️ Paths
#         w2v_model_path = f"data/bert/documents_{dataset_id}/model.pkl"
#         w2v_doc_vecs_path = f"data/word2vec/documents_{dataset_id}/doc_vectors.pkl"
#         tfidf_matrix_path = f"data/tfidf/documents_{dataset_id}/tfidf_matrix.pkl"
#         output_path = f"data/hybrid/documents_{dataset_id}/hybrid_vectors.pkl"

#         # ✅ Checks
#         if not all(os.path.exists(p) for p in [w2v_doc_vecs_path, tfidf_matrix_path]):
#             return jsonify({"error": "Required Word2Vec or TF-IDF files are missing"}), 404

#         os.makedirs(os.path.dirname(output_path), exist_ok=True)

#         # ✅ Load data
#         tfidf_matrix = joblib.load(tfidf_matrix_path)
#         w2v_doc_vectors = joblib.load(w2v_doc_vecs_path)

#         documents = get_documents(dataset_id)
#         doc_ids = [str(doc[0]) for doc in documents]
#         w2v_matrix = np.array([w2v_doc_vectors[doc_id] for doc_id in doc_ids])

#         # ✅ Dump hybrid
#         joblib.dump({
#             "tfidf": tfidf_matrix,
#             "w2v": w2v_matrix,
#             "doc_ids": doc_ids
#         }, output_path)

#         return jsonify({"message": f"✅ Hybrid vectors saved for dataset {dataset_id}"}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
