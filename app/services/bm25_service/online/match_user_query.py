from flask import Blueprint, request, jsonify
from app.services.bm25_service.offline.build_bm25 import load_inverted_index, compute_bm25

bp = Blueprint('bm25', __name__, url_prefix='/bm25')
@bp.route('/search', methods=['POST'])
def bm25_search():
    try:
        data = request.get_json()
        dataset_id = data.get("dataset_id")
        text = data.get("text")
        top_k = data.get("top_k", 20)

        k1 = data.get("k1", 1.5)
        b = data.get("b", 0.75)

        if dataset_id is None or not text:
            return jsonify({"error": "يرجى تمرير dataset_id و query"}), 400

        index_data = load_inverted_index(dataset_id)
        if not index_data:
            return jsonify({"error": f"لا يوجد فهرس لهذا الـ dataset_id: {dataset_id}"}), 404

        results = compute_bm25(text, index_data, top_k=top_k, k1=k1, b=b)

        return jsonify(results)

    except Exception as e:
        print(f"[BM25 Error]: {str(e)}")
        return jsonify({"error": str(e)}), 500
