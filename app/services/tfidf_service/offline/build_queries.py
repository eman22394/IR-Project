from flask import Blueprint, request, jsonify
from app.services.tfidf_service.utils import calculate_query_tfidf
import joblib
import os
import requests

bp = Blueprint('tfidf_queries', __name__, url_prefix='/tfidf')
@bp.route('/transform_queries', methods=['POST'])
def transform_queries():
    try:
        data = request.json
        dataset_id = data.get('dataset_id')

        preprocess_url = "http://127.0.0.1:5000/preprocess/bulk"
        response = requests.post(preprocess_url, json={
            "dataset_id": dataset_id,
            "table_name": "queries",
        })
        if response.status_code != 200:
            return jsonify({"error": "Failed to preprocess queries", "details": response.text}), 500

        result = response.json()
        processed_data = result["processed_data"]

        if not processed_data:
            return jsonify({"error": "No processed queries returned"}), 404

        # تحميل vectorizer من ملفات المستندات (dataset_id الخاص بالمستندات)
        model_dir = f"data/tfidf/documents_{dataset_id}"
        vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.pkl"))

        # استخدام vectorizer لتحويل الاستعلامات فقط (transform)
          # ✅ حساب TF-IDF
        tfidf_matrix = calculate_query_tfidf(processed_data, vectorizer)

        # 💾 حفظ النتائج
        model_dir = f"data/tfidf/queries{dataset_id}"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(tfidf_matrix, os.path.join(model_dir, "tfidf_matrix.pkl"))

        return jsonify({
            "message": f"✅ TF-IDF model built for dataset_id={dataset_id}",
            "num_documents": len(processed_data)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
