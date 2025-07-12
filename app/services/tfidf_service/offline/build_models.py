from flask import Blueprint, request, jsonify
from app.services.tfidf_service.utils import calculate_tfidf, calculate_query_tfidf
import joblib
import os
import requests

bp = Blueprint('tfidf_documents', __name__, url_prefix='/tfidf')

@bp.route('/build', methods=['POST'])
def build_tfidf_using_api():
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        table_name = data.get('table_name', 'documents') 

        if not dataset_id or table_name not in ['documents', 'queries']:
            return jsonify({"error": "Invalid dataset_id or table_name"}), 400

        preprocess_url = "http://127.0.0.1:5000/preprocess/bulk"
        response = requests.post(preprocess_url, json={
            "dataset_id": dataset_id,
            "table_name": table_name
        })

        if response.status_code != 200:
            return jsonify({"error": "Failed to preprocess data", "details": response.text}), 500

        result = response.json()
        processed_data = result.get("processed_data", [])

        if not processed_data:
            return jsonify({"error": "No processed data returned"}), 404

        if table_name == "documents":
            tfidf_matrix, vectorizer = calculate_tfidf(processed_data)
            model_dir = f"data/tfidf/documents_{dataset_id}"
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(tfidf_matrix, os.path.join(model_dir, "tfidf_matrix.pkl"))
            joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))

        elif table_name == "queries":
            model_dir = f"data/tfidf/documents_{dataset_id}"
            vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

            if not os.path.exists(vectorizer_path):
                return jsonify({"error": "Vectorizer not found. Build TF-IDF for documents first."}), 404

            vectorizer = joblib.load(vectorizer_path)
            tfidf_matrix = calculate_query_tfidf(processed_data, vectorizer)

            query_dir = f"data/tfidf/queries{dataset_id}"
            os.makedirs(query_dir, exist_ok=True)
            joblib.dump(tfidf_matrix, os.path.join(query_dir, "tfidf_matrix.pkl"))

        return jsonify({
            "message": f"✅ TF-IDF model built for dataset_id={dataset_id}, table={table_name}",
            "num_documents": len(processed_data)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# from flask import Blueprint, request, jsonify
# from app.services.tfidf_service.utils import calculate_tfidf
# from app.database.models import get_documents, get_queries  # تأكدي من وجودهم
# import joblib
# import os

# bp = Blueprint('tfidf_documents', __name__, url_prefix='/tfidf')

# @bp.route('/build_raw', methods=['POST'])
# def build_tfidf_without_preprocessing():
#     try:
#         data = request.json
#         dataset_id = data.get('dataset_id')
#         table_name = data.get('table_name', 'documents')

#         if not dataset_id or table_name not in ['documents', 'queries']:
#             return jsonify({"error": "Invalid dataset_id or table_name"}), 400

#         # ✅ جلب النصوص الخام من قاعدة البيانات
#         if table_name == 'documents':
#             raw_data = get_documents(dataset_id)  # [(doc_id, text), ...]
#         else:
#             raw_data = get_queries(dataset_id)  # [(query_id, text), ...]

#         if not raw_data:
#             return jsonify({"error": f"No {table_name} found for dataset_id={dataset_id}"}), 404

#         # ⏬ تحويل إلى الشكل المطلوب لـ TF-IDF
#         processed_data = {str(doc_id): text for doc_id, text in raw_data}
#         print("⚠️ البيانات الأصلية بدون معالجة:")
#         print(processed_data)

#         # ✅ حساب TF-IDF
#         tfidf_matrix, vectorizer = calculate_tfidf(processed_data)

#         # 💾 حفظ النتائج
#         model_dir = f"data/tfidf/{table_name}_{dataset_id}"
#         os.makedirs(model_dir, exist_ok=True)
#         joblib.dump(tfidf_matrix, os.path.join(model_dir, "tfidf_matrix.pkl"))
#         joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))

#         return jsonify({
#             "message": f"✅ Raw TF-IDF model built for {table_name} dataset_id={dataset_id}",
#             "num_documents": len(processed_data)
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
