# file: app/routes/mbert_build.py

from flask import Blueprint, request, jsonify
import os
import joblib
import requests
from sentence_transformers import SentenceTransformer

bp = Blueprint('mbert_documents', __name__, url_prefix='/mbert')

@bp.route('/build', methods=['POST'])
def build_mbert():
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        table_name = data.get('table_name', 'documents')

        if not dataset_id or table_name not in ['documents', 'queries']:
            return jsonify({"error": "Missing or invalid 'dataset_id' or 'table_name'"}), 400

        # 🔁 معالجة مسبقة
        preprocess_url = "http://127.0.0.1:5000/preprocess/bulk"
        response = requests.post(preprocess_url, json={
            "dataset_id": dataset_id,
            "table_name": table_name,
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
            return jsonify({"error": "Failed to preprocess data"}), 500

        processed_data = response.json().get("processed_data", {})
        if not processed_data:
            return jsonify({"error": "No processed data returned"}), 404

        print(f"🚀 Encoding {len(processed_data)} {table_name} with multilingual BERT...")

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        doc_vectors = {}
        for i, (doc_id, tokens) in enumerate(processed_data.items(), 1):
            text = " ".join(tokens)
            doc_vectors[doc_id] = model.encode(text, convert_to_numpy=True)

            if i % 1000 == 0:
                print(f"✅ Encoded {i} / {len(processed_data)} {table_name}")

        print(f"🎉 Finished encoding all {len(doc_vectors)} {table_name}")

        model_dir = f"data/mbert/{table_name}_{dataset_id}"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(doc_vectors, os.path.join(model_dir, "doc_vectors.pkl"))
        model.save(os.path.join(model_dir, "model"))

        return jsonify({
            "message": f"✅ Multilingual BERT vectors built for {table_name} of dataset {dataset_id}",
            "num_documents": len(doc_vectors)
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500
