from flask import Blueprint, request, jsonify
import os
import joblib
import requests
from sentence_transformers import SentenceTransformer

bp = Blueprint('mbert', __name__, url_prefix='/mbert')

@bp.route('/build', methods=['POST'])
def build_bert():
    try:
        print("📥 Received request to /mbert/build")

        data = request.json
        print(f"📦 Request data: {data}")

        dataset_id = data.get('dataset_id')
        table_name = data.get('table_name', 'documents')

        if not dataset_id or table_name not in ['documents', 'queries']:
            print("❌ Invalid dataset_id or table_name")
            return jsonify({"error": "Missing or invalid 'dataset_id' or 'table_name'"}), 400

        # 🔁 استدعاء خدمة المعالجة المسبقة
        print("🔄 Sending request to preprocessing service...")
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

        print(f"🧾 Preprocessing response status: {response.status_code}")

        if response.status_code != 200:
            print("❌ Failed to preprocess data")
            return jsonify({"error": "Failed to preprocess data", "details": response.text}), 500

        result = response.json()
        processed_data = result.get("processed_data", {})

        if not processed_data:
            print("❌ No processed data returned")
            return jsonify({"error": "No processed data returned"}), 404

        total_docs = len(processed_data)
        print(f"🚀 Starting BERT encoding for {total_docs} {table_name} in dataset {dataset_id}")

        # 🧠 تحميل النموذج المتعدد اللغات
        print("📥 Loading multilingual BERT model...")
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print("✅ Model loaded")

        doc_vectors = {}
        for idx, (doc_id, tokens) in enumerate(processed_data.items(), 1):
            text = " ".join(tokens)
            doc_vectors[doc_id] = model.encode(text, convert_to_numpy=True)

            if idx % 100 == 0 or idx == total_docs:
                print(f"📊 Encoded {idx} / {total_docs} {table_name}")

        print(f"🎉 Finished encoding all {total_docs} {table_name}")

        # 💾 حفظ التمثيلات
        model_dir = f"data/mbert/{table_name}_{dataset_id}"
        print(f"💾 Saving vectors to {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(doc_vectors, os.path.join(model_dir, "doc_vectors.pkl"))
        print("✅ Saved doc_vectors.pkl")

        model.save(os.path.join(model_dir, "model"))
        print("✅ Model saved")

        return jsonify({
            "message": f"✅ Multilingual BERT vectors built for {table_name} of dataset {dataset_id}",
            "num_documents": total_docs
        })

    except Exception as e:
        print("❌ Error during BERT vector building:", e)
        return jsonify({"error": str(e)}), 500
