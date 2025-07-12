from flask import Blueprint, request, jsonify
import os
import joblib
import requests
from sentence_transformers import SentenceTransformer
from app.database.models import get_documents
import faiss
import numpy as np
import time

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
bp = Blueprint('vector_store_', __name__, url_prefix='/vector_store')

@bp.route('/match_query', methods=['POST'])
def match_user_query():
    try:
        start = time.time()  # ⏱️ لحساب الوقت الكلي

        data = request.json
        dataset_id = data.get('dataset_id')
        query_text = data.get('text')

        print("📥 Received request with dataset_id:", dataset_id)
        print("📝 Original query text:", query_text)

        if not dataset_id or not query_text:
            return jsonify({"error": "Missing 'dataset_id' or 'text'"}), 400

        # 🔁 1. معالجة الاستعلام
        preprocess_url = "http://127.0.0.1:5000/preprocess/query"
        print("🔁 Sending request to:", preprocess_url)
        response = requests.post(preprocess_url, json={"text": query_text})
        if response.status_code != 200:
            return jsonify({"error": "Failed to preprocess query"}), 500

        tokens = response.json().get("tokens")
        print("🔤 Received tokens:", tokens)

        if not tokens:
            return jsonify({"error": "No tokens returned from preprocess"}), 500

        # 🧠 2. تمثيل الاستعلام كـ vector باستخدام BERT
        print("🧠 Encoding query with BERT model...")
        query_vector = model.encode(query_text, convert_to_numpy=True).astype('float32').reshape(1, -1)
        print("✅ Query vector shape:", query_vector.shape)

        # 📦 3. تحميل فهرس FAISS و doc_ids
        model_dir = f"data/bert/documents_{dataset_id}"
        index_path = os.path.join(model_dir, "faiss.index")
        doc_ids_path = os.path.join(model_dir, "doc_ids.pkl")

        print("📂 Loading FAISS index from:", index_path)
        if not os.path.exists(index_path) or not os.path.exists(doc_ids_path):
            return jsonify({"error": "FAISS index or doc_ids not found"}), 404

        index = faiss.read_index(index_path)
        print("📌 FAISS index type:", type(index))

        doc_ids = joblib.load(doc_ids_path)
        print("🆔 Loaded", len(doc_ids), "doc IDs")

        # 📄 4. تحميل النصوص الأصلية للوثائق من قاعدة البيانات
        print("🗃️ Fetching documents from database...")
        all_docs = get_documents(dataset_id)
        doc_text_map = {str(doc[0]): doc[1] for doc in all_docs}
        print("📄 Total documents in DB:", len(doc_text_map))

        # 🔍 5. البحث عن أقرب 10 وثائق
        top_k = 10
        print("🔍 Searching for top", top_k, "similar documents...")
        distances, indices = index.search(query_vector, top_k)

        results = []
        for i in range(top_k):
            doc_idx = indices[0][i]
            results.append({
                "doc_id": doc_ids[doc_idx],
                "score": float(distances[0][i]),
                "text": doc_text_map.get(doc_ids[doc_idx], "")
            })

        print("✅ Search complete. Found", len(results), "matches.")
        print("⏱️ Total time:", round(time.time() - start, 2), "sec")

        return jsonify({
            "query_tokens": tokens,
            "top_matches": results
        })

    except Exception as e:
        print("❌ Exception occurred:", str(e))
        return jsonify({"error": str(e)}), 500
