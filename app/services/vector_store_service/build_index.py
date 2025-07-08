# file: app/routes/faiss_builder.py

from flask import Blueprint, request, jsonify
import os
import joblib
import numpy as np
import faiss

bp = Blueprint('faiss_builder', __name__, url_prefix='/faiss')

@bp.route('/build_index', methods=['POST'])
def build_generic_faiss_index():
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        model_type = data.get('model_type', 'mbert')  # tfidf, word2vec, mbert, hybrid
        table_name = data.get('table_name', 'documents')  # documents / queries
        index_type = data.get('index_type', 'flat_l2')  # flat_l2 / flat_ip / hnsw

        if not dataset_id or not model_type:
            return jsonify({"error": "Missing 'dataset_id' or 'model_type'"}), 400

        # 🗂️ مسارات الملفات
        base_dir = f"data/{model_type}/{table_name}_{dataset_id}"
        doc_vectors_path = os.path.join(base_dir, "hybrid_vectors.pkl")
        print("🔍 Looking for:", doc_vectors_path)
        index_path = os.path.join(base_dir, "faiss.index")
        doc_ids_path = os.path.join(base_dir, "doc_ids.pkl")

        if not os.path.exists(doc_vectors_path):
            return jsonify({"error": f"Vector file not found in {base_dir}"}), 404

        # 📦 تحميل التمثيلات الشعاعية
        doc_vectors = joblib.load(doc_vectors_path)
        doc_ids = list(doc_vectors.keys())

        vectors = []
        for doc_id in doc_ids:
            vec = doc_vectors[doc_id]

            if isinstance(vec, (list, tuple)) and len(vec) == 2:
                part1 = np.array(vec[0], dtype='float32')
                part2 = np.array(vec[1], dtype='float32')
                combined = np.concatenate([part1, part2])

                # تحقق من طول المتجه المدمج
                vectors.append(combined)
            else:
                vectors.append(np.array(vec, dtype='float32'))

        # تحقق من أبعاد كل متجه
        lengths = [v.shape[0] for v in vectors]
        print("Vector lengths:", lengths)

        # إذا كلهم نفس الطول
        if len(set(lengths)) != 1:
            raise ValueError(f"Found vectors with different lengths: {set(lengths)}")

        vectors = np.vstack(vectors)

        print("First 3 vectors samples:")
        for i in range(3):
         print(vectors[i])

        print(f"🔍 First vector shape: {vectors[0].shape}, Total vectors: {len(vectors)}")
        dim = vectors.shape[1]

        # ⚙️ اختيار نوع الفهرس
        if index_type == 'flat_l2':
            index = faiss.IndexFlatL2(dim)
        elif index_type == 'flat_ip':
            index = faiss.IndexFlatIP(dim)
        elif index_type == 'hnsw':
            index = faiss.IndexHNSWFlat(dim, 32)
        else:
            return jsonify({"error": f"Invalid index_type: {index_type}"}), 400

        index.add(vectors)

        # 💾 حفظ الفهرس و doc_ids
        faiss.write_index(index, index_path)
        joblib.dump(doc_ids, doc_ids_path)

        return jsonify({
            "message": f"✅ FAISS index built for model '{model_type}' dataset {dataset_id}",
            "num_vectors": len(doc_ids),
            "index_type": index_type
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
