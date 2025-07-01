from flask import Blueprint, request, jsonify
import os
import joblib
import requests
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

bp = Blueprint('bert_documents', __name__, url_prefix='/bert')

@bp.route('/build', methods=['POST'])
def build_bert():
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        table_name = data.get('table_name', 'documents')

        if not dataset_id or table_name not in ['documents', 'queries']:
            return jsonify({"error": "Missing or invalid 'dataset_id' or 'table_name'"}), 400

        # 🔁 استدعاء خدمة المعالجة المسبقة
        preprocess_url = "http://127.0.0.1:5000/preprocess/bulk"
        response = requests.post(preprocess_url, json={
            "dataset_id": dataset_id,
            "table_name": table_name
        })

        if response.status_code != 200:
            return jsonify({"error": "Failed to preprocess data", "details": response.text}), 500

        result = response.json()
        processed_data = result["processed_data"]  # {doc_id: [tokens], ...}

        if not processed_data:
            return jsonify({"error": "No processed data returned"}), 404

        total_docs = len(processed_data)
        print(f"🚀 Starting BERT encoding for {total_docs} {table_name} in dataset {dataset_id}")

        # 🧠 تحميل نموذج BERT مخصص للاسترجاع
        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

        # ⚙️ إعداد التوجيهات وفلترة الكلمات
        stop_words = set(stopwords.words('english'))
        prefix = "query: " if table_name == "queries" else "passage: "

        doc_vectors = {}
        for idx, (doc_id, tokens) in enumerate(processed_data.items(), 1):
            filtered_tokens = [t for t in tokens if t.lower() not in stop_words]
            text = prefix + " ".join(filtered_tokens)
            doc_vectors[doc_id] = model.encode(text, convert_to_numpy=True)

            if idx % 1000 == 0:
                print(f"✅ Encoded {idx} / {total_docs} {table_name}")

        print(f"🎉 Finished encoding all {total_docs} {table_name}")

        # 💾 حفظ التمثيلات
        model_dir = f"data/bert/{table_name}_{dataset_id}"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(doc_vectors, os.path.join(model_dir, "doc_vectors.pkl"))
        model.save(os.path.join(model_dir, "model"))

        return jsonify({
            "message": f"✅ BERT vectors built for {table_name} of dataset {dataset_id}",
            "num_documents": total_docs
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

# from flask import Blueprint, request, jsonify
# import os
# import joblib
# import requests
# from sentence_transformers import SentenceTransformer

# bp = Blueprint('bert_documents', __name__, url_prefix='/bert')

# @bp.route('/build', methods=['POST'])
# def build_bert():
#     try:
#         data = request.json
#         dataset_id = data.get('dataset_id')
#         table_name = data.get('table_name', 'documents')

#         if not dataset_id or table_name not in ['documents', 'queries']:
#             return jsonify({"error": "Missing or invalid 'dataset_id' or 'table_name'"}), 400

#         # 🔁 استدعاء خدمة المعالجة المسبقة
#         preprocess_url = "http://127.0.0.1:5000/preprocess/bulk"
#         response = requests.post(preprocess_url, json={
#             "dataset_id": dataset_id,
#             "table_name": table_name
#         })

#         if response.status_code != 200:
#             return jsonify({"error": "Failed to preprocess data", "details": response.text}), 500

#         result = response.json()
#         processed_data = result["processed_data"]  # {doc_id: [tokens], ...}

#         if not processed_data:
#             return jsonify({"error": "No processed data returned"}), 404

#         total_docs = len(processed_data)
#         print(f"🚀 Starting BERT encoding for {total_docs} {table_name} in dataset {dataset_id}")

#         # 🧠 تحميل نموذج BERT
#         model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

#         doc_vectors = {}
#         for idx, (doc_id, tokens) in enumerate(processed_data.items(), 1):
#             prefix = "query: " if table_name == "queries" else "passage: "
#             text = prefix + " ".join(tokens)
#             doc_vectors[doc_id] = model.encode(text, convert_to_numpy=True)

#             if idx % 1000 == 0:
#                 print(f"✅ Encoded {idx} / {total_docs} {table_name}")

#         print(f"🎉 Finished encoding all {total_docs} {table_name}")

#         # 💾 حفظ التمثيلات
#         model_dir = f"data/bert/{table_name}_{dataset_id}"
#         os.makedirs(model_dir, exist_ok=True)
#         joblib.dump(doc_vectors, os.path.join(model_dir, "doc_vectors.pkl"))

#         return jsonify({
#             "message": f"✅ BERT vectors built for {table_name} of dataset {dataset_id}",
#             "num_documents": total_docs
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# from flask import Blueprint, request, jsonify
# import os
# import joblib
# import requests
# from app.services.embedding_service.utils import train_word2vec_model, save_word2vec_model, get_mean_vector
 
 
# bp = Blueprint('word2vec_documents', __name__, url_prefix='/word2vec')

# @bp.route('/build', methods=['POST'])
# def build_word2vec():
#     try:
#         data = request.json
#         dataset_id = data.get('dataset_id')
#         table_name = data.get('table_name', 'documents')

#         if not dataset_id or table_name not in ['documents', 'queries']:
#             return jsonify({"error": "Missing or invalid 'dataset_id' or 'table_name'"}), 400

#         # 🔁 استدعاء سيرفس المعالجة المسبقة (bulk)
#         preprocess_url = "http://127.0.0.1:5000/preprocess/bulk"
#         response = requests.post(preprocess_url, json={
#             "dataset_id": dataset_id,
#             "table_name": table_name
#         })

#         if response.status_code != 200:
#             return jsonify({"error": "Failed to preprocess data", "details": response.text}), 500

#         result = response.json()
#         processed_data = result["processed_data"]  # {doc_id: [tokens], ...}

#         if not processed_data:
#             return jsonify({"error": "No processed data returned"}), 404

#         tokenized_corpus = list(processed_data.values())
#         model = train_word2vec_model(tokenized_corpus)

#         # ✅ بناء تمثيل متوسط لكل مستند
#         doc_vectors = {
#             doc_id: get_mean_vector(model, tokens)
#             for doc_id, tokens in processed_data.items()
#         }

#         # 💾 الحفظ
#         model_dir = f"data/word2vec/{table_name}_{dataset_id}"
#         os.makedirs(model_dir, exist_ok=True)

#         save_word2vec_model(model, os.path.join(model_dir, "model.pkl"))
#         joblib.dump(doc_vectors, os.path.join(model_dir, "doc_vectors.pkl"))

#         return jsonify({
#             "message": f"✅ Word2Vec model built for {table_name} of dataset {dataset_id}",
#             "num_documents": len(processed_data)
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
