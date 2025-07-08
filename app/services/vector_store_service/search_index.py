# file: app/services/mbert_faiss/search_faiss_index.py
import faiss
import joblib
import numpy as np
import os

def search_faiss(query_vector, index_path, top_k=10):
    index = faiss.read_index(os.path.join(index_path, "faiss.index"))
    doc_ids = joblib.load(os.path.join(index_path, "doc_ids.pkl"))

    query_vector = np.array(query_vector).astype('float32').reshape(1, -1)

    distances, indices = index.search(query_vector, top_k)

    results = []
    for i in range(top_k):
        results.append({
            "doc_id": doc_ids[indices[0][i]],
            "distance": float(distances[0][i])
        })

    return results
