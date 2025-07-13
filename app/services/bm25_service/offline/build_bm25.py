import os
import json
import math
import heapq
from app.services.data_pre_processing.processing import preprocess_text

def load_inverted_index(dataset_id):
    dataset_map = {
        1: "antique",
        18: "quora"
    }
    dataset_name = dataset_map.get(dataset_id)
    if not dataset_name:
        return None

    index_path = os.path.join("data", "indexes", dataset_name, f"{dataset_name}_inverted_index.json")
    if not os.path.exists(index_path):
        return None

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    return index_data

def compute_bm25(query, index_data, top_k=None, k1=1.5, b=0.75):
    inverted_index = index_data["inverted_index"]
    total_docs = index_data["total_documents"]
    doc_lengths = index_data["doc_lengths"]
    documents = index_data["documents"]

    avgdl = sum(doc_lengths.values()) / total_docs if total_docs > 0 else 1

    query_terms = preprocess_text(query)
    scores = {}

    for term in query_terms:
        if term not in inverted_index:
            continue

        postings = inverted_index[term]
        df = len(postings)

        # تجاهل الكلمات الشائعة جدًا
        if df / total_docs > 0.9:
            continue

        idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))

        for entry in postings:
            doc_id = entry["doc_id"]
            tf = entry["tf"]
            dl = doc_lengths.get(str(doc_id), 1)

            score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl)))
            scores[doc_id] = scores.get(doc_id, 0) + score

    # ترتيب النتائج
    if top_k:
        top_results = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
    else:
        top_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = [{
        "doc_id": doc_id,
        "document": documents.get(str(doc_id), ""),
        "score": round(score, 4)
    } for doc_id, score in top_results]

    return results

# import os
# import json
# import math
# from app.services.data_pre_processing.processing import preprocess_text

# def load_inverted_index(dataset_id):
#     dataset_map = {
#         1: "antique",
#         18: "quora"
#     }
#     dataset_name = dataset_map.get(dataset_id)
#     if not dataset_name:
#         return None

#     # مسار ملف الفهرس مع اسم dataset الصحيح
#     index_path = os.path.join("data", "indexes", dataset_name, f"{dataset_name}_inverted_index.json")
#     if not os.path.exists(index_path):
#         return None

#     with open(index_path, "r", encoding="utf-8") as f:
#         index_data = json.load(f)
#     return index_data

# def compute_bm25(query, index_data, top_k=20):
#     k1 = 1.5
#     b = 0.75

#     inverted_index = index_data["inverted_index"]
#     total_docs = index_data["total_documents"]
#     doc_lengths = index_data["doc_lengths"]
#     documents = index_data["documents"]

#     avgdl = sum(doc_lengths.values()) / total_docs if total_docs > 0 else 1

#     query_terms = preprocess_text(query)
#     scores = {}

#     for term in query_terms:
#         if term not in inverted_index:
#             continue

#         postings = inverted_index[term]
#         df = len(postings)
#         if df == 0:
#             continue

#         idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))

#         for entry in postings:
#             doc_id = entry["doc_id"]
#             tf = entry["tf"]
#             dl = doc_lengths.get(str(doc_id), 1)

#             score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl)))
#             scores[doc_id] = scores.get(doc_id, 0) + score

#     sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

#     results = [{
#         "doc_id": doc_id,
#         "document": documents.get(str(doc_id), ""),
#         "score": round(score, 4)
#     } for doc_id, score in sorted_results[:top_k]]

#     return results
