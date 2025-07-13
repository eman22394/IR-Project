# app/services/index_builder_service/index_builder.py

import os
import json
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from app.database.connection import get_connection
from app.services.data_pre_processing.processing import preprocess_text


def get_documents(dataset_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, text FROM documents WHERE dataset_id = %s", (dataset_id,))
    docs = cursor.fetchall()
    cursor.close()
    conn.close()
    return docs


def process_document(args):
    doc_id, text = args
    if not isinstance(text, str) or not text.strip():
        return None
    tokens = preprocess_text(text)
    return (doc_id, tokens)


def build_inverted_index_parallel(docs):
    index = defaultdict(lambda: defaultdict(int))
    with Pool(processes=min(cpu_count(), 4)) as pool:
        results = pool.map(process_document, docs)

    for result in results:
        if result is None:
            continue
        doc_id, tokens = result
        for term in tokens:
            index[term][doc_id] += 1

    formatted_index = {
        term: [{"doc_id": doc_id, "tf": tf} for doc_id, tf in postings.items()]
        for term, postings in index.items()
    }
    return formatted_index


def compute_total_term_frequencies(inverted_index):
    return {
        term: sum(posting["tf"] for posting in postings)
        for term, postings in inverted_index.items()
    }


def save_index(index, total_term_freqs, dataset_name, total_docs, timing_log, doc_tokens_map, docs, subdir):
    output_dir = os.path.join("data", "indexes", subdir)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{dataset_name}_inverted_index.json")

    doc_lengths = {doc_id: len(tokens) for doc_id, tokens in doc_tokens_map.items()}
    documents = {doc_id: text for doc_id, text in docs if isinstance(text, str)}

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "inverted_index": index,
            "total_term_frequencies": total_term_freqs,
            "total_documents": total_docs,
            "doc_lengths": doc_lengths,
            "documents": documents,
            "timing": timing_log
        }, f, indent=2, ensure_ascii=False)

    return filepath


def run_index_builder(dataset_id):
    if dataset_id == 1:
        dataset_name = "antique"
        output_subdir = "antique"
    elif dataset_id == 18:
        dataset_name = "quora"
        output_subdir = "quora"
    else:
        return {"error": "❌ dataset_id غير مدعوم"}

    timing = {}
    t0 = time.time()
    docs = get_documents(dataset_id)
    timing["fetch_documents_sec"] = round(time.time() - t0, 2)

    t1 = time.time()
    index = build_inverted_index_parallel(docs)
    timing["build_index_sec"] = round(time.time() - t1, 2)

    t2 = time.time()
    total_term_freqs = compute_total_term_frequencies(index)
    timing["compute_tf_sec"] = round(time.time() - t2, 2)

    doc_tokens_map = defaultdict(list)
    for term, postings in index.items():
        for posting in postings:
            doc_tokens_map[posting["doc_id"]].append(term)

    t4 = time.time()
    filepath = save_index(index, total_term_freqs, dataset_name, len(docs), timing, doc_tokens_map, docs, output_subdir)
    timing["save_sec"] = round(time.time() - t4, 2)

    return {
        "message": "✅ تم بناء الفهرس بنجاح",
        "index_file": filepath,
        "timing": timing,
        "total_documents": len(docs)
    }
