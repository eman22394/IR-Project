from sklearn.metrics import average_precision_score

def mean_average_precision(qrels, predictions, scores):
    total = 0.0
    count = 0
    for qid, docs in predictions.items():
        rel_docs = qrels.get(qid, {})
        if not rel_docs:
            continue

        y_true = [1 if doc in rel_docs else 0 for doc in docs]
        y_score = scores.get(qid, [1]*len(docs)) 
        if sum(y_true) == 0:
            continue

        try:
            total += average_precision_score(y_true, y_score)
            count += 1
        except Exception as e:
            print(f"⚠️ Error for query {qid}: {e}")

    return total / count if count else 0.0

def mean_reciprocal_rank(qrels, predictions):
    mrr = 0.0
    for qid, docs in predictions.items():
        for rank, doc_id in enumerate(docs, 1):
            if doc_id in qrels.get(qid, []):
                mrr += 1 / rank
                break
    return mrr / len(predictions)

def precision_at_k(qrels, predictions, k):
    total = 0.0
    count = 0
    for qid, docs in predictions.items():
        rel_docs = qrels.get(qid, [])
        if not rel_docs:
            continue
        total += len(set(docs[:k]) & set(rel_docs)) / k
        count += 1
    return total / count if count else 0.0

def recall_at_k(qrels, predictions, k):
    total = 0.0
    count = 0
    for qid, docs in predictions.items():
        rel_docs = qrels.get(qid, [])
        if not rel_docs:
            continue
        total += len(set(docs[:k]) & set(rel_docs)) / len(rel_docs)
        count += 1
    return total / count if count else 0.0
