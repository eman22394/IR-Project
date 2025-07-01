from app.database.connection import get_connection

def get_documents(dataset_id):
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT doc_id, text FROM documents WHERE dataset_id = %s"
    cursor.execute(query, (dataset_id,))
    docs = cursor.fetchall()
    cursor.close()
    conn.close()
    return docs

def get_queries(dataset_id):
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT query_id, text FROM queries WHERE dataset_id = %s"
    cursor.execute(query, (dataset_id,))
    queries = cursor.fetchall()
    cursor.close()
    conn.close()
    return queries

def get_qrels(dataset_id):
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT query_id, doc_id, relevance FROM qrels WHERE dataset_id = %s"
    cursor.execute(query, (dataset_id,))
    qrels = cursor.fetchall()
    cursor.close()
    conn.close()
    return qrels

def get_queries_from_qrels(dataset_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT q.query_id, q.text
        FROM queries q
        JOIN qrels r ON q.query_id = r.query_id
        WHERE q.dataset_id = %s
    """, (dataset_id,))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results  # [(query_id, text), ...]
