import mysql.connector
import os

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "ir_project"
}

BASE_DIR = os.path.expanduser("C:/Users/Classic/.ir_datasets/antique")
COLLECTION_PATH = os.path.join(BASE_DIR, "collection.tsv")
QUERIES_PATH = os.path.join(BASE_DIR, "test", "queries.txt")
QRELS_PATH = os.path.join(BASE_DIR, "test", "qrels")

dataset_name = "antique"

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()


cursor.execute("SELECT dataset_id FROM datasets WHERE dataset_name = %s", (dataset_name,))
row = cursor.fetchone()

if row is None:
    cursor.execute("INSERT INTO datasets (dataset_name) VALUES (%s)", (dataset_name,))
    dataset_id = cursor.lastrowid
else:
    dataset_id = row[0]

with open(COLLECTION_PATH, encoding="utf-8") as f:
    for line in f:
        doc_id, text = line.strip().split("\t", 1)
        cursor.execute(
            "INSERT IGNORE INTO documents (doc_id, text, dataset_id) VALUES (%s, %s, %s)",
            (doc_id, text, dataset_id)
        )

        count = 0
for doc in dataset.docs_iter():
    count += 1
print("عدد المستندات في الداتا سيت:", count)


with open(QUERIES_PATH, encoding="utf-8") as f:
    for line in f:
        query_id, text = line.strip().split("\t", 1)
        cursor.execute(
            "INSERT IGNORE INTO queries (query_id, text, dataset_id) VALUES (%s, %s, %s)",
            (query_id, text, dataset_id)
        )

with open(QRELS_PATH, encoding="utf-8") as f:
    for line in f:
        query_id, _, doc_id, relevance = line.strip().split()
        cursor.execute(
            "INSERT INTO qrels (query_id, doc_id, relevance, dataset_id) VALUES (%s, %s, %s, %s)",
            (query_id, doc_id, int(relevance), dataset_id)
        )

conn.commit()
cursor.close()
conn.close()
