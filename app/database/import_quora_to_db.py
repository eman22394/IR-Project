

import ir_datasets
import mysql.connector

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "ir_project"
}

dataset_name = "beir/quora/dev"

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

cursor.execute("SELECT dataset_id FROM datasets WHERE dataset_name = %s", (dataset_name,))
row = cursor.fetchone()

if row is None:
    cursor.execute("INSERT INTO datasets (dataset_name) VALUES (%s)", (dataset_name,))
    dataset_id = cursor.lastrowid
else:
    dataset_id = row[0]

# # جمل الإدخال مع dataset_id
doc_insert = "INSERT IGNORE INTO documents (doc_id, text, dataset_id) VALUES (%s, %s, %s)"
query_insert = "INSERT IGNORE INTO queries (query_id, text, dataset_id) VALUES (%s, %s, %s)"
qrels_insert = "INSERT INTO qrels (query_id, doc_id, relevance, dataset_id) VALUES (%s, %s, %s, %s)"

# تحميل مجموعة البيانات
dataset = ir_datasets.load(dataset_name)

# 1. تخزين المستندات
print("⏳ جاري تخزين المستندات ...")
for doc in dataset.docs_iter():
    cursor.execute(doc_insert, (doc.doc_id, doc.text, dataset_id))

# 2. تخزين الاستعلامات
print("⏳ جاري تخزين الاستعلامات ...")
for query in dataset.queries_iter():
    cursor.execute(query_insert, (query.query_id, query.text, dataset_id))

try:
    print("⏳ جاري تخزين qrels ...")
    for qrel in dataset.qrels_iter():
        cursor.execute(qrels_insert, (qrel.query_id, qrel.doc_id, qrel.relevance, dataset_id))
except AttributeError:
    print("⚠️ مجموعة البيانات لا تحتوي على qrels.")


conn.commit()
cursor.close()
conn.close()

# print("✅ تم تخزين مجموعة beir/quora باستخدام ir_datasets بنجاح في قاعدة البيانات.")
