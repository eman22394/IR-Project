from multiprocessing import Pool, cpu_count
from flask import Blueprint, request, jsonify
from ..processing import preprocess_text
from app.database.connection import get_connection

bp = Blueprint('preprocessing', __name__, url_prefix='/preprocess')

def process_row(row, id_col):
    row_id = row[id_col]
    text = row['text']
    tokens = preprocess_text(text)
    return (' '.join(tokens), row_id)

def chunkify(lst, size):
    """يقسم القائمة إلى دفعات بحجم ثابت"""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

@bp.route('/db-parallel', methods=['POST'])
def preprocess_db_parallel():
    try:
        data = request.json
        dataset_id = data.get('dataset_id', 1)
        table_name = data.get('table_name', 'documents')

        if table_name not in ['documents', 'queries']:
            return jsonify({"error": "Invalid table_name"}), 400

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        id_col = 'doc_id' if table_name == 'documents' else 'query_id'

        query = f"SELECT {id_col}, text FROM {table_name} WHERE dataset_id = %s"
        cursor.execute(query, (dataset_id,))
        rows = cursor.fetchall()

        print(f"🚀 Parallel processing {len(rows)} rows in chunks of 100...")

        all_results = []
        chunk_size = 100
        total = len(rows)

        for idx, chunk in enumerate(chunkify(rows, chunk_size), start=1):
            with Pool(processes=min(cpu_count(), 8)) as pool:
                chunk_results = pool.starmap(process_row, [(row, id_col) for row in chunk])
                all_results.extend(chunk_results)

            print(f"✅ Processed {min(idx * chunk_size, total)} / {total}")

        cursor.executemany(
            f"UPDATE {table_name} SET processed_text=%s WHERE {id_col}=%s",
            all_results
        )
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "message": f"✅ All {len(all_results)} rows processed with progress tracking.",
            "sample": all_results[:100]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
