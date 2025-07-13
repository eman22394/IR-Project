from multiprocessing import Pool, cpu_count
from flask import Blueprint, request, jsonify
from ..processing import preprocess_text
from app.database.connection import get_connection

bp = Blueprint('preprocessing_bulk', __name__, url_prefix='/preprocess')

def process_row(row, id_col, options):
    row_id = row[id_col]
    text = row['text']
    tokens = preprocess_text(text, options)
    return (row_id, tokens)

@bp.route('/bulk', methods=['POST'])
def preprocess_bulk_only():
    try:
        data = request.json
        dataset_id = data.get('dataset_id', 1)
        table_name = data.get('table_name', 'documents')
        limit = data.get('limit', None)
        options = data.get('options', None) 
        if table_name not in ['documents', 'queries']:
            return jsonify({"error": "Invalid table_name"}), 400

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        id_col = 'doc_id' if table_name == 'documents' else 'query_id'
        query = f"SELECT {id_col}, text FROM {table_name} WHERE dataset_id = %s"
        params = [dataset_id]

        if limit:
            query += " LIMIT %s"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        print(f"🚀 Processing {len(rows)} rows (no storage)...")

        with Pool(processes=min(cpu_count(), 2)) as pool:
            results = pool.starmap(
                process_row, [(row, id_col, options) for row in rows]
            )
        processed_data = {str(doc_id): tokens for doc_id, tokens in results}

        cursor.close()
        conn.close()

        return jsonify({
            "processed_data": processed_data,
            "count": len(processed_data)
        })

   

    except Exception as e:
        import traceback
        print("❌ Error during bulk preprocessing:")
        traceback.print_exc()  
        return jsonify({"error": repr(e)}), 500


