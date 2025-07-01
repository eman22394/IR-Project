from flask import Blueprint, jsonify
from app.database.connection import get_connection

bp = Blueprint('datasets', __name__, url_prefix='/datasets')

@bp.route('', methods=['GET'])
def list_datasets():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM datasets")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        return jsonify({"datasets": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
