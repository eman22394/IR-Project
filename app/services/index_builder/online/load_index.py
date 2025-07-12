# app/services/index_builder_service/main.py
from flask import Blueprint, request, jsonify
from app.services.index_builder.offline.build_index import run_index_builder

index_builder_bp = Blueprint("index_builder", __name__)

@index_builder_bp.route("/build-index", methods=["POST"])
def build_index():
    data = request.get_json()
    dataset_id = data.get("dataset_id")

    if dataset_id not in [1, 2]:
        return jsonify({"error": "❌ dataset_id غير مدعوم. فقط 1 (antique) أو 18 (quora)"}), 400

    result = run_index_builder(dataset_id)
    return jsonify(result)
