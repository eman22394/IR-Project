from flask import Blueprint, request, jsonify
from sentence_transformers import SentenceTransformer

bp = Blueprint('embed', __name__, url_prefix='/embed')
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@bp.route('/query', methods=['POST'])
def embed_query():
    data = request.json
    tokens = data.get("tokens", [])
    if not tokens:
        return jsonify({"error": "No tokens"}), 400

    query_text = " ".join(tokens)
    vector = model.encode(query_text, convert_to_numpy=True).tolist()
    return jsonify({"vector": vector})
