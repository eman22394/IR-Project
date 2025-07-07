from flask import Blueprint, request, jsonify
from ..processing import preprocess_text

bp = Blueprint('preprocessing_query', __name__, url_prefix='/preprocess')

@bp.route('/query', methods=['POST'])
def preprocess_text_endpoint():
    try:
        data = request.json
        text = data.get('text')
        options = data.get('options', None)
        
        if not text:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        tokens = preprocess_text(text , options)
        
        return jsonify({
            "tokens": tokens,
            # "clean_text": ' '.join(tokens)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
