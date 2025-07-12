from flask import Flask, Blueprint, request, jsonify
from app.services.query_refinement.offline.query_processor import refine_query

query_refinement_bp = Blueprint('query_refinement', __name__, url_prefix='/query_refinement')

@query_refinement_bp.route('/refine', methods=['POST'])
def refine():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400
    
    input_query = data.get('query', '')
    dataset_id = data.get('dataset_id', None)
    
    if not input_query or dataset_id is None:
        return jsonify({"error": "'query' and 'dataset_id' are required"}), 400
    
    try:
        dataset_id = int(dataset_id)
    except ValueError:
        return jsonify({"error": "'dataset_id' must be an integer"}), 400
    
    try:
        result = refine_query(input_query, dataset_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    if isinstance(result, tuple) and len(result) == 2:
        body, status_code = result
        return jsonify(body), status_code
    
    return jsonify(result)


app = Flask(__name__)
app.register_blueprint(query_refinement_bp)

if __name__ == '__main__':
    app.run(debug=True)
