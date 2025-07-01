# app/main.py
from flask import Flask
from app.services.data_pre_processing.offline.preprocess_bulk import bp as preprocessing_bp
from app.services.data_pre_processing.offline.process_store_db import bp as preprocessing_db
from app.services.data_pre_processing.online.preprocess_query import bp as preprocessing_text
from app.services.tfidf_service.offline.build_documents import bp as tfidf_documents
from app.services.tfidf_service.offline.build_queries import bp as tfidf_query
from app.services.tfidf_service.offline.match_queries_to_docs import bp as tfidf_eval
from app.services.tfidf_service.online.match_user_query import bp as match_user_query
from app.services.embedding_service.offline.build_documents import bp as word2vec_bp
from app.services.embedding_service.online.match_user_query import bp as word2vec_match_user_query
from app.services.embedding_service.offline.match_all_queries import bp as word2vec_match_all_query
from app.services.embedding_service.offline.eval import bp as word2vec_eval
from app.services.hybrid_service.endpoints import bp as hybrid
from app.database.datasets_list import bp as dataset_list_bp
from app.services.hybrid_service.offline.hybrid_vector_builder import bp as hybrid_builder
from app.evaluation.endpoints import bp as bp_evaluation

app = Flask(__name__, static_folder="../templates", static_url_path="")

# واجهة المستخدم
@app.route("/")
def index():
    return app.send_static_file("index.html")


# تسجيل جميع الـ APIs
app.register_blueprint(bp_evaluation)
app.register_blueprint(word2vec_eval)
app.register_blueprint(tfidf_eval)
app.register_blueprint(dataset_list_bp)
app.register_blueprint(hybrid)
app.register_blueprint(preprocessing_db)
app.register_blueprint(preprocessing_bp)
app.register_blueprint(tfidf_documents)
app.register_blueprint(tfidf_query)
app.register_blueprint(preprocessing_text)
app.register_blueprint(match_user_query)
app.register_blueprint(word2vec_bp)
app.register_blueprint(word2vec_match_user_query)
app.register_blueprint(word2vec_match_all_query)
app.register_blueprint(hybrid_builder)


if __name__ == '__main__':
    app.run(debug=True)
