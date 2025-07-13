# app/main.py
from flask import Flask
from app.services.data_pre_processing.offline.preprocess_bulk import bp as preprocessing_bp
from app.services.data_pre_processing.offline.process_store_db import bp as preprocessing_db
from app.services.data_pre_processing.online.preprocess_query import bp as preprocessing_text
from app.services.tfidf_service.offline.build_models import bp as tfidf_documents
from app.services.tfidf_service.offline.tfidf_eval import bp as tfidf_eval
from app.services.tfidf_service.online.match_user_query import bp as match_user_query
from app.services.embedding_service.offline.build_models import bp as word2vec_bp
from app.services.embedding_service.online.match_user_query import bp as word2vec_match_user_query
from app.services.embedding_service.offline.bert_eval import bp as word2vec_eval
from app.database.datasets_list import bp as dataset_list_bp
from app.services.hybrid_service.offline.build_models import bp as hybrid_builder
from app.services.hybrid_service.offline.hybrid_eval import bp as hybrid_eval
from app.services.hybrid_service.offline.emb import bp as emb
from app.services.hybrid_service.online.match_user_query import bp as hybrid_query
from app.services.Multilingual_retrieval_system.offline.multilingual_build import bp as multilingual
from app.services.Multilingual_retrieval_system.online.match_user_query import bp as multilingual_query
from app.services.Multilingual_retrieval_system.offline.mbert_eval import bp as mbert_eval
from app.services.vector_store_service.build_index  import bp as build_index
from app.services.vector_store_service.match_user_query  import bp as match_user_query_use_vector
from app.services.index_builder.online.load_index import index_builder_bp
from app.services.bm25_service.online.match_user_query import bp as bm25_bp
from app.services.query_refinement.online.endpoints import query_refinement_bp

app = Flask(__name__, static_folder="../templates", static_url_path="")
@app.route("/")
def index():
    return app.send_static_file("index.html")
app.register_blueprint(build_index)
app.register_blueprint(match_user_query_use_vector)
app.register_blueprint(multilingual)
app.register_blueprint(multilingual_query)
app.register_blueprint(mbert_eval)
app.register_blueprint(hybrid_eval)
app.register_blueprint(hybrid_query)
app.register_blueprint(emb)
app.register_blueprint(word2vec_eval)
app.register_blueprint(tfidf_eval)
app.register_blueprint(dataset_list_bp)
app.register_blueprint(preprocessing_db)
app.register_blueprint(preprocessing_bp)
app.register_blueprint(tfidf_documents)
app.register_blueprint(preprocessing_text)
app.register_blueprint(match_user_query)
app.register_blueprint(word2vec_bp)
app.register_blueprint(word2vec_match_user_query)
app.register_blueprint(hybrid_builder)
app.register_blueprint(index_builder_bp)
app.register_blueprint(bm25_bp)
app.register_blueprint(query_refinement_bp)


if __name__ == '__main__':
    app.run(debug=True)
