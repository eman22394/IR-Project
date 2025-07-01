from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import os

def train_word2vec_model(tokenized_corpus, vector_size=300, window=10, min_count=1, sg=1):
    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg
    )
    return model

def save_word2vec_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_word2vec_model(path):
    return joblib.load(path)

def get_mean_vector(model, tokens):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def compute_similarities(query_vector, doc_vectors):
    similarities = cosine_similarity([query_vector], doc_vectors).flatten()
    return similarities
