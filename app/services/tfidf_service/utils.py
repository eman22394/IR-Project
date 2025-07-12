from sklearn.feature_extraction.text import TfidfVectorizer

def convert_str(corpus):
    str_corpus = {}
    for doc_id, value in corpus.items():
        if isinstance(value, list):
            str_value = " ".join(value)
        else:  # raw text (str)
            str_value = value
        str_corpus[doc_id] = str_value
        # print(str_corpus)
    return str_corpus


def calculate_tfidf(corpus):
    corpus = convert_str(corpus)
    vectorizer = TfidfVectorizer(
    tokenizer=None,      
    preprocessor=None,           
    lowercase=False,             
    stop_words=None 
    )
    documents = list(corpus.values())
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer


def calculate_query_tfidf(corpus, vectorizer):
    corpus = convert_str(corpus)
    queries = list(corpus.values())
    tfidf_matrix = vectorizer.transform(queries)  
    return tfidf_matrix
