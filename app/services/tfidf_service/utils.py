# tfidf_service/utils.py
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
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),  # استخدام n-grams
    max_df=0.85,         # تجاهل الكلمات الشائعة جدًا
    min_df=2             # تجاهل الكلمات النادرة
)

    # TfidfVectorizer(
    # tokenizer=None,      
    # preprocessor=None,           
    # lowercase=False,             
    # stop_words=None 
    # )
    documents = list(corpus.values())
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer


def calculate_query_tfidf(corpus, vectorizer):
    corpus = convert_str(corpus)
    queries = list(corpus.values())
    tfidf_matrix = vectorizer.transform(queries)  # تحويل فقط
    return tfidf_matrix
