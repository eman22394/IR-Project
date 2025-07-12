from app.services.data_pre_processing.processing import preprocess_text
from app.database.connection import get_connection
from nltk.corpus import wordnet
from spellchecker import SpellChecker
import json
import os

spell = SpellChecker()

def load_inverted_index(dataset_id):
    dataset_map = {
        1: "antique",
        18: "quora"
    }
    dataset_name = dataset_map.get(dataset_id)
    if not dataset_name:
        return None

    index_path = os.path.join("data", "indexes", dataset_name, f"{dataset_name}_inverted_index.json")
    if not os.path.exists(index_path):
        return None

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    return index_data

def correct_spelling(query):
    corrected_words = [spell.correction(word) for word in query.split()]
    return ' '.join(corrected_words)

# def expand_query_with_wordnet(query_tokens):
#     expanded_terms = set()
#     synonyms_dict = {}

#     for token in query_tokens:
#         synonyms = set()
#         try:
#             for syn in wordnet.synsets(token):
#                 for lemma in syn.lemmas():
#                     clean_lemma = lemma.name().replace("_", " ").lower()
#                     if clean_lemma != token:
#                         # نطبق preprocess_text على كل مرادف، لأنه ممكن يكون عدة كلمات أو بحاجة لتنظيف
#                         processed_lemmas = preprocess_text(clean_lemma)
#                         for pl in processed_lemmas:
#                             synonyms.add(pl)
#         except Exception:
#             pass
        
#         if synonyms:
#             synonyms_dict[token] = list(synonyms)
#             expanded_terms.update(synonyms)
#         else:
#             synonyms_dict[token] = []
#         expanded_terms.add(token)
#     return list(expanded_terms), synonyms_dict
def expand_query_with_wordnet(query_tokens, max_synonyms_per_word=5):
    expanded_terms = set()
    synonyms_dict = {}

    for token in query_tokens:
        synonyms = set()
        try:
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    clean_lemma = lemma.name().replace("_", " ").lower()
                    if clean_lemma != token:
                        processed_lemmas = preprocess_text(clean_lemma)
                        for pl in processed_lemmas:
                            if (
                                pl.isalpha() and               # فقط حروف
                                len(pl) > 2 and                # تجاهل القصيرة
                                pl not in synonyms and         # بدون تكرار
                                pl != token
                            ):
                                synonyms.add(pl)
                            if len(synonyms) >= max_synonyms_per_word:
                                break
                if len(synonyms) >= max_synonyms_per_word:
                    break
        except Exception:
            pass

        if synonyms:
            synonyms_dict[token] = list(synonyms)[:max_synonyms_per_word]
            expanded_terms.update(synonyms_dict[token])
        else:
            synonyms_dict[token] = []

        expanded_terms.add(token)  # أضف الكلمة الأصلية
    return list(expanded_terms), synonyms_dict

def find_documents_for_word(word, inverted_index):
    postings = inverted_index["inverted_index"].get(word)
    if postings is None:
        return []
    if isinstance(postings, list):
        doc_ids = []
        for item in postings:
            if isinstance(item, dict) and "doc_id" in item:
                doc_ids.append(item["doc_id"])
        return doc_ids
    return []

def get_document_content(doc_ids, inverted_index):
    documents = inverted_index.get("documents", {})
    contents = {}
    for doc_id in doc_ids:
        content = documents.get(doc_id, None)
        if content:
            contents[doc_id] = content
    return contents

def get_similar_queries(user_query, dataset_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM queries WHERE dataset_id = %s", (dataset_id,))
    all_queries = cursor.fetchall()
    conn.close()

    similar = []
    for q in all_queries:
        q_text = q[0]
        if user_query.lower() in q_text.lower() or q_text.lower() in user_query.lower():
            similar.append(q_text)
    return similar[:5]

def refine_query(input_query, dataset_id):
    if not input_query or not dataset_id:
        return {"error": "Both 'query' and 'dataset_id' are required"}, 400

    inverted_index = load_inverted_index(dataset_id)
    if not inverted_index:
        return {"error": "Inverted index not found for the given dataset_id"}, 404

    # تصحيح الإملاء وتنظيف الاستعلام الأصلي
    corrected_query = correct_spelling(input_query)
    corrected_tokens = preprocess_text(corrected_query)

    # توسيع الاستعلام بالمرادفات بعد تنظيفها
    expanded_terms, synonyms_dict = expand_query_with_wordnet(corrected_tokens)

    # فلترة الكلمات الموسعة بحيث نبحث فقط عن الكلمات الموجودة في الفهرس
    valid_terms = [term for term in expanded_terms if term in inverted_index["inverted_index"]]

    similar_queries = get_similar_queries(input_query, dataset_id)

    occurrences_in_documents = {}
    for token in valid_terms:
        doc_ids = find_documents_for_word(token, inverted_index)
        doc_contents = get_document_content(doc_ids, inverted_index)
        occurrences_in_documents[token] = doc_contents

    # Debug prints (يمكن تعطيلها لاحقًا)
    print("Corrected tokens:", corrected_tokens)
    print("Expanded terms:", expanded_terms)
    print("Valid search terms in index:", valid_terms)

    return {
        "input_query": input_query,
        "corrected_query": corrected_query,
        "expanded_query_terms": expanded_terms,
        "synonyms": synonyms_dict,
        "similar_queries": similar_queries,
        "occurrences_in_documents": occurrences_in_documents
    }
