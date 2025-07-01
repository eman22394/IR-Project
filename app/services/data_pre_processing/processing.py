# app/preprocessing_service/processing.py

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from spellchecker import SpellChecker
import spacy
import datefinder

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
# To correct spelling
spell = SpellChecker()
# Convert words to their original form
lemmatizer = WordNetLemmatizer()
# For text processing (to recognize entities such as dates)
# example:
# text = "Barack Obama was born on August 4, 1961 in Honolulu, Hawaii."
# Output: Barack Obama → PERSON
# August 4, 1961 → DATE
# Honolulu → GPE
# Hawaii → GPE
# sp = spacy.load('en_core_web_sm') 

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# "I'm Happy." → "im happy"
def normalize_text(text):
    return text.lower().replace('.', '').replace("'", '')

def correct_terms(text):
    if not text:
        return ""
    terms = text.split()
    corrected_terms = []
    for term in terms:
        if term[0].isupper():  # نحافظ على الأسماء الكبيرة مثل Sherlock، Holms
            corrected_terms.append(term)
        else:
            corrected = spell.correction(term)
            corrected_terms.append(corrected if corrected else term)
    return " ".join(corrected_terms)


def process_dates(text):
    matches = datefinder.find_dates(text)
    for match in matches:
        # Replace the original date with the ISO 8601 formatted date
        text = text.replace(str(match.date()), match.strftime("%Y-%m-%d"))
    return text


def tokenize(text):
    return word_tokenize(text)

def remove_stopwords(words):
    filtered = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 2]
    return filtered

# Reducing words to their root or stem (Stemming).
def stem_words(words):
    return [ps.stem(w) for w in words]
# Words are returned to their correct original dictionary form according to their type.
# "was" → "be"
def lemmatize_words(words):
    pos_tags = pos_tag(words)
    return [lemmatizer.lemmatize(w, get_wordnet_pos(pos)) for w, pos in pos_tags]

def preprocess_text(text):
    # text = correct_terms(text)
    text = normalize_text(text)
    # text = process_dates(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_words(tokens)
    tokens = stem_words(tokens)
    tokens = [t for t in tokens if t is not None]
    return tokens
