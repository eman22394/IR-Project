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
spell = SpellChecker()
lemmatizer = WordNetLemmatizer()

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

def normalize_text(text):
    return text.lower().replace('.', '').replace("'", '')

def correct_terms(text):
    if not text:
        return ""
    terms = text.split()
    corrected_terms = []
    for term in terms:
        if term[0].isupper():  
            corrected_terms.append(term)
        else:
            corrected = spell.correction(term)
            corrected_terms.append(corrected if corrected else term)
    return " ".join(corrected_terms)


def process_dates(text):
    matches = datefinder.find_dates(text)
    for match in matches:
        text = text.replace(str(match.date()), match.strftime("%Y-%m-%d"))
    return text


def tokenize(text):
    return word_tokenize(text)

def remove_stopwords(words):
    filtered = [w for w in words if w.lower() not in stop_words and len(w) > 2]
    return filtered


# Reducing words to their root or stem (Stemming).
def stem_words(words):
    return [ps.stem(w) for w in words]
# Words are returned to their correct original dictionary form according to their type.
# "was" → "be"
def lemmatize_words(words):
    pos_tags = pos_tag(words)
    return [lemmatizer.lemmatize(w, get_wordnet_pos(pos)) for w, pos in pos_tags]

def preprocess_text(text, options=None):
    if options is None:
        options = {
            "normalize": True,
            "spell_correction": False,
            "process_dates": False,
            "tokenize": True,
            "remove_stopwords": True,
            "lemmatize": True,
            "stem": True,
        }

    if options.get("spell_correction"):
        text = correct_terms(text)
    
    if options.get("process_dates"):
        text = process_dates(text)

    if options.get("normalize"):
        text = normalize_text(text)

    if options.get("tokenize"):
        tokens = tokenize(text)
    else:
        tokens = text.split()

    if options.get("remove_stopwords"):
        tokens = remove_stopwords(tokens)

    if options.get("lemmatize"):
        tokens = lemmatize_words(tokens)

    if options.get("stem"):
        tokens = stem_words(tokens)

    return [t for t in tokens if t]
