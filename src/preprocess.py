import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from num2words import num2words


CUSTOM_STOPWORDS = {"ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii"}
ABBREVIATIONS = {
    r"\bfig\.?\b": "figure",
    r"\bref\.?\b": "reference",
    r"\bapprox\.?\b": "approximately",
    r"\beq\.?\b": "equation",
    r"\bsq\.?\b": "square",
    r"\bno\.?\b": "number",
    r"\be\.g\.?\b": "for example",
    r"\bi\.e\.?\b": "that is",
    r"\bsec\.?\b": "section",
}


def buildStopWords(custom_stopwords=None):
    words = set(stopwords.words("english"))
    words.update(CUSTOM_STOPWORDS)
    if custom_stopwords:
        words.update(custom_stopwords)
    return words


def replaceNumber(match):
    try:
        return num2words(float(match.group())).replace("-", " ").replace(",", "")
    except Exception:
        return match.group()


def processDocument(document, stop_words=None, stemmer=None):
    if stop_words is None:
        stop_words = buildStopWords()
    if stemmer is None:
        stemmer = SnowballStemmer("english")

    expanded_text = document.lower().replace("-", " ")
    for pattern, replacement in ABBREVIATIONS.items():
        expanded_text = re.sub(pattern, replacement, expanded_text)

    text_with_words = re.sub(r"\b\d+(?:\.\d+)?\b", replaceNumber, expanded_text)
    text = re.sub(r"[^a-z0-9\s]", "", text_with_words)
    tokens = word_tokenize(text)
    tokens = [
        word
        for word in tokens
        if not (len(word) <= 2 and word.isalpha()) and word not in stop_words
    ]
    return [stemmer.stem(token) for token in tokens]


def processDocuments(docID_list, documents, stop_words=None):
    if stop_words is None:
        stop_words = buildStopWords()
    stemmer = SnowballStemmer("english")
    all_tokens = []
    processed_documents = {}

    for docID in docID_list:
        tokens = processDocument(documents[docID], stop_words, stemmer)
        processed_documents[docID] = tokens
        all_tokens.extend(tokens)

    return processed_documents, all_tokens


def buildVocabulary(process_documents, docID_list=None):
    all_terms = set()
    iterable = docID_list if docID_list is not None else process_documents.keys()
    for docID in iterable:
        all_terms.update(process_documents[docID])
    vocab = sorted(all_terms)
    vocab_index = {term: i for i, term in enumerate(vocab)}
    return vocab, vocab_index
