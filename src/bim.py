import math
from collections import defaultdict

from .preprocess import processDocument


def createInvertedIndexForBIM(documents, vocab_index):
    df_dict = defaultdict(set)

    for docID, terms in documents.items():
        for term in set(terms):
            df_dict[term].add(docID)

    inverted_index = defaultdict(dict)
    for term, docIDs in df_dict.items():
        nDoc = len(docIDs)
        u = nDoc / len(documents) if len(documents) > 0 else 0
        p = 0.5

        if u in [0, 1]:
            continue

        w = math.log(p / (1 - p)) - math.log(u / (1 - u))
        posting_list = [(docID, w) for docID in docIDs if term in vocab_index]

        inverted_index[term] = {
            "nDoc": nDoc,
            "postings": posting_list,
        }

    return inverted_index


def computeBIM(query, inverted_index, stop_words=None, k=20):
    query_tokens = processDocument(query, stop_words)
    score = defaultdict(float)

    for term in set(query_tokens):
        if term in inverted_index:
            postings = inverted_index[term]["postings"]
            for docID, w in postings:
                score[docID] += w

    sorted_scores = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:k]


def searchBIM(query, inverted_index_bim, stop_words=None, k=20):
    return computeBIM(query, inverted_index_bim, stop_words, k)
