import math
from collections import defaultdict

import numpy as np

from .preprocess import processDocument


def computeQueryVector(query, vocab_index, stop_words=None):
    query_tokens = processDocument(query, stop_words)
    query_vector = np.zeros(len(vocab_index))

    for token in query_tokens:
        if token in vocab_index:
            query_vector[vocab_index[token]] += 1

    norm = len(query_tokens)
    if norm > 0:
        query_vector /= norm
    return query_vector


def computeTF(tokens, vocab_index):
    tf_vector = np.zeros(len(vocab_index))
    for token in tokens:
        if token in vocab_index:
            tf_vector[vocab_index[token]] += 1
    n = len(tokens)
    if n > 0:
        tf_vector = tf_vector / n
    return tf_vector


def computeGlobalDF(documents, vocab_index):
    df_vector = np.zeros(len(vocab_index))
    for tokens in documents.values():
        for token in set(tokens):
            if token in vocab_index:
                df_vector[vocab_index[token]] += 1
    return df_vector


def computeTFIDF(documents, vocab_index):
    n_docs = len(documents)
    tfidf_matrix = np.zeros((n_docs, len(vocab_index)))
    global_df = computeGlobalDF(documents, vocab_index)
    vocab = sorted(vocab_index, key=vocab_index.get)

    for docID, tokens in documents.items():
        tf_vector = computeTF(tokens, vocab_index)
        for i, _term in enumerate(vocab):
            if tf_vector[i] > 0 and global_df[i] > 0:
                idf = math.log(n_docs / global_df[i])
                tfidf_matrix[docID - 1][i] = round(tf_vector[i] * idf, 6)

    return tfidf_matrix


def getCandidateDocuments(query, inverted_index, stop_words=None):
    query_tokens = processDocument(query, stop_words)
    candidate_documents = set()

    for term in query_tokens:
        if term in inverted_index:
            candidate_documents.update([docID for docID, _ in inverted_index[term]["postings"]])
    return candidate_documents


def cosineSimilarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (norm1 * norm2) if norm1 and norm2 else 0.0


def createInvertedIndexForVS(tfidf_matrix, process_documents, vocab_index):
    inverted_index = defaultdict(dict)
    df_dict = defaultdict(set)

    for docID, terms in process_documents.items():
        for term in set(terms):
            df_dict[term].add(docID)

    for term, docIDs in df_dict.items():
        term_index = vocab_index.get(term)
        if term_index is not None:
            posting_list = []
            for docID in docIDs:
                tfidf_value = tfidf_matrix[docID - 1][term_index]
                if tfidf_value > 0:
                    posting_list.append((docID, tfidf_value))

            inverted_index[term] = {
                "nDoc": len(docIDs),
                "postings": posting_list,
            }

    return inverted_index


def searchVectorSpace(query, tfidf_matrix, inverted_index, vocab_index, stop_words=None, k=20):
    query_vector = computeQueryVector(query, vocab_index, stop_words)
    candidate_documents = getCandidateDocuments(query, inverted_index, stop_words)

    # NTC Query weighting: multiply query TF by IDF
    N = tfidf_matrix.shape[0]
    query_tokens = processDocument(query, stop_words)
    for term in set(query_tokens):
        if term in inverted_index:
            term_idx = vocab_index[term]
            df = inverted_index[term]["nDoc"]
            idf = math.log(N / df)
            query_vector[term_idx] *= idf

    similarities = []
    for docID in candidate_documents:
        doc_vector = tfidf_matrix[docID - 1]
        similarity = cosineSimilarity(query_vector, doc_vector)
        similarities.append((docID, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]
