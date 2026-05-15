import math
from collections import Counter

from .preprocess import processDocument


class BM25:
    def __init__(self, processed_docs, k1=2.0, b=0.6, delta=0.0):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.doc_len = {}
        self.doc_freqs = {}
        self.idf = {}
        self.nd = {}
        self.N = len(processed_docs)
        self.avgdl = 0
        self._initialize(processed_docs)

    def _initialize(self, processed_docs):
        total_len = 0
        for docID, tokens in processed_docs.items():
            self.doc_len[docID] = len(tokens)
            total_len += len(tokens)

            frequencies = Counter(tokens)
            self.doc_freqs[docID] = frequencies

            for word in frequencies.keys():
                self.nd[word] = self.nd.get(word, 0) + 1

        self.avgdl = total_len / self.N if self.N > 0 else 0

        for word, freq in self.nd.items():
            self.idf[word] = math.log(((self.N - freq + 0.5) / (freq + 0.5)) + 1.0)

    def get_scores(self, query_tokens, k=20):
        scores = {}
        for docID, freq_dict in self.doc_freqs.items():
            score = 0.0
            doc_len = self.doc_len[docID]
            for q in query_tokens:
                if q not in freq_dict:
                    continue
                f = freq_dict[q]
                idf_val = self.idf.get(q, 0)
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                term_score = idf_val * ((numerator / denominator) + self.delta)
                score += term_score
            if score > 0:
                scores[docID] = score

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:k]


def searchBM25(query_text, bm25_model, stop_words=None, k=20):
    query_tokens = processDocument(query_text, stop_words)
    return bm25_model.get_scores(query_tokens, k=k)
