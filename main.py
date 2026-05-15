from pathlib import Path
import sys



ROOT = Path(__file__).resolve().parent
CRANFIELD_PATH = ROOT / "Cranfield"
TEST_PATH = ROOT / "TEST"
RESULTS_PATH = TEST_PATH / "RES"
QUERY_FILE = TEST_PATH / "query.txt"


def ensure_nltk_data():
    import nltk

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)


def print_metrics(name, metrics):
    print(f"\n{name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


def main():
    try:
        from src.bim import createInvertedIndexForBIM, searchBIM
        from src.bm25 import BM25, searchBM25
        from src.data_loader import loadDocuments
        from src.evaluator import evaluateModel
        from src.preprocess import buildStopWords, buildVocabulary, processDocuments
        from src.query_io import loadQueries, processTest
        from src.vector_space import computeTFIDF, createInvertedIndexForVS, searchVectorSpace
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}")
        print("Please run this with the cs419 environment, for example:")
        print("  conda activate cs419")
        print("  python main.py")
        return 1

    ensure_nltk_data()

    doc_ids, documents = loadDocuments(CRANFIELD_PATH)
    stop_words = buildStopWords()
    processed_docs, all_tokens = processDocuments(doc_ids, documents, stop_words)
    vocab, vocab_index = buildVocabulary(processed_docs, doc_ids)

    print("Dataset")
    print(f"Documents: {len(doc_ids)}")
    print(f"Tokens: {len(all_tokens)}")
    print(f"Vocabulary: {len(vocab)}")

    query_rels = processTest(RESULTS_PATH)
    queries = loadQueries(QUERY_FILE)
    print(f"Queries: {len(queries)}")
    print(f"Relevance files: {len(query_rels)}")

    tfidf_matrix = computeTFIDF(processed_docs, vocab_index)
    vector_index = createInvertedIndexForVS(tfidf_matrix, processed_docs, vocab_index)
    vector_results = {}
    for query_id, query_text in queries.items():
        results = searchVectorSpace(
            query_text,
            tfidf_matrix,
            vector_index,
            vocab_index,
            stop_words,
            k=20,
        )
        vector_results[query_id] = [doc_id for doc_id, _score in results]
    print_metrics("Vector Space Model", evaluateModel(vector_results, query_rels, k=10, verbose=False))

    bim_index = createInvertedIndexForBIM(processed_docs, vocab_index)
    bim_results = {}
    for query_id, query_text in queries.items():
        results = searchBIM(query_text, bim_index, stop_words, k=20)
        bim_results[query_id] = [doc_id for doc_id, _score in results]
    print_metrics("Binary Independence Model", evaluateModel(bim_results, query_rels, k=10, verbose=False))

    bm25_model = BM25(processed_docs)
    bm25_results = {}
    for query_id, query_text in queries.items():
        results = searchBM25(query_text, bm25_model, stop_words, k=20)
        bm25_results[query_id] = [doc_id for doc_id, _score in results]
    print_metrics("BM25", evaluateModel(bm25_results, query_rels, k=10, verbose=False))


if __name__ == "__main__":
    sys.exit(main())
