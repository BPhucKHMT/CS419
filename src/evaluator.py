import numpy as np


def precisionAtK(retrieved, relevant, k=10):
    if k == 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    relevant_count = sum(1 for doc in retrieved_at_k if doc in relevant_set)
    return relevant_count / k


def recallAtK(retrieved, relevant, k=10):
    if k == 0 or not relevant:
        return 0.0
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    relevant_count = sum(1 for doc in retrieved_at_k if doc in relevant_set)
    return relevant_count / len(relevant)


def avaragePrecision(retrieved, relevant):
    if not relevant:
        return 0.0
    precision_sum = 0.0
    for rank, docID in enumerate(retrieved, start=1):
        if docID in relevant:
            precision_sum += precisionAtK(retrieved, relevant, rank)
    return precision_sum / len(relevant)


def meanAveragePrecision(retrieved, relevant):
    total_ap = 0.0
    for query_id in retrieved:
        total_ap += avaragePrecision(retrieved[query_id], relevant[query_id])
    return total_ap / len(retrieved) if retrieved else 0.0


def meanReciprocalRank(retrieved, relevant):
    total_mrr = 0.0
    for query_id in retrieved:
        for rank, doc in enumerate(retrieved[query_id], start=1):
            if doc in relevant[query_id]:
                total_mrr += 1 / rank
                break
    return total_mrr / len(retrieved) if retrieved else 0.0


RECALL_POINTS = np.linspace(0.0, 1.0, 11)


def interpolatedRecallPrecision(retrieved, relevant):
    precision = []
    recall = []

    for i, docID in enumerate(retrieved, start=1):
        if docID in relevant:
            precision.append(precisionAtK(retrieved, relevant, i))
            recall.append(recallAtK(retrieved, relevant, i))

    interpolated = []
    for rp in RECALL_POINTS:
        p_at_r = [p for p, r in zip(precision, recall) if r >= rp]
        interpolated.append(max(p_at_r) if p_at_r else 0.0)

    return interpolated


def meanInterpolatedRecallPrecision(all_retrieved, all_relevant):
    interpolated_all = []

    for qid in all_retrieved:
        retrieved = all_retrieved[qid]
        relevant = all_relevant.get(qid, set())
        if relevant:
            interpolated_all.append(interpolatedRecallPrecision(retrieved, relevant))

    if not interpolated_all:
        return [0.0] * 11

    return np.mean(interpolated_all, axis=0)


def evaluateModel(retrieved, relevant, k=10, verbose=True):
    aps, ps, rs = [], [], []

    for query_id in retrieved:
        if query_id not in relevant:
            continue

        rel_docs = set(relevant[query_id])
        if not rel_docs:
            continue

        res = retrieved[query_id]
        retrieved_k = res[:k]
        hits_k = sum(1 for d in retrieved_k if d in rel_docs)
        ps.append(hits_k / k)
        rs.append(hits_k / len(rel_docs))

        score, hits = 0.0, 0
        for i, d in enumerate(res, 1):
            if d in rel_docs:
                hits += 1
                score += hits / i
        aps.append(score / len(rel_docs))

    map_score = float(np.mean(aps)) if aps else 0.0
    p_at_k = float(np.mean(ps)) if ps else 0.0
    r_at_k = float(np.mean(rs)) if rs else 0.0

    if verbose:
        print(f"Mean Average Precision (MAP): {map_score:.4f}")
        print(f"P@{k}: {p_at_k:.4f}")
        print(f"Recall@{k}: {r_at_k:.4f}")

    return {
        "MAP": map_score,
        f"P@{k}": p_at_k,
        f"Recall@{k}": r_at_k,
    }
