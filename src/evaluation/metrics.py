from math import log2

from src.data_utils.query import Query

def calculateRecall(relevancies: list[bool], query: Query):
    return sum(relevancies) / min(len(relevancies), query.getNumberOfRelevantDocuments())

def calculatePrecision(relevancies: list[bool], k=None):
    if k is None:
        k = len(relevancies)
    if k == 0:
        return 0.0
    if k < 0:
        raise ValueError('k must be positive')

    relevant_count = 0
    for index in range(min(k, len(relevancies))):
        if relevancies[index]:
            relevant_count += 1

    return relevant_count / k

def calculateMRR(relevancies: list[list[bool]]):
    if len(relevancies) == 0:
        return 0.0

    total_reciprocal_rank = 0.0
    for query_relevancies in relevancies:
        for index in range(len(query_relevancies)):
            if query_relevancies[index]:
                rank = index + 1
                total_reciprocal_rank += 1 / rank
                break

    return total_reciprocal_rank / len(relevancies)

def calculateNDCG(relevancies: list[bool], query: Query):
    """Binary-relevance nDCG over the supplied ranking (cut it to k first), as BEIR computes it."""
    dcg = sum(1 / log2(rank + 2) for rank, hit in enumerate(relevancies) if hit)
    ideal_hits = min(len(relevancies), query.getNumberOfRelevantDocuments())
    idcg = sum(1 / log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def Evaluate(relevancies: list[list[bool]], queries: list[Query], k=None):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    scores = {'recall': None}
    n_queries = len(queries)
    total_recall = 0
    total_precision = 0
    total_ndcg = 0

    for i in range(n_queries):
        total_recall += calculateRecall(relevancies[i], queries[i])
        total_precision += calculatePrecision(relevancies[i], k)
        total_ndcg += calculateNDCG(relevancies[i], queries[i])

    scores['recall'] = total_recall / n_queries
    scores['precision'] = total_precision / n_queries
    scores['mrr'] = calculateMRR(relevancies)
    scores['ndcg'] = total_ndcg / n_queries
    return scores
