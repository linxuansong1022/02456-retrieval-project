from src.data_utils.query import Query


def calculateRecall(relevancies: list[bool], query: Query) -> float:
    """Return the project's recall@k: hits / min(k, relevant documents)."""
    denominator = min(len(relevancies), query.getNumberOfRelevantDocuments())
    return sum(relevancies) / denominator if denominator else 0.0


def calculateReciprocalRank(relevancies: list[bool]) -> float:
    """Return 1 / the first relevant result's rank, or 0 if none was retrieved."""
    for rank, relevant in enumerate(relevancies, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def Evaluate(relevancies: list[list[bool]], queries: list[Query]):
    """Average recall and reciprocal rank over queries using the supplied top-k lists."""
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    n_queries = len(queries)
    if not n_queries:
        return {'recall': 0.0, 'mrr': 0.0}

    total_recall = 0
    total_reciprocal_rank = 0
    for i in range(n_queries):
        total_recall += calculateRecall(relevancies[i], queries[i])
        total_reciprocal_rank += calculateReciprocalRank(relevancies[i])

    return {
        'recall': total_recall / n_queries,
        'mrr': total_reciprocal_rank / n_queries,
    }
