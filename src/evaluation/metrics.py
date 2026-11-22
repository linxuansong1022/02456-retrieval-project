from src.data_utils.query import Query

def calculateRecall(relevancies: list[bool], query: Query):
    return sum(relevancies) / min(len(relevancies), query.getNumberOfRelevantDocuments())

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

def Evaluate(relevancies: list[list[bool]], queries: list[Query]):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    scores = {'recall': None}
    n_queries = len(queries)
    total_recall = 0
    
    for i in range(n_queries):
        total_recall += calculateRecall(relevancies[i], queries[i])
    
    scores['recall'] = total_recall / n_queries
    scores['mrr'] = calculateMRR(relevancies)
    return scores
