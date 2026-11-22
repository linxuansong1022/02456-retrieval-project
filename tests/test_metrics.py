from math import isclose

from src.data_utils.query import Query
from src.evaluation.metrics import Evaluate, calculateMRR


def test_first_result_is_relevant():
    assert calculateMRR([[True, False]]) == 1.0


def test_second_result_is_relevant():
    assert calculateMRR([[False, True]]) == 0.5


def test_only_first_relevant_result_counts():
    assert calculateMRR([[False, True, True]]) == 0.5


def test_no_relevant_result():
    assert calculateMRR([[False, False]]) == 0.0


def test_average_includes_queries_with_no_relevant_result():
    relevancies = [[True, False], [False, True], [False, False]]
    assert calculateMRR(relevancies) == 0.5


def test_empty_inputs():
    assert calculateMRR([]) == 0.0
    assert calculateMRR([[]]) == 0.0
    assert calculateMRR([[True], []]) == 0.5


def test_only_supplied_top_k_results_count():
    results = [False, False, True]
    assert calculateMRR([results[:2]]) == 0.0
    assert isclose(calculateMRR([results[:3]]), 1 / 3)


def test_evaluate_returns_recall_and_mrr():
    queries = [
        Query("first", "q0", ["d0"]),
        Query("second", "q1", ["d1"]),
        Query("third", "q2", ["d2"]),
    ]
    relevancies = [[True, False], [False, True], [False, False]]
    scores = Evaluate(relevancies, queries)
    assert isclose(scores["recall"], 2 / 3)
    assert scores["mrr"] == 0.5
