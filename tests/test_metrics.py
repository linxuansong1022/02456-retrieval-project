from math import isclose

from src.data_utils.query import Query
from src.evaluation.metrics import Evaluate, calculateMRR, calculatePrecision


def test_precision_counts_relevant_results():
    assert calculatePrecision([True, False, True, False, False], k=5) == 0.4


def test_precision_when_all_results_are_relevant():
    assert calculatePrecision([True, True], k=2) == 1.0


def test_precision_when_no_results_are_relevant():
    assert calculatePrecision([False, False], k=2) == 0.0


def test_precision_ignores_results_after_k():
    assert calculatePrecision([False, True, True], k=2) == 0.5


def test_precision_keeps_k_when_fewer_results_are_returned():
    assert calculatePrecision([True], k=5) == 0.2


def test_precision_empty_results():
    assert calculatePrecision([], k=5) == 0.0
    assert calculatePrecision([]) == 0.0


def test_evaluate_averages_precision_at_k():
    queries = [Query("first", "q0", ["d0"]), Query("second", "q1", ["d1"])]
    scores = Evaluate([[True, False], [False, False]], queries, k=5)
    assert scores["precision"] == 0.1


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


def test_evaluate_returns_recall_precision_and_mrr():
    queries = [
        Query("first", "q0", ["d0"]),
        Query("second", "q1", ["d1"]),
        Query("third", "q2", ["d2"]),
    ]
    relevancies = [[True, False], [False, True], [False, False]]
    scores = Evaluate(relevancies, queries)
    assert isclose(scores["recall"], 2 / 3)
    assert isclose(scores["precision"], 1 / 3)
    assert scores["mrr"] == 0.5
