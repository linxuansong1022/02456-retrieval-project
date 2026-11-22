"""Recall and MRR on ordered relevance lists, without model downloads."""
import pytest

from src.data_utils.query import Query
from src.evaluation.metrics import Evaluate, calculateRecall, calculateReciprocalRank


@pytest.mark.parametrize(
    ("relevancies", "expected"),
    [
        ([True, False], 1.0),
        ([False, True], 0.5),
        ([False, False, True], 1 / 3),
        ([False, True, True], 0.5),
        ([False, False], 0.0),
        ([], 0.0),
    ],
)
def test_reciprocal_rank(relevancies, expected):
    assert calculateReciprocalRank(relevancies) == pytest.approx(expected)


def test_evaluate_averages_every_query_including_misses():
    queries = [Query("question", str(i), ["relevant"]) for i in range(3)]
    scores = Evaluate([[True, False], [False, True], [False, False]], queries)
    assert scores == pytest.approx({"recall": 2 / 3, "mrr": 0.5})


def test_cutoff_excludes_relevant_results_after_k():
    ranking = [False, False, True]
    query = Query("question", "q", ["relevant"])
    assert Evaluate([ranking[:2]], [query]) == {"recall": 0.0, "mrr": 0.0}
    assert Evaluate([ranking[:3]], [query]) == pytest.approx({"recall": 1.0, "mrr": 1 / 3})


def test_recall_preserves_project_denominator():
    query = Query("question", "q", ["d1", "d2", "d3"])
    assert calculateRecall([True, True], query) == 1.0


def test_empty_rankings_and_queries_without_relevant_documents():
    queries = [Query("question", "q1", ["d1"]), Query("question", "q2", [])]
    assert Evaluate([[], [False, False]], queries) == {"recall": 0.0, "mrr": 0.0}
    assert Evaluate([], []) == {"recall": 0.0, "mrr": 0.0}


def test_query_count_must_match_rankings():
    with pytest.raises(AssertionError):
        Evaluate([[]], [])
