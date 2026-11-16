"""Tests for pair construction, deduplication and collation."""
from __future__ import annotations

import torch
from datasets import Dataset

from src.training.pairs import Pair, PairCollator, PairDataset, documents_from_corpus, pairs_from_corpus


class FakeCorpus:
    def __init__(self, train_ds, all_documents_ds):
        self.train = train_ds
        self.all_documents = all_documents_ds


def _train_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "question": ["q0", "q1", "q2"],
            "query_id": ["id0", "id1", "id2"],
            "document_id": [["d0"], ["d1", "d2"], ["d3"]],
            "text": [["hello world"], ["foo", "bar"], ["baz"]],
        }
    )


def test_pair_count_and_flatten():
    pairs = pairs_from_corpus(FakeCorpus(_train_dataset(), None), dedupe=False)
    assert len(pairs) == 4
    assert pairs[0] == Pair("id0", "q0", "d0", "hello world")
    assert pairs[1] == Pair("id1", "q1", "d1", "foo")
    assert pairs[2] == Pair("id1", "q1", "d2", "bar")
    assert pairs[3] == Pair("id2", "q2", "d3", "baz")


def test_dedupe_by_id_and_text():
    ds = Dataset.from_dict(
        {
            "question": ["q0", "q0", "q1"],
            "query_id": ["id0", "id1", "id2"],
            "document_id": [["d0", "d0"], ["d1"], ["d2"]],
            "text": [["same text", "same text"], ["same text"], ["other"]],
        }
    )
    pairs = pairs_from_corpus(FakeCorpus(ds, None), dedupe=True)
    # the id dedupe drops one id0/d0, the text dedupe collapses the two "same text" pairs
    assert len(pairs) == 2
    texts = sorted((p.query_id, p.document_text) for p in pairs)
    assert texts == [("id0", "same text"), ("id2", "other")]


def test_documents_from_corpus_flattens():
    all_docs = Dataset.from_dict(
        {"text": [["a", "b"], ["c"]], "document_id": [["d0", "d1"], ["d2"]]}
    )
    docs = documents_from_corpus(FakeCorpus(None, all_docs))
    assert docs == [("d0", "a"), ("d1", "b"), ("d2", "c")]


def test_pair_dataset():
    pairs = [Pair("q0", "hello", "d0", "world"), Pair("q1", "foo", "d1", "bar")]
    ds = PairDataset(pairs)
    assert len(ds) == 2
    assert ds[1] == pairs[1]


def test_collator_shapes_masks_truncation(tiny_tokenizer):
    collator = PairCollator(tiny_tokenizer, query_max_len=4, passage_max_len=4)
    pairs = [
        Pair("q0", "hello world", "d0", "the quick brown fox"),  # 4 passage tokens
        Pair("q1", "hello", "d1", "jumps over the lazy dog"),  # 5 passage tokens -> truncated
    ]
    batch = collator(pairs)

    assert batch["query_input_ids"].dtype == torch.long
    assert batch["query_attention_mask"].dtype == torch.long
    assert batch["document_input_ids"].dtype == torch.long
    assert batch["document_attention_mask"].dtype == torch.long

    assert batch["query_input_ids"].shape == (2, 2)  # "hello world" (2) / "hello" (1)
    assert batch["document_input_ids"].shape == (2, 4)  # both capped at 4

    for ids, mask in (
        (batch["query_input_ids"], batch["query_attention_mask"]),
        (batch["document_input_ids"], batch["document_attention_mask"]),
    ):
        assert torch.all((mask == 0) | (mask == 1))
        assert mask.sum(dim=1).min().item() >= 1  # every row has at least one real token
        assert torch.all(ids[mask == 1] != tiny_tokenizer.pad_token_id)
        assert torch.all(ids[mask == 0] == tiny_tokenizer.pad_token_id)
