"""Build (query, passage) training pairs and collate them into tower inputs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.data_utils.corpus import Corpus


@dataclass(frozen=True)
class Pair:
    """One (query, relevant passage) training pair."""

    query_id: str
    query_text: str
    document_id: str
    document_text: str


def build_tokenizer(name: str = "bert-base-uncased") -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(name)


def pairs_from_corpus(corpus: Corpus, split: str = "train", dedupe: bool = True) -> list[Pair]:
    """Flatten one split into one Pair per (query, relevant passage).

    Repeated (query_id, document_id) pairs are always dropped. With dedupe=True,
    repeated (query_text, document_text) pairs are dropped too so a duplicated
    passage cannot act as an in-batch negative against its own positive.
    """
    split_ds = getattr(corpus, split)
    if split_ds is None:
        raise ValueError(f"split {split!r} is not present in the corpus")

    questions = split_ds["question"]
    query_ids = split_ds["query_id"]
    document_ids = split_ds["document_id"]
    texts = split_ds["text"]

    rows = split_ds.num_rows
    pairs: list[Pair] = []
    for i in range(rows):
        for doc_id, doc_text in zip(document_ids[i], texts[i]):
            pairs.append(Pair(query_ids[i], questions[i], doc_id, doc_text))

    pairs_before = len(pairs)

    seen_qd: set[tuple[str, str]] = set()
    deduped: list[Pair] = []
    for pair in pairs:
        key = (pair.query_id, pair.document_id)
        if key not in seen_qd:
            seen_qd.add(key)
            deduped.append(pair)
    pairs = deduped

    text_counts = Counter(p.document_text for p in pairs)
    repeated_passage_texts = sum(1 for count in text_counts.values() if count > 1)

    if dedupe:
        seen_qt: set[tuple[str, str]] = set()
        deduped = []
        for pair in pairs:
            key = (pair.query_text, pair.document_text)
            if key not in seen_qt:
                seen_qt.add(key)
                deduped.append(pair)
        pairs = deduped

    print(
        f"pairs_from_corpus[{split}]: rows={rows} pairs_before={pairs_before} "
        f"pairs_after={len(pairs)} repeated_passage_texts={repeated_passage_texts}"
    )
    return pairs


def documents_from_corpus(corpus: Corpus) -> list[tuple[str, str]]:
    """Return the eval pool as (document_id, text) tuples."""
    documents: list[tuple[str, str]] = []
    for texts, ids in zip(corpus.all_documents["text"], corpus.all_documents["document_id"]):
        documents.extend(zip(ids, texts))
    return documents


class PairDataset(Dataset):
    def __init__(self, pairs: list[Pair]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Pair:
        return self.pairs[idx]


class PairCollator:
    """Tokenize a batch of Pair objects into padded input ids and masks.

    Queries and passages use separate truncation lengths.
    """

    def __init__(self, tokenizer, query_max_len: int = 64, passage_max_len: int = 256):
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len

    def __call__(self, batch: list[Pair]) -> dict:
        query_texts = [pair.query_text for pair in batch]
        document_texts = [pair.document_text for pair in batch]

        queries = self.tokenizer(query_texts, truncation=True, max_length=self.query_max_len)
        documents = self.tokenizer(document_texts, truncation=True, max_length=self.passage_max_len)

        queries = self.tokenizer.pad(queries, padding="longest", return_tensors="pt")
        documents = self.tokenizer.pad(documents, padding="longest", return_tensors="pt")

        return {
            "query_input_ids": queries["input_ids"].long(),
            "query_attention_mask": queries["attention_mask"].long(),
            "document_input_ids": documents["input_ids"].long(),
            "document_attention_mask": documents["attention_mask"].long(),
        }
