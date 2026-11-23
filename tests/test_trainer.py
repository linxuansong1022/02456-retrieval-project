"""Tests for the Trainer: loss reduction, recall, checkpoint round-trip, non-finite."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_utils.query import Query
from src.training.build import load_towers
from src.training.pairs import Pair, PairCollator, PairDataset
from src.training.trainer import Trainer
from tests.tokenizer_util import make_tiny_tokenizer


class ToyTower(nn.Module):
    """Mean-pooled embedding + linear, forward(input_ids, attention_mask) -> (B, d)."""

    def __init__(self, vocab_size: int, emb_dim: int, out_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.linear = nn.Linear(emb_dim, out_dim)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = x * attention_mask.unsqueeze(-1)
        denominator = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        x = x.sum(dim=1) / denominator
        return self.linear(x)

    def attention_maps(self):
        return None


def _make_loader(tokenizer, n_pairs: int = 4, batch_size: int = 4):
    pairs = [Pair(f"q{i}", f"tok{i}", f"d{i}", f"tok{i}") for i in range(n_pairs)]
    dataset = PairDataset(pairs)
    collator = PairCollator(tokenizer, query_max_len=8, passage_max_len=8)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)


def test_forty_steps_halve_loss():
    torch.manual_seed(0)
    tokenizer = make_tiny_tokenizer(tuple(f"tok{i}" for i in range(4)))
    loader = _make_loader(tokenizer)
    trainer = Trainer(
        ToyTower(len(tokenizer), 16, 16),
        ToyTower(len(tokenizer), 16, 16),
        lr=0.05,
        warmup_ratio=0.0,
        seed=0,
    )
    first, last = trainer.fit_overfit(loader, steps=40)
    assert last < first / 2


def test_save_load_roundtrip(tmp_path, monkeypatch):
    torch.manual_seed(0)
    vocab_size, emb_dim, out_dim = 64, 16, 32
    query_encoder = ToyTower(vocab_size, emb_dim, out_dim)
    document_encoder = ToyTower(vocab_size, emb_dim, out_dim)
    trainer = Trainer(query_encoder, document_encoder)
    trainer.save_checkpoint(
        str(tmp_path),
        meta={
            "tower": "hf",
            "model_name": "bert-base-uncased",
            "pooling_type": "cls",
            "embedding_dim": out_dim,
            "query_max_len": 64,
            "passage_max_len": 256,
            "epochs": 1,
            "final_train_loss": 1.0,
            "val": None,
        },
    )

    monkeypatch.setattr(
        "src.training.build.HFEncoder",
        lambda model_name, pooling_type, **kw: ToyTower(vocab_size, emb_dim, out_dim),
    )
    query_loaded, document_loaded, meta, tokenizer = load_towers(str(tmp_path), device="cpu")

    ids = torch.randint(0, vocab_size, (5, 8))
    mask = torch.ones(5, 8, dtype=torch.long)
    query_encoder.eval()
    document_encoder.eval()
    query_loaded.eval()
    document_loaded.eval()
    assert torch.allclose(query_encoder(ids, mask), query_loaded(ids, mask), atol=1e-6)
    assert torch.allclose(document_encoder(ids, mask), document_loaded(ids, mask), atol=1e-6)
    assert meta["model_name"] == "bert-base-uncased"


def test_evaluate_perfect():
    torch.manual_seed(0)
    tokenizer = make_tiny_tokenizer(tuple(f"tok{i}" for i in range(3)))
    shared = ToyTower(len(tokenizer), 16, 16)
    trainer = Trainer(shared, shared, device="cpu")
    trainer.set_tokenizer(tokenizer, query_max_len=8, passage_max_len=8)

    # Each query's relevant document carries the same text as the query, and the
    # same tower encodes both, so the relevant document scores exactly 1.0.
    queries = [
        Query("tok0", id="q0", relevant_document_ids=["d0"]),
        Query("tok1", id="q1", relevant_document_ids=["d1"]),
    ]
    documents = [{"text": ["tok0", "tok1", "tok2"], "document_id": ["d0", "d1", "d2"]}]
    result = trainer.evaluate(queries, documents, k=(1, 5), batch_size=2)
    assert result[1]["recall"] == 1.0
    assert result[1]["mrr"] == 1.0
    assert result[5]["recall"] == 1.0


def test_non_finite_loss_raises():
    class NaNTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))

        def forward(self, input_ids, attention_mask):
            return torch.full((input_ids.shape[0], 4), float("nan"))

        def attention_maps(self):
            return None

    trainer = Trainer(NaNTower(), NaNTower(), device="cpu")
    batch = [
        {
            "query_input_ids": torch.ones(2, 3, dtype=torch.long),
            "query_attention_mask": torch.ones(2, 3, dtype=torch.long),
            "document_input_ids": torch.ones(2, 3, dtype=torch.long),
            "document_attention_mask": torch.ones(2, 3, dtype=torch.long),
        }
    ]
    try:
        trainer.fit_overfit(batch, steps=1)
    except RuntimeError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for non-finite loss")
