"""Tests for tower construction from the Hydra config."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn

import src.training.build as build_mod
from src.training.build import build_towers


def _hf_cfg():
    return SimpleNamespace(
        model=SimpleNamespace(
            tower="hf",
            model_name="bert-base-uncased",
            pooling_type="cls",
            freeze=False,
            emb_dim=128,
            n_heads=4,
            n_layers=6,
            feed_forward_dim=768,
            dropout=0.1,
        )
    )


def _scratch_cfg():
    return SimpleNamespace(
        model=SimpleNamespace(
            tower="scratch",
            emb_dim=128,
            n_heads=4,
            n_layers=6,
            feed_forward_dim=768,
            pooling_type="cls",
            dropout=0.1,
        )
    )


class _FakeTokenizer:
    def __len__(self):
        return 30522


def test_hf_builds_two_distinct_modules(monkeypatch):
    class FakeHF(nn.Module):
        def __init__(self, model_name, pooling_type, freeze=False, collect_attention=False):
            super().__init__()
            self.model_name = model_name
            self.pooling_type = pooling_type

        def forward(self, input_ids, attention_mask):
            return input_ids[:, :1].float()

        def attention_maps(self):
            return None

    monkeypatch.setattr(build_mod, "HFEncoder", FakeHF)
    query_encoder, document_encoder = build_towers(_hf_cfg(), tokenizer=_FakeTokenizer())
    assert isinstance(query_encoder, FakeHF)
    assert isinstance(document_encoder, FakeHF)
    assert query_encoder is not document_encoder
    assert query_encoder.model_name == "bert-base-uncased"


def test_scratch_success_with_stub(monkeypatch):
    class StubEncoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

        def forward(self, input_ids, attention_mask):
            return None

        def attention_maps(self):
            return None

    monkeypatch.setattr(build_mod, "TransformerEncoder", StubEncoder)
    query_encoder, document_encoder = build_towers(_scratch_cfg(), tokenizer=_FakeTokenizer())
    assert query_encoder is not document_encoder
    assert query_encoder.kwargs["vocab_size"] == 30522
    assert query_encoder.kwargs["n_layers"] == 6
