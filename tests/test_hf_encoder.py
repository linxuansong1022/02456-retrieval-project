"""Tests for the Hugging Face encoder tower. Skips when the model is unavailable."""
from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from src.retrievers.models.hf_encoder import HFEncoder


def _hf_available() -> bool:
    try:
        from transformers import AutoModel

        AutoModel.from_pretrained("bert-base-uncased")
        return True
    except Exception:
        return False


_HF_AVAILABLE = _hf_available()

skip_if_no_model = pytest.mark.skipif(not _HF_AVAILABLE, reason="bert-base-uncased model cannot be downloaded")


@skip_if_no_model
@pytest.mark.parametrize("pooling", ["cls", "mean", "max"])
def test_hf_encoder_shapes(pooling):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    texts = ["hello world", "the quick brown fox jumps over the lazy dog"]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")

    encoder = HFEncoder(model_name="bert-base-uncased", pooling_type=pooling)
    with torch.no_grad():
        out = encoder(encoded["input_ids"], encoded["attention_mask"])
    assert out.shape == (2, 768)
    assert encoder.embedding_dim == 768


@skip_if_no_model
def test_hf_attention_maps_enabled_and_disabled():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    texts = ["hello world", "the quick brown fox"]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=16, return_tensors="pt")

    enabled = HFEncoder(model_name="bert-base-uncased", pooling_type="cls", collect_attention=True)
    with torch.no_grad():
        enabled(encoded["input_ids"], encoded["attention_mask"])
    maps = enabled.attention_maps()
    assert maps is not None and len(maps) == 12
    for layer in maps:
        assert layer.shape[0] == 2 and layer.shape[1] == 12 and layer.ndim == 4

    disabled = HFEncoder(model_name="bert-base-uncased", pooling_type="cls", collect_attention=False)
    with torch.no_grad():
        disabled(encoded["input_ids"], encoded["attention_mask"])
    assert disabled.attention_maps() is None
