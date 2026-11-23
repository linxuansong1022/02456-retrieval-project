"""Build the two encoder towers from config (model.tower: hf or scratch)."""
from __future__ import annotations

import json
import os

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.models.hf_encoder import HFEncoder


def build_towers(cfg, tokenizer: PreTrainedTokenizer):
    """Return (query_encoder, document_encoder).

    With model.tower == "hf" both towers start from the same pretrained
    checkpoint and are not tied. With "scratch" they are TransformerEncoder
    instances built from cfg.model.
    """
    if cfg.model.tower == "hf":
        query_encoder = HFEncoder(
            model_name=cfg.model.model_name,
            pooling_type=cfg.model.pooling_type,
            freeze=cfg.model.freeze,
        )
        document_encoder = HFEncoder(
            model_name=cfg.model.model_name,
            pooling_type=cfg.model.pooling_type,
            freeze=cfg.model.freeze,
        )
        return query_encoder, document_encoder

    if cfg.model.tower == "scratch":
        query_encoder = TransformerEncoder(
            vocab_size=len(tokenizer),
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            emb_dim=cfg.model.emb_dim,
            feed_forward_dim=cfg.model.feed_forward_dim,
            pooling_type=cfg.model.pooling_type,
            dropout=cfg.model.dropout,
        )
        document_encoder = TransformerEncoder(
            vocab_size=len(tokenizer),
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            emb_dim=cfg.model.emb_dim,
            feed_forward_dim=cfg.model.feed_forward_dim,
            pooling_type=cfg.model.pooling_type,
            dropout=cfg.model.dropout,
        )
        return query_encoder, document_encoder

    raise ValueError(f"unknown model.tower {cfg.model.tower!r}; expected 'hf' or 'scratch'")


def load_towers(checkpoint_dir: str, device: str = "cpu"):
    """Return (query_encoder, document_encoder, meta, tokenizer) from a train_dpr.py checkpoint."""
    with open(os.path.join(checkpoint_dir, "meta.json")) as f:
        meta = json.load(f)

    tokenizer_dir = os.path.join(checkpoint_dir, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) if os.path.isdir(tokenizer_dir) else None

    # older checkpoints only saved the hf model name and pooling
    model = meta.get("model") or {
        "tower": "hf",
        "model_name": meta.get("model_name", "bert-base-uncased"),
        "pooling_type": meta.get("pooling_type", "cls"),
        "freeze": False,
    }
    query_encoder, document_encoder = build_towers(OmegaConf.create({"model": model}), tokenizer)

    query_encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "query_encoder.pt"), map_location=device))
    document_encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "document_encoder.pt"), map_location=device))
    return query_encoder.to(device).eval(), document_encoder.to(device).eval(), meta, tokenizer
