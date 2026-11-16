"""Build the two encoder towers from config (model.tower: hf or scratch)."""
from __future__ import annotations

from transformers import PreTrainedTokenizer

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
        try:
            query_encoder = TransformerEncoder(
                vocab_size=len(tokenizer),
                n_layers=cfg.model.n_layers,
                n_heads=cfg.model.n_heads,
                emb_dim=cfg.model.emb_dim,
                hidden_dim=cfg.model.hidden_dim,
                feed_forward_dim=cfg.model.feed_forward_dim,
                pooling_type=cfg.model.pooling_type,
                dropout=cfg.model.dropout,
            )
            document_encoder = TransformerEncoder(
                vocab_size=len(tokenizer),
                n_layers=cfg.model.n_layers,
                n_heads=cfg.model.n_heads,
                emb_dim=cfg.model.emb_dim,
                hidden_dim=cfg.model.hidden_dim,
                feed_forward_dim=cfg.model.feed_forward_dim,
                pooling_type=cfg.model.pooling_type,
                dropout=cfg.model.dropout,
            )
        except NotImplementedError as exc:
            raise RuntimeError(
                "The from-scratch tower is not available yet: "
                "src/retrievers/models/encoder.py still raises NotImplementedError. "
                "Use model.tower=hf for the pretrained tower."
            ) from exc
        return query_encoder, document_encoder

    raise ValueError(f"unknown model.tower {cfg.model.tower!r}; expected 'hf' or 'scratch'")
