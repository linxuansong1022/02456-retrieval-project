
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    # adapted from 02456-2026 "Week6 Transformers/6.1 Deep Learning with Transformers.ipynb", cell 44 (PositionalEncoding)
    def __init__(self, emb_dim: int, max_len: int, dropout: float = 0.1):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


def attention(q, k, v, mask=None, dropout=None):
    # adapted from 02456-2026 "Week6 Transformers/6.1 Deep Learning with Transformers.ipynb", cell 46 (attention)
    raise NotImplementedError()


class MultiHeadAttention(nn.Module):
    # adapted from 02456-2026 "Week6 Transformers/6.1 Deep Learning with Transformers.ipynb", cell 47 (MultiHeadedAttention)
    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, feed_forward_dim: int, dropout: float = 0.1):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class EncoderLayer(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, feed_forward_dim: int, dropout: float = 0.1):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int,
                 feed_forward_dim: int, pooling_type: str = "cls", max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.setup_model()

    def setup_model(self):
        raise NotImplementedError()

    def pool(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "cls":
            raise NotImplementedError()
        elif self.pooling_type == "mean":
            raise NotImplementedError()
        elif self.pooling_type == "max":
            raise NotImplementedError()
        raise ValueError(f"unknown pooling_type {self.pooling_type}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def attention_maps(self) -> list:
        raise NotImplementedError()
