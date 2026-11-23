
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.retrievers.models.pooling import pool


class PositionalEncoding(nn.Module):
    # adapted from 02456-2026 "Week6 Transformers/6.1 Deep Learning with Transformers.ipynb", cell 44 (PositionalEncoding)
    def __init__(self, emb_dim: int, max_len: int, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def attention(q, k, v, mask=None, dropout=None):
    # adapted from 02456-2026 "Week6 Transformers/6.1 Deep Learning with Transformers.ipynb", cell 46 (attention)
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
    p_attn = F.softmax(scores, dim=-1)
    weights = p_attn if dropout is None else dropout(p_attn)
    return torch.matmul(weights, v), p_attn


class MultiHeadAttention(nn.Module):
    # adapted from 02456-2026 "Week6 Transformers/6.1 Deep Learning with Transformers.ipynb", cell 47 (MultiHeadedAttention)
    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.d_k = emb_dim // n_heads
        self.n_heads = n_heads
        self.linears = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        if mask is not None:
            mask = mask[:, None, None, :]
        query, key, value = [linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
                             for linear in self.linears[:3]]
        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        self.attn = attn.detach()
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, feed_forward_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, feed_forward_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feed_forward_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, feed_forward_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(emb_dim, n_heads, dropout)
        self.feed_forward = FeedForward(emb_dim, feed_forward_dim, dropout)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), mask))
        return x + self.dropout(self.feed_forward(self.norm2(x)))


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, n_heads: int, emb_dim: int, feed_forward_dim: int,
                 pooling_type: str = "cls", max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.feed_forward_dim = feed_forward_dim
        self.max_len = max_len
        self.dropout = dropout
        self.pooling_type = pooling_type
        self.setup_model()

    def setup_model(self):
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.positional_encoding = PositionalEncoding(self.emb_dim, self.max_len, self.dropout)
        self.layers = nn.ModuleList([EncoderLayer(self.emb_dim, self.n_heads, self.feed_forward_dim, self.dropout)
                                     for _ in range(self.n_layers)])
        self.norm = LayerNorm(self.emb_dim)

    @property
    def embedding_dim(self) -> int:
        return self.emb_dim

    def pool(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return pool(x, attention_mask, self.pooling_type)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(self.embedding(input_ids) * math.sqrt(self.emb_dim))
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.pool(self.norm(x), attention_mask)

    def attention_maps(self) -> list:
        # one (B, heads, T, T) tensor per layer, from the last forward pass
        return [layer.self_attn.attn for layer in self.layers]
