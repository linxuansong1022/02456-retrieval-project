"""InfoNCE loss for DPR (project brief section 2.2)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPRLoss(nn.Module):
    """loss_i = -log( exp(s(q_i, d_i) / tau) / sum_j exp(s(q_i, d_j) / tau) ).

    Scores are Q @ D.T, so rows are queries. ``documents`` holds the B positives
    first (the positive of query i is row i) and may append extra passages, such
    as mined hard negatives, after them; every query is scored against all of
    them. ``ignore`` is an optional (B, P) bool mask of passages to leave out of
    query i's softmax, used for other gold passages of the same question that
    landed in the pool. Returns the mean over the batch.
    """

    def __init__(self, temperature: float = 1.0, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(
        self,
        queries: torch.Tensor,
        documents: torch.Tensor,
        ignore: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if queries.dim() != 2 or documents.dim() != 2:
            raise ValueError("queries and documents must be 2D tensors of shape (B, d) and (P, d)")
        if documents.shape[0] < queries.shape[0]:
            raise ValueError(
                f"need at least one passage per query, got {queries.shape[0]} queries and {documents.shape[0]} passages"
            )
        if queries.shape[1] != documents.shape[1]:
            raise ValueError(
                f"queries and documents must share embedding dim, got {queries.shape[1]} vs {documents.shape[1]}"
            )

        if self.normalize:
            queries = F.normalize(queries, p=2, dim=-1)
            documents = F.normalize(documents, p=2, dim=-1)

        scores = queries @ documents.transpose(0, 1)
        scores = scores / self.temperature
        if ignore is not None:
            scores = scores.masked_fill(ignore, float("-inf"))

        targets = torch.arange(scores.shape[0], device=scores.device)
        return F.cross_entropy(scores, targets)
