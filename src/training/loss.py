"""InfoNCE loss for DPR (project brief section 2.2)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPRLoss(nn.Module):
    """loss_i = -log( exp(D_i . Q_i / tau) / sum_j exp(D_j . Q_i / tau) ).

    Scores are Q @ D.T, so rows are queries and the positive of query i is
    column i. Returns the mean over the batch.
    """

    def __init__(self, temperature: float = 1.0, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(
        self,
        queries: torch.Tensor,
        documents: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if queries.dim() != 2 or documents.dim() != 2:
            raise ValueError("queries and documents must be 2D tensors of shape (B, d)")
        if queries.shape[0] != documents.shape[0]:
            raise ValueError(
                f"queries and documents must share batch size, got {queries.shape[0]} vs {documents.shape[0]}"
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

        targets = torch.arange(scores.shape[0], device=scores.device)
        return F.cross_entropy(scores, targets)
