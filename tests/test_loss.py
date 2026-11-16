"""Tests for the InfoNCE/DPR loss."""
from __future__ import annotations

import math

import pytest
import torch

from src.training.loss import DPRLoss


def test_random_unit_vectors_about_ln_b():
    torch.manual_seed(95)
    batch, dim = 8, 768
    queries = torch.randn(batch, dim)
    documents = torch.randn(batch, dim)
    loss = DPRLoss(temperature=1.0, normalize=True)(queries, documents)
    assert abs(loss.item() - math.log(batch)) < 1e-3


def test_matched_orthogonal_pairs_near_zero():
    torch.manual_seed(0)
    batch = 8
    queries = torch.eye(batch)
    documents = queries.clone()
    loss = DPRLoss(temperature=0.01, normalize=True)(queries, documents)
    assert loss.item() < 1e-3


def test_gradients_are_non_zero():
    queries = torch.nn.Parameter(torch.randn(8, 16))
    documents = torch.nn.Parameter(torch.randn(8, 16))
    loss = DPRLoss()(queries, documents)
    loss.backward()
    assert queries.grad is not None and queries.grad.abs().sum().item() > 0
    assert documents.grad is not None and documents.grad.abs().sum().item() > 0


def test_mis_shaped_input_raises():
    criterion = DPRLoss()
    with pytest.raises(ValueError):
        criterion(torch.randn(8, 16), torch.randn(7, 16))  # batch mismatch
    with pytest.raises(ValueError):
        criterion(torch.randn(8, 16), torch.randn(8, 20))  # dim mismatch
    with pytest.raises(ValueError):
        criterion(torch.randn(8), torch.randn(8, 16))  # not 2D
