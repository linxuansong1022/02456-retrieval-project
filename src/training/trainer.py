"""The DPR training loop, evaluation, checkpointing and a 64-pair overfit check."""
from __future__ import annotations

import datetime
import itertools
import json
import os
import subprocess
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.evaluation.metrics import Evaluate
from src.retrievers.dpr import DPR
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.training.loss import DPRLoss


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


class Trainer:
    def __init__(
        self,
        query_encoder,
        document_encoder,
        device: str = "cpu",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        temperature: float = 1.0,
        normalize: bool = True,
        grad_clip: float = 1.0,
        warmup_ratio: float = 0.1,
        log_every: int = 50,
        seed: int = 0,
    ):
        self.query_encoder = query_encoder
        self.document_encoder = document_encoder
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.normalize = normalize
        self.grad_clip = grad_clip
        self.warmup_ratio = warmup_ratio
        self.log_every = log_every
        self.seed = seed
        self.criterion = DPRLoss(temperature=temperature, normalize=normalize)
        self.tokenizer = None
        self.query_max_len = 64
        self.passage_max_len = 256

    def set_tokenizer(self, tokenizer, query_max_len: int = 64, passage_max_len: int = 256) -> None:
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len

    def _optimizer_and_scheduler(self, total_steps: int):
        params = list(self.query_encoder.parameters()) + list(self.document_encoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return optimizer, scheduler, params

    def _step(self, optimizer, params, batch) -> float:
        query_ids = batch["query_input_ids"].to(self.device)
        query_mask = batch["query_attention_mask"].to(self.device)
        document_ids = batch["document_input_ids"].to(self.device)
        document_mask = batch["document_attention_mask"].to(self.device)

        optimizer.zero_grad()
        query_emb = self.query_encoder(query_ids, query_mask)
        document_emb = self.document_encoder(document_ids, document_mask)
        loss = self.criterion(query_emb, document_emb)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss.item()}")
        loss.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(params, self.grad_clip)
        optimizer.step()
        return loss.item()

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_fn=None,
        val_loader=None,
    ) -> list[dict]:
        """Train for a number of epochs and return one history entry per epoch."""
        torch.manual_seed(self.seed)
        self.query_encoder.to(self.device)
        self.document_encoder.to(self.device)
        self.query_encoder.train()
        self.document_encoder.train()

        total_steps = len(train_loader) * epochs
        optimizer, scheduler, params = self._optimizer_and_scheduler(total_steps)

        history: list[dict] = []
        for epoch in range(epochs):
            start = time.time()
            step_losses: list[float] = []
            progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
            for step, batch in enumerate(progress):
                loss_value = self._step(optimizer, params, batch)
                scheduler.step()
                step_losses.append(loss_value)
                progress.set_postfix(loss=f"{loss_value:.6f}")
                if self.log_every and (step + 1) % self.log_every == 0:
                    print(f"epoch {epoch + 1} step {step + 1}/{len(train_loader)} loss {loss_value:.6f}")

            seconds = time.time() - start
            train_loss = sum(step_losses) / len(step_losses) if step_losses else 0.0
            tail = max(1, int(len(step_losses) * 0.2))
            losses_tail = sum(step_losses[-tail:]) / tail if step_losses else 0.0

            if val_fn is not None:
                optimizer.zero_grad(set_to_none=True)
                self.query_encoder.eval()
                self.document_encoder.eval()
                val = val_fn()
                self.query_encoder.train()
                self.document_encoder.train()
            else:
                val = None

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "losses_tail": losses_tail,
                    "seconds": seconds,
                    "val": val,
                }
            )
        return history

    def fit_overfit(self, loader: DataLoader, steps: int) -> tuple[float, float]:
        """Run ``steps`` steps on a small loader and return ``(first_loss, last_loss)``."""
        torch.manual_seed(self.seed)
        self.query_encoder.to(self.device)
        self.document_encoder.to(self.device)
        self.query_encoder.train()
        self.document_encoder.train()

        optimizer, scheduler, params = self._optimizer_and_scheduler(steps)

        first_loss: float | None = None
        last_loss: float | None = None
        for batch in itertools.islice(itertools.cycle(loader), steps):
            loss_value = self._step(optimizer, params, batch)
            scheduler.step()
            if first_loss is None:
                first_loss = loss_value
            last_loss = loss_value
        return first_loss, last_loss

    def evaluate(self, queries, documents, k: tuple = (1, 5, 10, 20), batch_size: int = 128) -> dict:
        """Score val queries the same way main.py does: DPR retriever, then Evaluate at each k."""
        if self.tokenizer is None:
            raise RuntimeError("tokenizer not set; call set_tokenizer before evaluate")

        retriever = DPR(documents, self.query_encoder, self.document_encoder, self.tokenizer,
                        self.query_max_len, self.passage_max_len, batch_size, self.device)
        relevancies = retrieveQueryAndGetRelevancies(retriever, queries, k=max(k))
        return {kk: Evaluate([row[:kk] for row in relevancies], queries, k=kk) for kk in k}

    def save_checkpoint(self, dir: str, meta: dict | None = None) -> str:
        """Save both encoders, the tokenizer and a meta.json describing them."""
        os.makedirs(dir, exist_ok=True)
        torch.save(self.query_encoder.state_dict(), os.path.join(dir, "query_encoder.pt"))
        torch.save(self.document_encoder.state_dict(), os.path.join(dir, "document_encoder.pt"))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(os.path.join(dir, "tokenizer"))

        meta = dict(meta or {})
        meta.setdefault("git_commit", _git_commit())
        meta.setdefault("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
        with open(os.path.join(dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return dir
