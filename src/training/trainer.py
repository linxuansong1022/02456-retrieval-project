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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data_utils.query import Query
from src.evaluation.metrics import Evaluate
from src.retrievers.models.hf_encoder import HFEncoder
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

    def _encode_texts(self, encoder, texts: list[str], max_len: int, batch_size: int) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                ids = encoded["input_ids"].to(self.device)
                mask = encoded["attention_mask"].to(self.device)
                # contiguous() drops the reference to the (B, T, d) parent
                embeddings.append(encoder(ids, mask).contiguous())
        return torch.cat(embeddings, dim=0)

    def evaluate_recall(
        self,
        queries: list[tuple],
        documents: list[tuple[str, str]],
        k: tuple = (1, 5, 10, 20),
        batch_size: int = 128,
    ) -> dict[int, float]:
        """Return recall@k, retaining the original recall-only interface."""
        return self.evaluate_metrics(queries, documents, k=k, batch_size=batch_size)["recall"]

    def evaluate_metrics(
        self,
        queries: list[tuple],
        documents: list[tuple[str, str]],
        k: tuple = (1, 5, 10, 20),
        batch_size: int = 128,
    ) -> dict[str, dict[int, float]]:
        """Return {"recall": {k: score}, "mrr": {k: score}} from one ranking pass.

        ``queries`` holds ``(query_id, query_text, relevant_document_ids)``
        tuples and ``documents`` holds ``(document_id, text)`` tuples.
        MRR@k counts only the first relevant result within the top k.
        """
        if not k or any(kk <= 0 for kk in k):
            raise ValueError("k must contain positive cutoffs")
        if self.tokenizer is None:
            raise RuntimeError("tokenizer not set; call set_tokenizer before evaluation")

        results = {metric: {kk: 0.0 for kk in k} for metric in ("recall", "mrr")}
        if not queries or not documents:
            return results

        self.query_encoder.eval()
        self.document_encoder.eval()

        document_ids = [doc_id for doc_id, _ in documents]
        document_texts = [text for _, text in documents]
        document_emb = self._encode_texts(self.document_encoder, document_texts, self.passage_max_len, batch_size)
        document_emb = F.normalize(document_emb, p=2, dim=-1)

        query_objs = [Query(text=q[1], id=q[0], relevant_document_ids=q[2]) for q in queries]
        query_texts = [q[1] for q in queries]
        query_emb = self._encode_texts(self.query_encoder, query_texts, self.query_max_len, batch_size)
        query_emb = F.normalize(query_emb, p=2, dim=-1)

        scores = query_emb @ document_emb.transpose(0, 1)  # (n_queries, n_documents)
        top = min(max(k), scores.shape[1])
        topk_indices = scores.topk(top, dim=1).indices  # (n_queries, top)

        for kk in k:
            relevancies = []
            width = min(kk, top)
            for i, query_obj in enumerate(query_objs):
                ranked_ids = [document_ids[j] for j in topk_indices[i, :width].tolist()]
                relevancies.append([doc_id in query_obj.relevant_document_ids for doc_id in ranked_ids])
            metrics = Evaluate(relevancies, query_objs)
            for metric, value in metrics.items():
                results[metric][kk] = value
        return results

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

    def load_checkpoint(self, dir: str, tower: str = "hf", device: str = "cpu"):
        """Return ``(query_encoder, document_encoder, meta, tokenizer)``."""
        meta_path = os.path.join(dir, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        tokenizer_dir = os.path.join(dir, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) if os.path.isdir(tokenizer_dir) else None

        if tower == "hf":
            query_encoder = HFEncoder(
                model_name=meta.get("model_name", "bert-base-uncased"),
                pooling_type=meta.get("pooling_type", "cls"),
            )
            document_encoder = HFEncoder(
                model_name=meta.get("model_name", "bert-base-uncased"),
                pooling_type=meta.get("pooling_type", "cls"),
            )
        elif tower == "scratch":
            raise RuntimeError(
                "The from-scratch tower is not available yet: "
                "src/retrievers/models/encoder.py still raises NotImplementedError."
            )
        else:
            raise ValueError(f"unknown tower {tower!r}; expected 'hf' or 'scratch'")

        query_encoder.load_state_dict(torch.load(os.path.join(dir, "query_encoder.pt"), map_location=device))
        document_encoder.load_state_dict(torch.load(os.path.join(dir, "document_encoder.pt"), map_location=device))
        query_encoder.to(device)
        document_encoder.to(device)
        return query_encoder, document_encoder, meta, tokenizer
