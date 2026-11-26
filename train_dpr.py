"""DPR training entry point.

    python train_dpr.py device=cuda
    python train_dpr.py overfit=64 train.epochs=30 train.batch_size=16 device=cuda

Checkpoints land in <output_dir>/<tower>_<pooling>/ with a run.json next to them.
"""
from __future__ import annotations

import os

# The default allocator fragments on a 6 GB card with two BERT towers.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import datetime
import json
import subprocess
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data_utils.corpus import Corpus
from src.training.build import build_towers
from src.training.pairs import (
    ContrastiveCollator,
    PairCollator,
    PairDataset,
    QuestionDataset,
    build_tokenizer,
    pairs_from_corpus,
    passages_from_corpus,
    questions_from_corpus,
)
from src.training.trainer import Trainer

# Hydra 1.3 passes a non-str help value that Python 3.14's argparse rejects.
import argparse

if hasattr(argparse.ArgumentParser, "_check_help") and not hasattr(argparse.ArgumentParser, "_hydra_py314_shim"):
    _original_check_help = argparse.ArgumentParser._check_help

    def _check_help(self, action):
        if not isinstance(action.help, str):
            return
        _original_check_help(self, action)

    argparse.ArgumentParser._check_help = _check_help
    argparse.ArgumentParser._hydra_py314_shim = True


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return device_cfg


@hydra.main(config_path="config", config_name="train_dpr.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    start = time.time()
    git_commit = _git_commit()  # at start: the worktree may move on while a long run is going
    device = _resolve_device(cfg.device)
    train_cfg = cfg.train

    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f"{cfg.base_path}/retrieval_data")
    tokenizer = build_tokenizer(cfg.tokenizer_name)
    torch.manual_seed(train_cfg.seed)
    query_encoder, document_encoder = build_towers(cfg, tokenizer)
    generator = torch.Generator().manual_seed(train_cfg.seed)

    unit = train_cfg.get("unit", "pairs")
    if unit == "pairs":
        # One item per (question, gold passage), in-batch negatives only (the first run).
        examples = pairs_from_corpus(corpus, split="train")
        if cfg.limit_pairs > 0:
            examples = examples[: cfg.limit_pairs]
        if cfg.overfit > 0:
            examples = examples[: cfg.overfit]
        dataset = PairDataset(examples)
        collator = PairCollator(
            tokenizer, query_max_len=train_cfg.query_max_len, passage_max_len=train_cfg.passage_max_len
        )
    elif unit == "questions":
        # One item per question; a gold passage and n_hard mined negatives are drawn each epoch.
        hard_negatives = None
        if train_cfg.n_hard > 0:
            with open(train_cfg.hard_negatives_file) as f:
                hard_negatives = json.load(f)["negatives"]
        examples = questions_from_corpus(corpus, split="train", hard_negatives=hard_negatives)
        if cfg.limit_pairs > 0:
            examples = examples[: cfg.limit_pairs]
        if cfg.overfit > 0:
            examples = examples[: cfg.overfit]
        dataset = QuestionDataset(
            examples,
            passages_from_corpus(corpus),
            n_hard=train_cfg.n_hard,
            hard_pool=train_cfg.hard_pool,
            seed=train_cfg.seed,
        )
        collator = ContrastiveCollator(
            tokenizer,
            query_max_len=train_cfg.query_max_len,
            passage_max_len=train_cfg.passage_max_len,
            chunk_tokens=train_cfg.chunk_tokens,
            chunk_rows=train_cfg.chunk_rows,
        )
        print(
            f"questions[train]: {len(examples)} batch={train_cfg.batch_size} "
            f"passages_per_batch={train_cfg.batch_size * (1 + train_cfg.n_hard)} "
            f"hard_negatives_per_question={train_cfg.n_hard} drawn_from_top={train_cfg.hard_pool}"
        )
    else:
        raise ValueError(f"unknown train.unit {unit!r}; expected 'pairs' or 'questions'")

    train_loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=train_cfg.get("num_workers", 0),
    )

    trainer = Trainer(
        query_encoder,
        document_encoder,
        device=device,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        temperature=train_cfg.temperature,
        normalize=train_cfg.normalize,
        grad_clip=train_cfg.grad_clip,
        warmup_ratio=train_cfg.warmup_ratio,
        log_every=train_cfg.log_every,
        seed=train_cfg.seed,
        bf16=train_cfg.get("bf16", False),
    )
    trainer.set_tokenizer(
        tokenizer,
        query_max_len=train_cfg.query_max_len,
        passage_max_len=train_cfg.passage_max_len,
    )

    val_queries = corpus.queries("val")
    documents = corpus.all_documents

    def val_fn():
        return trainer.evaluate(
            val_queries, documents, k=tuple(cfg.eval.k), batch_size=cfg.eval.batch_size
        )

    tower = cfg.model.tower
    pooling_type = cfg.model.pooling_type
    checkpoint_dir = os.path.join(cfg.output_dir, f"{tower}_{pooling_type}")

    def save(epoch, final_train_loss, val):
        trainer.save_checkpoint(
            checkpoint_dir,
            meta={
                "tower": tower,
                "model": OmegaConf.to_container(cfg.model, resolve=True),
                "model_name": cfg.model.get("model_name", None) or cfg.tokenizer_name,
                "pooling_type": pooling_type,
                "embedding_dim": getattr(query_encoder, "embedding_dim", None),
                "query_max_len": train_cfg.query_max_len,
                "passage_max_len": train_cfg.passage_max_len,
                "epochs": cfg.train.epochs,
                "epoch": epoch,
                "final_train_loss": final_train_loss,
                "val": val,
                "git_commit": git_commit,
            },
        )

    # Early stopping on one validation number, e.g. "10.ndcg" for nDCG@10.
    select_key = train_cfg.get("select", None)
    select = None
    if select_key:
        k, metric = select_key.split(".")
        select = lambda val: val[int(k)][metric]  # noqa: E731

    def write_run(history, finished):
        run = {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "n_pairs": len(examples),
            "unit": unit,
            "history": history,
            "selected_epoch": next((e["epoch"] for e in reversed(history) if e.get("best")), None),
            "finished": finished,
            "wall_clock_seconds": time.time() - start,
            "device": device,
            "git_commit": git_commit,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "run.json"), "w") as f:
            json.dump(run, f, indent=2)

    def on_epoch_end(history):
        write_run(history, finished=False)

    if cfg.overfit > 0:
        steps = cfg.train.epochs * len(train_loader)
        first_loss, last_loss = trainer.fit_overfit(train_loader, steps)
        history = []
        final_train_loss = last_loss
        val = None
        print(f"overfit: first_loss={first_loss:.6f} last_loss={last_loss:.6f} steps={steps}")
        save(cfg.train.epochs, final_train_loss, val)
    elif select is not None:
        # The checkpoint on disk is always the best validation epoch so far, not the last one.
        history = trainer.train(
            train_loader,
            epochs=cfg.train.epochs,
            val_fn=val_fn,
            select=select,
            patience=train_cfg.get("patience", None),
            on_best=lambda entry: save(entry["epoch"], entry["train_loss"], entry["val"]),
            on_epoch_end=on_epoch_end,
        )
        chosen = next(e for e in reversed(history) if e.get("best"))
        final_train_loss = chosen["train_loss"]
        val = chosen["val"]
    else:
        history = trainer.train(train_loader, epochs=cfg.train.epochs, val_fn=val_fn, on_epoch_end=on_epoch_end)
        final_train_loss = history[-1]["train_loss"]
        val = history[-1]["val"]
        save(cfg.train.epochs, final_train_loss, val)

    write_run(history, finished=True)

    wall = time.time() - start
    print("=" * 60)
    print(f"{unit}: {len(examples)}")
    print(f"epochs run: {len(history) or cfg.train.epochs}")
    print(f"train loss (kept epoch): {final_train_loss:.6f}")
    if val is not None:
        for k, scores in val.items():
            print(f"@{k}: " + " ".join(f"{name}={value:.4f}" for name, value in scores.items()))
    print(f"seconds: {wall:.1f}")
    print(f"checkpoint: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
