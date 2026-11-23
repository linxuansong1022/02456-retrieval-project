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
    PairCollator,
    PairDataset,
    build_tokenizer,
    pairs_from_corpus,
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
    device = _resolve_device(cfg.device)

    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f"{cfg.base_path}/retrieval_data")
    pairs = pairs_from_corpus(corpus, split="train")
    if cfg.limit_pairs > 0:
        pairs = pairs[: cfg.limit_pairs]
    if cfg.overfit > 0:
        pairs = pairs[: cfg.overfit]

    tokenizer = build_tokenizer(cfg.tokenizer_name)
    torch.manual_seed(cfg.train.seed)
    query_encoder, document_encoder = build_towers(cfg, tokenizer)

    dataset = PairDataset(pairs)
    generator = torch.Generator().manual_seed(cfg.train.seed)
    collator = PairCollator(
        tokenizer,
        query_max_len=cfg.train.query_max_len,
        passage_max_len=cfg.train.passage_max_len,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )

    trainer = Trainer(
        query_encoder,
        document_encoder,
        device=device,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        temperature=cfg.train.temperature,
        normalize=cfg.train.normalize,
        grad_clip=cfg.train.grad_clip,
        warmup_ratio=cfg.train.warmup_ratio,
        log_every=cfg.train.log_every,
        seed=cfg.train.seed,
    )
    trainer.set_tokenizer(
        tokenizer,
        query_max_len=cfg.train.query_max_len,
        passage_max_len=cfg.train.passage_max_len,
    )

    val_queries = corpus.queries("val")
    documents = corpus.all_documents

    def val_fn():
        return trainer.evaluate(
            val_queries, documents, k=tuple(cfg.eval.k), batch_size=cfg.eval.batch_size
        )

    if cfg.overfit > 0:
        steps = cfg.train.epochs * len(train_loader)
        first_loss, last_loss = trainer.fit_overfit(train_loader, steps)
        history = []
        final_train_loss = last_loss
        val = None
        print(f"overfit: first_loss={first_loss:.6f} last_loss={last_loss:.6f} steps={steps}")
    else:
        history = trainer.train(train_loader, epochs=cfg.train.epochs, val_fn=val_fn)
        final_train_loss = history[-1]["train_loss"]
        val = history[-1]["val"]

    tower = cfg.model.tower
    pooling_type = cfg.model.pooling_type
    checkpoint_dir = os.path.join(cfg.output_dir, f"{tower}_{pooling_type}")
    trainer.save_checkpoint(
        checkpoint_dir,
        meta={
            "tower": tower,
            "model": OmegaConf.to_container(cfg.model, resolve=True),
            "model_name": cfg.model.get("model_name", None) or cfg.tokenizer_name,
            "pooling_type": pooling_type,
            "embedding_dim": getattr(query_encoder, "embedding_dim", None),
            "query_max_len": cfg.train.query_max_len,
            "passage_max_len": cfg.train.passage_max_len,
            "epochs": cfg.train.epochs,
            "final_train_loss": final_train_loss,
            "val": val,
        },
    )

    run = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "n_pairs": len(pairs),
        "history": history,
        "wall_clock_seconds": time.time() - start,
        "device": device,
        "git_commit": _git_commit(),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(checkpoint_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=2)

    wall = time.time() - start
    print("=" * 60)
    print(f"pairs: {len(pairs)}")
    print(f"epochs: {cfg.train.epochs}")
    print(f"final loss: {final_train_loss:.6f}")
    if val is not None:
        for k, scores in val.items():
            print(f"@{k}: " + " ".join(f"{name}={value:.4f}" for name, value in scores.items()))
    print(f"seconds: {wall:.1f}")
    print(f"checkpoint: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
