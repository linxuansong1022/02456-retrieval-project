"""Exercise retrieval output and DPR metric persistence with a download-free corpus."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

import main as retrieval_entrypoint
import train_dpr as training_entrypoint


@pytest.fixture
def tiny_corpus():
    split = Dataset.from_dict({
        "question": ["hello", "world"],
        "query_id": ["q0", "q1"],
        "document_id": [["d0"], ["d1"]],
        "text": [["hello"], ["world"]],
    })
    return SimpleNamespace(
        train=split,
        val=split,
        test=split,
        all_documents=split,
        relevant_ids={"q0": ["d0"], "q1": ["d1"]},
    )


@pytest.mark.parametrize("method", ["tf-idf", "bm25"])
def test_retrieval_reports_recall_and_mrr(method, tiny_corpus, monkeypatch, capsys):
    monkeypatch.setattr(retrieval_entrypoint, "Corpus", lambda **kwargs: tiny_corpus)
    cfg = OmegaConf.create({
        "dataset": {"name": "tiny"},
        "base_path": ".",
        "top_k": 1,
        "retriever": {"method": method, "k1": 1.5, "b": 0.75},
    })
    retrieval_entrypoint.main.__wrapped__(cfg)
    output = capsys.readouterr().out
    for split in ("Train", "Val", "Test"):
        assert f"{split} results: {{'recall': 1.0, 'mrr': 1.0}}" in output


def test_dpr_training_reports_and_saves_metrics(tiny_corpus, tiny_tokenizer, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(training_entrypoint, "Corpus", lambda **kwargs: tiny_corpus)
    monkeypatch.setattr(training_entrypoint, "build_tokenizer", lambda name: tiny_tokenizer)
    config_path = Path(__file__).resolve().parents[1] / "config" / "train_dpr.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.model = {
        "tower": "scratch",
        "n_layers": 1,
        "n_heads": 2,
        "emb_dim": 8,
        "hidden_dim": 8,
        "feed_forward_dim": 16,
        "pooling_type": "mean",
        "dropout": 0.0,
    }
    cfg.device = "cpu"
    cfg.train.epochs = 1
    cfg.train.batch_size = 2
    cfg.train.log_every = 0
    cfg.eval.k = [1, 2]
    cfg.output_dir = str(tmp_path)

    training_entrypoint.main.__wrapped__(cfg)

    checkpoint = tmp_path / "scratch_mean"
    meta = json.loads((checkpoint / "meta.json").read_text())
    run = json.loads((checkpoint / "run.json").read_text())
    assert meta["val"] == run["history"][0]["val"]
    assert set(meta["val"]) == {"recall", "mrr"}
    for scores in meta["val"].values():
        assert set(scores) == {"1", "2"}
        assert all(0 <= score <= 1 for score in scores.values())
    assert (checkpoint / "query_encoder.pt").is_file()
    assert (checkpoint / "document_encoder.pt").is_file()
    output = capsys.readouterr().out
    assert "recall@1/2: " in output
    assert "mrr@1/2: " in output
