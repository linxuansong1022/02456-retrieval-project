# 02456 Retrieval Project

Starter code for information retrieval project work in the 02456 Deep Learning course (Fall 2026).

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py            # Run TF-IDF on FIQA
python main.py top_k=20   # Override a config value
```

This prints recall@k for the train, validation and test splits. Settings are managed with Hydra in `config/`.

## Training DPR

```bash
python train_dpr.py device=cuda                                   # first run: BERT-base pairs, in-batch negatives
python scripts/mine_hard_negatives.py --out hard_negatives.json   # BM25 negatives for the training questions
python train_dpr.py --config-name train_dpr_modernbert train.hard_negatives_file=hard_negatives.json
```

The last two commands are the ModernBERT run (`config/train_dpr_modernbert.yaml`): one shared tower,
one BM25 hard negative per question, 128 questions per step through a gradient cache, tau 0.02,
512-token passages and early stopping on validation nDCG@10. `scripts/training_run_data.py` scores
DPR, BM25 and TF-IDF with the same metric code and writes the numbers into `docs/dpr-training-run.html`.

## Structure

```
config/           Experiment, dataset, retriever, and model settings.
main.py           Runs retrieval using TF-IDF and a preliminary evaluation pipeline.
src/
├── data_utils/   Dataset loading, corpus, document and query classes.
├── retrievers/   Retriever base class, TF-IDF, BM25 and DPR.
├── training/     DPR loss and trainer.
└── evaluation/   Recall metric.
```
