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

This prints recall@k and MRR@k for the train, validation and test splits. Settings are managed with Hydra in `config/`.

MRR@k averages the reciprocal rank of the first relevant result for each query.
A first relevant result at rank 2 scores 0.5; no relevant result in the top k scores 0.
The project's recall@k divides the number of relevant results by `min(k, number of relevant documents)`.
Empty rankings and queries with no relevant documents score 0; an empty query split reports 0 for both metrics.

DPR validation reports both metrics at each configured cutoff (defaults: 1, 5, 10, 20):

```bash
python train_dpr.py device=cuda
python train_dpr.py 'eval.k=[1,5,10]'
```

Validation metrics in checkpoint `meta.json` and the history in `run.json` use
`{"recall": {"1": 0.5, "5": 0.8}, "mrr": {"1": 0.5, "5": 0.6}}`.
`Trainer.evaluate_recall()` continues to return the original `{k: recall}` mapping;
`Trainer.evaluate_metrics()` returns both metric mappings from the same retrieval pass.

## Structure

```
config/           Experiment, dataset, retriever, and model settings.
main.py           Runs retrieval using TF-IDF and a preliminary evaluation pipeline.
src/
├── data_utils/   Dataset loading, corpus, document and query classes.
├── retrievers/   Retriever base class, TF-IDF, BM25 and DPR.
├── training/     DPR loss and trainer.
└── evaluation/   Recall and MRR metrics.
```
