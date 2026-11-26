# Dense Passage Retrieval on FiQA and SQuAD
![Static Badge](https://img.shields.io/badge/project_status-done-green)

This repository was made for the final project in [02456 Deep Learning](https://kurser.dtu.dk/course/02456) at the Technical University of Denmark.

## Content
<!-- TOC start -->
- [Introduction](#introduction)
   * [Dense retrieval](#dense-retrieval)
   * [Dense Passage Retrieval](#dense-passage-retrieval)
   * [What this project does](#what-this-project-does)
- [Data](#data)
   * [FiQA](#fiqa)
   * [SQuAD](#squad)
   * [How the repo builds retrieval_data](#how-the-repo-builds-retrieval_data)
- [Results](#results)
- [Weights and logs](#weights-and-logs)
- [Reproducing results](#reproducing-results)
   * [Setup](#setup)
   * [Baselines](#baselines)
   * [Hard-negative mining](#hard-negative-mining)
   * [Training DPR](#training-dpr)
   * [Evaluation](#evaluation)
   * [Tests](#tests)
- [Run report](#run-report)
- [Repository structure](#repository-structure)
<!-- TOC end -->

## Introduction

### Dense retrieval
Retrieval means ranking a collection of passages so that the ones relevant to a query come first. TF-IDF and BM25 do this from word statistics: they score the terms a query and a passage share, weighted by how rare each term is (BM25 also saturates term frequency and normalises for passage length). They are fast and strong, but they cannot match a question to an answer that uses different words.

Dense retrieval learns the matching instead. A neural encoder maps every query and every passage to a vector, and relevance is the similarity of the two vectors. Passages are encoded once into an index; at query time only the query is encoded, and the nearest passage vectors are the answer.

### Dense Passage Retrieval
[Dense Passage Retrieval](https://aclanthology.org/2020.emnlp-main.550/) (DPR, EMNLP 2020) trains two Transformer encoders, one for queries and one for passages, with a contrastive loss. A batch holds $B$ questions, each with its gold passage. Every question is scored against every passage in the batch, so the other $B-1$ gold passages act as free negatives (in-batch negatives). DPR adds a hard negative per question as well: a passage BM25 ranks high for that question that is not a gold answer, which is much harder to separate than a random passage.

The loss for question $q_i$ is a softmax cross-entropy over all passages $p_j$ in the batch, with its own gold passage $p_i$ as the target:

$$
\mathcal{L}_i = -\log \frac{\exp\big(s(q_i, p_i) / \tau\big)}{\sum_{j} \exp\big(s(q_i, p_j) / \tau\big)}
$$

The course brief states it with the raw inner product $s(q, p) = f_\theta(p)^\top g_\phi(q)$ and no temperature. This repo's `DPRLoss` (`src/training/loss.py`) L2-normalises both vectors, so $s$ is a cosine, and divides by a temperature $\tau$. When a passage in the batch is also a gold answer for $q_i$ (FiQA questions often have several), `ContrastiveCollator` masks it out of that question's softmax so a right answer never counts as a wrong one.

### What this project does
The project brief asks for TF-IDF and BM25 from scratch, the key parts of a Transformer encoder, the DPR training setup, a vector index for dense search, retrieval metrics (precision@k, recall@k, MRR) and a comparison of the methods on FiQA and SQuAD. This repo has:

- TF-IDF and BM25 retrievers (`src/retrievers/tf_idf.py`, `src/retrievers/bm25.py`).
- A Transformer encoder written from scratch (`src/retrievers/models/encoder.py`) and a wrapper around pretrained Hugging Face encoders (`src/retrievers/models/hf_encoder.py`), both with CLS, mean or max pooling.
- The DPR loss, BM25 hard-negative mining, a gradient-cached trainer with early stopping, and a DPR retriever that does exact cosine search over the encoded corpus.
- Recall, precision, MRR and nDCG at k (`src/evaluation/metrics.py`).

Three DPR models were trained and scored on the test split against TF-IDF and BM25 with the same metric code:

1. FiQA, BERT-base, two towers, in-batch negatives only, batch 16, $\tau = 1$. It ended below both lexical baselines.
2. FiQA, ModernBERT-base, one shared tower, mean pooling, one BM25 hard negative per question, 128 questions per step, $\tau = 0.02$, 512-token passages. It beats both BM25 settings on every reported metric.
3. SQuAD, the same recipe as 2. It beats BM25 at recall@10 and recall@20, and stays behind it at nDCG@10, MRR@10, recall@1 and recall@5.

## Data
Both datasets come from the Hugging Face Hub and are downloaded by the code on first use (`src/data_utils/download_retrieval_datasets.py`). Nothing needs to be placed by hand.

### FiQA
[`vibrantlabsai/fiqa`](https://huggingface.co/datasets/vibrantlabsai/fiqa), config `main`: personal finance questions from forum threads, each with the answers that were posted to it (`ground_truths`). The repo keeps the published splits.

| Split | Questions |
|---|---|
| train | 5,500 |
| validation | 500 |
| test | 648 |

The search corpus is the union of the gold answers of all three splits: 17,110 passages, every one of them the right answer to some question.

> [!NOTE]
> The [BEIR](https://github.com/beir-cellar/beir) version of FiQA searches 57,638 passages. Numbers here come from the smaller corpus above, so they are only comparable to each other, not to published FiQA results.

### SQuAD
[`rajpurkar/squad`](https://huggingface.co/datasets/rajpurkar/squad) (SQuAD v1.1): crowd-sourced questions on Wikipedia paragraphs; the paragraph (`context`) is the gold passage. SQuAD has no test split, so the repo builds one:

| Split | Source | Questions |
|---|---|---|
| train | SQuAD train, minus the held-out articles | 78,872 |
| validation | SQuAD train, every 10th article title in sorted order | 8,727 |
| test | SQuAD validation | 10,570 |

Splitting by article keeps every paragraph of an article in one split. The corpus is the union of the context paragraphs of all three splits: 20,958 passages.

### How the repo builds retrieval_data
`Corpus(dataset_name, base_dir=f"{base_path}/retrieval_data")` in `src/data_utils/corpus.py` does the whole build the first time a dataset is used:

1. Downloads the dataset into `retrieval_data/raw/` (for SQuAD, splits it as above).
2. Renames the answer column (`ground_truths` or `context`) to `text`, a list of gold passages per question.
3. Gives every question an id `q<n>` and every distinct passage text an id `doc<n>`, shared across splits, and records the gold passage ids of each question in `relevant_ids.json`.
4. Saves the splits with `save_to_disk` and deletes `retrieval_data/raw/`.

```
retrieval_data/
├── fiqa
│   ├── dataset_dict.json
│   ├── relevant_ids.json
│   ├── train/
│   ├── validation/
│   └── test/
└── squad
    └── (same layout)
```

Later runs load from disk. With `HF_OFFLINE=1` set, the download only reads the local Hugging Face cache.

## Results
Test split, every passage in the corpus searched exactly, same metric code for every row. Full val and test tables at k = 1, 5, 10 and 20 are in [`experiments/`](experiments).

**FiQA** (648 questions, 17,110 passages)

| Method | nDCG@10 | MRR@10 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Hit@20 |
|---|---|---|---|---|---|---|---|
| DPR, ModernBERT-base (second run) | 0.3658 | 0.4215 | 0.3148 | 0.3745 | 0.4576 | 0.5533 | 0.7377 |
| DPR, BERT-base (first run) | 0.0608 | 0.0802 | 0.0540 | 0.0584 | 0.0803 | 0.1152 | 0.2238 |
| BM25, k1 1.5, b 0.75 | 0.2803 | 0.3481 | 0.2685 | 0.2855 | 0.3437 | 0.4186 | 0.6188 |
| BM25, k1 0.9, b 0.4 | 0.2576 | 0.3184 | 0.2423 | 0.2569 | 0.3187 | 0.3920 | 0.5880 |
| TF-IDF | 0.1222 | 0.1579 | 0.1003 | 0.1164 | 0.1639 | 0.2399 | 0.4151 |

All FiQA rows come from one evaluation pass (`experiments/fiqa-modernbert/metrics/eval.csv`), the same one behind the second-run table in the run report. Section 05 of the report shows an earlier pass over the same BERT checkpoint whose scores differ from this row in the third or fourth decimal.

**SQuAD** (10,570 questions, 20,958 passages)

| Method | nDCG@10 | MRR@10 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Hit@20 |
|---|---|---|---|---|---|---|---|
| DPR, ModernBERT-base | 0.7739 | 0.7306 | 0.6340 | 0.8588 | 0.9079 | 0.9420 | 0.9420 |
| BM25, k1 0.9, b 0.4 | 0.8002 | 0.7684 | 0.6974 | 0.8606 | 0.8987 | 0.9265 | 0.9265 |
| BM25, k1 1.5, b 0.75 | 0.7935 | 0.7602 | 0.6851 | 0.8571 | 0.8966 | 0.9256 | 0.9256 |
| TF-IDF | 0.4208 | 0.3667 | 0.2750 | 0.4903 | 0.5950 | 0.7048 | 0.7048 |

> [!IMPORTANT]
> Recall@k here is the repo's `calculateRecall`: hits in the top k divided by min(k, number of gold passages), not by the number of gold passages. Hit@k is the share of questions with at least one gold passage in the top k. Every SQuAD question has one gold passage, so the two coincide there.

## Weights and logs
The two ModernBERT models are on the Hugging Face Hub. Each repository holds the shared encoder (`query_encoder.pt`), `meta.json`, `run.json`, the tokenizer and a model card with the recipe and test scores.

| Model | Repository |
|---|---|
| FiQA, ModernBERT-base | [hananinas/02456-dpr-fiqa-modernbert](https://huggingface.co/hananinas/02456-dpr-fiqa-modernbert) |
| SQuAD, ModernBERT-base | [hananinas/02456-dpr-squad-modernbert](https://huggingface.co/hananinas/02456-dpr-squad-modernbert) |

The logs and metrics of every run are committed under [`experiments/`](experiments):

```
experiments/
├── fiqa-bert/                 first run: BERT-base, in-batch negatives
├── fiqa-temperature-check/    one epoch on 1,024 questions at tau 1.0, 0.05 and 0.02
├── fiqa-modernbert/           second run: ModernBERT-base, the FiQA model on the Hub
└── squad-modernbert/          the same recipe on SQuAD, the SQuAD model on the Hub
```

Each run has `logs/` (the `run.json` and `meta.json` written by `train_dpr.py`, the raw training log, and the mining, baseline and evaluation logs where the run had them) and `metrics/` (`history.csv` with validation scores after every epoch, and for the two ModernBERT runs `eval.csv` with val and test scores of every method at k = 1, 5, 10 and 20). The FiQA `eval.csv` also holds the test scores of the BERT run.

## Reproducing results
> [!NOTE]
> Every run here was trained on one 6 GB laptop GPU (NVIDIA RTX A3000) with Python 3.14. Wall clock from `run.json`: 6,002 s for the BERT run, 7,387 s for the FiQA ModernBERT run and 11,844 s for the SQuAD run.

### Setup
Clone the repository and switch to this branch:
```bash
git clone https://github.com/linxuansong1022/02456-retrieval-project.git
cd 02456-retrieval-project
git checkout feat/training-visualization
```
**(Optional)** Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate
```
**(Required)** Install the requirements (Python 3.10 or newer):
```bash
python -m pip install -r requirements.txt
```

All commands below run from the repository root and write into `out/`:
```bash
mkdir -p out/diag out/squad
```

### Baselines
`main.py` runs one retriever from `config/fiqa_experiment.yaml` and prints recall, precision, MRR and nDCG at `top_k` (default 10) for the val and test splits. Hydra overrides pick the retriever and dataset:
```bash
python main.py                                   # TF-IDF on FiQA
python main.py retriever=bm25                    # BM25, k1 1.5, b 0.75 from config/retriever/bm25.yaml
python main.py dataset=squad retriever=bm25 top_k=20
```

The baseline rows in the results tables come from `scripts/training_run_data.py`, which scores at k = 1, 5, 10 and 20 and merges each method into one JSON file:
```bash
python scripts/training_run_data.py evaluate --methods bm25 tf-idf --out out/eval.json
python scripts/training_run_data.py evaluate --methods bm25 --k1 1.5 --b 0.75 --label bm25-1.5-0.75 --out out/eval.json
```
Add `--dataset squad` and `--out out/squad/eval.json` for SQuAD. `--methods bm25` defaults to k1 0.9, b 0.4.

### Hard-negative mining
`scripts/mine_hard_negatives.py` runs BM25 (k1 0.9, b 0.4) for every training question and keeps the 30 best passages that are not gold. Every validation and test gold passage is left out of the pool, so training never sees an evaluation answer.
```bash
python scripts/mine_hard_negatives.py --out out/hard_negatives.json
python scripts/mine_hard_negatives.py --dataset squad --out out/squad/hard_negatives.json
```
On the logged runs this takes about 10 s for FiQA and 72 s for SQuAD.

### Training DPR
`train_dpr.py` trains from a Hydra config in `config/` and writes `<output_dir>/<tower>_<pooling>/` with the checkpoint, `meta.json` and `run.json`. With `train.select` set, the epoch with the best validation score is the one kept on disk.

The ModernBERT recipe lives in `config/train_dpr_modernbert.yaml` and `config/model/modernbert.yaml`:
```yaml
# config/model/modernbert.yaml
tower: hf
model_name: answerdotai/ModernBERT-base
pooling_type: mean  # cls | mean | max
freeze: false
shared: true  # two trainable towers need about 4.8 GB for fp32 weights, grads and Adam state alone
```
```yaml
# config/train_dpr_modernbert.yaml (train block, abridged)
train:
  unit: questions
  epochs: 7
  batch_size: 128        # questions per optimiser step
  n_hard: 1              # BM25 negatives per question, so 256 passages per step
  hard_pool: 10          # drawn each epoch from the 10 best-ranked non-gold BM25 passages
  chunk_tokens: 16384    # micro-batch size in padded tokens (32 passages of 512)
  lr: 5e-5
  temperature: 0.02
  passage_max_len: 512
  select: 10.ndcg
  patience: 3
```

> [!TIP]
> The 128-question batch fits in 6 GB because the trainer uses a gradient cache: it embeds the whole batch without keeping activations, computes the loss and its gradient with respect to the embeddings, then replays one micro-batch at a time through the encoder. `tests/test_trainer.py` checks that the update matches plain autograd.

**FiQA, ModernBERT-base.** The temperature check (one epoch on 1,024 questions per temperature), then the full run:
```bash
for tau in 1.0 0.05 0.02; do
  python train_dpr.py --config-name train_dpr_modernbert \
    train.hard_negatives_file=out/hard_negatives.json output_dir=out/diag/tau-$tau \
    limit_pairs=1024 train.epochs=1 train.temperature=$tau \
    2>&1 | tee out/diag/tau-$tau.log
done
python train_dpr.py --config-name train_dpr_modernbert \
  train.hard_negatives_file=out/hard_negatives.json output_dir=out/full \
  2>&1 | tee out/train.log
```
All 7 epochs ran; epoch 5 had the best validation nDCG@10 (0.3985) and is the one kept.

**SQuAD, ModernBERT-base.** `config/train_dpr_modernbert_squad.yaml` reuses the FiQA recipe with `dataset: squad`. One pass over SQuAD is 617 steps, so `train.steps_per_epoch: 100` validates every 100 steps, for 9 rounds:
```bash
python train_dpr.py --config-name train_dpr_modernbert_squad \
  train.hard_negatives_file=out/squad/hard_negatives.json output_dir=out/squad \
  2>&1 | tee out/squad/train.log
```
Every round improved on the last; round 9 is kept (validation nDCG@10 0.7675).

**FiQA, BERT-base (first run).** `config/train_dpr.yaml` with two untied `bert-base-uncased` towers, CLS pooling, in-batch negatives only:
```bash
python train_dpr.py model=hf_encoder train.epochs=3 train.batch_size=16 train.lr=2e-5 device=cuda \
  output_dir=out/bert 2>&1 | tee out/bert.log
```
Batch 16 is what fits two BERT towers on 6 GB with gradient checkpointing. On a larger GPU, `python train_dpr.py device=cuda` runs the config default of batch 32. The repo's earlier SLURM job used that variant and was never submitted to a cluster, so no result here comes from it.

**Pooling ablation (optional).** One BERT-base run per pooling type, each written to `checkpoints/hf_<pooling>/`. No results from it are committed.
```bash
mkdir -p checkpoints
for p in cls mean max; do
  python train_dpr.py model.pooling_type=$p train.batch_size=16 device=cuda 2>&1 | tee checkpoints/ablation_$p.log
done
```

### Evaluation
Score a checkpoint on the val and test splits and merge it into the baseline file:
```bash
python scripts/training_run_data.py evaluate --methods dpr --label dpr-modernbert \
  --checkpoint out/full/hf_mean --device cuda --out out/eval.json
python scripts/training_run_data.py evaluate --dataset squad --methods dpr \
  --checkpoint out/squad/hf_mean --device cuda --out out/squad/eval.json
```
`main.py` takes a checkpoint too:
```bash
python main.py retriever=dpr retriever.checkpoint_dir=out/full/hf_mean device=cuda
```

To score the published weights instead of training, download them and pass the folder as the checkpoint:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('hananinas/02456-dpr-fiqa-modernbert', local_dir='out/hub/fiqa')"
python scripts/training_run_data.py evaluate --methods dpr --checkpoint out/hub/fiqa --device cuda --out out/hub/fiqa-eval.json
```

> [!WARNING]
> DPR evaluation encodes the whole corpus first: about 17k passages for FiQA and 21k for SQuAD at up to 512 tokens. Use `--device cuda` when a GPU is available.

### Tests
```bash
pytest
```

## Run report
[`docs/dpr-training-run.html`](docs/dpr-training-run.html) walks through all three runs: the model and loss, every training step, validation after each epoch, the comparison with TF-IDF and BM25, the questions where DPR and BM25 disagree, and the SQuAD run. It is a single static file; open it in a browser. A full-page capture is in [`docs/dpr-training-run.png`](docs/dpr-training-run.png).

Every number on the page is written into it by `scripts/training_run_data.py` (`page`, `second` and `squad` subcommands) from the run's `train.log`, `run.json` and `eval.json`. `scripts/capture_training_page.mjs` drives the page's controls with Playwright, fails on any console error or warning, and saves the capture.

## Repository structure
```
config/                 Hydra configs: experiments, datasets, retrievers and encoders
docs/                   run report and its capture
experiments/            committed logs and metrics of every run
scripts/
├── mine_hard_negatives.py      BM25 hard negatives for the training questions
├── training_run_data.py        evaluation at k = 1, 5, 10, 20 and the report data
└── capture_training_page.mjs   report capture and console check
src/
├── data_utils/         dataset download, corpus build, query and document classes
├── retrievers/         TF-IDF, BM25 and DPR retrievers; encoders and pooling in models/
├── training/           loss, pair and question datasets, collators, trainer
└── evaluation/         recall, precision, MRR and nDCG
tests/                  pytest suite
main.py                 run one retriever and print its scores
train_dpr.py            train a DPR model
```
