"""Mine BM25 hard negatives for the FiQA training questions.

    python scripts/mine_hard_negatives.py --base-path <repo with retrieval_data> --out hard_negatives.json

Keeps the top --depth BM25 passages (k1 = 0.9, b = 0.4, the Anserini setting BEIR
reports) that are not gold. The pool leaves out every val/test gold passage, so
training never sees an evaluation answer. train_dpr.py samples from the front.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-path", default=".")
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from src.data_utils.corpus import Corpus
    from src.retrievers.bm25 import BM25

    start = time.time()
    corpus = Corpus(dataset_name="fiqa", base_dir=f"{args.base_path}/retrieval_data")
    held_out = {doc for split in ("val", "test") for query in corpus.queries(split) for doc in query.relevant_document_ids}
    documents = corpus.all_documents
    keep = [i for i, ids in enumerate(documents["document_id"]) if ids[0] not in held_out]
    pool = documents.select(keep)
    retriever = BM25(documents=pool, k1=args.k1, b=args.b)
    print(f"pool: {pool.num_rows} of {documents.num_rows} passages ({len(held_out)} val/test gold passages left out)")

    queries = corpus.queries("train")
    # ask wider so there are still --depth non-gold passages after dropping gold
    width = args.depth + max(query.getNumberOfRelevantDocuments() for query in queries)
    retrieved = retriever.Lookup([query.getQuery() for query in queries], k=width)
    negatives, gold_ranks = {}, []
    for query, row in zip(queries, retrieved):
        ids = [doc.GetId() for doc in row]
        gold_ranks.append(next((rank for rank, doc in enumerate(ids, 1) if doc in query.relevant_document_ids), None))
        negatives[query.id] = [doc for doc in ids if doc not in query.relevant_document_ids][: args.depth]

    hit_at_1 = sum(rank == 1 for rank in gold_ranks) / len(queries)
    Path(args.out).write_text(json.dumps({
        "params": {"k1": args.k1, "b": args.b, "depth": args.depth, "pool_passages": pool.num_rows},
        "train_bm25_hit_at_1": hit_at_1,
        "seconds": round(time.time() - start, 1),
        "negatives": negatives,
    }))
    print(f"mined {args.depth} negatives for {len(negatives)} questions in {time.time() - start:.0f}s; "
          f"BM25 puts a gold passage first for {hit_at_1:.3f} of training questions")


if __name__ == "__main__":
    main()
