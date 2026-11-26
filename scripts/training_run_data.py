"""Collect the numbers behind docs/dpr-training-run.html.

    # retrieval scores at k = 1, 5, 10, 20 on val and test (merged into --out)
    python scripts/training_run_data.py evaluate --methods bm25 tf-idf --out eval.json
    python scripts/training_run_data.py evaluate --methods dpr --checkpoint <dir> --device cuda --out eval.json

    # parse train.log and run.json, then write the data block into the page
    python scripts/training_run_data.py page --log train.log --run run.json --eval eval.json --base-path <repo with retrieval_data>

    # the same for the second (ModernBERT) run, plus its temperature check and mining stats
    python scripts/training_run_data.py second --log train.log --run run.json --eval eval.json \
        --diag <dir with tau-*/hf_*/run.json> --mining hard_negatives.json

Every number on the page comes from one of these files.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KS = (1, 5, 10, 20)
SPLITS = ("val", "test")
PAGE = ROOT / "docs" / "dpr-training-run.html"
DATA_OPEN = '<script id="run-data" type="application/json">'
DATA_CLOSE = "</script>"


def evaluate(args) -> None:
    from src.data_utils.corpus import Corpus
    from src.evaluation.metrics import Evaluate

    corpus = Corpus(dataset_name="fiqa", base_dir=f"{args.base_path}/retrieval_data")
    documents = corpus.all_documents
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() else {}
    results["n_passages"] = documents.num_rows

    for method in args.methods:
        start = time.time()
        if method == "bm25":
            from src.retrievers.bm25 import BM25
            retriever = BM25(documents=documents, k1=args.k1, b=args.b)
            params = {"k1": args.k1, "b": args.b}
        elif method == "tf-idf":
            from src.retrievers.tf_idf import TFIDF
            retriever = TFIDF(documents=documents)
            params = {}
        elif method == "dpr":
            from src.retrievers.dpr import DPR
            retriever = DPR.from_checkpoint(documents, args.checkpoint, batch_size=128, device=args.device)
            params = {"checkpoint": str(args.checkpoint), "device": args.device}
        else:
            raise ValueError(f"unknown method {method!r}")
        index_seconds = time.time() - start

        scores = {}
        for split in SPLITS:
            queries = corpus.queries(split)
            # Same as retrieveQueryAndGetRelevancies, but keeping the retrieved ids for the examples.
            retrieved = retriever.Lookup(queries=[query.getQuery() for query in queries], k=max(KS))
            relevancies = [[query.isDocumentRelevant(doc) for doc in row] for query, row in zip(queries, retrieved)]
            # Then as Trainer.evaluate: retrieve max(k) once and cut the list per k.
            scores[split] = {
                "n_queries": len(queries),
                "scores": {str(k): Evaluate([row[:k] for row in relevancies], queries, k=k) for k in KS},
                # Share of queries with at least one gold passage in the top k: the paper's "top-k accuracy".
                "hit": {str(k): sum(any(row[:k]) for row in relevancies) / len(queries) for k in KS},
                "per_query": {
                    query.id: {
                        "first_gold": next((rank for rank, hit in enumerate(row, 1) if hit), None),
                        "top": [doc.GetId() for doc in docs[:3]],
                    }
                    for query, row, docs in zip(queries, relevancies, retrieved)
                },
            }
        results.setdefault("methods", {})[args.label or method] = {
            "params": params,
            "index_seconds": round(index_seconds, 1),
            "total_seconds": round(time.time() - start, 1),
            **scores,
        }
        out.write_text(json.dumps(results, indent=2))
        print(args.label or method, json.dumps({split: scores[split]["scores"] for split in SPLITS}))


# tqdm prints each step twice: once from set_postfix with the new loss, once from
# the counter update. Both carry "loss=<value>", so consecutive repeats collapse.
_SEGMENT = re.compile(r"epoch (\d+)/(\d+):.*?\| *(\d+)/(\d+) \[.*?loss=([0-9.]+)\]")


def parse_step_losses(log_text: str) -> dict[int, list[float]]:
    losses: dict[int, list[float]] = {}
    last: tuple[int, str] | None = None
    for segment in re.split(r"[\r\n]", log_text):
        match = _SEGMENT.search(segment)
        if not match:
            continue
        epoch, value = int(match.group(1)), match.group(5)
        if last == (epoch, value):
            continue
        last = (epoch, value)
        losses.setdefault(epoch, []).append(float(value))
    return losses


def _clip(text: str, limit: int = 420) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def build_examples(evaluation: dict, base_path: str, per_group: int = 3) -> dict | None:
    """Compare DPR and BM25 per test question: a 2x2 count on hit@20 and a few real questions per cell."""
    methods = evaluation.get("methods", {})
    if "dpr" not in methods or "bm25" not in methods:
        return None
    from src.data_utils.corpus import Corpus

    corpus = Corpus(dataset_name="fiqa", base_dir=f"{base_path}/retrieval_data")
    passages = {}
    for texts, ids in zip(corpus.all_documents["text"], corpus.all_documents["document_id"]):
        passages[ids[0]] = texts[0]

    dpr, bm25 = methods["dpr"]["test"]["per_query"], methods["bm25"]["test"]["per_query"]
    groups = {"dpr_only": [], "bm25_only": [], "both": [], "neither": []}
    counts = dict.fromkeys(groups, 0)
    for query in corpus.queries("test"):
        d, b = dpr[query.id]["first_gold"], bm25[query.id]["first_gold"]
        group = "both" if d and b else "dpr_only" if d else "bm25_only" if b else "neither"
        counts[group] += 1
        # Examples are the first questions in test order whose winner put a gold passage at rank 1.
        if len(groups[group]) >= per_group or (group == "dpr_only" and d != 1) or (group == "bm25_only" and b != 1):
            continue
        if group == "both" and not (d == 1 and b == 1):
            continue
        groups[group].append({
            "query_id": query.id,
            "question": query.getQuery(),
            "n_gold": query.getNumberOfRelevantDocuments(),
            "gold": _clip(passages[query.relevant_document_ids[0]]),
            "methods": {
                name: {
                    "first_gold": ranks[query.id]["first_gold"],
                    "top1": _clip(passages[ranks[query.id]["top"][0]]),
                    "top1_is_gold": ranks[query.id]["top"][0] in query.relevant_document_ids,
                }
                for name, ranks in (("dpr", dpr), ("bm25", bm25))
            },
        })
    return {"counts": counts, "groups": groups}


def page(args) -> None:
    log_text = Path(args.log).read_text(errors="replace")
    steps = parse_step_losses(log_text)
    pairs = re.search(r"pairs_from_corpus\[train\]: rows=(\d+) pairs_before=(\d+) pairs_after=(\d+)", log_text)

    run = json.loads(Path(args.run).read_text()) if args.run and Path(args.run).exists() else None
    evaluation = json.loads(Path(args.eval).read_text()) if args.eval and Path(args.eval).exists() else None
    examples = build_examples(evaluation, args.base_path) if evaluation else None
    if evaluation:
        for method in evaluation["methods"].values():
            for split in SPLITS:
                method[split].pop("per_query", None)

    steps_per_epoch = int(re.search(r"epoch \d+/\d+:.*?\| *\d+/(\d+) \[", log_text).group(1))
    total_epochs = int(re.search(r"epoch \d+/(\d+):", log_text).group(1))
    if run is not None:
        for entry in run["history"]:
            logged = steps.get(entry["epoch"], [])
            mean = sum(logged) / len(logged)
            if len(logged) != steps_per_epoch or abs(mean - entry["train_loss"]) > 1e-5:
                raise SystemExit(
                    f"epoch {entry['epoch']}: parsed {len(logged)} steps with mean {mean:.6f}, "
                    f"run.json says {entry['train_loss']:.6f}"
                )

    data = {
        "steps_per_epoch": steps_per_epoch,
        "epochs": total_epochs,
        "train_rows": int(pairs.group(1)),
        "pairs_before": int(pairs.group(2)),
        "pairs_after": int(pairs.group(3)),
        "step_losses": {str(epoch): values for epoch, values in sorted(steps.items())},
        "run": None,
        "eval": evaluation,
        "examples": examples,
    }
    if run is not None:
        data["run"] = {
            "config": {
                "model": run["config"]["model"],
                "train": run["config"]["train"],
                "eval": run["config"]["eval"],
                "tokenizer_name": run["config"]["tokenizer_name"],
            },
            "n_pairs": run["n_pairs"],
            "history": run["history"],
            "wall_clock_seconds": run["wall_clock_seconds"],
            "device": run["device"],
            "git_commit": run["git_commit"],
            "created_at": run["created_at"],
        }

    html = PAGE.read_text()
    start = html.index(DATA_OPEN) + len(DATA_OPEN)
    end = html.index(DATA_CLOSE, start)
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    PAGE.write_text(html[:start] + blob + html[end:])
    done = {epoch: len(values) for epoch, values in steps.items()}
    print(f"steps parsed per epoch: {done}; run.json: {'yes' if run else 'no'}; eval: {'yes' if evaluation else 'no'}")


RUN2_OPEN = '<script id="run2-data" type="application/json">'


def _scores_only(method: dict) -> dict:
    return {key: value for key, value in method.items() if key not in SPLITS} | {
        split: {key: value for key, value in method[split].items() if key != "per_query"} for split in SPLITS
    }


def steps_of(log: Path) -> list[float]:
    if not log.exists():
        return []
    return [value for values in parse_step_losses(log.read_text(errors="replace")).values() for value in values]


def second(args) -> None:
    """Write the data block for the second run (section 08 of the page)."""
    log_text = Path(args.log).read_text(errors="replace")
    steps = parse_step_losses(log_text)
    run = json.loads(Path(args.run).read_text())
    for entry in run["history"]:
        logged = steps.get(entry["epoch"], [])
        mean = sum(logged) / len(logged)
        if abs(mean - entry["train_loss"]) > 1e-5:
            raise SystemExit(f"epoch {entry['epoch']}: parsed mean {mean:.6f}, run.json says {entry['train_loss']:.6f}")

    diag = []
    for path in sorted(Path(args.diag).glob("tau-*/hf_*/run.json")) if args.diag else []:
        d = json.loads(path.read_text())
        last = d["history"][-1]
        diag.append({
            "temperature": d["config"]["train"]["temperature"],
            "questions": d["n_pairs"],
            "steps": len(steps_of(path.parents[2] / f"{path.parents[1].name}.log")),
            "train_loss": last["train_loss"],
            "losses_tail": last["losses_tail"],
            "val": last["val"],
        })
    diag.sort(key=lambda row: -row["temperature"])

    mining = None
    if args.mining:
        m = json.loads(Path(args.mining).read_text())
        mining = {key: m[key] for key in ("params", "train_bm25_hit_at_1")}

    # Passage lengths in the second run's tokenizer, to show what each truncation length cuts.
    from src.data_utils.corpus import Corpus
    from transformers import AutoTokenizer

    corpus = Corpus(dataset_name="fiqa", base_dir=f"{args.base_path}/retrieval_data")
    tokenizer = AutoTokenizer.from_pretrained(run["config"]["tokenizer_name"])
    lengths = sorted(len(ids) for ids in tokenizer([texts[0] for texts in corpus.all_documents["text"]])["input_ids"])
    passage_tokens = {
        "tokenizer": run["config"]["tokenizer_name"],
        "mean": sum(lengths) / len(lengths),
        "median": lengths[len(lengths) // 2],
        "p95": lengths[int(0.95 * (len(lengths) - 1))],
        "max": lengths[-1],
        "over_256": sum(n > 256 for n in lengths) / len(lengths),
        "over_512": sum(n > 512 for n in lengths) / len(lengths),
    }

    evaluation = json.loads(Path(args.eval).read_text())
    data = {
        "steps_per_epoch": int(re.search(r"epoch \d+/\d+:.*?\| *\d+/(\d+) \[", log_text).group(1)),
        "step_losses": {str(epoch): values for epoch, values in sorted(steps.items())},
        "run": {key: run[key] for key in ("n_pairs", "history", "selected_epoch", "wall_clock_seconds", "device",
                                          "git_commit", "created_at")}
        | {"config": {key: run["config"][key] for key in ("model", "train", "eval", "tokenizer_name")}},
        "diag": diag,
        "mining": mining,
        "passage_tokens": passage_tokens,
        "eval": {"n_passages": evaluation["n_passages"],
                 "methods": {name: _scores_only(method) for name, method in evaluation["methods"].items()}},
    }
    html = PAGE.read_text()
    start = html.index(RUN2_OPEN) + len(RUN2_OPEN)
    end = html.index(DATA_CLOSE, start)
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    PAGE.write_text(html[:start] + blob + html[end:])
    print(f"second run: {len(run['history'])} epochs, {sum(map(len, steps.values()))} steps, "
          f"{len(diag)} diagnostic runs, methods {sorted(evaluation['methods'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--methods", nargs="+", required=True, choices=("bm25", "tf-idf", "dpr"))
    ev.add_argument("--base-path", default=".")
    ev.add_argument("--checkpoint")
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--k1", type=float, default=0.9)
    ev.add_argument("--b", type=float, default=0.4)
    ev.add_argument("--label", help="key to store the scores under, default the method name (one method only)")
    ev.add_argument("--out", required=True)

    pg = sub.add_parser("page")
    pg.add_argument("--log", required=True)
    pg.add_argument("--run")
    pg.add_argument("--eval")
    pg.add_argument("--base-path", default=".")

    sc = sub.add_parser("second")
    sc.add_argument("--log", required=True)
    sc.add_argument("--run", required=True)
    sc.add_argument("--eval", required=True)
    sc.add_argument("--diag")
    sc.add_argument("--mining")
    sc.add_argument("--base-path", default=".")

    args = parser.parse_args()
    {"evaluate": evaluate, "page": page, "second": second}[args.command](args)


if __name__ == "__main__":
    main()
