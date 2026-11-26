import hydra
from src.data_utils.corpus import Corpus
from src.retrievers.tf_idf import TFIDF
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate
from src.retrievers.bm25 import BM25
from src.retrievers.dpr import DPR

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

@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')

    ### Initialize the configured retriever
    if cfg.retriever.method == 'bm25':
        retriever = BM25(
            documents=corpus.all_documents,
            k1=cfg.retriever.k1,
            b=cfg.retriever.b,
        )
    elif cfg.retriever.method == 'tf-idf':
        retriever = TFIDF(documents=corpus.all_documents)
    elif cfg.retriever.method == 'dpr':
        retriever = DPR.from_checkpoint(
            corpus.all_documents,
            cfg.retriever.checkpoint_dir,
            batch_size=cfg.retriever.batch_size,
            device=cfg.device,
        )
    else:
        raise ValueError(f'Unsupported retriever: {cfg.retriever.method}')

    ### Retrieve top-k documents for each query and evaluate
    for split in ('val', 'test'):
        queries = corpus.queries(split)
        relevancies = retrieveQueryAndGetRelevancies(retriever, queries, k=cfg.top_k)
        print(f'{split} results: {Evaluate(relevancies, queries, k=cfg.top_k)}')

if __name__ == '__main__':
    main()
