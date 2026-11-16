import os
import pickle
import numpy as np
from abc import ABC, abstractmethod
from src.data_utils.dataset import DocumentDataset

class Retriever(ABC):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        if not index_path is None:
            self.index = self.__LoadIndex(index_path)
        else:
            self.index = self.__BuildIndex(documents)
    
    def __BuildIndex(self, documents: list[dict]):
        """
        @param dataset: The dataset for which an index containing embeddings should be built.
        """
        index = DocumentDataset(documents)
        return index
    
    def __LoadIndex(self, index_path: str):
        """
        @param index_path: The path to the pre-computed index.
        """
        file = open(index_path, 'rb')
        index = pickle.load(file)
        file.close()
        return index
    
    def SaveIndex(self, index_path: str):
        """
        @param index_path: The path to save the pre-computed index.
        """
        if not os.path.exists(os.path.dirname(index_path)):
            os.makedirs(os.path.dirname(index_path))
        file = open(index_path, 'wb')
        pickle.dump(self.index, file)
        file.close()

        
    @abstractmethod
    def CalculateScores(self, query: str):
        raise NotImplementedError("Must overwrite")
    
    def Lookup(self, queries: list[str], k: int) -> list:
        """
        @param queries: The input text(s) to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve for each input query.
        """
        documents = self.index.GetDocuments()
        k = min(k, len(documents))
        results = []
        for start in range(0, len(queries), 1000):
            scores = np.asarray(self.CalculateScores(queries[start:start + 1000]))
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
            top = np.take_along_axis(top, order, axis=1)
            results.extend([documents[i] for i in row] for row in top)
        return results