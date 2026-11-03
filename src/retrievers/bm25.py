from scipy.sparse import lil_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document
from math import log

def bm25_term_score(term_count,document_length,average_document_length,document_count,document_frequency, k1 = 1.5, b= 0.75):
    if term_count == 0:
        return 0.0
    idf = log((document_count + 1) / (document_frequency + 0.5))
    length_adjustment = k1 * (1 - b + b * document_length / average_document_length)
    term_weight = term_count * (k1 + 1) / (term_count + length_adjustment)
    score = idf * term_weight
    return score

class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        self.bm25_matrix = self.GetDocumentBM25Vectors()

    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.PreprocessText(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        document_count = len(document_lengths)
        return document_lengths, average_document_length / document_count if document_count else 0.0

    def GetDocumentBM25Vectors(self):
        documents = self.index.GetDocuments()
        document_lengths, average_document_length = self.GetDocumentLengths()
        document_count = len(documents)
        document_frequencies = self.GetDocumentFrequencies()
        bm25_matrix = lil_matrix(
            (document_count, len(self.term_to_index)), dtype=float
        ) #lil_matrix -> sparse matrix
        for row, document in enumerate(documents):
            term_counts = self.GetDocumentTermCounts(document)
            for term, count in term_counts.items():
                column = self.term_to_index[term]
                bm25_matrix[row, column] = bm25_term_score(
                    term_count = count,
                    document_length = document_lengths[document],
                    average_document_length = average_document_length,
                    document_count = document_count,
                    document_frequency = document_frequencies[term],
                    k1 = self.k1,
                    b = self.b,
                )
        return bm25_matrix.tocsr()

    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query

        Args:
            query (str): The query to calculate scores for

        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores
