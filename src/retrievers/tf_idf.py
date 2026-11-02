from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
import numpy as np
from scipy.sparse import lil_matrix


class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.term_to_index = {}
        sorted_terms = sorted(self.corpus_vocabulary)
        for column, term in enumerate(sorted_terms):
            self.term_to_index[term] = column

        # Only run if we are in TFIDF class
        if type(self) == TFIDF:
            self.idf = self.GetInverseDocumentFrequencies()
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted charatcters from text and lowercase
        """
        characters = []
        for character in text.lower():
            if character.isalnum():
                characters.append(character)
            else:
                characters.append(" ")

        cleaned_text = "".join(characters)
        return cleaned_text.split()
        
    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """
        tokens = self.PreprocessText(query)
        unique_terms = set()
        for term in tokens:
            unique_terms.add(term)
        return sorted(unique_terms)

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """
        text = document.GetText()
        tokens = self.PreprocessText(text)
        unique_terms = set()
        for term in tokens:
            unique_terms.add(term)
        return sorted(unique_terms)
    
    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        corpus_vocabulary = set()
        documents = self.index.GetDocuments()
        for document in documents:
            terms = self.GetDocumentVocabulary(document)
            for term in terms:
                corpus_vocabulary.add(term)
        return corpus_vocabulary

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """
        frequencies = {}
        documents = self.index.GetDocuments()
        for document in documents:
            unique_terms = self.GetDocumentVocabulary(document)
            for term in unique_terms:
                if term not in frequencies:
                    frequencies[term] = 0
                frequencies[term] += 1
        return frequencies
       
    
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        raise NotImplementedError("Implement TF-IDF inverse document frequencies.")
    
    
    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """
        text = document.GetText()
        tokens = self.PreprocessText(text)
        counts = {}
        for term in tokens:
            if term not in counts:
                counts[term] = 0
            counts[term] += 1
        return counts

    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        raise NotImplementedError("Implement the TF-IDF document weight matrix.")

    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.
        
        Args:
            query (str): The query to convert
        
        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)

        for row, query in enumerate(queries):
            tokens = self.PreprocessText(query)
            for term in tokens:
                if term in self.term_to_index:
                    column = self.term_to_index[term]
                    query_vector[row, column] += 1

        return query_vector.tocsr()

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        return scores
    