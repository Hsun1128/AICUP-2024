from .faiss_retrieval import FaissRetrieval
from .position_score import PositionScore
from .query_coverage import QueryCoverage
from .semantic_similarity import SemanticSimilarity
from .term_density import TermDensity
from .context_similarity import ContextSimilarity
from .term_importance import TermImportance
from typing import List, Tuple, Dict, Set
from collections import Counter
import numpy as np
from gensim.models import KeyedVectors
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class RetrievalSystem:
    def __init__(self, word2vec_model: KeyedVectors, embedding_model: HuggingFaceBgeEmbeddings):
        self.faiss_retrieval_instance = FaissRetrieval(embedding_model)
        self.position_score_instance = PositionScore()
        self.query_coverage_instance = QueryCoverage()
        self.semantic_similarity_instance = SemanticSimilarity(word2vec_model)
        self.term_density_instance = TermDensity()
        self.context_similarity_instance = ContextSimilarity(word2vec_model)
        self.term_importance_instance = TermImportance()
        
    def retrieve_with_faiss(self, query: str, chunked_corpus: List[str], key_idx_map: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        return self.faiss_retrieval_instance.faiss_retrieve(query, chunked_corpus, key_idx_map)

    def calculate_position_score(self, doc_tokens: List[str], intersection: Set[str]) -> float:
        return self.position_score_instance.calculate_position_score(doc_tokens, intersection)

    def calculate_query_coverage(self, intersection: Set[str], expanded_query: Set[str], tokenized_corpus: List[List[str]], doc_freq: Counter[str]) -> float:
        return self.query_coverage_instance.calculate_query_coverage(intersection, expanded_query, tokenized_corpus, doc_freq)
    
    def calculate_semantic_similarity(self, doc_tokens: List[str], query_vector: np.ndarray,
                                      doc_freq: Counter[str], tokenized_corpus: List[List[str]]) -> float:
        return self.semantic_similarity_instance.calculate_semantic_similarity(doc_tokens, query_vector, doc_freq, tokenized_corpus)
    
    def calculate_term_density(self, doc_tokens: List[str], intersection: Set[str], tokenized_corpus: List[List[str]],
                               doc_freq: Counter[str]) -> float:
        return self.term_density_instance.calculate_term_density(doc_tokens, intersection, tokenized_corpus, doc_freq)
    
    def calculate_context_similarity(self, doc_tokens: List[str], intersection: Set[str]) -> float:
        return self.context_similarity_instance.calculate_context_similarity(doc_tokens, intersection)
    
    def calculate_term_importance(self, intersection: Set[str], doc_tokens: List[str], frequency: Counter[str], doc_freq: Counter[str], tokenized_corpus: List[List[str]]) -> float:
        return self.term_importance_instance.calculate_term_importance(intersection, doc_tokens, frequency, doc_freq, tokenized_corpus)

__all__ = ['RetrievalSystem']