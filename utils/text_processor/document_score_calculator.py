import logging
from typing import List, Set, Tuple
import numpy as np
from collections import Counter
from utils.text_processor.retrieval_system import RetrievalSystem
from gensim.models import KeyedVectors
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

logger = logging.getLogger(__name__)

class DocumentScoreCalculator:
    """
    文檔評分計算器，負責計算文檔的各項評分指標
    """
    def __init__(self, word2vec_model: KeyedVectors, embeddings: HuggingFaceBgeEmbeddings):
        """
        初始化文檔評分計算器
        
        Args:
            word2vec_model: Word2Vec模型，用於計算語義相似度
            embeddings: 嵌入模型，用於檢索
        """
        self.retrieval_system = RetrievalSystem(word2vec_model, embeddings)

    def calculate_document_scores(self, tokenized_corpus: List[List[str]], 
                                 doc_tokens: List[str],
                                 expanded_query: Set[str], 
                                 doc_freq: Counter[str],
                                 query_vector: np.ndarray) -> Tuple[float, float, float, float, float, float, Set[str], Counter[str]]:
        """
        計算文檔的各項評分指標
        
        Args:
            tokenized_corpus (List[List[str]]): 所有文檔的分詞結果列表,例如 [["台灣", "總統府",...], [...],...]
            doc_tokens (List[str]): 當前文檔的分詞結果,例如 ["台灣", "總統府",...]
            expanded_query (Set[str]): 擴展後的查詢詞集合,例如 {"台灣", "總統府", "位置",...}
            doc_freq (Counter[str]): 詞頻統計,例如 Counter({"台灣": 10, "總統府": 5,...})
            query_vector (np.ndarray): 查詢向量,例如 array([0.2, 0.3, 0.1,...])
            
        Returns:
            Tuple[float, float, float, float, float, float, Set[str], Counter[str]]: 包含以下8個評分指標:
                - term_importance (float): 詞項重要性分數 (0-1),例如 0.8
                - semantic_similarity (float): 語義相似度分數 (0-1),例如 0.7
                - query_coverage (float): 查詢覆蓋率分數 (0-1),例如 0.6
                - position_score (float): 詞位置分數 (0-1),例如 0.5
                - term_density (float): 詞密度分數 (0-1),例如 0.4
                - context_similarity (float): 上下文相似度分數 (0-1),例如 0.6
                - intersection (Set[str]): 查詢詞和文檔詞的交集,例如 {"台灣", "總統府"}
                - frequency (Counter[str]): 交集詞在文檔中的頻率統計,例如 Counter({"台灣": 2, "總統府": 1})
                
        Example:
            >>> processor = TextProcessor(...)
            >>> scores = processor._calculate_document_scores(
            ...     tokenized_corpus=[["台灣", "總統府"], ["台北", "101"]],
            ...     doc_tokens=["台灣", "總統府", "位置"],
            ...     expanded_query={"台灣", "總統府", "位置"},
            ...     doc_freq=Counter({"台灣": 2, "總統府": 1}),
            ...     query_vector=np.array([0.2, 0.3, 0.1])
            ... )
            >>> print(scores)
            (0.8, 0.7, 0.6, 0.5, 0.4, 0.6, {"台灣", "總統府"}, Counter({"台灣": 2, "總統府": 1}))
        """
        # 計算查詢詞和文檔詞的交集
        intersection = set(expanded_query) & set(doc_tokens)
        
        # 計算交集中每個詞在文檔中的頻率
        frequency = Counter(token for token in doc_tokens if token in intersection)
        
        # 計算各項評分指標
        term_importance = self.retrieval_system.calculate_term_importance(
            intersection, doc_tokens, frequency, doc_freq, tokenized_corpus)
            
        semantic_similarity = self.retrieval_system.calculate_semantic_similarity(
            doc_tokens, query_vector, doc_freq, tokenized_corpus)
            
        query_coverage = self.retrieval_system.calculate_query_coverage(
            intersection, expanded_query, tokenized_corpus, doc_freq)
            
        position_score = self.retrieval_system.calculate_position_score(
            doc_tokens, intersection)
            
        term_density = self.retrieval_system.calculate_term_density(
            doc_tokens, intersection, tokenized_corpus, doc_freq)
            
        context_similarity = self.retrieval_system.calculate_context_similarity(
            doc_tokens, intersection)
            
        return (term_importance, semantic_similarity, query_coverage, 
                position_score, term_density, context_similarity,
                intersection, frequency)

__all__ = ['DocumentScoreCalculator']