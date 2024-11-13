from typing import Dict, Tuple, List, Any, Set
from collections import Counter
import numpy as np
from .base_scorer import BaseScorer
from utils.rag_processor import DocumentProcessor
import logging

logger = logging.getLogger(__name__)

class WeightedScorer(BaseScorer):
    """處理多維度加權評分"""
    
    def __init__(self, config, doc_processor: DocumentProcessor):
        super().__init__()
        self.config = config
        self.doc_processor = doc_processor
        
    def calculate_weighted_scores(self, 
        bm25_results: Dict[Tuple[int, int], float],
        faiss_results: Dict[Tuple[int, int], float],
        term_importance: Dict[Tuple[int, int], float],
        semantic_similarity: Dict[Tuple[int, int], float],
        query_coverage: Dict[Tuple[int, int], float],
        position_scores: Dict[Tuple[int, int], float],
        term_density: Dict[Tuple[int, int], float],
        context_similarity: Dict[Tuple[int, int], float],
        query_length: int,
        query_diversity: float
    ) -> Dict[Tuple[int, int], float]:
        """
        計算多指標加權分數
        
        Args:
            bm25_results (Dict[Tuple[int, int], float]): BM25檢索結果分數，格為{(文檔ID,chunk_idx): 分數}
            faiss_results (Dict[Tuple[int, int], float]): FAISS檢索結果分數，格式為{(文檔ID,chunk_idx): 分數}
            term_importance (Dict[Tuple[int, int], float]): 詞項重要性分數，格式為{(文檔ID,chunk_idx): 分數}
            semantic_similarity (Dict[Tuple[int, int], float]): 語義相似度分數，格式為{(文檔ID,chunk_idx): 分數}
            query_coverage (Dict[Tuple[int, int], float]): 查詢覆蓋率分數，格式為{(文檔ID,chunk_idx): 分數}
            position_scores (Dict[Tuple[int, int], float]): 詞位置分數，格式為{(文檔ID,chunk_idx): 分數}
            term_density (Dict[Tuple[int, int], float]): 詞密度分數，格式為{(文檔ID,chunk_idx): 分數}
            context_similarity (Dict[Tuple[int, int], float]): 上下文相似度分數，格式為{(文檔ID,chunk_idx): 分數}
            query_length (int): 查詢長度，例如 3
            query_diversity (float): 查詢多樣性，例如 1.5
            
        Returns:
            Dict[Tuple[int, int], float]: 包含每個文檔最終加權分數的字典，格式為{(文檔ID,chunk_idx): 加權分數}
            
        Example:
            >>> processor = TextProcessor()
            >>> bm25_results = {(1,0): 0.8, (1,1): 0.6}
            >>> faiss_results = {(1,0): 0.7, (1,1): 0.5}
            >>> term_importance = {(1,0): 0.6, (1,1): 0.4}
            >>> semantic_similarity = {(1,0): 0.7, (1,1): 0.5}
            >>> query_coverage = {(1,0): 0.8, (1,1): 0.6}
            >>> position_scores = {(1,0): 0.7, (1,1): 0.5}
            >>> term_density = {(1,0): 0.6, (1,1): 0.4}
            >>> context_similarity = {(1,0): 0.7, (1,1): 0.5}
            >>> weighted_scores = processor._calculate_weighted_scores(
            ...     bm25_results, faiss_results, term_importance,
            ...     semantic_similarity, query_coverage, position_scores,
            ...     term_density, context_similarity, 3, 1.5
            ... )
            >>> print(weighted_scores)
            {(1,0): 0.7125, (1,1): 0.5125}
            
        Notes:
            - 對每個指標進行正規化處理
            - 根據查詢特徵動態調整權重
            - 計算加權總分
        """
        # 正規化分數的輔助函數
        def normalize_scores(scores):
            max_score = max(scores.values())
            min_score = min(scores.values())
            return {k: (v - min_score)/(max_score - min_score) if max_score != min_score else 1 
                   for k, v in scores.items()}

        # 正規化所有指標的分數
        normalized_bm25 = normalize_scores(bm25_results)
        normalized_faiss = {k: 1 - normalize_scores(faiss_results)[k] for k in faiss_results}
        normalized_importance = normalize_scores(term_importance)
        normalized_semantic = normalize_scores(semantic_similarity) if semantic_similarity else {}
        normalized_coverage = normalize_scores(query_coverage)
        normalized_position = normalize_scores(position_scores) if position_scores else {}
        normalized_density = normalize_scores(term_density) if term_density else {}
        normalized_context = normalize_scores(context_similarity) if context_similarity else {}

        # 設定基礎權重
        base_weights = {
            'bm25': self.config.base_weights['bm25'],
            'faiss': self.config.base_weights['faiss'],
            'importance': self.config.base_weights['importance'],
            'semantic': self.config.base_weights['semantic'],
            'coverage': self.config.base_weights['coverage'],
            'position': self.config.base_weights['position'],
            'density': self.config.base_weights['density'],
            'context': self.config.base_weights['context']
        }
        
        # 根據查詢特徵調整權重
        adjusted_weights = base_weights.copy()
        if query_length <= 2:  # 短查詢
            adjusted_weights['semantic'] *= 1.2  # 增加語義相似度的權重
            adjusted_weights['context'] *= 1.2   # 增加上下文相似度的權重
            adjusted_weights['bm25'] *= 0.8      # 降低BM25的權重
        else:  # 長查詢
            adjusted_weights['bm25'] *= 1.2      # 增加BM25的權重
            adjusted_weights['coverage'] *= 1.2   # 增加查詢覆蓋率的權重
            adjusted_weights['semantic'] *= 0.8   # 降低語義相似度的權重
        
        if query_diversity > 1.5:  # 查詢擴展效果明顯
            adjusted_weights['semantic'] *= 1.2   # 增加語義相似度的權重
            adjusted_weights['context'] *= 1.2    # 增加上下文相似度的權重
        
        # 正規化調整後的權重
        weight_sum = sum(adjusted_weights.values())
        adjusted_weights = {k: v/weight_sum for k, v in adjusted_weights.items()}
        
        # 計算加權分數
        weighted_scores = {}
        for key in set(bm25_results.keys()) | set(faiss_results.keys()):
            score = (
                adjusted_weights['bm25'] * normalized_bm25.get(key, 0) +
                adjusted_weights['faiss'] * normalized_faiss.get(key, 0) +
                adjusted_weights['importance'] * normalized_importance.get(key, 0) +
                adjusted_weights['semantic'] * normalized_semantic.get(key, 0) +
                adjusted_weights['coverage'] * normalized_coverage.get(key, 0) +
                adjusted_weights['position'] * normalized_position.get(key, 0) +
                adjusted_weights['density'] * normalized_density.get(key, 0) +
                adjusted_weights['context'] * normalized_context.get(key, 0)
            )
            weighted_scores[key] = score

        # 記錄權重和分數詳情
        logger.info(f'Query Features - Length: {query_length}, Diversity: {query_diversity:.2f}')
        logger.info(f'Adjusted Weights: {adjusted_weights}')
        for key in sorted(weighted_scores.keys(), key=lambda k: weighted_scores[k], reverse=True):
            logger.info(
                f'PDF {key[0]}, Chunk {key[1]}: '
                f'Final Score={weighted_scores[key]:.4f}, '
                f'BM25={normalized_bm25.get(key, 0):.4f}, '
                f'FAISS={normalized_faiss.get(key, 0):.4f}, '
                f'Importance={normalized_importance.get(key, 0):.4f}, '
                f'Semantic={normalized_semantic.get(key, 0):.4f}, '
                f'Coverage={normalized_coverage.get(key, 0):.4f}, '
                f'Position={normalized_position.get(key, 0):.4f}, '
                f'Density={normalized_density.get(key, 0):.4f}, '
                f'Context={normalized_context.get(key, 0):.4f}'
            )

        return weighted_scores

    
    def calculate_retrieval_scores(self, ans: List[str], chunked_corpus: List[str], 
                                  key_idx_map: List[Tuple[int, int]], tokenized_corpus: List[List[str]],
                                  scores: List[float], expanded_query: Set[str], 
                                  doc_freq: Counter[str], query_vector: np.ndarray) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], Dict[str, float]]]:
        """
        計算檢索結果的各項評分指標
        
        Args:
            ans (List[str]): 檢索到的文檔內容列表，例如 ["台灣總統府位於台北市中正區...", "總統府是一棟巴洛克式建築..."]
            chunked_corpus (List[str]): 分塊後的語料庫，例如 ["台灣總統府位於台北市中正區...", "總統府是一棟巴洛克式建築..."]
            key_idx_map (List[Tuple[int, int]]): 每個文本片段對應的(file_key, chunk_index)列表，例如 [(1, 0), (1, 1)]
            tokenized_corpus (List[List[str]]): 分詞後的語料庫，例如 [["台灣", "總統府", ...], ["總統府", "巴洛克", ...]]
            scores (List[float]): BM25分數列表，例如 [0.8, 0.6]
            expanded_query (Set[str]): 擴展後的查詢詞集合，例如 {"台灣", "總統府", "位置"}
            doc_freq (Counter[str]): 詞頻統計，例如 Counter({"台灣": 10, "總統府": 5})
            query_vector (np.ndarray): 查詢向量，例如 array([0.2, 0.3, 0.1])
            
        Returns:
            Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], Dict[str, float]]]:
                - retrieved_keys: 檢索到的文檔鍵值列表，例如 [(1, 0), (1, 1)]
                - score_dicts: 包含各項評分指標的字典，例如
                    {
                        (1, 0): {
                            "bm25_score": 0.8,
                            "term_importance": 0.7,
                            "semantic_similarity": 0.6,
                            "query_coverage": 0.5,
                            "position_score": 0.4,
                            "term_density": 0.3,
                            "context_similarity": 0.5
                        },
                        (1, 1): {...}
                    }
                
        Notes:
            計算的評分指標包括:
            - BM25分數 (0-1): 基於詞頻和文檔長度的相關性分數
            - 詞項重要性 (0-1): 基於TF-IDF的詞重要性分數
            - 語義相似度 (0-1): 基於詞向量的語義相似度
            - 查詢覆蓋率 (0-1): 查詢詞在文檔中的覆蓋程度
            - 詞位置分數 (0-1): 查詢詞在文檔中的位置分布
            - 詞密度 (0-1): 查詢詞在文檔中的密集程度
            - 上下文相似度 (0-1): 查詢詞周圍上下文的相似度
            
        Example:
            >>> processor = TextProcessor()
            >>> ans = ["台灣總統府位於台北市中正區", "總統府是一棟巴洛克式建築"]
            >>> chunked_corpus = ["台灣總統府位於台北市中正區", "總統府是一棟巴洛克式建築"]
            >>> key_idx_map = [(1, 0), (1, 1)]
            >>> tokenized_corpus = [["台灣", "總統府"], ["總統府", "巴洛克"]]
            >>> scores = [0.8, 0.6]
            >>> expanded_query = {"台灣", "總統府", "位置"}
            >>> doc_freq = Counter({"台灣": 10, "總統府": 5})
            >>> query_vector = np.array([0.2, 0.3, 0.1])
            >>> keys, scores = processor._calculate_retrieval_scores(
            ...     ans, chunked_corpus, key_idx_map, tokenized_corpus,
            ...     scores, expanded_query, doc_freq, query_vector
            ... )
            >>> print(keys)
            [(1, 0), (1, 1)]
            >>> print(scores[(1, 0)]["bm25_score"])
            0.8
        """
        retrieved_keys: List[Tuple[int, int]] = []
        score_dicts: Dict[Tuple[int, int], Dict[str, float]] = {}
        
        for index, doc in enumerate(ans):
            chunk_key, scores_dict = self.doc_processor.process_retrieved_document(
                index, doc, chunked_corpus, key_idx_map, tokenized_corpus,
                scores, expanded_query, doc_freq, query_vector
            )
            
            if chunk_key:
                retrieved_keys.append(chunk_key)
                score_dicts[chunk_key] = scores_dict
                
        return retrieved_keys, score_dicts 
