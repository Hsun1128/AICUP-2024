import os
from typing import List, Set, Tuple, Dict, Counter, Any, Optional, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .rag_processor import RAGProcessorConfig, ResourceLoader, QueryProcessor, DocumentProcessor
from .rag_processor.retrieval_system import RetrievalSystem
import logging
from .rag_processor.scoring.bm25_scorer import BM25Scorer
from .rag_processor.scoring.weighted_scorer import WeightedScorer
# 設定日誌記錄
logging.basicConfig(level=logging.INFO, filename='retrieve.log', filemode='w', format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class Retrieval:
    """
    文本處理器主類，負責文本處理的核心功能
    """
    def __init__(self, config: RAGProcessorConfig = RAGProcessorConfig()):
        """
        初始化文本處理器

        Args:
            config (RAGProcessorConfig): 配置對象，包含所有必要的參數設置
        """
        # 保存配置參數
        self.config = config
        
        # 加載所需資源
        self.stopwords = ResourceLoader.load_stopwords(config.stopwords_filepath)
        self.word2vec_model = ResourceLoader.load_word2vec_model(
            os.path.join(os.path.dirname(__file__), config.expansion_model_path)
        )
        self.embeddings = ResourceLoader.load_embeddings(embedding_model_name=config.embedding_model_name, use_faiss=config.use_faiss)
        
        # 初始化查詢處理器
        self.query_processor = QueryProcessor(
            stopwords=self.stopwords,
            word2vec_model=self.word2vec_model,
            use_expansion=config.use_expansion,
            expanded_topn=config.expanded_topn
        )
        
        # 初始化文本分割器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.overlap,
            length_function=len,
            is_separator_regex=False,
            keep_separator=False,
            separators=['\n\n', '\n', '!', '?', '。', ';']
        )

        # 初始化文檔處理器
        self.doc_processor = DocumentProcessor(word2vec_model=self.word2vec_model, embeddings=self.embeddings)
        
        # 初始化評分器
        self.bm25_scorer = BM25Scorer(config, self.doc_processor)
        self.weighted_scorer = WeightedScorer(config, self.doc_processor)
        self.retrieval_system = RetrievalSystem(self.word2vec_model, self.embeddings)

    def BM25_retrieve(self, qs: str, source: List[int], corpus_dict: dict) -> Optional[int]:
        """
        使用BM25算法進行文檔檢索
        """
        # 1. 文檔預處理
        chunked_corpus, key_idx_map = self.doc_processor.prepare_corpus(source, corpus_dict, self.config)
        
        # 2. BM25基礎檢索
        bm25_results = self.bm25_scorer.process_and_score(chunked_corpus, qs, self.query_processor)
        
        # 3. 特徵計算
        feature_results = self.query_processor._analyze_query_features(bm25_results['tokenized_query'], bm25_results['expanded_query'], bm25_results['tokenized_corpus'])
        doc_freq = feature_results['doc_freq']
        query_vector = feature_results['query_vector']
        query_length = feature_results['query_length']
        query_diversity = feature_results['query_diversity']
        
        # 4. 多維度評分
        retrieved_keys, score_dicts = self.weighted_scorer.calculate_retrieval_scores(
            bm25_results['top_docs'], chunked_corpus, key_idx_map, bm25_results['tokenized_corpus'],
            bm25_results['scores'], bm25_results['expanded_query'], doc_freq, query_vector
        )
        
        # 將評分結果分配到對應類別
        score_results = self._distribute_scores(score_dicts)
        
        # 5. 結果整合與返回
        return self._integrate_results(
            qs, chunked_corpus, key_idx_map, retrieved_keys, 
            score_results, query_length, query_diversity
        )

    def _get_faiss_or_bm25_result(self, retrieved_keys: List[Tuple[int, int]], score_results: Dict[str, Dict[Tuple[int, int], float]],
                                  faiss_results: Dict[Tuple[int, int], float], query_length: int,
                                  query_diversity: float, weighted_scores: Optional[Dict[Tuple[int, int], float]]) -> Optional[Tuple[int, int]]:
        """
        根據FAISS和BM25結果決定最終返回的文檔ID
        
        Args:
            retrieved_keys (List[Tuple[int, int]]): BM25檢索到的文檔鍵值列表，每個元素為(文檔ID, chunk_idx)的元組
            score_results (Dict[str, Dict[Tuple[int, int], float]]): 包含各項BM25評分指標的字典，格式為{'bm25_results': {(doc_id, chunk_idx): score}, ...}
            faiss_results (Dict[Tuple[int, int], float]): FAISS檢索結果字典，格式為{(文檔ID,chunk_idx): 相似度分數}
            query_length (int): 查詢長度，即查詢詞的數量
            query_diversity (float): 查詢多樣性分數，範圍0-1
            weighted_scores (Optional[Dict[Tuple[int, int], float]]): 計算後的加權分數字典，格式為{(文檔ID, chunk_idx): 加權分數}
            
        Returns:
            Optional[Tuple[int, int]]: 返回得分最高的文檔鍵值(doc_id, chunk_idx)，如果沒有找到任何結果則返回None
            
        Notes:
            1. 如果同時有FAISS和BM25結果，返回加權分數最高的文檔
            2. 如果只有BM25結果，返回BM25最佳結果
            3. 如果沒有任何結果，返回None
        """
        # 如果有FAISS結果，使用加權分數
        if score_results['bm25_results'] and faiss_results:
            # 記錄最終排序結果
            logger.info('-'*100)
            logger.info('Final Rankings (Adaptive Weights):')
            logger.info(f'Query Features - Length: {query_length}, Diversity: {query_diversity:.2f}')
            
            # 返回得分最高的文檔鍵值
            if weighted_scores:
                best_key = max(weighted_scores.items(), key=lambda x: x[1])[0]
                return best_key[0]

        # 如果只有BM25結果，返回BM25最佳結果
        elif retrieved_keys:
            return retrieved_keys[0][0]
            
        # 如果沒有找到任何結果，返回None
        return None
   
    
    def _integrate_results(self, query: str, chunked_corpus: List[str], key_idx_map: List[Tuple[int, int]],
                          retrieved_keys: List[Tuple[int, int]], score_results: Dict[str, Dict[Tuple[int, int], float]], 
                          query_length: int, query_diversity: float) -> Optional[Tuple[int, int]]:
        """
        整合檢索結果，結合FAISS和BM25的結果
        
        Args:
            query (str): 原始查詢字符串，例如 "台灣總統府在哪裡?"
            chunked_corpus (List[str]): 切分後的文檔列表，例如 ["台灣總統府...", "位於台北市..."]
            key_idx_map (List[Tuple[int, int]]): 文檔ID與索引的映射關係，例如 [(1, 0), (1, 1), (2, 0)]
            retrieved_keys (List[Tuple[int, int]]): BM25檢索到的文檔鍵值列表，例如 [(1, 0), (2, 0)]
            score_results (Dict[str, Dict[Tuple[int, int], float]]): 包含各項評分結果的字典，格式如:
                {
                    'bm25_results': {(1, 0): 0.8, (2, 0): 0.6},
                    'term_importance': {(1, 0): 0.7, (2, 0): 0.5},
                    'semantic_similarity': {(1, 0): 0.6, (2, 0): 0.4},
                    'query_coverage': {(1, 0): 0.8, (2, 0): 0.3},
                    'position_scores': {(1, 0): 0.9, (2, 0): 0.7},
                    'term_density': {(1, 0): 0.5, (2, 0): 0.4},
                    'context_similarity': {(1, 0): 0.7, (2, 0): 0.6}
                }
            query_length (int): 查詢長度，例如 5
            query_diversity (float): 查詢多樣性分數，例如 0.8
            
        Returns:
            Optional[Tuple[int, int]]: 最終選擇的文檔ID和chunk索引，例如 (1, 0)，如果沒有找到則返回None
            
        Example:
            >>> processor = TextProcessor()
            >>> result = processor._integrate_results(
            ...     "台灣總統府在哪裡?",
            ...     ["台灣總統府...", "位於台北市..."],
            ...     [(1, 0), (1, 1)],
            ...     [(1, 0)],
            ...     {'bm25_results': {(1, 0): 0.8}, ...},
            ...     5,
            ...     0.8
            ... )
            >>> print(result)
            (1, 0)
        """
        faiss_results: Dict[Tuple[int, int], float] = {}
        if self.config.use_faiss:
            faiss_results = self.retrieval_system.retrieve_with_faiss(query, chunked_corpus, key_idx_map)

        weighted_scores: Optional[Dict[Tuple[int, int], float]] = None
        if score_results['bm25_results'] and faiss_results:
            weighted_scores = self.weighted_scorer.calculate_weighted_scores(
                score_results['bm25_results'], faiss_results, 
                score_results['term_importance'], score_results['semantic_similarity'],
                score_results['query_coverage'], score_results['position_scores'], 
                score_results['term_density'], score_results['context_similarity'],
                query_length, query_diversity
            )

        return self._get_faiss_or_bm25_result(retrieved_keys, score_results, faiss_results,
                                             query_length, query_diversity, weighted_scores)

    def _distribute_scores(self, score_dicts: Dict[Tuple[int, int], Dict[str, float]]) -> Dict[str, Dict[Tuple[int, int], float]]:
        """
        將計算的評分結果分配到對應的字典中
        
        Args:
            score_dicts (Dict[Tuple[int, int], Dict[str, float]]): 包含每個文檔所有評分指標的字典
                格式: {(doc_id, chunk_idx): {
                    'bm25_score': float,
                    'term_importance': float,
                    'semantic_similarity': float,
                    'query_coverage': float,
                    'position_score': float,
                    'term_density': float,
                    'context_similarity': float
                }}
                
        Returns:
            Dict[str, Dict[Tuple[int, int], float]]: 包含所有評分類型的字典
                格式: {
                    'bm25_results': {(doc_id, chunk_idx): score},
                    'term_importance': {(doc_id, chunk_idx): score},
                    'semantic_similarity': {(doc_id, chunk_idx): score},
                    'query_coverage': {(doc_id, chunk_idx): score},
                    'position_scores': {(doc_id, chunk_idx): score},
                    'term_density': {(doc_id, chunk_idx): score},
                    'context_similarity': {(doc_id, chunk_idx): score}
                }
                
        Example:
            >>> processor = TextProcessor()
            >>> scores = {
            ...     (1, 0): {
            ...         'bm25_score': 0.8,
            ...         'term_importance': 0.7,
            ...         'semantic_similarity': 0.6,
            ...         'query_coverage': 0.5,
            ...         'position_score': 0.4,
            ...         'term_density': 0.3,
            ...         'context_similarity': 0.5
            ...     }
            ... }
            >>> results = processor._distribute_scores(scores)
            >>> print(results['bm25_results'][(1, 0)])
            0.8
        """
        results: Dict[str, Dict[Tuple[int, int], float]] = {
            'bm25_results': {},
            'term_importance': {},
            'semantic_similarity': {},
            'query_coverage': {},
            'position_scores': {},
            'term_density': {},
            'context_similarity': {}
        }
        
        for chunk_key, scores in score_dicts.items():
            results['bm25_results'][chunk_key] = scores['bm25_score']
            results['term_importance'][chunk_key] = scores['term_importance']
            results['semantic_similarity'][chunk_key] = scores['semantic_similarity']
            results['query_coverage'][chunk_key] = scores['query_coverage']
            results['position_scores'][chunk_key] = scores['position_score']
            results['term_density'][chunk_key] = scores['term_density']
            results['context_similarity'][chunk_key] = scores['context_similarity']
            
        return results

