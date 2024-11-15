import logging
from typing import List, Set, Tuple, Optional, Dict, Union, Any
import numpy as np
from collections import Counter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from utils.rag_processor.document_score_calculator import DocumentScoreCalculator
from utils.rag_processor.config import RAGProcessorConfig
import os
from datetime import datetime

logger = logging.getLogger(__name__)
chunk_logger = logging.getLogger('chunk')

# 在創建 FileHandler 之前，先確保日誌目錄存在
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)  # 如果目錄不存在則創建

# 然後再創建 FileHandler
chunk_handler = logging.FileHandler(f'{log_dir}/chunk_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
chunk_logger.addHandler(chunk_handler)
chunk_logger.propagate = False

class DocumentProcessor:
    def __init__(self, word2vec_model = None, embeddings: HuggingFaceBgeEmbeddings = None):
        """
        初始化文檔處理器
        
        Args:
            word2vec_model: Word2Vec模型實例，用於計算語義相似度
            embedding_model_name: embedding模型路徑
        """
        # 保存word2vec模型
        self.word2vec_model = word2vec_model
        self.embeddings = embeddings
        
        # 初始化文檔評分計算器
        self.score_calculator = DocumentScoreCalculator(word2vec_model=self.word2vec_model, embeddings=self.embeddings)
        
    def process_retrieved_document(self, doc_index: int, doc_content: str, 
                                  chunked_corpus: List[str], key_idx_map: List[Tuple[int, int]], 
                                  tokenized_corpus: List[List[str]], scores: List[float],
                                  expanded_query: Set[str], doc_freq: Counter[str], 
                                  query_vector: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Dict[str, float]]:
        """
        處理單個檢索到的文檔，計算各項評分指標並記錄日誌
        
        Args:
            doc_index (int): 文檔在檢索結果中的排名索引,例如 0 表示第一個檢索結果
            doc_content (str): 文檔內容,例如 "台灣總統府位於台北市中正區重慶南路一段122號..."
            chunked_corpus (List[str]): 切分後的語料庫,例如 ["文檔1內容", "文檔2內容",...]
            key_idx_map (List[Tuple[int, int]]): 文檔索引到(PDF ID, chunk ID)的映射,例如 [(1, 0), (1, 1), (2, 0)]
            tokenized_corpus (List[List[str]]): 分詞後的語料庫,例如 [["台灣", "總統府",...], [...],...]
            scores (List[float]): BM25分數列表,例如 [0.8, 0.6, 0.4,...]
            expanded_query (Set[str]): 擴展後的查詢詞集合,例如 {"台灣", "總統府", "位置",...}
            doc_freq (Counter[str]): 詞頻統計,例如 Counter({"台灣": 10, "總統府": 5,...})
            query_vector (np.ndarray): 查詢向量,例如 array([0.2, 0.3, 0.1,...])
            
        Returns:
            Tuple[Optional[Tuple[int, int]], Dict[str, float]]:
                - 文檔鍵值(PDF ID, chunk ID)或None(處理失敗時),例如 (1, 2) 或 None
                - 包含各項評分的字典
        """
        try:
            # 1. 定位文檔
            chunk_key = self._locate_document(doc_content, chunked_corpus, key_idx_map)
            if chunk_key is None:
                return None, {}
                
            # 2. 計算文檔評分
            scores_dict, frequency = self._calculate_document_scores(
                doc_content, chunked_corpus, tokenized_corpus,
                scores, expanded_query, doc_freq, query_vector
            )
            
            # 3. 記錄日誌
            self._log_document_scores(doc_index, chunk_key, scores_dict, frequency)
            
            return chunk_key, scores_dict
            
        except Exception as e:
            logger.error(f'處理文檔時發生錯誤: {str(e)}')
            return None, {}
            
    def _locate_document(self, doc_content: str, chunked_corpus: List[str], 
                        key_idx_map: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """定位文檔在語料庫中的位置並返回對應的鍵值"""
        try:
            actual_index = chunked_corpus.index(doc_content)
            return key_idx_map[actual_index]
        except ValueError:
            logger.error(f'在語料庫中找不到文檔: {doc_content[:100]}...')
            return None
            
    def _calculate_document_scores(self, doc_content: str,
                                 chunked_corpus: List[str], tokenized_corpus: List[List[str]], 
                                 scores: List[float], expanded_query: Set[str],
                                 doc_freq: Counter[str], query_vector: np.ndarray) -> Tuple[Dict[str, float], Counter[str]]:
        """
        計算文檔的各項評分指標
        
        Args:
            doc_content (str): 文檔內容
            chunked_corpus (List[str]): 切分後的語料庫
            tokenized_corpus (List[List[str]]): 分詞後的語料庫
            scores (List[float]): BM25分數列表
            expanded_query (Set[str]): 擴展後的查詢詞集合
            doc_freq (Counter[str]): 詞頻統計
            query_vector (np.ndarray): 查詢向量
            
        Returns:
            Tuple[Dict[str, float], Counter[str]]:
                - 包含各項評分的字典
                - 詞頻統計結果
        """
        actual_index = chunked_corpus.index(doc_content)
        doc_tokens = tokenized_corpus[actual_index]
        
        scores_dict: Dict[str, float] = {}
        
        # 計算各項評分指標
        (scores_dict['term_importance'], scores_dict['semantic_similarity'],
         scores_dict['query_coverage'], scores_dict['position_score'],
         scores_dict['term_density'], scores_dict['context_similarity'],
         intersection, frequency) = self.score_calculator.calculate_document_scores(
            tokenized_corpus, doc_tokens, expanded_query, doc_freq, query_vector)
            
        # 記錄BM25分數
        scores_dict['bm25_score'] = scores[actual_index]
        
        return scores_dict, frequency
        
    def _log_document_scores(self, doc_index: int, chunk_key: Tuple[int, int], 
                           scores_dict: Dict[str, float], frequency: Counter[str]) -> None:
        """記錄文檔評分的詳細日誌"""
        logger.info(
            f'BM25 Rank {doc_index + 1}: PDF {chunk_key[0]}, Chunk {chunk_key[1]}, '
            f'Score: {scores_dict["bm25_score"]:.4f}, '
            f'Term Importance: {scores_dict["term_importance"]:.4f}, '
            f'Semantic Similarity: {scores_dict["semantic_similarity"]:.4f}, '
            f'Query Coverage: {scores_dict["query_coverage"]:.4f}, '
            f'Position Score: {scores_dict["position_score"]:.4f}, '
            f'Term Density: {scores_dict["term_density"]:.4f}, '
            f'Context Similarity: {scores_dict["context_similarity"]:.4f}, '
            f'Frequency: {frequency.most_common(10)}'
        )

    def prepare_corpus(self, source: List[int], corpus_dict: Dict[int, Union[str, List[Any]]], config: RAGProcessorConfig) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        準備語料庫數據，將原始文檔切分並建立索引映射
        
        Args:
            source (List[int]): 來源文件ID列表，例如 [1, 2, 3]
            corpus_dict (Dict[int, Union[str, List[Any]]]): 語料庫字典，key為文件ID，value為文件內容或Document列表
                例如 {1: "文件1內容", 2: [Document對象1, Document對象2]}
            
        Returns:
            Tuple[List[str], List[Tuple[int, int]]]: 包含以下兩個元素:
                - chunked_corpus (List[str]): 切分後的文本片段列表，例如 ["片段1", "片段2"]
                - key_idx_map (List[Tuple[int, int]]): 每個文本片段對應的(file_key, chunk_index)列表
                    例如 [(1, 0), (1, 1), (2, 0)]
                    
        Example:
            >>> processor = TextProcessor()
            >>> source = [1, 2]
            >>> corpus_dict = {
            ...     1: "文件1內容",
            ...     2: [Document(page_content="片段1"), Document(page_content="片段2")]
            ... }
            >>> chunked, key_map = processor._prepare_corpus(source, corpus_dict)
            >>> print(chunked)
            ["文件1內容", "片段1", "片段2"]
            >>> print(key_map)
            [(1, 0), (2, 0), (2, 1)]
        """
        chunked_corpus: List[str] = []  # 存儲所有切分後的文本段落
        key_idx_map: List[Tuple[int, int]] = []  # 存儲每個文本片段對應的(file_key, chunk_index)
        
        # 遍歷每個來源文件ID
        for file_key in source:
            # 獲取對應文件的內容
            corpus = corpus_dict[int(file_key)]
            # 對每個文件內容進行切分
            for idx, chunk in enumerate(corpus):
                key_idx_map.append((file_key, idx))
                try:    
                    # 如果是Document對象,取其page_content屬性
                    chunked_corpus.append(chunk.page_content)
                    if config.chunk_preview:
                        chunk_logger.info(f'file {file_key}, Chunk {idx}: {chunk.page_content}')
                except AttributeError:
                    # 如果不是Document對象,直接添加文本內容
                    chunked_corpus.append(corpus)
                    if config.chunk_preview:
                        chunk_logger.info(f'file {file_key}, Chunk {idx}: {corpus}')
                    break
                    
        return chunked_corpus, key_idx_map


__all__ = ['DocumentProcessor']
