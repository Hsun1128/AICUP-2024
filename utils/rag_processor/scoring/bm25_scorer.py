from typing import Dict, List, Set, Tuple, Any
from rank_bm25 import BM25Okapi
from .base_scorer import BaseScorer
from utils.rag_processor import DocumentProcessor

class BM25Scorer(BaseScorer):
    """處理BM25相關的評分"""
    
    def __init__(self, config, doc_processor: DocumentProcessor):
        super().__init__()
        self.config = config
        self.doc_processor = doc_processor
    
    def process_and_score(self, chunked_corpus: List[str], query: str, 
                         query_processor) -> Dict[str, Any]:
        """
        BM25評分的主要處理邏輯
        
        Args:
            chunked_corpus (List[str]): 切分後的文檔列表，例如 ["文檔1內容", "文檔2內容", ...]
            query (str): 查詢字符串，例如 "如何申請專利?"
            
        Returns:
            Dict[str, Any]: 包含以下鍵值的字典:
                - scores (np.ndarray): BM25分數列表，例如 [0.8, 0.6, 0.4, ...]
                - top_docs (List[str]): 得分最高的文檔列表，例如 ["文檔1內容", "文檔2內容", ...]
                - expanded_query (Set[str]): 擴展後的查詢詞集合，例如 {"專利", "申請", "流程"}
                - tokenized_query (Set[str]): 原始查詢分詞結果，例如 {"專利", "申請"}
                - tokenized_corpus (List[List[str]]): 分詞後的文檔列表，例如 [["文檔", "內容"], ["文檔", "內容"]]
                
        Example:
            >>> processor = TextProcessor(...)
            >>> corpus = ["專利申請流程說明", "商標註冊相關規定"]
            >>> query = "如何申請專利?"
            >>> result = processor._process_and_score_bm25(corpus, query)
            >>> print(result)
            {
                'scores': array([0.8, 0.2]),
                'top_docs': ['專利申請流程說明'],
                'expanded_query': {'專利', '申請', '流程'},
                'tokenized_query': {'專利', '申請'},
                'tokenized_corpus': [['專利', '申請', '流程'], ['商標', '註冊']]
            }
        """
        tokenized_corpus = [query_processor.jieba_cut_with_stopwords(doc) 
                          for doc in chunked_corpus]
        bm25 = BM25Okapi(tokenized_corpus, k1=self.config.bm25_k1, 
                         b=self.config.bm25_b, epsilon=self.config.bm25_epsilon)
        tokenized_query, expanded_query = query_processor.process_query(query)
        scores = bm25.get_scores(expanded_query)
        top_docs = self._get_top_documents(scores, chunked_corpus)
        
        return {
            'scores': scores,
            'top_docs': top_docs,
            'expanded_query': expanded_query,
            'tokenized_query': tokenized_query,
            'tokenized_corpus': tokenized_corpus
        }
    
    def _get_top_documents(self, scores: List[float], 
                          chunked_corpus: List[str], n: int = 5) -> List[str]:
        """
        根據BM25分數獲取前N個最相關的文檔
        
        Args:
            scores (List[float]): 每個文檔的BM25分數列表，例如 [0.8, 0.6, 0.4]
            chunked_corpus (List[str]): 分塊後的文檔內容列表，例如 ["文檔1內容", "文檔2內容"]
            n (int, optional): 要返回的最相關文檔數量. 默認為5
            
        Returns:
            List[str]: 包含前N個最相關文檔內容的列表，例如 ["最相關文檔內容", "次相關文檔內容"]
            
        Example:
            >>> processor = TextProcessor()
            >>> scores = [0.8, 0.6, 0.4]
            >>> corpus = ["文檔1", "文檔2", "文檔3"]
            >>> top_docs = processor._get_top_documents(scores, corpus, n=2)
            >>> print(top_docs)
            ["文檔1", "文檔2"]
            
        Notes:
            1. 使用sorted和lambda函數對分數進行降序排序
            2. 獲取前N個最高分數的文檔索引
            3. 根據索引從語料庫中提取對應的文檔內容
        """
        top_n_indices = sorted(range(len(scores)), 
                             key=lambda i: scores[i], reverse=True)[:n]
        return [chunked_corpus[i] for i in top_n_indices] 