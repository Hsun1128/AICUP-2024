import logging
from typing import List, Set, Tuple, Counter, Any, Dict
import numpy as np
import jieba
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    查詢處理器類，負責處理和擴展用戶查詢
    
    主要功能:
    - 查詢分詞
    - 查詢擴展
    - 停用詞過濾
    
    Attributes:
        stopwords (Set[str]): 停用詞集合
        word2vec_model (KeyedVectors): word2vec模型
        use_expansion (bool): 是否使用查詢擴展
        expanded_topn (int): 每個詞擴展的相似詞數量
    """
    
    def __init__(self, stopwords: Set[str], word2vec_model: KeyedVectors,
                 use_expansion: bool = True, expanded_topn: int = 3):
        """
        初始化查詢處理器
        
        Args:
            stopwords (Set[str]): 停用詞集合
            word2vec_model (KeyedVectors): 預訓練的word2vec模型
            use_expansion (bool, optional): 是否使用查詢擴展. Defaults to True.
            expanded_topn (int, optional): 每個詞擴展的相似詞數量. Defaults to 3.
        """
        self.stopwords = stopwords
        self.word2vec_model = word2vec_model
        self.use_expansion = use_expansion
        self.expanded_topn = expanded_topn

    def process_query(self, query: str) -> Tuple[Set[str], Set[str]]:
        """
        處理查詢字串，包括分詞和查詢擴展
        
        Args:
            query (str): 原始查詢字串，例如"台灣的總統府在哪裡?"
            
        Returns:
            Tuple[Set[str], Set[str]]:
                - tokenized_query (Set[str]): 分詞後的原始查詢詞集合，例如 {"台灣", "總統府", "哪裡"}
                - expanded_query (Set[str]): 擴展後的查詢詞集合，例如 {"台灣", "總統府", "哪裡", "中華民國", "政府", "位置"}
        """
        # 對查詢語句進行分詞
        tokenized_query = set(self.jieba_cut_with_stopwords(query))
        
        # 使用word2vec進行查詢擴展
        expanded_query = self._expand_query(tokenized_query)

        # 記錄查詢相關信息
        logger.info(f'query分詞結果: {"/".join(tokenized_query)}')
        unique_expanded_query = set(expanded_query) - set(tokenized_query)
        logger.info(f'query擴展詞: {"/".join(unique_expanded_query)}')
        
        return tokenized_query, expanded_query

    def _expand_query(self, tokenized_query: Set[str]) -> Set[str]:
        """
        對輸入的分詞後查詢進行擴展，增加相似詞以提高召回率
        
        Args:
            tokenized_query (Set[str]): 分詞後的查詢詞集合，例如 {"台灣", "總統府"}
            
        Returns:
            Set[str]: 擴展後的查詢詞集合，例如 {"台灣", "總統府", "中華民國", "政府"}
        """
        if not self.use_expansion:
            return tokenized_query  # 如果不使用擴展，返回原始查詢
        
        expanded_query = []
        for word in tokenized_query:
            if word in self.word2vec_model.key_to_index:
                topn_words = self.word2vec_model.most_similar(word, topn=self.expanded_topn)
                topn_words = set([w[0] for w in topn_words if w[0] not in self.stopwords and w[0].strip() and len(w[0]) != 1])
                expanded_query.extend([word] + list(topn_words))  # 包含原始詞和擴展詞
        return set(expanded_query)

    def jieba_cut_with_stopwords(self, text: str) -> List[str]:
        """
        使用jieba進行分詞並移除停用詞
        
        Args:
            text (str): 待分詞的文本
            
        Returns:
            List[str]: 分詞結果列表，已移除停用詞
        """
        words = jieba.cut_for_search(text)
        return [word for word in words if word not in self.stopwords and word.strip()]
    
    def _analyze_query_features(self, tokenized_query: List[str], expanded_query: Set[str], 
                              tokenized_corpus: List[List[str]]) -> Dict[str, Any]:
        """
        分析查詢特徵，包括文檔頻率、查詢向量和查詢特徵
        
        Args:
            tokenized_query (List[str]): 分詞後的原始查詢，例如 ["台灣", "總統府"]
            expanded_query (Set[str]): 擴展後的查詢，例如 {"台灣", "總統府", "政府"} 
            tokenized_corpus (List[List[str]]): 分詞後的文檔列表，例如 [["台灣", "總統府"], ["政府", "機關"]]
            
        Returns:
            Dict[str, Any]: 包含以下鍵值的字典:
                - doc_freq (Counter[str]): 文檔頻率字典，例如 Counter({"台灣": 2, "總統府": 1})
                - query_vector (np.ndarray): 查詢向量，例如 array([0.2, 0.3, 0.1])
                - query_length (int): 查詢長度，例如 2
                - query_diversity (float): 查詢多樣性分數，例如 0.8
                
        Example:
            >>> processor = TextProcessor()
            >>> result = processor._analyze_query_features(
            ...     ["台灣", "總統府"],
            ...     {"台灣", "總統府", "政府"},
            ...     [["台灣", "總統府"], ["政府", "機關"]]
            ... )
            >>> print(result)
            {
                'doc_freq': Counter({"台灣": 2, "總統府": 1}),
                'query_vector': array([0.2, 0.3, 0.1]),
                'query_length': 2,
                'query_diversity': 0.8
            }
        """
        doc_freq = self._calculate_doc_frequencies(tokenized_corpus)
        query_vector = self._calculate_query_vector(tokenized_query, tokenized_corpus, doc_freq)
        query_length, query_diversity = self._calculate_query_features(tokenized_query, expanded_query)
        
        return {
            'doc_freq': doc_freq,
            'query_vector': query_vector,
            'query_length': query_length,
            'query_diversity': query_diversity
        }

    
    def _calculate_doc_frequencies(self, tokenized_corpus: List[List[str]]) -> Counter[str]:
        """
        計算文檔頻率(每個詞出現在多少文檔中)

        Args:
            tokenized_corpus (List[List[str]]): 分詞後的文檔集合，每個文檔是一個詞列表
                例如: [["台灣", "總統府"], ["台北", "101"]]

        Returns:
            Counter[str]: 包含每個詞的文檔頻率的Counter對象
                例如: Counter({"台灣": 1, "總統府": 1, "台北": 1, "101": 1})

        Example:
            >>> processor = TextProcessor()
            >>> corpus = [["台灣", "總統府"], ["台北", "101"]]
            >>> doc_freq = processor._calculate_doc_frequencies(corpus)
            >>> print(doc_freq)
            Counter({"台灣": 1, "總統府": 1, "台北": 1, "101": 1})

        Notes:
            - 對每個文檔中的詞進行去重，確保每個詞在一個文檔中只被計算一次
            - 使用Counter累加每個詞在不同文檔中的出現次數
            - 返回的Counter對象可用於計算IDF值和其他相關指標
        """
        doc_freq = Counter()
        for doc in tokenized_corpus:
            # 對每個文檔中的詞進行去重
            doc_freq.update(set(doc))
        return doc_freq

    def _calculate_query_vector(self, tokenized_query: Set[str], tokenized_corpus: List[List[str]], doc_freq: Counter[str]) -> np.ndarray:
        """
        計算查詢向量，使用word2vec加權平均

        Args:
            tokenized_query (Set[str]): 分詞後的查詢詞集合，例如 {"台灣", "總統府"}
            tokenized_corpus (List[List[str]]): 分詞後的文檔集合，例如 [["台灣", "總統府"], ["台北", "101"]]
            doc_freq (Counter[str]): 詞頻統計字典，例如 Counter({"台灣": 2, "總統府": 1})

        Returns:
            np.ndarray: 查詢向量，使用word2vec加權平均計算得到，例如 array([0.2, 0.3, 0.1,...])
            
        Example:
            >>> processor = TextProcessor()
            >>> query = {"台灣", "總統府"}
            >>> corpus = [["台灣", "總統府"], ["台北", "101"]]
            >>> doc_freq = Counter({"台灣": 2, "總統府": 1})
            >>> vector = processor._calculate_query_vector(query, corpus, doc_freq)
            >>> print(vector.shape)
            (300,)
            
        Notes:
            - 使用IDF作為權重計算查詢向量
            - 對於不在word2vec模型中的詞會被忽略
            - 如果沒有任何詞的權重，返回零向量
        """
        query_vector: np.ndarray = np.zeros(self.word2vec_model.vector_size)
        total_weight: float = 0.0
        
        for word in tokenized_query:
            if word in self.word2vec_model:
                # 使用IDF作為權重
                weight: float = np.log(len(tokenized_corpus) / (doc_freq[word] + 1))
                query_vector += weight * self.word2vec_model[word]
                total_weight += weight
                
        if total_weight > 0:
            query_vector /= total_weight
            
        return query_vector
    
    def _calculate_query_features(self, tokenized_query: Set[str], expanded_query: Set[str]) -> Tuple[int, float]:
        """
        計算查詢的複雜度特徵
        
        Args:
            tokenized_query (Set[str]): 分詞後的原始查詢詞集合，例如 {"台灣", "總統府", "哪裡"}
            expanded_query (Set[str]): 擴展後的查詢詞集合，例如 {"台灣", "總統府", "哪裡", "中華民國", "政府", "位置"}
            
        Returns:
            Tuple[int, float]:
                - query_length (int): 原始查詢的詞數量，例如 3 (對應上述例子)
                - query_diversity (float): 查詢擴展的多樣性,計算為擴展詞數量與原始詞數量的比值，例如 2.0 (6/3)
                
        Example:
            >>> processor = TextProcessor(...)
            >>> tokenized = {"台灣", "總統府", "哪裡"}
            >>> expanded = {"台灣", "總統府", "哪裡", "中華民國", "政府", "位置"}
            >>> length, diversity = processor._calculate_query_features(tokenized, expanded)
            >>> print(length, diversity)
            3 2.0
        """
        query_length = len(tokenized_query)  # 查詢長度
        query_diversity = len(expanded_query) / (len(tokenized_query) + 1)  # 查詢擴展的多樣性
        return query_length, query_diversity
    


__all__ = ['QueryProcessor']