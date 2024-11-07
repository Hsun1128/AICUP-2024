import logging
from typing import List, Set, Tuple
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
        expanded_query = self.expand_query(tokenized_query)

        # 記錄查詢相關信息
        logger.info(f'query分詞結果: {"/".join(tokenized_query)}')
        unique_expanded_query = set(expanded_query) - set(tokenized_query)
        logger.info(f'query擴展詞: {"/".join(unique_expanded_query)}')
        
        return tokenized_query, expanded_query

    def expand_query(self, tokenized_query: Set[str]) -> Set[str]:
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

__all__ = ['QueryProcessor']