import os

# 設置全局參數
RANGE = range(0, 150)  # 處理問題的範圍

STOPWORDS_FILEPATH = './custom_dicts/stopwords.txt'  # 停用詞文件路徑
USE_EXPANSION = 0  # 是否使用查詢擴展
EXPANSION_MODEL_PATH = '../word2vec/wiki.zh.bin'  # word2vec模型路徑
EXPANDED_TOPN = 2  # 查詢擴展時每個詞的相似詞數量

USE_FAISS = 1  # 是否使用FAISS向量檢索

BM25_K1 = 0.5  # BM25算法的k1參數
BM25_B = 0.7  # BM25算法的b參數

CHUNK_SIZE = 500  # 文本分塊大小
OVERLAP = 100  # 文本分塊重疊大小

class TextProcessorConfig:
    """
    文本處理器的配置類，用於集中管理所有配置參數
    此類可在主程式中實例化並傳入TextProcessor，方便統一管理和修改配置
    
    Attributes:
        stopwords_filepath (str): 停用詞文件路徑
        bm25_k1 (float): BM25算法的k1參數,用於控制詞頻的影響程度
        bm25_b (float): BM25算法的b參數,用於控制文檔長度的影響程度
        use_expansion (bool): 是否啟用查詢擴展功能
        expanded_topn (int): 查詢擴展時每個詞選取的相似詞數量
        chunk_size (int): 文本分塊的大小(字符數)
        overlap (int): 相鄰分塊間的重疊字符數
        expansion_model_path (str): word2vec模型文件路徑
        use_faiss (bool): 是否使用FAISS向量檢索
        
    Example:
        >>> config = TextProcessorConfig(
        ...     stopwords_filepath="custom_stopwords.txt",
        ...     bm25_k1=0.8,
        ...     use_faiss=True
        ... )
        >>> processor = TextProcessor(config)
    """
    def __init__(self,
                 stopwords_filepath: str = STOPWORDS_FILEPATH,
                 bm25_k1: float = BM25_K1,
                 bm25_b: float = BM25_B,
                 use_expansion: bool = USE_EXPANSION,
                 expanded_topn: int = EXPANDED_TOPN,
                 chunk_size: int = CHUNK_SIZE,
                 overlap: int = OVERLAP,
                 expansion_model_path: str = EXPANSION_MODEL_PATH,
                 use_faiss: bool = USE_FAISS):
        self.stopwords_filepath = stopwords_filepath
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.use_expansion = use_expansion
        self.expanded_topn = expanded_topn
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.expansion_model_path = expansion_model_path
        self.use_faiss = use_faiss

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TextProcessorConfig':
        """
        從字典創建配置實例，方便從配置文件加載
        
        Args:
            config_dict (dict): 包含配置參數的字典
            
        Returns:
            TextProcessorConfig: 配置實例
        """
        return cls(**config_dict)

__all__ = ['TextProcessorConfig']