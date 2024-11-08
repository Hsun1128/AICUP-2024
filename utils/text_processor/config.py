import os
import yaml
from dataclasses import dataclass, asdict

# 設置全局參數
RANGE = range(0, 150)  # 處理問題的範圍

# 默認配置
DEFAULT_CONFIG = {
    'stopwords_filepath': './custom_dicts/stopwords.txt',  # 停用詞文件路徑
    'embedding_model_name': 'BAAI/bge-m3',  # embedding模型路徑

    'bm25_k1': 0.5,  # BM25算法的k1參數
    'bm25_b': 0.7,  # BM25算法的b參數
    'bm25_epsilon': 0.5,  # BM25算法的epsilon參數

    'use_expansion': False,  # 是否使用查詢擴展
    'expansion_model_path': '../word2vec/wiki.zh.bin',  # word2vec模型路徑
    'expanded_topn': 2,  # 查詢擴展時每個詞的相似詞數量

    'chunk_size': 500,  # 文本分塊大小
    'overlap': 100,  # 文本分塊重疊大小

    'use_faiss': True,  # 是否使用FAISS向量檢索
    'use_term_importance': True,  # 是否使用詞項重要性評分
    'use_semantic_search': True,  # 是否使用語義相似度評分
    'use_term_density': True,  # 是否使用詞密度評分
    'use_query_coverage': True,  # 是否使用查詢覆蓋率評分
    'use_position_score': True,  # 是否使用位置得分評分
    'use_context_similarity': True  # 是否使用上下文相似度評分
}

@dataclass
class TextProcessorConfig:
    """
    文本處理器的配置類，用於集中管理所有配置參數
    此類可在主程式中實例化並傳入TextProcessor，方便統一管理和修改配置
    
    Attributes:
        stopwords_filepath (str): 停用詞文件路徑
        embedding_model_name (str): embedding模型路徑
        bm25_k1 (float): BM25算法的k1參數,用於控制詞頻的影響程度
        bm25_b (float): BM25算法的b參數,用於控制文檔長度的影響程度
        bm25_epsilon (float): BM25算法的epsilon參數,用於控制文檔長度的影響程度
        use_expansion (bool): 是否啟用查詢擴展功能
        expansion_model_path (str): word2vec模型文件路徑
        expanded_topn (int): 查詢擴展時每個詞選取的相似詞數量
        chunk_size (int): 文本分塊的大小(字符數)
        overlap (int): 相鄰分塊間的重疊字符數
        use_faiss (bool): 是否使用FAISS向量檢索
        use_term_importance (bool): 是否使用詞項重要性評分
        use_semantic_search (bool): 是否使用語義相似度評分
        use_term_density (bool): 是否使用詞密度評分
        use_query_coverage (bool): 是否使用查詢覆蓋率評分
        use_position_score (bool): 是否使用位置得分評分
        use_context_similarity (bool): 是否使用上下文相似度評分
        
    Example:
        >>> config = TextProcessorConfig.from_yaml("config.yaml")
        >>> processor = TextProcessor(config)
    """
    stopwords_filepath: str = DEFAULT_CONFIG['stopwords_filepath']
    embedding_model_name: str = DEFAULT_CONFIG['embedding_model_name']

    bm25_k1: float = DEFAULT_CONFIG['bm25_k1']
    bm25_b: float = DEFAULT_CONFIG['bm25_b']
    bm25_epsilon: float = DEFAULT_CONFIG['bm25_epsilon']

    use_expansion: bool = DEFAULT_CONFIG['use_expansion']
    expansion_model_path: str = DEFAULT_CONFIG['expansion_model_path']
    expanded_topn: int = DEFAULT_CONFIG['expanded_topn']

    chunk_size: int = DEFAULT_CONFIG['chunk_size']
    overlap: int = DEFAULT_CONFIG['overlap']
    
    use_faiss: bool = DEFAULT_CONFIG['use_faiss']
    use_term_importance: bool = DEFAULT_CONFIG['use_term_importance']
    use_semantic_search: bool = DEFAULT_CONFIG['use_semantic_search']
    use_term_density: bool = DEFAULT_CONFIG['use_term_density']
    use_query_coverage: bool = DEFAULT_CONFIG['use_query_coverage']
    use_position_score: bool = DEFAULT_CONFIG['use_position_score']
    use_context_similarity: bool = DEFAULT_CONFIG['use_context_similarity']


    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TextProcessorConfig':
        """
        從YAML文件加載配置
        
        Args:
            yaml_path (str): YAML配置文件路徑
            
        Returns:
            TextProcessorConfig: 配置實例
            
        Example:
            config.yaml:
            ```yaml
            stopwords_filepath: ./custom_dicts/stopwords.txt
            bm25_k1: 0.5
            bm25_b: 0.7
            use_expansion: 0
            expanded_topn: 2
            chunk_size: 500
            overlap: 100
            expansion_model_path: ../word2vec/wiki.zh.bin
            use_faiss: 1
            ```
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f'配置文件不存在: {yaml_path}')
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        # 使用默認配置填充缺失的值
        for key, default_value in DEFAULT_CONFIG.items():
            if key not in config_dict:
                config_dict[key] = default_value
                
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str = './config.yaml') -> None:
        """
        將當前配置保存為YAML文件
        
        Args:
            yaml_path (str): 要保存的YAML文件路徑
        """
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, allow_unicode=True, sort_keys=False)

if __name__ == '__main__':
    config = TextProcessorConfig()
    config.to_yaml('./config.yaml')
    print(config)

__all__ = ['TextProcessorConfig']