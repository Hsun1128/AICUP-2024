from typing import Optional
import logging
from gensim.models import KeyedVectors
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch

logger = logging.getLogger(__name__)

class ResourceLoader:
    """
    資源加載器類，負責加載和初始化所需的外部資源
    此類設計為獨立的工具類，可在不同模組中重用
    
    Methods:
        load_stopwords: 加載停用詞列表
        load_word2vec_model: 加載word2vec模型
        load_embeddings: 加載FAISS使用的詞嵌入模型
    """
    @staticmethod
    def load_stopwords(filepath: str) -> set[str]:
        """
        從文件加載停用詞列表
        
        Args:
            filepath (str): 停用詞文件的路徑
            
        Returns:
            set[str]: 停用詞集合
            
        Raises:
            FileNotFoundError: 當停用詞文件不存在時
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            stopwords = set(line.strip() for line in file)
        logger.info('Loading stopwords success')
        logger.info(f'stopwords: {stopwords}')
        return stopwords

    @staticmethod
    def load_word2vec_model(model_path: str) -> Optional[KeyedVectors]:
        """
        加載word2vec模型
        
        Args:
            model_path (str): 模型文件路徑
            
        Returns:
            KeyedVectors: 加載的word2vec模型
            
        Raises:
            FileNotFoundError: 當模型文件不存在時
        """
        model = KeyedVectors.load(model_path)
        logger.info('Word2Vec model loaded successfully')
        return model
        
    @staticmethod
    def load_embeddings(embedding_model_name: str, use_faiss: bool) -> Optional[HuggingFaceBgeEmbeddings]:
        """
        加載FAISS使用的詞嵌入模型
        
        Args:
            use_faiss (bool): 是否使用FAISS
            
        Returns:
            Optional[HuggingFaceBgeEmbeddings]: 詞嵌入模型實例或None
        """
        if use_faiss:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f'Using device: {device}')
            return HuggingFaceBgeEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
        return None

__all__ = ['ResourceLoader']