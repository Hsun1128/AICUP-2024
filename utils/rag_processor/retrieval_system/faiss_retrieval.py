from typing import List, Tuple, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import logging

logger = logging.getLogger(__name__)

class FaissRetrieval:
    def __init__(self, embedding_model: HuggingFaceBgeEmbeddings):
        self.embedding_model = embedding_model
        
    def faiss_retrieve(self, query: str, chunked_corpus: List[str], key_idx_map: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """
        使用FAISS進行向量檢索
        
        Args:
            query (str): 查詢文本，例如 "台灣總統府在哪裡?"
            chunked_corpus (List[str]): 切分後的文本列表，例如 ["台灣總統府位於台北市中正區...", "總統府是一棟巴洛克式建築..."]
            key_idx_map (List[Tuple[int, int]]): 文本片段對應的(file_key, chunk_index)列表，例如 [(1, 0), (1, 1)]
            
        Returns:
            Dict[Tuple[int, int], float]: 包含檢索結果的字典，格式為 {(file_key, chunk_index): score}
                 例如 {(1, 0): 0.85, (1, 1): 0.75}
                 - file_key (int): 文件ID，例如 1 
                 - chunk_index (int): 文本片段索引，例如 0
                 - score (float): 相似度分數，範圍0-1，例如 0.85
                 
        Example:
            >>> processor = TextProcessor()
            >>> query = "台灣總統府在哪裡?"
            >>> corpus = ["台灣總統府位於台北市中正區...", "總統府是一棟巴洛克式建築..."]
            >>> key_idx_map = [(1, 0), (1, 1)]
            >>> results = processor._faiss_retrieve(query, corpus, key_idx_map)
            >>> print(results)
            {(1, 0): 0.85, (1, 1): 0.75}
        """
        # 初始化結果字典
        faiss_results: Dict[Tuple[int, int], float] = {}
        
        # 使用FAISS建立向量索引
        vector_store = FAISS.from_texts(chunked_corpus, self.embeddings, normalize_L2=True)
        
        # 進行相似度搜索，返回前5個最相似的文檔及其分數
        faiss_ans = vector_store.similarity_search_with_score(query, k=5)

        # 記錄分隔線
        logger.info('-'*100)
        
        # 處理每個檢索結果
        for doc, score in faiss_ans:
            # 找到文檔在chunked_corpus中的索引
            faiss_actual_index: int = chunked_corpus.index(doc.page_content)
            # 獲取對應的文檔鍵值
            faiss_chunk_key: Tuple[int, int] = key_idx_map[faiss_actual_index]
            # 儲存分數
            faiss_results[faiss_chunk_key] = float(score)
            # 記錄檢索結果
            logger.info(f'FAISS Score: [{score:.4f}], PDF: {faiss_chunk_key[0]}, '
                      f'Chunk: {faiss_chunk_key[1]}, metadata: [{doc.metadata}]')
                      
        return faiss_results
    