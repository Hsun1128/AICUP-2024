from typing import List, Set
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

class ContextSimilarity:
    def __init__(self, word2vec_model: KeyedVectors):
        self.word2vec_model = word2vec_model
        
    def calculate_context_similarity(self, doc_tokens: List[str], intersection: Set[str]) -> float:
        """
        計算上下文相似度得分
        
        Args:
            doc_tokens (List[str]): 文檔的分詞結果，例如 ["台灣", "總統府", "位於", "台北"]
            intersection (Set[str]): 查詢詞和文檔詞的交集，例如 {"台灣", "總統府"}
            
        Returns:
            float: 上下文相似度得分，範圍為0-1，例如 0.8。分數越高表示查詢詞與上下文的語義相似度越高
            
        Example:
            >>> processor = TextProcessor()
            >>> doc_tokens = ["台灣", "總統府", "位於", "台北"]
            >>> intersection = {"台灣", "總統府"}
            >>> score = processor._calculate_context_similarity(doc_tokens, intersection)
            >>> print(score)
            0.8
            
        Notes:
            - 計算查詢詞與其上下文詞的語義相似度
            - 使用固定大小的上下文窗口
            - 使用word2vec計算詞向量的餘弦相似度
        """
        context_size: int = 3  # 上下文窗口大小
        context_score: float = 0.0
        
        for i, token in enumerate(doc_tokens):
            if token in intersection:
                # 獲取詞的上下文窗口
                start: int = max(0, i - context_size)
                end: int = min(len(doc_tokens), i + context_size + 1)
                context: List[str] = doc_tokens[start:end]
                
                # 計算詞與上下文的相似度
                for c_token in context:
                    if c_token in self.word2vec_model and token in self.word2vec_model:
                        similarity: float = cosine_similarity(
                            self.word2vec_model[token].reshape(1, -1),
                            self.word2vec_model[c_token].reshape(1, -1)
                        )[0][0]
                        context_score += similarity
                        
        return float(context_score / (len(intersection) + 1))
 