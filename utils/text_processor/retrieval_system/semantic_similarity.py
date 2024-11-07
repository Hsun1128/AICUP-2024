import numpy as np
from typing import List, Counter
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSimilarity:
    def __init__(self, word2vec_model: KeyedVectors):
        self.word2vec_model = word2vec_model
        
    def calculate_semantic_similarity(self, doc_tokens: List[str], query_vector: np.ndarray,
                                        doc_freq: Counter[str], tokenized_corpus: List[List[str]]) -> float:
            """
            計算語義相似度得分
            
            Args:
                doc_tokens (List[str]): 文檔的分詞結果，例如 ["台灣", "總統府", "位於", "台北"]
                query_vector (np.ndarray): 查詢向量，例如 array([0.2, 0.3, 0.1, ...])
                doc_freq (Counter[str]): 每個詞在所有文檔中的出現次數，例如 Counter({"台灣": 10, "總統府": 5})
                tokenized_corpus (List[List[str]]): 所有文檔的分詞結果，例如 [["台灣", "總統府"], ["台北", "101"]]
                
            Returns:
                float: 語義相似度得分，範圍為0-1，例如 0.75
                
            Example:
                >>> processor = TextProcessor()
                >>> doc_tokens = ["台灣", "總統府", "位於", "台北"]
                >>> query_vector = np.array([0.2, 0.3, 0.1])
                >>> doc_freq = Counter({"台灣": 10, "總統府": 5})
                >>> tokenized_corpus = [["台灣", "總統府"], ["台北", "101"]]
                >>> score = processor._calculate_semantic_similarity(
                ...     doc_tokens, query_vector, doc_freq, tokenized_corpus)
                >>> print(score)
                0.75
                
            Notes:
                - 使用加權文檔向量計算與查詢向量的餘弦相似度
                - 權重使用IDF值
            """
            doc_vector: np.ndarray = np.zeros(self.word2vec_model.vector_size)
            total_weight: float = 0.0
            for word in doc_tokens:
                if word in self.word2vec_model:
                    weight: float = np.log(len(tokenized_corpus) / (doc_freq[word] + 1))
                    doc_vector += weight * self.word2vec_model[word]
                    total_weight += weight
            if total_weight > 0:
                doc_vector /= total_weight
                return float(cosine_similarity(query_vector.reshape(1, -1), doc_vector.reshape(1, -1))[0][0])
            return 0.0
