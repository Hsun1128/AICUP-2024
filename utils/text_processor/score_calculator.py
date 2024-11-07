import logging
from typing import List, Set, Tuple
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class DocumentScoreCalculator:
    """
    文檔評分計算器，負責計算文檔的各項評分指標
    """
    def __init__(self, word2vec_model):
        """
        初始化文檔評分計算器
        
        Args:
            word2vec_model: Word2Vec模型，用於計算語義相似度
        """
        self.word2vec_model = word2vec_model

    def calculate_document_scores(self, tokenized_corpus: List[List[str]], 
                                 doc_tokens: List[str],
                                 expanded_query: Set[str], 
                                 doc_freq: Counter[str],
                                 query_vector: np.ndarray) -> Tuple[float, float, float, float, float, float, Set[str], Counter[str]]:
        """
        計算文檔的各項評分指標
        
        Args:
            tokenized_corpus (List[List[str]]): 所有文檔的分詞結果列表,例如 [["台灣", "總統府",...], [...],...]
            doc_tokens (List[str]): 當前文檔的分詞結果,例如 ["台灣", "總統府",...]
            expanded_query (Set[str]): 擴展後的查詢詞集合,例如 {"台灣", "總統府", "位置",...}
            doc_freq (Counter[str]): 詞頻統計,例如 Counter({"台灣": 10, "總統府": 5,...})
            query_vector (np.ndarray): 查詢向量,例如 array([0.2, 0.3, 0.1,...])
            
        Returns:
            Tuple[float, float, float, float, float, float, Set[str], Counter[str]]: 包含以下8個評分指標:
                - term_importance (float): 詞項重要性分數 (0-1),例如 0.8
                - semantic_similarity (float): 語義相似度分數 (0-1),例如 0.7
                - query_coverage (float): 查詢覆蓋率分數 (0-1),例如 0.6
                - position_score (float): 詞位置分數 (0-1),例如 0.5
                - term_density (float): 詞密度分數 (0-1),例如 0.4
                - context_similarity (float): 上下文相似度分數 (0-1),例如 0.6
                - intersection (Set[str]): 查詢詞和文檔詞的交集,例如 {"台灣", "總統府"}
                - frequency (Counter[str]): 交集詞在文檔中的頻率統計,例如 Counter({"台灣": 2, "總統府": 1})
                
        Example:
            >>> processor = TextProcessor(...)
            >>> scores = processor._calculate_document_scores(
            ...     tokenized_corpus=[["台灣", "總統府"], ["台北", "101"]],
            ...     doc_tokens=["台灣", "總統府", "位置"],
            ...     expanded_query={"台灣", "總統府", "位置"},
            ...     doc_freq=Counter({"台灣": 2, "總統府": 1}),
            ...     query_vector=np.array([0.2, 0.3, 0.1])
            ... )
            >>> print(scores)
            (0.8, 0.7, 0.6, 0.5, 0.4, 0.6, {"台灣", "總統府"}, Counter({"台灣": 2, "總統府": 1}))
        """
        # 計算查詢詞和文檔詞的交集
        intersection = set(expanded_query) & set(doc_tokens)
        
        # 計算交集中每個詞在文檔中的頻率
        frequency = Counter(token for token in doc_tokens if token in intersection)
        
        # 計算各項評分指標
        term_importance = self._calculate_term_importance(
            intersection, doc_tokens, frequency, doc_freq, tokenized_corpus)
            
        semantic_similarity = self._calculate_semantic_similarity(
            doc_tokens, query_vector, doc_freq, tokenized_corpus)
            
        query_coverage = self._calculate_query_coverage(
            intersection, expanded_query, tokenized_corpus, doc_freq)
            
        position_score = self._calculate_position_score(
            doc_tokens, intersection)
            
        term_density = self._calculate_term_density(
            doc_tokens, intersection, tokenized_corpus, doc_freq)
            
        context_similarity = self._calculate_context_similarity(
            doc_tokens, intersection)
            
        return (term_importance, semantic_similarity, query_coverage, 
                position_score, term_density, context_similarity,
                intersection, frequency)


    def _calculate_term_importance(self, intersection: Set[str], doc_tokens: List[str], 
                                 frequency: Counter[str], doc_freq: Counter[str], 
                                 tokenized_corpus: List[List[str]]) -> float:
        """
        計算詞項重要性得分
        
        Args:
            intersection (Set[str]): 查詢詞和文檔詞的交集，例如 {"台灣", "總統府"}
            doc_tokens (List[str]): 文檔的分詞結果，例如 ["台灣", "總統府", "位於", "台北"]
            frequency (Counter[str]): 交集中每個詞在文檔中的頻率，例如 Counter({"台灣": 2, "總統府": 1})
            doc_freq (Counter[str]): 每個詞在所有文檔中的出現次數，例如 Counter({"台灣": 10, "總統府": 5})
            tokenized_corpus (List[List[str]]): 所有文檔的分詞結果，例如 [["台灣", "總統府"], ["台北", "101"]]
            
        Returns:
            float: 詞項重要性得分，範圍為0-1，例如 0.75
            
        Example:
            >>> processor = TextProcessor()
            >>> intersection = {"台灣", "總統府"}
            >>> doc_tokens = ["台灣", "總統府", "位於", "台北"]
            >>> frequency = Counter({"台灣": 2, "總統府": 1})
            >>> doc_freq = Counter({"台灣": 10, "總統府": 5})
            >>> tokenized_corpus = [["台灣", "總統府"], ["台北", "101"]]
            >>> score = processor._calculate_term_importance(
            ...     intersection, doc_tokens, frequency, doc_freq, tokenized_corpus)
            >>> print(score)
            0.75
            
        Notes:
            - 考慮詞頻(TF)、逆文檔頻率(IDF)和位置權重
            - 位置權重：詞的位置越靠前，權重越大
            - 最終得分是TF-IDF和位置權重的乘積
        """
        importance_score = 0.0
        for term in intersection:
            tf = float(frequency[term])  # 詞頻
            idf = float(np.log(len(tokenized_corpus) / doc_freq[term]))  # 逆文檔頻率
            # 獲取詞在文檔中的所有位置
            positions = [i for i, t in enumerate(doc_tokens) if t == term]
            # 計算位置權重(位置越靠前權重越大)
            position_weight = float(sum(1/(pos + 1) for pos in positions))
            importance_score += tf * idf * position_weight
        return float(importance_score)

    def _calculate_semantic_similarity(self, doc_tokens: List[str], query_vector: np.ndarray,
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

    def _calculate_query_coverage(self, intersection: Set[str], expanded_query: Set[str],
                                tokenized_corpus: List[List[str]], doc_freq: Counter[str]) -> float:
        """
        計算查詢覆蓋率得分
        
        Args:
            intersection (Set[str]): 查詢詞和文檔詞的交集，例如 {"台灣", "總統府"}
            expanded_query (Set[str]): 擴展後的查詢詞集合，例如 {"台灣", "總統府", "中華民國", "政府"}
            tokenized_corpus (List[List[str]]): 所有文檔的分詞結果，例如 [["台灣", "總統府"], ["台北", "101"]]
            doc_freq (Counter[str]): 每個詞在所有文檔中的出現次數，例如 Counter({"台灣": 10, "總統府": 5})
            
        Returns:
            float: 查詢覆蓋率得分，範圍為0-1，例如 0.75
            
        Example:
            >>> processor = TextProcessor()
            >>> intersection = {"台灣", "總統府"}
            >>> expanded_query = {"台灣", "總統府", "中華民國", "政府"}
            >>> tokenized_corpus = [["台灣", "總統府"], ["台北", "101"]]
            >>> doc_freq = Counter({"台灣": 10, "總統府": 5})
            >>> score = processor._calculate_query_coverage(
            ...     intersection, expanded_query, tokenized_corpus, doc_freq)
            >>> print(score)
            0.75
            
        Notes:
            - 考慮查詢詞在文檔中的覆蓋程度
            - 使用IDF加權計算覆蓋率
        """
        if len(expanded_query) == 0:
            return 0.0
            
        weighted_coverage: float = sum(np.log(len(tokenized_corpus) / doc_freq[term]) 
                                     for term in intersection)
        return float(weighted_coverage / len(expanded_query))

    def _calculate_position_score(self, doc_tokens: List[str], intersection: Set[str]) -> float:
        """
        計算詞位置得分
        
        Args:
            doc_tokens (List[str]): 文檔的分詞結果，例如 ["台灣", "總統府", "位於", "台北"]
            intersection (Set[str]): 查詢詞和文檔詞的交集，例如 {"台灣", "總統府"}
            
        Returns:
            float: 詞位置得分，範圍為0-1，例如 0.8。分數越高表示查詢詞位置越靠前且分布越集中
            
        Example:
            >>> processor = TextProcessor()
            >>> doc_tokens = ["台灣", "總統府", "位於", "台北"]
            >>> intersection = {"台灣", "總統府"}
            >>> score = processor._calculate_position_score(doc_tokens, intersection)
            >>> print(score)
            0.8
            
        Notes:
            - 考慮查詢詞在文檔中的位置分布
            - 位置越靠前且分布越集中，得分越高
            - 使用平均位置和標準差計算分數
        """
        positions: List[int] = []
        for i, token in enumerate(doc_tokens):
            if token in intersection:
                positions.append(i)
        if positions:
            avg_pos: float = np.mean(positions)  # 平均位置
            std_pos: float = np.std(positions) if len(positions) > 1 else 0  # 位置標準差
            return float(1 / (1 + std_pos + avg_pos/len(doc_tokens)))
        return 0.0

    def _calculate_term_density(self, doc_tokens: List[str], intersection: Set[str],
                              tokenized_corpus: List[List[str]], doc_freq: Counter[str]) -> float:
        """
        計算詞密度得分
        
        Args:
            doc_tokens (List[str]): 文檔的分詞結果，例如 ["台灣", "總統府", "位於", "台北"]
            intersection (Set[str]): 查詢詞和文檔詞的交集，例如 {"台灣", "總統府"}
            tokenized_corpus (List[List[str]]): 所有文檔的分詞結果，例如 [["台灣", "總統府"], ["台北", "101"]]
            doc_freq (Counter[str]): 每個詞在所有文檔中的出現次數，例如 Counter({"台灣": 2, "總統府": 1})
            
        Returns:
            float: 詞密度得分，範圍為0-1，例如 0.8。分數越高表示查詢詞在局部區域的密度越大
            
        Example:
            >>> processor = TextProcessor()
            >>> doc_tokens = ["台灣", "總統府", "位於", "台北"]
            >>> intersection = {"台灣", "總統府"}
            >>> tokenized_corpus = [["台灣", "總統府"], ["台北", "101"]]
            >>> doc_freq = Counter({"台灣": 2, "總統府": 1})
            >>> score = processor._calculate_term_density(
            ...     doc_tokens, intersection, tokenized_corpus, doc_freq)
            >>> print(score)
            0.8
            
        Notes:
            - 使用滑動窗口計算查詢詞的局部密度
            - 考慮IDF權重
        """
        window_size = min(20, len(doc_tokens))  # 動態窗口大小
        max_density = 0.0
        for i in range(len(doc_tokens) - window_size + 1):
            window = doc_tokens[i:i + window_size]
            weighted_matches = sum(np.log(len(tokenized_corpus) / doc_freq[w]) 
                                for w in window if w in intersection)
            density = weighted_matches / window_size
            max_density = max(max_density, density)
        return float(max_density)

    def _calculate_context_similarity(self, doc_tokens: List[str], intersection: Set[str]) -> float:
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
 
__all__ = ['DocumentScoreCalculator']