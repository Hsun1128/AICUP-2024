import numpy as np
from typing import Set, List, Counter

class TermImportance:
    def calculate_term_importance(self, intersection: Set[str], doc_tokens: List[str], 
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
            >>> score = processor.calculate_term_importance(
            ...     intersection, doc_tokens, frequency, doc_freq, tokenized_corpus)
            >>> print(score)
            0.75

        Notes:
            - 考慮詞頻(TF)、逆文檔頻率(IDF)和位置權重
            - 位置權重：詞的位置越靠前，權重越大
            - 最終得分是TF-IDF和位置權重的乘積
        """
        importance_score = 0.0  # 初始化詞項重要性得分

        # 遍歷交集中的每個詞
        for term in intersection:
            # 計算詞頻 (TF)
            tf = float(frequency[term])

            # 計算逆文檔頻率 (IDF)
            idf = float(np.log(len(tokenized_corpus) / doc_freq[term]))

            # 獲取詞在文檔中的所有位置
            positions = [i for i, t in enumerate(doc_tokens) if t == term]

            # 計算位置權重 (位置越靠前權重越大)
            position_weight = float(sum(1 / (pos + 1) for pos in positions))

            # 累加計算詞項重要性得分
            importance_score += tf * idf * position_weight

        return float(importance_score)  # 返回最終的詞項重要性得分
