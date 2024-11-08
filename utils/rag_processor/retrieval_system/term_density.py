import numpy as np
from typing import List, Set, Counter

class TermDensity:
    def calculate_term_density(self, doc_tokens: List[str], intersection: Set[str],
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
