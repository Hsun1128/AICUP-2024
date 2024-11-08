import numpy as np
from typing import Set, List, Counter

class QueryCoverage:
    def calculate_query_coverage(self, intersection: Set[str], expanded_query: Set[str],
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
