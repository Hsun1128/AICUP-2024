import numpy as np
from typing import List, Set

class PositionScore:
    def calculate_position_score(self, doc_tokens: List[str], intersection: Set[str]) -> float:
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
