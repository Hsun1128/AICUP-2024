from typing import Dict, Tuple, List, Set, Optional
from collections import Counter
import numpy as np
import logging

class BaseScorer:
    """基礎評分器類，提供共用的評分功能"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def normalize_scores(self, scores: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """正規化分數的通用方法"""
        if not scores:
            return {}
        max_score = max(scores.values())
        min_score = min(scores.values())
        return {k: (v - min_score)/(max_score - min_score) if max_score != min_score else 1 
               for k, v in scores.items()} 