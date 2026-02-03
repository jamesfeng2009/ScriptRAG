"""Adaptive Threshold Strategy - 动态阈值与悬崖截断

功能：
1. 相对分值截断（悬崖截断）：检测分数"悬崖"，自动截断无关结果
2. 自适应阈值：根据查询复杂度动态调整最低分数
3. Top-k + Min-Score 混合策略

核心算法：
- 悬崖检测：计算相邻结果分数下降幅度，超过阈值则截断
- 自适应阈值：基于查询复杂度估计调整基础阈值
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveThresholdConfig:
    """自适应阈值配置"""
    base_min_score: float = 0.65
    cliff_drop_threshold: float = 0.20
    min_score_floor: float = 0.55
    max_results: int = 20
    complexity_weight: float = 0.1


@dataclass
class ThresholdAnalysisResult:
    """阈值分析结果"""
    applied_threshold: float
    total_candidates: int
    retained_count: int
    cliff_detected: bool
    cliff_index: Optional[int]
    cliff_drop_rate: Optional[float]
    min_score_enforced: bool
    reasoning: str


class AdaptiveThresholdStrategy:
    """
    自适应阈值策略
    
    解决的问题：
    - 固定阈值无法适应不同查询
    - 简单查询需要更高阈值，复杂查询需要更低阈值
    
    解决方案：
    - 悬崖截断：检测分数骤降点
    - 自适应调整：基于查询复杂度动态调整
    """
    
    def __init__(self, config: Optional[AdaptiveThresholdConfig] = None):
        self.config = config or AdaptiveThresholdConfig()
    
    def analyze_and_filter(
        self,
        results: List[float],
        query_complexity: float = 0.5
    ) -> Tuple[List[int], ThresholdAnalysisResult]:
        """
        分析并过滤结果
        
        Args:
            results: 按相似度排序的结果分数列表
            query_complexity: 查询复杂度 (0.0-1.0)
                - 0.0: 简单查询（特定术语、函数名）
                - 0.5: 中等查询
                - 1.0: 复杂查询（概念理解、流程查询）
            
        Returns:
            - 保留的结果索引列表
            - 分析结果对象
        """
        if not results:
            return [], ThresholdAnalysisResult(
                applied_threshold=self.config.base_min_score,
                total_candidates=0,
                retained_count=0,
                cliff_detected=False,
                cliff_index=None,
                cliff_drop_rate=None,
                min_score_enforced=False,
                reasoning="No results to process"
            )
        
        # 计算自适应阈值
        adjusted_threshold = self._calculate_adaptive_threshold(query_complexity)
        
        # 第一层过滤：低于最低分数的结果
        below_threshold_indices = [
            i for i, score in enumerate(results) 
            if score < adjusted_threshold
        ]
        
        # 悬崖检测
        cliff_info = self._detect_cliff(results, below_threshold_indices)
        
        # 确定截断位置
        if cliff_info['detected']:
            cutoff_index = cliff_info['index']
            reasoning = f"Cliff detected at index {cutoff_index} with drop rate {cliff_info['drop_rate']:.1%}"
        elif below_threshold_indices:
            cutoff_index = below_threshold_indices[0]
            adjusted_threshold = max(adjusted_threshold, self.config.min_score_floor)
            reasoning = f"Enforced minimum threshold {adjusted_threshold:.3f}"
        else:
            cutoff_index = len(results)
            reasoning = f"All {len(results)} results meet threshold {adjusted_threshold:.3f}"
        
        # 限制最大返回数量
        cutoff_index = min(cutoff_index, self.config.max_results)
        if cutoff_index < len(results):
            reasoning += f", capped at {self.config.max_results} results"
        
        retained_indices = list(range(cutoff_index))
        
        result = ThresholdAnalysisResult(
            applied_threshold=adjusted_threshold,
            total_candidates=len(results),
            retained_count=len(retained_indices),
            cliff_detected=cliff_info['detected'],
            cliff_index=cliff_info['index'],
            cliff_drop_rate=cliff_info.get('drop_rate'),
            min_score_enforced=any(
                results[i] < self.config.base_min_score 
                for i in retained_indices
            ),
            reasoning=reasoning
        )
        
        logger.info(
            f"Adaptive threshold analysis: threshold={adjusted_threshold:.3f}, "
            f"retained={len(retained_indices)}/{len(results)}, "
            f"cliff={cliff_info['detected']}, reason={reasoning}"
        )
        
        return retained_indices, result
    
    def _calculate_adaptive_threshold(self, complexity: float) -> float:
        """
        计算自适应阈值
        
        策略：
        - 简单查询（低复杂度）：提高阈值，更严格筛选
        - 复杂查询（高复杂度）：降低阈值，提高召回
        """
        # 复杂度调整：范围 [0,1]
        # 简单查询 complexity=0 → threshold 提高 0.05
        # 复杂查询 complexity=1 → threshold 降低 0.10
        complexity_adjustment = (
            (0.5 - complexity) * self.config.complexity_weight * 2
        )
        
        adjusted = self.config.base_min_score + complexity_adjustment
        
        # 确保不低于地板值
        return max(adjusted, self.config.min_score_floor)
    
    def _detect_cliff(
        self,
        results: List[float],
        below_threshold_indices: List[int]
    ) -> dict:
        """
        检测悬崖（分数骤降点）
        
        悬崖定义：
        相邻结果的分数下降超过配置阈值（默认20%）
        """
        if len(results) < 2:
            return {'detected': False, 'index': None, 'drop_rate': None}
        
        # 如果第一个结果就低于阈值，不是悬崖
        if results[0] < self.config.base_min_score:
            return {'detected': False, 'index': None, 'drop_rate': None}
        
        # 检查相邻分数的下降幅度
        for i in range(1, len(results)):
            prev_score = results[i - 1]
            curr_score = results[i]
            
            if prev_score <= 0:
                continue
            
            drop_rate = (prev_score - curr_score) / prev_score
            
            if drop_rate > self.config.cliff_drop_threshold:
                return {
                    'detected': True,
                    'index': i,
                    'drop_rate': drop_rate
                }
        
        return {'detected': False, 'index': None, 'drop_rate': None}
    
    async def estimate_query_complexity(self, query: str) -> float:
        """
        估计查询复杂度
        
        启发式规则：
        - 短查询（<10字符）+ 特定术语 → 简单 (0.0-0.3)
        - 中等长度查询 → 普通 (0.4-0.6)
        - 长查询 + 概念性词汇 → 复杂 (0.7-1.0)
        """
        complexity_indicators = {
            'simple': [
                'how to use', 'how do i', 'what is', 'usage of',
                'function', 'method', 'api', 'class', 'module'
            ],
            'complex': [
                'explain', 'understand', 'difference between',
                'best practice', 'architecture', 'design pattern',
                'performance', 'optimization', 'comparison'
            ]
        }
        
        query_lower = query.lower()
        query_length = len(query.split())
        
        # 基于长度的初步估计
        if query_length <= 3:
            base_complexity = 0.2
        elif query_length <= 6:
            base_complexity = 0.5
        else:
            base_complexity = 0.7
        
        # 简单查询指标（降低复杂度）
        simple_count = sum(
            1 for indicator in complexity_indicators['simple']
            if indicator in query_lower
        )
        
        # 复杂查询指标（增加复杂度）
        complex_count = sum(
            1 for indicator in complexity_indicators['complex']
            if indicator in query_lower
        )
        
        # 调整复杂度
        complexity = base_complexity - (simple_count * 0.1) + (complex_count * 0.1)
        
        return max(0.0, min(1.0, complexity))


class CliffEdgeCutoff:
    """
    悬崖截断器
    
    专门用于检测和截断分数"悬崖"
    """
    
    def __init__(
        self,
        drop_threshold: float = 0.20,
        min_score: float = 0.55,
        max_results: int = 10
    ):
        self.drop_threshold = drop_threshold
        self.min_score = min_score
        self.max_results = max_results
    
    def cutoff(
        self,
        results: List[float]
    ) -> Tuple[List[int], Optional[int]]:
        """
        执行悬崖截断
        
        Args:
            results: 按相似度排序的分数列表
            
        Returns:
            - 保留的索引列表
            - 悬崖位置（None表示未检测到悬崖）
        """
        if not results:
            return [], None
        
        # 限制最大数量
        if len(results) > self.max_results:
            results = results[:self.max_results]
        
        cliff_index = None
        
        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]
            
            if prev <= 0:
                continue
            
            drop_rate = (prev - curr) / prev
            
            if drop_rate > self.drop_threshold:
                cliff_index = i
                logger.info(
                    f"Cliff edge detected: score dropped from "
                    f"{prev:.3f} to {curr:.3f} ({drop_rate:.1%})"
                )
                break
        
        # 返回悬崖之前的所有结果
        if cliff_index is not None:
            return list(range(cliff_index)), cliff_index
        else:
            # 无悬崖，返回所有结果
            return list(range(len(results))), None


class HybridCutoffStrategy:
    """
    混合截断策略
    
    组合悬崖截断和最低分数限制
    """
    
    def __init__(
        self,
        min_score: float = 0.60,
        drop_threshold: float = 0.20,
        max_results: int = 15
    ):
        self.min_score = min_score
        self.drop_threshold = drop_threshold
        self.max_results = max_results
    
    def filter(
        self,
        scores: List[float]
    ) -> Tuple[List[int], dict]:
        """
        混合过滤
        
        Returns:
            - 保留的索引
            - 过滤元数据
        """
        metadata = {
            'total': len(scores),
            'cliff_detected': False,
            'cliff_index': None,
            'min_score_enforced': False,
            'max_limit_enforced': False
        }
        
        if not scores:
            return [], metadata
        
        # 悬崖检测
        cliff_cutter = CliffEdgeCutoff(
            drop_threshold=self.drop_threshold,
            min_score=self.min_score,
            max_results=self.max_results
        )
        
        cliff_indices, cliff_index = cliff_cutter.cutoff(scores)
        
        if cliff_index is not None:
            metadata['cliff_detected'] = True
            metadata['cliff_index'] = cliff_index
            return cliff_indices, metadata
        
        # 无悬崖，应用最低分数限制
        min_indices = [
            i for i, s in enumerate(scores)
            if s >= self.min_score
        ]
        
        if len(min_indices) < len(scores):
            metadata['min_score_enforced'] = True
        
        # 限制最大数量
        if len(min_indices) > self.max_results:
            metadata['max_limit_enforced'] = True
            min_indices = min_indices[:self.max_results]
        
        return min_indices, metadata
