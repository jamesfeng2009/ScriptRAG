"""智能跳过优化 - 根据内容质量智能跳过不必要的操作

本模块实现智能跳过功能：
1. 质量评估：评估内容质量，决定是否跳过事实检查
2. 复杂度阈值：基于复杂度决定是否跳过某些处理步骤
3. 缓存命中跳过：如果缓存命中，可以跳过重复操作
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class SkipDecision:
    """跳过决策结果"""
    should_skip: bool
    skip_type: str
    confidence: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class QualityAssessor:
    """
    内容质量评估器
    
    评估内容质量，决定是否可以跳过某些处理步骤。
    """

    def __init__(
        self,
        high_quality_threshold: float = 0.85,
        low_quality_threshold: float = 0.5,
        cache_ttl: int = 300
    ):
        self.high_quality_threshold = high_quality_threshold
        self.low_quality_threshold = low_quality_threshold
        self.cache_ttl = cache_ttl

        self._quality_cache: Dict[str, Tuple[float, float]] = {}

    def assess_quality(
        self,
        content: str,
        context: Optional[str] = None
    ) -> float:
        """
        评估内容质量分数

        Args:
            content: 要评估的内容
            context: 上下文信息

        Returns:
            质量分数 (0.0 - 1.0)
        """
        cache_key = self._generate_cache_key(content, context)

        if cache_key in self._quality_cache:
            quality, timestamp = self._quality_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return quality

        quality = self._calculate_quality(content, context)

        self._quality_cache[cache_key] = (quality, time.time())

        return quality

    def _calculate_quality(
        self,
        content: str,
        context: Optional[str]
    ) -> float:
        """计算质量分数"""
        if not content or len(content.strip()) == 0:
            return 0.0

        score = 0.0

        length_factor = min(len(content) / 5000, 1.0) * 0.2
        score += length_factor

        has_code_blocks = content.count('```') >= 2
        if has_code_blocks:
            score += 0.2

        has_structure = any(marker in content for marker in ['##', '###', '**', '- '])
        if has_structure:
            score += 0.15

        technical_terms = [
            'function', 'class', 'method', 'api', 'interface',
            'async', 'await', 'import', 'export', 'def ', 'const '
        ]
        term_count = sum(1 for term in technical_terms if term in content)
        term_factor = min(term_count / 10, 1.0) * 0.2
        score += term_factor

        if context:
            context_match = self._check_context_relevance(content, context)
            score += context_match * 0.25

        return min(1.0, score)

    def _check_context_relevance(
        self,
        content: str,
        context: str
    ) -> float:
        """检查内容与上下文的关联度"""
        context_words = set(context.lower().split())
        content_words = set(content.lower().split())

        if not context_words:
            return 0.5

        overlap = len(context_words & content_words)
        return min(1.0, overlap / len(context_words))

    def _generate_cache_key(
        self,
        content: str,
        context: Optional[str]
    ) -> str:
        """生成缓存键"""
        content_hash = hash(content[:500])
        if context:
            context_hash = hash(context[:200])
            return f"{content_hash}:{context_hash}"
        return str(content_hash)

    def should_skip_fact_check(
        self,
        quality_score: float,
        skip_types: list = None
    ) -> SkipDecision:
        """
        决定是否应该跳过事实检查

        Args:
            quality_score: 内容质量分数
            skip_types: 允许跳过的检查类型

        Returns:
            SkipDecision 决策结果
        """
        skip_types = skip_types or ['fact_check']

        if 'fact_check' not in skip_types:
            return SkipDecision(
                should_skip=False,
                skip_type='none',
                confidence=1.0,
                reason='skip_types not allowed'
            )

        if quality_score >= self.high_quality_threshold:
            return SkipDecision(
                should_skip=True,
                skip_type='fact_check',
                confidence=quality_score,
                reason=f'High quality content (score={quality_score:.2f})',
                metadata={
                    'quality_score': quality_score,
                    'threshold': self.high_quality_threshold
                }
            )

        elif quality_score < self.low_quality_threshold:
            return SkipDecision(
                should_skip=False,
                skip_type='fact_check',
                confidence=1.0 - quality_score,
                reason=f'Low quality content (score={quality_score:.2f}), need verification',
                metadata={
                    'quality_score': quality_score,
                    'threshold': self.low_quality_threshold
                }
            )

        return SkipDecision(
            should_skip=False,
            skip_type='fact_check',
            confidence=0.5,
            reason=f'Medium quality content (score={quality_score:.2f}), requires verification',
            metadata={
                'quality_score': quality_score,
                'threshold': self.high_quality_threshold
            }
        )


class ComplexityBasedSkipper:
    """
    基于复杂度的跳过器

    根据内容复杂度决定是否跳过某些处理步骤。
    """

    def __init__(
        self,
        skip_threshold: float = 0.8,
        reduce_processing_threshold: float = 0.6
    ):
        self.skip_threshold = skip_threshold
        self.reduce_processing_threshold = reduce_processing_threshold

    def should_skip_detailed_processing(
        self,
        complexity_score: float
    ) -> Tuple[bool, str]:
        """
        决定是否应该跳过详细处理

        Args:
            complexity_score: 复杂度分数

        Returns:
            (是否跳过, 原因)
        """
        if complexity_score >= self.skip_threshold:
            return True, f"High complexity ({complexity_score:.2f}), skipping detailed processing"

        elif complexity_score >= self.reduce_processing_threshold:
            return False, f"Medium complexity ({complexity_score:.2f}), using reduced processing"

        return False, f"Low complexity ({complexity_score:.2f}), using standard processing"

    def get_processing_mode(
        self,
        complexity_score: float
    ) -> str:
        """
        获取处理模式

        Args:
            complexity_score: 复杂度分数

        Returns:
            处理模式: 'standard', 'reduced', 'minimal'
        """
        if complexity_score >= 0.8:
            return 'minimal'

        elif complexity_score >= 0.5:
            return 'reduced'

        return 'standard'


class CacheBasedSkipper:
    """
    基于缓存的跳过器

    如果缓存命中，可以跳过重复的操作。
    """

    def __init__(self):
        self._cache_hits: Dict[str, int] = {}
        self._total_lookups: int = 0

    def check_cache_hit(
        self,
        cache_key: str
    ) -> Tuple[bool, int]:
        """
        检查缓存是否命中

        Args:
            cache_key: 缓存键

        Returns:
            (是否命中, 命中次数)
        """
        self._total_lookups += 1

        if cache_key in self._cache_hits:
            self._cache_hits[cache_key] += 1
            return True, self._cache_hits[cache_key]

        self._cache_hits[cache_key] = 1
        return False, 1

    def should_skip_processing(
        self,
        cache_key: str,
        min_hits_for_skip: int = 2
    ) -> SkipDecision:
        """
        决定是否应该跳过处理（基于缓存命中历史）

        Args:
            cache_key: 缓存键
            min_hits_for_skip: 跳过的最小命中次数

        Returns:
            SkipDecision 决策结果
        """
        is_hit, hit_count = self.check_cache_hit(cache_key)

        if is_hit and hit_count >= min_hits_for_skip:
            return SkipDecision(
                should_skip=True,
                skip_type='cache_processing',
                confidence=min(1.0, hit_count / 5),
                reason=f'Cache hit {hit_count} times, skipping redundant processing',
                metadata={
                    'hit_count': hit_count,
                    'min_hits_for_skip': min_hits_for_skip
                }
            )

        return SkipDecision(
            should_skip=False,
            skip_type='cache_processing',
            confidence=0.0,
            reason='Cache miss or insufficient hits',
            metadata={
                'is_hit': is_hit,
                'hit_count': hit_count
            }
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self._total_lookups
        hits = sum(self._cache_hits.values())

        return {
            'total_lookups': total,
            'total_hits': hits,
            'hit_rate': hits / total if total > 0 else 0.0,
            'unique_keys': len(self._cache_hits)
        }


class SmartSkipOptimizer:
    """
    智能跳过优化器

    综合多个跳过策略，做出最优跳过决策。
    """

    def __init__(
        self,
        enable_quality_skip: bool = True,
        enable_complexity_skip: bool = True,
        enable_cache_skip: bool = True
    ):
        self.quality_assessor = QualityAssessor()
        self.complexity_skipper = ComplexityBasedSkipper()
        self.cache_skipper = CacheBasedSkipper()

        self.enable_quality_skip = enable_quality_skip
        self.enable_complexity_skip = enable_complexity_skip
        self.enable_cache_skip = enable_cache_skip

    def evaluate_skip_decision(
        self,
        content: str,
        complexity_score: float,
        cache_key: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, SkipDecision]:
        """
        评估跳过决策

        Args:
            content: 要评估的内容
            complexity_score: 复杂度分数
            cache_key: 缓存键
            context: 上下文

        Returns:
            各类型的跳过决策
        """
        decisions = {}

        if self.enable_quality_skip:
            quality_score = self.quality_assessor.assess_quality(content, context)
            decisions['quality'] = self.quality_assessor.should_skip_fact_check(
                quality_score,
                skip_types=['fact_check']
            )

        if self.enable_complexity_skip:
            skip, reason = self.complexity_skipper.should_skip_detailed_processing(
                complexity_score
            )
            decisions['complexity'] = SkipDecision(
                should_skip=skip,
                skip_type='complexity_processing',
                confidence=complexity_score,
                reason=reason,
                metadata={'complexity_score': complexity_score}
            )

        if self.enable_cache_skip and cache_key:
            decisions['cache'] = self.cache_skipper.should_skip_processing(
                cache_key,
                min_hits_for_skip=2
            )

        return decisions

    def get_overall_skip_decision(
        self,
        decisions: Dict[str, SkipDecision]
    ) -> SkipDecision:
        """
        综合所有决策，返回总体跳过决策

        Args:
            decisions: 各维度的跳过决策

        Returns:
            总体跳过决策
        """
        if not decisions:
            return SkipDecision(
                should_skip=False,
                skip_type='none',
                confidence=0.0,
                reason='No decisions available'
            )

        skip_count = sum(1 for d in decisions.values() if d.should_skip)
        total_count = len(decisions)

        if skip_count >= total_count // 2:
            avg_confidence = sum(d.confidence for d in decisions.values()) / total_count
            skip_types = [d.skip_type for d in decisions.values() if d.should_skip]

            return SkipDecision(
                should_skip=True,
                skip_type='combined',
                confidence=avg_confidence,
                reason=f'Skipping based on {skip_count}/{total_count} criteria: {", ".join(skip_types)}',
                metadata={'individual_decisions': {k: v.__dict__ for k, v in decisions.items()}}
            )

        return SkipDecision(
            should_skip=False,
            skip_type='combined',
            confidence=0.0,
            reason=f'Only {skip_count}/{total_count} skip criteria met',
            metadata={'individual_decisions': {k: v.__dict__ for k, v in decisions.items()}}
        )
