"""结果合并器 - 可配置的结果合并策略

该模块提供不同的策略来合并来自多个来源的检索结果，
支持可配置的权重和算法。
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from collections import Counter
from .strategies import RetrievalResult

logger = logging.getLogger(__name__)


class ResultMerger(ABC):
    """结果合并策略抽象基类

    所有合并策略必须实现这个接口。
    支持自定义合并算法扩展。
    """

    def __init__(self, name: str):
        """
        初始化合并策略

        Args:
            name: 策略名称
        """
        self.name = name

    @abstractmethod
    def merge(
        self,
        results_by_strategy: Dict[str, List[RetrievalResult]],
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        合并多个策略的检索结果

        Args:
            results_by_strategy: 各策略的检索结果字典
            weights: 各策略的权重映射
            top_k: 返回结果数量
            **kwargs: 额外参数

        Returns:
            合并后的结果列表
        """
        pass


class WeightedMerger(ResultMerger):
    """加权合并策略

    根据权重对不同策略的结果进行加权合并。
    这是默认的合并策略。
    """

    def __init__(self, dedup_threshold: float = 0.9):
        """
        初始化加权合并策略

        Args:
            dedup_threshold: 去重相似度阈值
        """
        super().__init__("weighted_merge")
        self.dedup_threshold = dedup_threshold

    def merge(
        self,
        results_by_strategy: Dict[str, List[RetrievalResult]],
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        执行加权合并

        Args:
            results_by_strategy: 各策略的检索结果
            weights: 权重映射（默认：各策略权重 0.5）
            top_k: 返回结果数量
            **kwargs: 额外参数

        Returns:
            合并后的结果
        """
        if not results_by_strategy:
            return []

        effective_weights = weights or {
            strategy: 1.0 / len(results_by_strategy)
            for strategy in results_by_strategy.keys()
        }

        result_dict: Dict[str, RetrievalResult] = {}
        strategy_contributions: Dict[str, List[str]] = {}

        for strategy, results in results_by_strategy.items():
            weight = effective_weights.get(strategy, 0.5)

            for result in results:
                if result.id in result_dict:
                    existing = result_dict[result.id]
                    existing.confidence = (
                        existing.confidence +
                        result.similarity * weight
                    )
                    if "strategy_contributions" in existing.metadata:
                        existing.metadata["strategy_contributions"].append(strategy)
                    else:
                        existing.metadata["strategy_contributions"] = [strategy]
                else:
                    result.confidence = result.similarity * weight
                    result.metadata["strategy_contributions"] = [strategy]
                    result.metadata["source"] = strategy
                    result_dict[result.id] = result

        merged = list(result_dict.values())
        merged.sort(key=lambda x: x.confidence, reverse=True)
        final_results = self._apply_diversity_filter(merged, top_k)

        logger.info(
            f"加权合并: 从 {len(results_by_strategy)} 个策略返回了 {len(final_results)} 个结果"
        )
        return final_results

    def _apply_diversity_filter(
        self,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """应用多样性过滤"""
        filtered: List[RetrievalResult] = []
        file_paths: Dict[str] = {}

        for result in results:
            is_duplicate = False
            for existing_path in file_paths:
                if self._calculate_similarity(result.content, existing_path) > self.dedup_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(result)
                file_paths[result.file_path] = True

            if len(filtered) >= top_k:
                break

        return filtered

    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度（简化版）"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0


class ReciprocalRankMerger(ResultMerger):
    """倒数排名合并策略（RRF）

    使用倒数排名算法合并多个检索结果列表。
    适合多策略融合场景。
    """

    def __init__(self, k: int = 60):
        """
        初始化 RRF 合并策略

        Args:
            k: RRF 常数（通常 60）
        """
        super().__init__("rrf_merge")
        self.k = k

    def merge(
        self,
        results_by_strategy: Dict[str, List[RetrievalResult]],
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        执行 RRF 合并

        算法：RRF(d) = 1 / (k + rank(d))
        最终得分 = sum(RRF(d) for all strategies)
        """
        if not results_by_strategy:
            return []

        scores: Dict[str, float] = {}
        rank_info: Dict[str, Dict[str, Any]] = {}

        for strategy, results in results_by_strategy.items():
            weight = weights.get(strategy, 1.0) if weights else 1.0

            for rank, result in enumerate(results, 1):
                rrf_score = self.k / (self.k + rank) * weight

                if result.id in scores:
                    scores[result.id] += rrf_score
                    rank_info[result.id]["strategies"].append(strategy)
                    rank_info[result.id]["max_rank"] = min(
                        rank_info[result.id]["max_rank"], rank
                    )
                else:
                    scores[result.id] = rrf_score
                    rank_info[result.id] = {
                        "strategies": [strategy],
                        "max_rank": rank,
                        "result": result
                    }

        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        merged: List[RetrievalResult] = []
        for result_id, score in sorted_results:
            result = rank_info[result_id]["result"]
            result.confidence = score
            result.metadata["source"] = "rrf"
            result.metadata["strategy_contributions"] = rank_info[result_id]["strategies"]
            merged.append(result)

        logger.info(f"RRF 合并: 从 {len(results_by_strategy)} 个策略返回了 {len(merged)} 个结果")
        return merged


class RoundRobinMerger(ResultMerger):
    """轮询合并策略

    按策略顺序轮询选择结果，确保各策略均衡贡献。
    """

    def __init__(self, max_per_strategy: int = 3):
        """
        初始化轮询合并策略

        Args:
            max_per_strategy: 每个策略最大贡献结果数
        """
        super().__init__("round_robin_merge")
        self.max_per_strategy = max_per_strategy

    def merge(
        self,
        results_by_strategy: Dict[str, List[RetrievalResult]],
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """执行轮询合并"""
        if not results_by_strategy:
            return []

        strategy_order = list(results_by_strategy.keys())
        used_ids: set = set()
        merged: List[RetrievalResult] = []
        round_num = 0

        while len(merged) < top_k and round_num < self.max_per_strategy * len(strategy_order):
            for strategy in strategy_order:
                if len(merged) >= top_k:
                    break

                results = results_by_strategy.get(strategy, [])
                for result in results:
                    if result.id not in used_ids:
                        result.metadata["source"] = "round_robin"
                        result.metadata["round"] = round_num
                        merged.append(result)
                        used_ids.add(result.id)
                        break

            round_num += 1

        merged.sort(key=lambda x: x.confidence, reverse=True)
        logger.info(f"轮询合并: 返回了 {len(merged)} 个结果")
        return merged


class FusionMerger(ResultMerger):
    """融合合并策略

    结合加权合并和 RRF 的优点，支持复杂的融合逻辑。
    """

    def __init__(
        self,
        alpha: float = 0.5,
        dedup_threshold: float = 0.9
    ):
        """
        初始化融合合并策略

        Args:
            alpha: 加权分数和 RRF 分数的融合权重
            dedup_threshold: 去重阈值
        """
        super().__init__("fusion_merge")
        self.alpha = alpha
        self.dedup_threshold = dedup_threshold
        self.rrf_merger = ReciprocalRankMerger()
        self.weighted_merger = WeightedMerger(dedup_threshold)

    def merge(
        self,
        results_by_strategy: Dict[str, List[RetrievalResult]],
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        执行融合合并

        融合公式：score = alpha * weighted_score + (1 - alpha) * rrf_score
        """
        if not results_by_strategy:
            return []

        weighted_results = self.weighted_merger.merge(
            results_by_strategy, weights, top_k * 2
        )
        rrf_results = self.rrf_merger.merge(
            results_by_strategy, weights, top_k * 2
        )

        weighted_dict = {r.id: r for r in weighted_results}
        rrf_dict = {r.id: r for r in rrf_results}

        all_ids = set(weighted_dict.keys()) | set(rrf_dict.keys())

        max_weighted = max((r.confidence for r in weighted_results), default=1.0)
        max_rrf = max((r.confidence for r in rrf_results), default=1.0)

        merged: List[RetrievalResult] = []
        scores: Dict[str, float] = {}

        for result_id in all_ids:
            weighted_score = weighted_dict.get(result_id, RetrievalResult(
                id=result_id,
                file_path="",
                content="",
                similarity=0,
                confidence=0,
                strategy_name="",
            )).confidence / max_weighted if max_weighted > 0 else 0

            rrf_score = rrf_dict.get(result_id, RetrievalResult(
                id=result_id,
                file_path="",
                content="",
                similarity=0,
                confidence=0,
                strategy_name="",
            )).confidence / max_rrf if max_rrf > 0 else 0

            fusion_score = self.alpha * weighted_score + (1 - self.alpha) * rrf_score
            scores[result_id] = fusion_score

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

        for result_id in sorted_ids:
            source_result = weighted_dict.get(result_id) or rrf_dict.get(result_id)
            if source_result:
                source_result.confidence = scores[result_id]
                source_result.metadata["source"] = "fusion"
                merged.append(source_result)

        logger.info(f"融合合并: 返回了 {len(merged)} 个结果")
        return merged


class MergerRegistry:
    """合并策略注册表

    管理所有可用的合并策略，支持动态注册。
    """

    _instance = None
    _mergers: Dict[str, type] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str):
        """装饰器：注册合并策略"""
        def decorator(merger_class):
            cls._mergers[name] = merger_class
            return merger_class
        return decorator

    @classmethod
    def get_merger_class(cls, name: str) -> Optional[type]:
        """获取合并策略类"""
        return cls._mergers.get(name)

    @classmethod
    def list_mergers(cls) -> List[str]:
        """列出所有已注册的合并策略"""
        return list(cls._mergers.keys())

    @classmethod
    def create_merger(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[ResultMerger]:
        """创建合并策略实例"""
        merger_class = cls.get_merger_class(name)
        if merger_class is None:
            logger.error(f"未知的合并策略: {name}")
            return None

        try:
            return merger_class(**(config or {}))
        except Exception as e:
            logger.error(f"创建合并策略 {name} 失败: {str(e)}")
            return None


MergerRegistry.register("weighted_merge")(WeightedMerger)
MergerRegistry.register("rrf_merge")(ReciprocalRankMerger)
MergerRegistry.register("round_robin_merge")(RoundRobinMerger)
MergerRegistry.register("fusion_merge")(FusionMerger)
