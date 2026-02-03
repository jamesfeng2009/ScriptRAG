"""Retrieval Service - Hybrid retrieval with pluggable strategies

This service implements the hybrid retrieval strategy using:
- Multiple retrieval strategies (vector, keyword, hybrid)
- Configurable result merging algorithms
- Query expansion and reranking support
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from .database.vector_db import IVectorDBService
from .llm.service import LLMService
from .query_expansion import QueryExpansion, QueryOptimizer
from .reranker import MultiFactorReranker, DiversityFilter, RetrievalQualityMonitor
from .cache.retrieval_cache import RetrievalCache, CacheConfig
from .monitoring.retrieval_monitor import RetrievalMonitor, MonitoringConfig

from .retrieval.config import (
    RetrievalConfig,
    RetrievalStrategyConfig,
    MergerConfig
)
from .retrieval.strategies import (
    RetrievalResult,
    RetrievalStrategy,
    VectorSearchStrategy,
    KeywordSearchStrategy,
    HybridSearchStrategy,
    StrategyRegistry
)
from .retrieval.mergers import (
    ResultMerger,
    WeightedMerger,
    ReciprocalRankMerger,
    FusionMerger,
    MergerRegistry
)


logger = logging.getLogger(__name__)


class RetrievalService:
    """
    检索服务

    功能：
    - 支持多种检索策略（向量、关键词、混合）
    - 可配置的结果合并算法
    - 查询扩展和重排序
    - 自定义关键词标记支持
    """

    def __init__(
        self,
        vector_db_service: IVectorDBService,
        llm_service: LLMService,
        config: Optional[RetrievalConfig] = None,
        query_expansion: Optional[QueryExpansion] = None,
        reranker: Optional[MultiFactorReranker] = None,
        diversity_filter: Optional[DiversityFilter] = None,
        quality_monitor: Optional[RetrievalQualityMonitor] = None,
        cache: Optional[RetrievalCache] = None,
        monitor: Optional[RetrievalMonitor] = None
    ):
        """
        初始化检索服务

        Args:
            vector_db_service: 向量数据库服务
            llm_service: LLM 服务
            config: 检索配置
            query_expansion: 查询扩展组件
            reranker: 重排序组件
            diversity_filter: 多样性过滤组件
            quality_monitor: 质量监控组件
            cache: 缓存组件
            monitor: 监控组件
        """
        self.vector_db = vector_db_service
        self.llm_service = llm_service
        self.config = config or RetrievalConfig()

        self._initialize_strategies()
        self._initialize_merger()

        self.query_expansion = query_expansion or QueryExpansion(llm_service)
        self.query_optimizer = QueryOptimizer()
        self.reranker = reranker or MultiFactorReranker()
        self.diversity_filter = diversity_filter or DiversityFilter()
        self.quality_monitor = quality_monitor or RetrievalQualityMonitor()

        self.cache = cache or RetrievalCache(CacheConfig())
        self.monitor = monitor or RetrievalMonitor(MonitoringConfig())

    def _initialize_strategies(self):
        """初始化检索策略"""
        self.strategies: Dict[str, RetrievalStrategy] = {}
        strategy_config = self.config.strategy

        if strategy_config.vector.enabled:
            self.strategies["vector_search"] = VectorSearchStrategy(
                vector_db_service=self.vector_db,
                llm_service=self.llm_service,
                similarity_threshold=strategy_config.vector.similarity_threshold
            )

        if strategy_config.keyword.enabled:
            self.strategies["keyword_search"] = KeywordSearchStrategy(
                vector_db_service=self.vector_db,
                llm_service=self.llm_service,
                markers=strategy_config.keyword.markers,
                boost_factors=strategy_config.keyword.boost_factors
            )

        if strategy_config.hybrid_merge.enabled:
            self.strategies["hybrid_search"] = HybridSearchStrategy(
                vector_db_service=self.vector_db,
                llm_service=self.llm_service,
                vector_strategy=self.strategies.get("vector_search"),
                keyword_strategy=self.strategies.get("keyword_search"),
                vector_weight=strategy_config.hybrid_merge.vector_weight,
                keyword_weight=strategy_config.hybrid_merge.keyword_weight
            )

        logger.info(f"Initialized {len(self.strategies)} retrieval strategies")

    def _initialize_merger(self):
        """初始化结果合并策略"""
        merger_config = self.config.merger

        merger_map = {
            "weighted_merge": WeightedMerger,
            "rrf_merge": ReciprocalRankMerger,
            "fusion_merge": FusionMerger
        }

        merger_class = merger_map.get(merger_config.type.value)
        if merger_class:
            if merger_config.type.value == "weighted_merge":
                self.merger = merger_class(
                    dedup_threshold=self.config.strategy.hybrid_merge.dedup_threshold
                )
            elif merger_config.type.value == "fusion_merge":
                self.merger = merger_class(
                    alpha=merger_config.fusion.alpha if merger_config.fusion else 0.5,
                    dedup_threshold=merger_config.fusion.dedup_threshold if merger_config.fusion else 0.9
                )
            else:
                self.merger = merger_class()
        else:
            self.merger = WeightedMerger(
                dedup_threshold=self.config.strategy.hybrid_merge.dedup_threshold
            )

        logger.info(f"Initialized merger: {self.merger.name}")

    def register_strategy(
        self,
        name: str,
        strategy: RetrievalStrategy
    ) -> bool:
        """
        注册自定义检索策略

        Args:
            name: 策略名称
            strategy: 策略实例

        Returns:
            是否注册成功
        """
        if not name or not strategy:
            return False

        self.strategies[name] = strategy
        logger.info(f"Registered custom strategy: {name}")
        return True

    def unregister_strategy(self, name: str) -> bool:
        """注销检索策略"""
        if name in self.strategies:
            del self.strategies[name]
            logger.info(f"Unregistered strategy: {name}")
            return True
        return False

    def get_strategy(self, name: str) -> Optional[RetrievalStrategy]:
        """获取检索策略"""
        return self.strategies.get(name)

    def list_strategies(self) -> List[str]:
        """列出所有可用策略"""
        return list(self.strategies.keys())

    def set_merger(self, merger: ResultMerger):
        """设置结果合并策略"""
        self.merger = merger
        logger.info(f"Set merger: {merger.name}")

    def add_custom_marker(self, marker: str) -> bool:
        """添加自定义关键词标记"""
        return self.config.add_custom_marker(marker)

    def remove_custom_marker(self, marker: str) -> bool:
        """移除自定义关键词标记"""
        return self.config.remove_custom_marker(marker)

    def get_all_markers(self) -> List[str]:
        """获取所有关键词标记"""
        return self.config.get_all_markers()

    async def hybrid_retrieve(
        self,
        workspace_id: str,
        query: str,
        top_k: Optional[int] = None,
        strategy_name: Optional[str] = None,
        merger_name: Optional[str] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        混合检索（使用配置的策略和合并算法）

        Args:
            workspace_id: 工作空间 ID
            query: 查询文本
            top_k: 返回结果数量
            strategy_name: 使用的检索策略（默认使用配置的主策略）
            merger_name: 使用的合并策略（默认使用配置的策略）
            **kwargs: 额外参数

        Returns:
            检索结果列表
        """
        final_top_k = top_k or self.config.strategy.vector.top_k

        try:
            optimized_query = self._optimize_query(query)
            expanded_queries = self._expand_query(optimized_query)

            results_by_strategy = await self._search_all_strategies(
                workspace_id=workspace_id,
                query=optimized_query,
                expanded_queries=expanded_queries,
                top_k=final_top_k * 2,
                **kwargs
            )

            if not results_by_strategy:
                logger.warning("No results from any strategy")
                return []

            merged_results = self._merge_results(
                results_by_strategy=results_by_strategy,
                strategy_name=strategy_name,
                merger_name=merger_name,
                top_k=final_top_k
            )

            reranked_results = self._rerank_results(
                query=optimized_query,
                results=merged_results,
                top_k=final_top_k
            )

            final_results = self._apply_diversity_filter(reranked_results, final_top_k)

            logger.info(f"Hybrid retrieval returned {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}")
            raise

    def _optimize_query(self, query: str) -> str:
        """优化查询"""
        if self.config.query_expansion.enabled:
            return self.query_optimizer.optimize_query(query)
        return query

    def _expand_query(self, query: str) -> List[str]:
        """扩展查询"""
        if self.config.query_expansion.enabled:
            max_queries = self.config.query_expansion.max_queries
            return [query] * max_queries
        return [query]

    async def _search_all_strategies(
        self,
        workspace_id: str,
        query: str,
        expanded_queries: List[str],
        top_k: int,
        **kwargs
    ) -> Dict[str, List[RetrievalResult]]:
        """并行执行所有策略的检索"""
        results_by_strategy: Dict[str, List[RetrievalResult]] = {}

        for strategy_name, strategy in self.strategies.items():
            try:
                query_embedding = await self.llm_service.embedding([query])
                if not query_embedding:
                    continue

                strategy_results = await strategy.search(
                    query=query,
                    query_embedding=query_embedding[0],
                    workspace_id=workspace_id,
                    top_k=top_k,
                    custom_markers=self.config.custom_markers,
                    **kwargs
                )

                if strategy_results:
                    results_by_strategy[strategy_name] = strategy_results
                    logger.debug(f"Strategy {strategy_name}: {len(strategy_results)} results")

            except Exception as e:
                logger.error(f"Strategy {strategy_name} failed: {str(e)}")
                continue

        return results_by_strategy

    def _merge_results(
        self,
        results_by_strategy: Dict[str, List[RetrievalResult]],
        strategy_name: Optional[str],
        merger_name: Optional[str],
        top_k: int
    ) -> List[RetrievalResult]:
        """合并检索结果"""
        if not results_by_strategy:
            return []

        if len(results_by_strategy) == 1:
            strategy_name = list(results_by_strategy.keys())[0]
            results = results_by_strategy[strategy_name]
            return results[:top_k]

        weights = {
            name: self.config.strategy.weights.get(name, 1.0)
            for name in results_by_strategy.keys()
        }

        if merger_name:
            merger = MergerRegistry.create_merger(merger_name)
            if merger:
                return merger.merge(results_by_strategy, weights, top_k)

        return self.merger.merge(results_by_strategy, weights, top_k)

    def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """重排序结果"""
        if not self.config.reranking.enabled or not results:
            return results

        reranked = self.reranker.rerank(
            query=query,
            results=results,
            top_k=top_k * 2
        )
        return reranked

    def _apply_diversity_filter(
        self,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """应用多样性过滤"""
        if not self.config.diversity.enabled or not results:
            return results[:top_k]

        filtered = self.diversity_filter.filter(
            results=results,
            threshold=self.config.diversity.threshold,
            top_k=top_k
        )
        return filtered

    async def retrieve_with_strategy(
        self,
        workspace_id: str,
        query: str,
        strategy_name: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """使用指定策略进行检索"""
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            logger.error(f"Unknown strategy: {strategy_name}")
            return []

        final_top_k = top_k or self.config.strategy.vector.top_k

        try:
            query_embedding = await self.llm_service.embedding([query])
            if not query_embedding:
                return []

            results = await strategy.search(
                query=query,
                query_embedding=query_embedding[0],
                workspace_id=workspace_id,
                top_k=final_top_k,
                custom_markers=self.config.custom_markers,
                **kwargs
            )

            logger.info(f"Strategy {strategy_name} returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Strategy {strategy_name} failed: {str(e)}")
            return []

    def get_strategy_stats(self) -> Dict[str, Any]:
        """获取策略统计信息"""
        return {
            "available_strategies": self.list_strategies(),
            "current_strategy": self.config.strategy.name.value,
            "current_merger": self.merger.name,
            "custom_markers": self.config.custom_markers,
            "total_strategies": len(self.strategies)
        }
