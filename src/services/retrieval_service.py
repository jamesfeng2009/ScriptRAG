"""检索服务 - 混合检索策略，支持可插拔策略

该服务实现混合检索策略，包括：
- 多种检索策略（向量、关键词、混合）
- 可配置的结果合并算法
- 查询扩展和重排序支持
- 高级增强流水线（查询改写、Cross-Encoder、自适应阈值、GraphRAG）
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
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

from .retrieval_enhancement import (
    RetrievalEnhancementPipeline,
    RetrievalPipelineBuilder,
    EnhancementConfig,
    EnhancementResult
)

from .graprag_engine import GraphRAGEngine


logger = logging.getLogger(__name__)


class RetrievalService:
    """
    检索服务

    功能：
    - 支持多种检索策略（向量、关键词、混合）
    - 可配置的结果合并算法
    - 查询扩展和重排序
    - 自定义关键词标记支持
    - 高级检索增强（查询改写、Cross-Encoder精排、悬崖截断、GraphRAG多跳）
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
        monitor: Optional[RetrievalMonitor] = None,
        enhancement_config: Optional[EnhancementConfig] = None,
        graprag_workspace_id: Optional[str] = None
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
            enhancement_config: 检索增强配置
            graprag_workspace_id: GraphRAG 工作空间 ID
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

        self._init_enhancement_pipeline(enhancement_config, graprag_workspace_id)

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

    def _init_enhancement_pipeline(
        self,
        config: Optional[EnhancementConfig],
        graprag_workspace_id: Optional[str]
    ):
        """初始化检索增强流水线"""
        if config is None:
            self.enhancement_pipeline = None
            logger.info("Enhancement pipeline not enabled (no config provided)")
            return

        graprag_id = graprag_workspace_id or f"retrieval_{id(self)}"

        if config.enable_graprag:
            graprag_engine = GraphRAGEngine(workspace_id=graprag_id)
        else:
            graprag_engine = None

        self.enhancement_pipeline = RetrievalEnhancementPipeline(
            llm_service=self.llm_service,
            vector_db_service=self.vector_db,
            config=config,
            graprag_engine=graprag_engine
        )

        logger.info(
            f"Enhancement pipeline initialized: "
            f"query_rewrite={config.enable_query_rewrite}, "
            f"cross_encoder={config.enable_cross_encoder}, "
            f"graprag={config.enable_graprag}"
        )

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
        stats = {
            "available_strategies": self.list_strategies(),
            "current_strategy": self.config.strategy.name.value,
            "current_merger": self.merger.name,
            "custom_markers": self.config.custom_markers,
            "total_strategies": len(self.strategies)
        }

        if self.enhancement_pipeline:
            stats["enhancement"] = self.enhancement_pipeline.get_stats()

        return stats

    def enable_enhancement_pipeline(
        self,
        enable_query_rewrite: bool = True,
        enable_cross_encoder: bool = False,
        enable_mmr: bool = True,
        enable_adaptive_threshold: bool = True,
        enable_graprag: bool = False,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> bool:
        """
        启用检索增强流水线

        Args:
            enable_query_rewrite: 启用查询改写
            enable_cross_encoder: 启用 Cross-Encoder 精排
            enable_mmr: 启用 MMR 多样性排序
            enable_adaptive_threshold: 启用自适应阈值截断
            enable_graprag: 启用 GraphRAG 多跳检索
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重

        Returns:
            是否成功启用
        """
        try:
            config = EnhancementConfig(
                enable_query_rewrite=enable_query_rewrite,
                enable_cross_encoder=enable_cross_encoder,
                enable_mmr=enable_mmr,
                enable_adaptive_threshold=enable_adaptive_threshold,
                enable_graprag=enable_graprag,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight
            )

            self._init_enhancement_pipeline(config, None)
            logger.info("Enhancement pipeline enabled successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to enable enhancement pipeline: {str(e)}")
            return False

    async def enhanced_retrieve(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 10,
        use_enhancement: bool = True,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        增强检索（使用检索增强流水线）

        如果启用了增强流水线，则使用完整的增强流程：
        1. 查询改写
        2. 查询扩展
        3. 混合检索
        4. Cross-Encoder/MMR 精排
        5. 悬崖截断
        6. GraphRAG 多跳扩展（可选）

        Args:
            workspace_id: 工作空间 ID
            query: 查询文本
            top_k: 返回结果数量
            use_enhancement: 是否使用增强流水线
            **kwargs: 额外参数

        Returns:
            检索结果列表
        """
        if not use_enhancement or not self.enhancement_pipeline:
            return await self.hybrid_retrieve(workspace_id, query, top_k, **kwargs)

        try:
            result: EnhancementResult = await self.enhancement_pipeline.enhanced_retrieve(
                workspace_id=workspace_id,
                query=query,
                top_k=top_k,
                filters=kwargs.get("filters")
            )

            logger.info(
                f"Enhanced retrieval: query={query[:50]}..., "
                f"results={len(result.final_results)}, "
                f"time={result.processing_time_ms:.1f}ms"
            )

            return result.final_results

        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {str(e)}")
            raise

    async def rewrite_query(self, query: str) -> Dict[str, Any]:
        """
        改写查询

        Args:
            query: 原始查询

        Returns:
            改写结果
        """
        if not self.enhancement_pipeline:
            return {
                "original_query": query,
                "rewritten_query": query,
                "sub_queries": [],
                "error": "Enhancement pipeline not enabled"
            }

        try:
            rewrite_result = await self.enhancement_pipeline.query_rewriter.rewrite(query)
            return {
                "original_query": query,
                "rewritten_query": rewrite_result.rewritten_query,
                "sub_queries": rewrite_result.sub_queries,
                "confidence": rewrite_result.confidence
            }

        except Exception as e:
            logger.error(f"Query rewrite failed: {str(e)}")
            return {
                "original_query": query,
                "rewritten_query": query,
                "sub_queries": [],
                "error": str(e)
            }

    async def rerank_results(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 10,
        use_cross_encoder: bool = False
    ) -> List[RetrievalResult]:
        """
        对检索结果进行重排序

        Args:
            query: 查询文本
            results: 待重排序的结果
            top_k: 返回结果数量
            use_cross_encoder: 是否使用 Cross-Encoder

        Returns:
            重排序后的结果
        """
        if not self.enhancement_pipeline:
            return results[:top_k]

        try:
            reranking_pipeline = self.enhancement_pipeline.reranking_pipeline

            if use_cross_encoder:
                reranked = await reranking_pipeline.rerank(
                    query=query,
                    results=results,
                    top_k=top_k * 2,
                    use_primary=True
                )
            else:
                reranked = await reranking_pipeline.rerank(
                    query=query,
                    results=results,
                    top_k=top_k * 2,
                    use_primary=False
                )

            return [
                RetrievalResult(
                    id=r.id,
                    file_path=r.file_path,
                    content=r.content,
                    similarity=r.fused_score,
                    confidence=r.fused_score,
                    strategy_name="reranked"
                )
                for r in reranked[:top_k]
            ]

        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            return results[:top_k]

    def apply_adaptive_threshold(
        self,
        results: List[RetrievalResult],
        query: Optional[str] = None
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        应用自适应阈值截断

        Args:
            results: 待处理的结果
            query: 查询文本（用于估计复杂度）

        Returns:
            (截断后的结果, 分析信息)
        """
        if not self.enhancement_pipeline:
            return results, {"error": "Enhancement pipeline not enabled"}

        try:
            strategy = self.enhancement_pipeline.threshold_strategy
            scores = [r.similarity for r in results]

            import asyncio
            complexity = asyncio.get_event_loop().run_until_complete(
                strategy.estimate_query_complexity(query or "")
            )

            indices, analysis = strategy.analyze_and_filter(scores, complexity)

            filtered_results = [results[i] for i in indices]

            return filtered_results, {
                "applied_threshold": analysis.applied_threshold,
                "retained_count": analysis.retained_count,
                "cliff_detected": analysis.cliff_detected,
                "complexity": complexity
            }

        except Exception as e:
            logger.error(f"自适应阈值截断失败: {str(e)}")
            return results, {"error": str(e)}

    def index_for_bm25(self, documents: List[Dict[str, str]]):
        """
        索引文档用于 BM25 检索

        Args:
            documents: 文档列表，每项包含 id, content, file_path
        """
        if self.enhancement_pipeline:
            self.enhancement_pipeline.index_for_bm25(documents)

    def index_for_graprag(self, content: str, file_path: str, doc_type: str = "code"):
        """
        索引文档用于 GraphRAG

        Args:
            content: 文档内容
            file_path: 文件路径
            doc_type: 文档类型
        """
        if self.enhancement_pipeline:
            self.enhancement_pipeline.index_for_graprag(content, file_path, doc_type)

    def get_related_via_graprag(
        self,
        doc_id: str,
        max_depth: int = 2
    ) -> List[Tuple[str, float]]:
        """
        通过 GraphRAG 获取关联文档

        Args:
            doc_id: 文档 ID
            max_depth: 最大跳数

        Returns:
            关联文档列表 (节点, 分数)
        """
        if not self.enhancement_pipeline or not self.enhancement_pipeline.graprag_engine:
            return []

        try:
            related = self.enhancement_pipeline.graprag_engine.get_related_documents(
                doc_id,
                max_depth=max_depth
            )
            return [(node, score) for node, score in related]

        except Exception as e:
            logger.error(f"GraphRAG 关联文档获取失败: {str(e)}")
            return []
