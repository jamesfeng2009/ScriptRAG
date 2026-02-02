"""Retrieval Strategies - Pluggable retrieval implementations

This module provides an extensible architecture for retrieval strategies,
allowing new retrieval methods to be added without modifying core logic.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from ..database.vector_db import IVectorDBService, VectorSearchResult
from ..llm.service import LLMService

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    id: str
    file_path: str
    content: str
    similarity: float
    confidence: float
    strategy_name: str
    has_deprecated: bool = False
    has_fixme: bool = False
    has_todo: bool = False
    has_security: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RetrievalStrategy(ABC):
    """检索策略抽象基类

    所有检索策略必须实现这个接口。
    支持自定义策略扩展：
    1. 继承 RetrievalStrategy
    2. 实现 search 方法
    3. 在 RetrievalConfig 中注册策略名称
    """

    def __init__(
        self,
        name: str,
        vector_db_service: IVectorDBService,
        llm_service: LLMService,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化检索策略

        Args:
            name: 策略名称（唯一标识）
            vector_db_service: 向量数据库服务
            llm_service: LLM 服务
            config: 策略特定配置
        """
        self.name = name
        self.vector_db = vector_db_service
        self.llm_service = llm_service
        self.config = config or {}

    @abstractmethod
    async def search(
        self,
        query: str,
        query_embedding: List[float],
        workspace_id: str,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        执行检索

        Args:
            query: 查询文本
            query_embedding: 查询嵌入向量
            workspace_id: 工作空间 ID
            top_k: 返回结果数量
            **kwargs: 额外参数

        Returns:
            检索结果列表
        """
        pass

    @property
    @abstractmethod
    def supported_markers(self) -> Set[str]:
        """返回支持的关键词标记集合"""
        pass

    def validate_query(self, query: str) -> bool:
        """验证查询有效性"""
        return query is not None and len(query.strip()) > 0

    async def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        embeddings = await self.llm_service.embedding([text])
        return embeddings[0] if embeddings else []


class VectorSearchStrategy(RetrievalStrategy):
    """向量搜索策略

    基于语义相似度的向量检索，使用余弦相似度计算。
    """

    def __init__(
        self,
        vector_db_service: IVectorDBService,
        llm_service: LLMService,
        similarity_threshold: float = 0.7,
        **kwargs
    ):
        """
        初始化向量搜索策略

        Args:
            vector_db_service: 向量数据库服务
            llm_service: LLM 服务
            similarity_threshold: 相似度阈值
        """
        super().__init__(
            name="vector_search",
            vector_db_service=vector_db_service,
            llm_service=llm_service,
            config={"similarity_threshold": similarity_threshold}
        )
        self.similarity_threshold = similarity_threshold

    async def search(
        self,
        query: str,
        query_embedding: List[float],
        workspace_id: str,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """执行向量搜索"""
        if not self.validate_query(query):
            logger.warning(f"Invalid query for vector search: {query}")
            return []

        try:
            db_results = await self.vector_db.vector_search(
                workspace_id=workspace_id,
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=self.similarity_threshold
            )

            results = []
            for db_result in db_results:
                results.append(RetrievalResult(
                    id=db_result.id,
                    file_path=db_result.file_path,
                    content=db_result.content,
                    similarity=db_result.similarity,
                    confidence=db_result.similarity,
                    strategy_name=self.name,
                    has_deprecated=db_result.has_deprecated,
                    has_fixme=db_result.has_fixme,
                    has_todo=db_result.has_todo,
                    has_security=db_result.has_security,
                    metadata=db_result.metadata
                ))

            logger.info(f"Vector search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    @property
    def supported_markers(self) -> Set[str]:
        """向量搜索不依赖关键词标记"""
        return set()


class KeywordSearchStrategy(RetrievalStrategy):
    """关键词搜索策略

    基于敏感标记（@deprecated, FIXME, TODO, Security 等）的检索。
    支持自定义关键词标记配置。
    """

    def __init__(
        self,
        vector_db_service: IVectorDBService,
        llm_service: LLMService,
        markers: Optional[List[str]] = None,
        boost_factors: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        初始化关键词搜索策略

        Args:
            vector_db_service: 向量数据库服务
            llm_service: LLM 服务
            markers: 关键词标记列表（默认：@deprecated, FIXME, TODO, Security）
            boost_factors: 标记加权因子映射
        """
        default_markers = ["@deprecated", "FIXME", "TODO", "Security"]
        default_boosts = {
            "@deprecated": 1.5,
            "Security": 1.5,
            "FIXME": 1.3,
            "TODO": 1.2
        }

        super().__init__(
            name="keyword_search",
            vector_db_service=vector_db_service,
            llm_service=llm_service,
            config={
                "markers": markers or default_markers,
                "boost_factors": boost_factors or default_boosts
            }
        )

        self.markers = self.config["markers"]
        self.boost_factors = self.config["boost_factors"]

    async def search(
        self,
        query: str,
        query_embedding: List[float],
        workspace_id: str,
        top_k: int = 5,
        custom_markers: Optional[List[str]] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """执行关键词搜索"""
        if not self.validate_query(query):
            logger.warning(f"Invalid query for keyword search: {query}")
            return []

        try:
            all_markers = self.markers + (custom_markers or [])

            keyword_filters = {}
            marker_mapping = {
                "@deprecated": "has_deprecated",
                "FIXME": "has_fixme",
                "TODO": "has_todo",
                "Security": "has_security"
            }

            for marker in all_markers:
                db_field = marker_mapping.get(marker)
                if db_field:
                    keyword_filters[db_field] = True

            if not keyword_filters:
                logger.warning("No valid keyword filters configured")
                return []

            db_results = await self.vector_db.hybrid_search(
                workspace_id=workspace_id,
                query_embedding=query_embedding,
                keyword_filters=keyword_filters,
                top_k=top_k
            )

            results = []
            for db_result in db_results:
                keyword_count = sum([
                    db_result.has_deprecated,
                    db_result.has_fixme,
                    db_result.has_todo,
                    db_result.has_security
                ])

                boost_factor = 1.0
                if db_result.has_deprecated:
                    boost_factor = self.boost_factors.get("@deprecated", 1.5)
                elif db_result.has_security:
                    boost_factor = self.boost_factors.get("Security", 1.5)
                elif db_result.has_fixme:
                    boost_factor = self.boost_factors.get("FIXME", 1.3)
                elif db_result.has_todo:
                    boost_factor = self.boost_factors.get("TODO", 1.2)

                weighted_similarity = db_result.similarity * boost_factor

                results.append(RetrievalResult(
                    id=db_result.id,
                    file_path=db_result.file_path,
                    content=db_result.content,
                    similarity=weighted_similarity,
                    confidence=weighted_similarity,
                    strategy_name=self.name,
                    has_deprecated=db_result.has_deprecated,
                    has_fixme=db_result.has_fixme,
                    has_todo=db_result.has_todo,
                    has_security=db_result.has_security,
                    metadata={
                        **db_result.metadata,
                        "keyword_count": keyword_count,
                        "boost_factor": boost_factor,
                        "matched_markers": [
                            m for m in all_markers
                            if (m == "@deprecated" and db_result.has_deprecated) or
                               (m == "FIXME" and db_result.has_fixme) or
                               (m == "TODO" and db_result.has_todo) or
                               (m == "Security" and db_result.has_security)
                        ]
                    }
                ))

            results.sort(key=lambda x: x.similarity, reverse=True)
            logger.info(f"Keyword search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []

    @property
    def supported_markers(self) -> Set[str]:
        """支持的关键词标记"""
        return set(self.markers)


class HybridSearchStrategy(RetrievalStrategy):
    """混合搜索策略

    同时使用向量搜索和关键词搜索，然后合并结果。
    这是默认的主检索策略。
    """

    def __init__(
        self,
        vector_db_service: IVectorDBService,
        llm_service: LLMService,
        vector_strategy: Optional[VectorSearchStrategy] = None,
        keyword_strategy: Optional[KeywordSearchStrategy] = None,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        **kwargs
    ):
        """
        初始化混合搜索策略

        Args:
            vector_db_service: 向量数据库服务
            llm_service: LLM 服务
            vector_strategy: 向量搜索策略实例
            keyword_strategy: 关键词搜索策略实例
            vector_weight: 向量结果权重
            keyword_weight: 关键词结果权重
        """
        super().__init__(
            name="hybrid_search",
            vector_db_service=vector_db_service,
            llm_service=llm_service,
            config={
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight
            }
        )

        self.vector_strategy = vector_strategy or VectorSearchStrategy(
            vector_db_service, llm_service
        )
        self.keyword_strategy = keyword_strategy or KeywordSearchStrategy(
            vector_db_service, llm_service
        )
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    async def search(
        self,
        query: str,
        query_embedding: List[float],
        workspace_id: str,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """执行混合搜索"""
        if not self.validate_query(query):
            logger.warning(f"Invalid query for hybrid search: {query}")
            return []

        try:
            vector_results = await self.vector_strategy.search(
                query=query,
                query_embedding=query_embedding,
                workspace_id=workspace_id,
                top_k=top_k * 2,
                **kwargs
            )

            keyword_results = await self.keyword_strategy.search(
                query=query,
                query_embedding=query_embedding,
                workspace_id=workspace_id,
                top_k=top_k * 2,
                **kwargs
            )

            merged_results = self._merge_results(vector_results, keyword_results, top_k)
            logger.info(f"Hybrid search returned {len(merged_results)} results")
            return merged_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            return []

    def _merge_results(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """合并向量和关键词搜索结果"""
        result_dict: Dict[str, RetrievalResult] = {}

        for result in vector_results:
            result.confidence = result.similarity * self.vector_weight
            result_dict[result.id] = result

        for result in keyword_results:
            result.confidence = result.similarity * self.keyword_weight

            if result.id in result_dict:
                existing = result_dict[result.id]
                combined_score = (
                    existing.similarity * self.vector_weight +
                    result.similarity * self.keyword_weight
                )
                existing.confidence = combined_score
                existing.metadata["source"] = "hybrid"
                if "keyword_count" in result.metadata:
                    existing.metadata["keyword_count"] = result.metadata["keyword_count"]
            else:
                result.metadata["source"] = "keyword"
                result_dict[result.id] = result

        merged = list(result_dict.values())
        merged.sort(key=lambda x: x.confidence, reverse=True)
        return merged[:top_k]

    @property
    def supported_markers(self) -> Set[str]:
        """支持的关键词标记"""
        return self.vector_strategy.supported_markers | self.keyword_strategy.supported_markers


class StrategyRegistry:
    """检索策略注册表

    管理所有可用的检索策略，支持动态注册和查找。
    """

    _instance = None
    _strategies: Dict[str, type] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str):
        """装饰器：注册检索策略"""
        def decorator(strategy_class):
            cls._strategies[name] = strategy_class
            return strategy_class
        return decorator

    @classmethod
    def get_strategy_class(cls, name: str) -> Optional[type]:
        """获取策略类"""
        return cls._strategies.get(name)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """列出所有已注册的策略"""
        return list(cls._strategies.keys())

    @classmethod
    def create_strategy(
        cls,
        name: str,
        vector_db_service: IVectorDBService,
        llm_service: LLMService,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[RetrievalStrategy]:
        """创建策略实例"""
        strategy_class = cls.get_strategy_class(name)
        if strategy_class is None:
            logger.error(f"Unknown strategy: {name}")
            return None

        try:
            return strategy_class(
                vector_db_service=vector_db_service,
                llm_service=llm_service,
                **(config or {})
            )
        except Exception as e:
            logger.error(f"Failed to create strategy {name}: {str(e)}")
            return None


StrategyRegistry.register("vector_search")(VectorSearchStrategy)
StrategyRegistry.register("keyword_search")(KeywordSearchStrategy)
StrategyRegistry.register("hybrid_search")(HybridSearchStrategy)
