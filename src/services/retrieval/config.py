"""Retrieval Configuration

This module provides configuration models for the retrieval system,
supporting flexible configuration of strategies, mergers, and markers.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class StrategyType(str, Enum):
    """支持的检索策略类型"""
    VECTOR_SEARCH = "vector_search"
    KEYWORD_SEARCH = "keyword_search"
    HYBRID_SEARCH = "hybrid_search"


class MergerType(str, Enum):
    """支持的结果合并策略类型"""
    WEIGHTED_MERGE = "weighted_merge"
    RRF_MERGE = "rrf_merge"
    ROUND_ROBIN_MERGE = "round_robin_merge"
    FUSION_MERGE = "fusion_merge"


class VectorSearchConfig(BaseModel):
    """向量搜索配置"""
    enabled: bool = True
    top_k: int = Field(default=5, description="返回结果数量")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class KeywordSearchConfig(BaseModel):
    """关键词搜索配置"""
    enabled: bool = True
    top_k: int = Field(default=5, description="返回结果数量")
    markers: List[str] = Field(
        default_factory=lambda: ["@deprecated", "FIXME", "TODO", "Security"],
        description="敏感标记列表"
    )
    custom_markers: List[str] = Field(
        default_factory=list,
        description="自定义关键词标记（可动态添加）"
    )
    boost_factors: Dict[str, float] = Field(
        default_factory=lambda: {
            "@deprecated": 1.5,
            "Security": 1.5,
            "FIXME": 1.3,
            "TODO": 1.2
        },
        description="标记加权因子"
    )


class HybridMergeConfig(BaseModel):
    """混合搜索合并配置"""
    enabled: bool = True
    vector_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    dedup_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


class RetrievalStrategyConfig(BaseModel):
    """检索策略配置"""
    name: StrategyType = StrategyType.HYBRID_SEARCH
    vector: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    keyword: KeywordSearchConfig = Field(default_factory=KeywordSearchConfig)
    hybrid_merge: HybridMergeConfig = Field(default_factory=HybridMergeConfig)
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "vector_search": 0.6,
            "keyword_search": 0.4
        },
        description="各策略的权重（用于合并）"
    )


class WeightedMergerConfig(BaseModel):
    """加权合并配置"""
    dedup_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


class RRFMergeConfig(BaseModel):
    """RRF 合并配置"""
    k: int = Field(default=60, description="RRF 常数")


class FusionMergerConfig(BaseModel):
    """融合合并配置"""
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    dedup_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


class MergerConfig(BaseModel):
    """合并策略配置"""
    type: MergerType = MergerType.WEIGHTED_MERGE
    weighted: Optional[WeightedMergerConfig] = None
    rrf: Optional[RRFMergeConfig] = None
    fusion: Optional[FusionMergerConfig] = None
    max_per_strategy: int = Field(default=3, description="轮询合并时每策略最大数量")


class QueryExpansionConfig(BaseModel):
    """查询扩展配置"""
    enabled: bool = True
    max_queries: int = Field(default=3, description="最大扩展查询数")


class RerankingConfig(BaseModel):
    """重排序配置"""
    enabled: bool = True
    top_k: int = Field(default=5, description="重排序后返回数量")


class DiversityConfig(BaseModel):
    """多样性配置"""
    enabled: bool = True
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_per_file: int = Field(default=2, description="每个文件的最大结果数")


class RetrievalConfig(BaseModel):
    """主检索配置

    包含所有检索相关的配置项。
    支持通过配置文件或代码进行配置。
    """

    version: str = "1.0"

    strategy: RetrievalStrategyConfig = Field(
        default_factory=RetrievalStrategyConfig,
        description="检索策略配置"
    )

    merger: MergerConfig = Field(
        default_factory=MergerConfig,
        description="结果合并策略配置"
    )

    query_expansion: QueryExpansionConfig = Field(
        default_factory=QueryExpansionConfig,
        description="查询扩展配置"
    )

    reranking: RerankingConfig = Field(
        default_factory=RerankingConfig,
        description="重排序配置"
    )

    diversity: DiversityConfig = Field(
        default_factory=DiversityConfig,
        description="多样性配置"
    )

    custom_strategies: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="自定义检索策略配置"
    )

    custom_markers: List[str] = Field(
        default_factory=list,
        description="全局自定义关键词标记"
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalConfig":
        """从字典创建配置"""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()

    def add_custom_marker(self, marker: str) -> bool:
        """添加自定义关键词标记

        Args:
            marker: 要添加的标记（如 "@important", "@review"）

        Returns:
            是否添加成功
        """
        if not marker or marker in self.custom_markers:
            return False

        self.custom_markers.append(marker)
        return True

    def remove_custom_marker(self, marker: str) -> bool:
        """移除自定义关键词标记

        Args:
            marker: 要移除的标记

        Returns:
            是否移除成功
        """
        if marker in self.custom_markers:
            self.custom_markers.remove(marker)
            return True
        return False

    def get_all_markers(self) -> List[str]:
        """获取所有关键词标记（默认 + 自定义）"""
        default_markers = self.strategy.keyword.markers
        return default_markers + [
            m for m in self.custom_markers if m not in default_markers
        ]

    def register_custom_strategy(
        self,
        name: str,
        strategy_config: Dict[str, Any]
    ) -> bool:
        """注册自定义检索策略

        Args:
            name: 策略名称
            strategy_config: 策略配置

        Returns:
            是否注册成功
        """
        if not name or not strategy_config:
            return False

        self.custom_strategies[name] = strategy_config
        return True

    def unregister_custom_strategy(self, name: str) -> bool:
        """注销自定义检索策略

        Args:
            name: 策略名称

        Returns:
            是否注销成功
        """
        if name in self.custom_strategies:
            del self.custom_strategies[name]
            return True
        return False
