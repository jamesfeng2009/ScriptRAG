"""Retrieval Module - Pluggable retrieval architecture

This module provides an extensible retrieval system with:
- Multiple retrieval strategies (vector, keyword, hybrid)
- Configurable result merging algorithms
- Support for custom strategies and mergers
"""

from .strategies import (
    RetrievalResult,
    RetrievalStrategy,
    VectorSearchStrategy,
    KeywordSearchStrategy,
    HybridSearchStrategy,
    StrategyRegistry
)

from .mergers import (
    ResultMerger,
    WeightedMerger,
    ReciprocalRankMerger,
    RoundRobinMerger,
    FusionMerger,
    MergerRegistry
)

from .config import RetrievalConfig, RetrievalStrategyConfig, MergerConfig

from .llm_query_strategy import (
    LLMGeneratedQueryStrategy,
    LLMQueryGenerationConfig,
    QueryVariant,
    MultiQuerySearchStrategy,
    MultiQueryConfig
)

__all__ = [
    "RetrievalResult",
    "RetrievalStrategy",
    "VectorSearchStrategy",
    "KeywordSearchStrategy",
    "HybridSearchStrategy",
    "StrategyRegistry",
    "ResultMerger",
    "WeightedMerger",
    "ReciprocalRankMerger",
    "RoundRobinMerger",
    "FusionMerger",
    "MergerRegistry",
    "RetrievalConfig",
    "RetrievalStrategyConfig",
    "MergerConfig",
    "LLMGeneratedQueryStrategy",
    "LLMQueryGenerationConfig",
    "QueryVariant",
    "MultiQuerySearchStrategy",
    "MultiQueryConfig"
]
