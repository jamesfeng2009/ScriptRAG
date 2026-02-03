"""Service Layer - LLM adapters, database services, parsing services

Retrieval Enhancement Modules:
- query_rewriter: Query intent disambiguation and complex query decomposition
- query_expansion: Query expansion using LLM
- hybrid_search: Hybrid retrieval with vector and BM25
- cross_encoder_reranker: Cross-Encoder and MMR reranking
- adaptive_threshold: Adaptive threshold and cliff edge cutoff
- enhanced_parent_retriever: Small-to-Big retrieval with context merging
- graprag_engine: GraphRAG multi-hop retrieval engine
- retrieval_enhancement: Unified enhancement pipeline
"""

from .llm.service import LLMService

from .query_rewriter import (
    QueryRewriter,
    QueryContext,
    RewriteResult,
    QueryType
)

from .query_expansion import QueryExpansion, QueryOptimizer

from .hybrid_search import (
    HybridSearchService,
    HybridSearchConfig,
    RRFEngine,
    BM25KeywordSearch,
    FusionResult
)

from .cross_encoder_reranker import (
    CrossEncoderReranker,
    MMMReranker,
    RerankingPipeline,
    RerankConfig,
    RerankResult
)

from .retrieval.adaptive_threshold import (
    AdaptiveThresholdStrategy,
    AdaptiveThresholdConfig,
    ThresholdAnalysisResult,
    CliffEdgeCutoff
)

from .enhanced_parent_retriever import (
    EnhancedParentDocumentRetriever,
    SmallToBigRetrievalPipeline,
    MergedContext,
    MergeConfig
)

from .graprag_engine import (
    GraphRAGEngine,
    DocumentDependencyGraph,
    GraphTraversalEngine,
    EntityExtractor,
    GraphNode,
    GraphEdge,
    NodeType,
    RelationType
)

from .retrieval_enhancement import (
    RetrievalEnhancementPipeline,
    RetrievalPipelineBuilder,
    EnhancementConfig,
    EnhancementResult
)

from .retrieval_service import RetrievalService

from .retrieval.config import RetrievalConfig
from .retrieval.strategies import RetrievalResult, RetrievalStrategy
from .retrieval.mergers import ResultMerger
from .retrieval.strategies import IVectorDBService

__all__ = [
    # LLM
    "LLMService",
    
    # Query Processing
    "QueryRewriter",
    "QueryContext",
    "RewriteResult",
    "QueryType",
    "QueryExpansion",
    "QueryOptimizer",
    
    # Hybrid Search
    "HybridSearchService",
    "HybridSearchConfig",
    "RRFEngine",
    "BM25KeywordSearch",
    "FusionResult",
    
    # Reranking
    "CrossEncoderReranker",
    "MMMReranker",
    "RerankingPipeline",
    "RerankConfig",
    "RerankResult",
    
    # Adaptive Threshold
    "AdaptiveThresholdStrategy",
    "AdaptiveThresholdConfig",
    "ThresholdAnalysisResult",
    "CliffEdgeCutoff",
    
    # Parent Retrieval
    "EnhancedParentDocumentRetriever",
    "SmallToBigRetrievalPipeline",
    "MergedContext",
    "MergeConfig",
    
    # GraphRAG
    "GraphRAGEngine",
    "DocumentDependencyGraph",
    "GraphTraversalEngine",
    "EntityExtractor",
    "GraphNode",
    "GraphEdge",
    "NodeType",
    "RelationType",
    
    # Enhancement Pipeline
    "RetrievalEnhancementPipeline",
    "RetrievalPipelineBuilder",
    "EnhancementConfig",
    "EnhancementResult",
    
    # Main Service
    "RetrievalService",
    
    # Common Types
    "RetrievalResult",
    "RetrievalStrategy",
    "ResultMerger",
    "RetrievalConfig",
    "IVectorDBService",
]
