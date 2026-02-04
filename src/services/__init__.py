"""服务层 - LLM 适配器、数据库服务、解析服务

检索增强模块：
- query_rewriter: 查询意图消歧和复杂查询分解（增强版：缓存+意图分类）
- query_expansion: 使用 LLM 进行查询扩展
- hybrid_search: 混合检索（向量 + BM25）
- cross_encoder_reranker: Cross-Encoder 和 MMR 重排序
- adaptive_threshold: 自适应阈值和悬崖边缘截止
- enhanced_parent_retriever: 小到大的检索与上下文合并
- graprag_engine: GraphRAG 多跳检索引擎
- advanced_retrieval: 高级检索流水线（查询改写、混合检索、CrossEncoder、GraphRAG）
- retrieval_quality_assessor: 检索质量评估（覆盖度、一致性、新鲜度、完整性）
- hallucination_detection: 幻觉检测完善（细粒度检测+预防+修复）
- session_manager: 会话状态持久化（断点续传）
- enhanced_cache: 增强检索缓存（多级缓存+预热）
"""

from .llm.service import LLMService

from .query_rewriter import (
    QueryRewriter,
    QueryContext,
    RewriteResult,
    QueryType,
    IntentClassifier,
    RewriteCacheEntry
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

from .advanced_retrieval import (
    AdvancedRetrievalPipeline,
    AdvancedPipelineBuilder,
    AdvancedRetrievalConfig,
    AdvancedRetrievalResult,
    # 向后兼容别名
    RetrievalEnhancementPipeline,
    RetrievalPipelineBuilder,
    EnhancementConfig,
    EnhancementResult
)

from .retrieval_service import RetrievalService

from .retrieval_quality_assessor import (
    RetrievalQualityAssessor,
    NegativeSampleFilter,
    QualityAssessment,
    CoverageAnalyzer,
    ConsistencyAnalyzer,
    FreshnessAnalyzer,
    CompletenessAnalyzer
)

from .hallucination_detection import (
    GranularHallucinationDetector,
    HallucinationClassifier,
    HallucinationPrevention,
    HallucinationRepairer,
    UnifiedHallucinationService,
    HallucinationType,
    HallucinationSeverity,
    SentenceHallucinationResult,
    FragmentHallucinationResult,
    CodeEntity
)

from .session_manager import (
    SessionManager,
    SessionState,
    SessionConfig,
    InMemorySessionStore
)

from .retrieval.config import RetrievalConfig
from .retrieval.strategies import RetrievalResult, RetrievalStrategy
from .retrieval.mergers import ResultMerger
from .retrieval.strategies import IVectorDBService

__all__ = [
    # LLM
    "LLMService",
    
    # 查询处理
    "QueryRewriter",
    "QueryContext",
    "RewriteResult",
    "QueryType",
    "IntentClassifier",
    "RewriteCacheEntry",
    "QueryExpansion",
    "QueryOptimizer",
    
    # 混合搜索
    "HybridSearchService",
    "HybridSearchConfig",
    "RRFEngine",
    "BM25KeywordSearch",
    "FusionResult",
    
    # 重排序
    "CrossEncoderReranker",
    "MMMReranker",
    "RerankingPipeline",
    "RerankConfig",
    "RerankResult",
    
    # 自适应阈值
    "AdaptiveThresholdStrategy",
    "AdaptiveThresholdConfig",
    "ThresholdAnalysisResult",
    "CliffEdgeCutoff",
    
    # 父文档检索
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
    
    # 高级检索流水线
    "AdvancedRetrievalPipeline",
    "AdvancedPipelineBuilder",
    "AdvancedRetrievalConfig",
    "AdvancedRetrievalResult",
    # 向后兼容别名
    "RetrievalEnhancementPipeline",
    "RetrievalPipelineBuilder",
    "EnhancementConfig",
    "EnhancementResult",
    
    # 质量评估
    "RetrievalQualityAssessor",
    "NegativeSampleFilter",
    "QualityAssessment",
    "CoverageAnalyzer",
    "ConsistencyAnalyzer",
    "FreshnessAnalyzer",
    "CompletenessAnalyzer",
    
    # 幻觉检测
    "GranularHallucinationDetector",
    "HallucinationClassifier",
    "HallucinationPrevention",
    "HallucinationRepairer",
    "UnifiedHallucinationService",
    "HallucinationType",
    "HallucinationSeverity",
    "SentenceHallucinationResult",
    "FragmentHallucinationResult",
    "CodeEntity",

    # 会话管理
    "SessionManager",
    "SessionState",
    "SessionConfig",
    "InMemorySessionStore",

    # 增强缓存
    "EnhancedRetrievalCache",
    "EnhancedCacheConfig",
    "CacheStats",
    "RewriteResultCacheEntry",

    # 主服务
    "RetrievalService",
    
    # 通用类型
    "RetrievalResult",
    "RetrievalStrategy",
    "ResultMerger",
    "RetrievalConfig",
    "IVectorDBService",
]
