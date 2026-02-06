"""
服务层 - LLM 适配器、数据库服务、解析服务

组织结构：
├── core/               # 核心服务（LLM、检索、RAG、会话）
├── retrieval/          # 检索相关（策略、合并、阈值）
├── knowledge/          # 知识处理（分块、索引、知识图谱）
├── quality/            # 质量保障（幻觉检测、质量评估）
├── cache/              # 缓存服务
├── monitoring/         # 监控服务
├── database/           # 数据库服务
├── llm/               # LLM 服务
├── rag/               # RAG 服务
├── parser/             # 解析服务
├── optimization/       # 优化服务
└── interfaces.py       # 服务接口定义
"""

from .llm.service import LLMService

# ============== Service Interfaces (服务接口) ==============
from .interfaces import (
    IDocument,
    IQueryResult,
    IRetrievalStrategy,
    IServiceStatus,
    IRetrievalService,
    ILLMService,
    IRAGService,
    ICacheService,
    IStorageService,
    IMonitoringService,
)

# ============== Service Errors (服务错误) ==============
from .errors import (
    ServiceError,
    ServiceErrorContext,
    ErrorSeverity,
    ErrorCategory,
    RetrievalServiceError,
    LLMServiceError,
    RAGServiceError,
    CacheServiceError,
    SessionServiceError,
    StorageServiceError,
    ValidationError,
    create_error_context,
)

# ============== Mock Services (Mock 服务 - 用于测试) ==============
from .mocks import (
    MockRetrievalService,
    MockLLMService,
    MockRAGService,
    MockCacheService,
    MockStorageService,
    MockMonitoringService,
    MockDocument,
    MockQueryResult,
    create_mock_services,
)

# ============== Core Services (核心服务) ==============
from .retrieval_service import RetrievalService
from .rag.rag_service import RAGService
from .core.summarization_service import SummarizationService

# ============== Retrieval Enhancement (检索增强) ==============
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

from .retrieval.strategies import RetrievalResult, RetrievalStrategy
from .retrieval.mergers import ResultMerger
from .retrieval.config import RetrievalConfig
from .retrieval.strategies import IVectorDBService

from .reranker import MultiFactorReranker

from .documents.enhanced_parent_retriever import (
    EnhancedParentDocumentRetriever,
    SmallToBigRetrievalPipeline,
    MergedContext,
    MergeConfig
)

from .documents.parent_document_retriever import ParentDocumentRetriever

from .knowledge.graprag_engine import (
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

# ============== Knowledge Processing (知识处理) ==============
from .documents.document_chunker import SmartChunker, Chunk
from .documents.document_chunks_service import DocumentChunksService
from .documents.document_persistence_service import DocumentService as DocumentPersistenceService
from .knowledge.knowledge_graph_service import KnowledgeGraphService, KnowledgeNodeModel as Entity, KnowledgeRelationModel as Relation

# ============== Quality Assurance (质量保障) ==============
from .quality.retrieval_quality_assessor import (
    RetrievalQualityAssessor,
    NegativeSampleFilter,
    QualityAssessment,
    CoverageAnalyzer,
    ConsistencyAnalyzer,
    FreshnessAnalyzer,
    CompletenessAnalyzer
)

from .quality.hallucination_detection import (
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

# ============== Cache & Storage (缓存与存储) ==============
from .cache.enhanced_cache import (
    EnhancedRetrievalCache,
    EnhancedCacheConfig,
    CacheStats,
    RewriteResultCacheEntry
)

from .documents.incremental_storage import (
    IncrementalStorageOptimizer,
    DeltaState,
    StateDiffCalculator
)

# ============== Session & Persistence (会话与持久化) ==============
from .persistence.chat_session_persistence_service import (
    ChatSessionPersistenceService,
    ChatSessionRecord
)

from .persistence.task_persistence_service import TaskService as TaskPersistenceService
from .persistence.skill_persistence_service import SkillService as SkillPersistenceService
from .retrieval_logs_service import RetrievalLogsService
from .llm_call_logger import LLMCallLogger

# ============== Skill & Routing (技能与路由) ==============
from .persistence.skill_routing_service import SkillRoutingService
from .rag.skill_aware_rag import SkillAwareRAGService as SkillAwareRAG

# ============== API & Monitoring (API与监控) ==============
from .api_usage_stats_service import APIUsageStatsService

from .monitoring.retrieval_monitor import RetrievalMonitor
from .monitoring.skill_monitoring import SkillMonitor as SkillMonitoringService

# ============== Utility Services (工具服务) ==============
from .retrieval_isolation import RetrievalIsolation

__all__ = [
    # ============== Interfaces (接口) ==============
    "IDocument",
    "IQueryResult",
    "IRetrievalStrategy",
    "IServiceStatus",
    "IRetrievalService",
    "ILLMService",
    "IRAGService",
    "ICacheService",
    "IStorageService",
    "IMonitoringService",

    # ============== Errors (错误) ==============
    "ServiceError",
    "ServiceErrorContext",
    "ErrorSeverity",
    "ErrorCategory",
    "RetrievalServiceError",
    "LLMServiceError",
    "RAGServiceError",
    "CacheServiceError",
    "SessionServiceError",
    "StorageServiceError",
    "ValidationError",
    "create_error_context",

    # ============== Mock Services (Mock 服务) ==============
    "MockRetrievalService",
    "MockLLMService",
    "MockRAGService",
    "MockCacheService",
    "MockStorageService",
    "MockMonitoringService",
    "MockDocument",
    "MockQueryResult",
    "create_mock_services",

    # ============== Core (核心) ==============
    "LLMService",
    "RetrievalService",
    "RAGService",
    "SummarizationService",

    # ============== Retrieval Enhancement ==============
    "QueryRewriter",
    "QueryContext",
    "RewriteResult",
    "QueryType",
    "IntentClassifier",
    "RewriteCacheEntry",
    "QueryExpansion",
    "QueryOptimizer",
    "HybridSearchService",
    "HybridSearchConfig",
    "RRFEngine",
    "BM25KeywordSearch",
    "FusionResult",
    "CrossEncoderReranker",
    "MMMReranker",
    "RerankingPipeline",
    "RerankConfig",
    "RerankResult",
    "AdaptiveThresholdStrategy",
    "AdaptiveThresholdConfig",
    "ThresholdAnalysisResult",
    "CliffEdgeCutoff",
    "RetrievalResult",
    "RetrievalStrategy",
    "ResultMerger",
    "RetrievalConfig",
    "IVectorDBService",
    "MultiFactorReranker",
    "EnhancedParentDocumentRetriever",
    "SmallToBigRetrievalPipeline",
    "MergedContext",
    "MergeConfig",
    "GraphRAGEngine",
    "DocumentDependencyGraph",
    "GraphTraversalEngine",
    "EntityExtractor",
    "GraphNode",
    "GraphEdge",
    "NodeType",
    "RelationType",
    "AdvancedRetrievalPipeline",
    "AdvancedPipelineBuilder",
    "AdvancedRetrievalConfig",
    "AdvancedRetrievalResult",
    "RetrievalEnhancementPipeline",
    "RetrievalPipelineBuilder",
    "EnhancementConfig",
    "EnhancementResult",

    # ============== Knowledge ==============
    "SmartChunker",
    "Chunk",
    "DocumentChunksService",
    "DocumentPersistenceService",
    "KnowledgeGraphService",
    "Entity",
    "Relation",

    # ============== Quality ==============
    "RetrievalQualityAssessor",
    "NegativeSampleFilter",
    "QualityAssessment",
    "CoverageAnalyzer",
    "ConsistencyAnalyzer",
    "FreshnessAnalyzer",
    "CompletenessAnalyzer",
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

    # ============== Cache & Storage ==============
    "EnhancedRetrievalCache",
    "EnhancedCacheConfig",
    "CacheStats",
    "RewriteResultCacheEntry",
    "IncrementalStorageOptimizer",
    "DeltaState",
    "StateDiffCalculator",

    # ============== Session & Persistence ==============
    "ChatSessionPersistenceService",
    "ChatSessionRecord",
    "TaskPersistenceService",
    "SkillPersistenceService",
    "RetrievalLogsService",
    "LLMCallLogger",

    # ============== Skill & Routing ==============
    "SkillRoutingService",
    "SkillAwareRAG",

    # ============== API & Monitoring ==============
    "APIUsageStatsService",
    "RetrievalMonitor",
    "SkillMonitoringService",

    # ============== Utility ==============
    "RetrievalIsolation",
    "ParentDocumentRetriever",
]
