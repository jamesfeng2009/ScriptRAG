"""Advanced Retrieval Pipeline - 高级检索流水线

高级检索功能模块，提供端到端的检索增强能力。

功能模块：
1. 查询改写与扩展 (QueryRewriter)
2. 混合检索 (HybridSearchService)
3. Cross-Encoder 精排 (CrossEncoderReranker)
4. 悬崖截断 (AdaptiveThresholdStrategy)
5. Small-to-Big 上下文组装 (SmallToBigRetrievalPipeline)
6. GraphRAG 多跳检索 (GraphRAGEngine)
7. MMR 多样性排序 (MMMReranker)

与 retrieval_service.py 的区别：
- retrieval_service.py: 主入口服务，负责整体编排
- advanced_retrieval.py: 高级检索功能实现，提供细粒度控制

使用方式：
```python
from src.services.advanced_retrieval import AdvancedRetrievalPipeline

pipeline = AdvancedRetrievalPipeline(
    llm_service=llm_service,
    vector_db=vector_db,
    enable_cross_encoder=True,
    enable_graprag=True
)

# 执行完整检索流程
results = await pipeline.enhanced_retrieve(
    workspace_id="my_project",
    query="如何在 Python 中处理异步请求",
    top_k=10
)
```
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .llm.service import LLMService
from .database.vector_db import IVectorDBService

from .query_rewriter import QueryRewriter, QueryContext
from .query_expansion import QueryExpansion

from .hybrid_search import (
    HybridSearchService,
    HybridSearchConfig,
    FusionResult,
    BM25KeywordSearch
)

from .cross_encoder_reranker import (
    CrossEncoderReranker,
    MMMReranker,
    RerankingPipeline,
    RerankConfig
)

from .retrieval.adaptive_threshold import AdaptiveThresholdStrategy, AdaptiveThresholdConfig

from .enhanced_parent_retriever import EnhancedParentDocumentRetriever, SmallToBigRetrievalPipeline

from .graprag_engine import GraphRAGEngine

from .retrieval.strategies import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class AdvancedRetrievalConfig:
    """高级检索配置"""

    # 查询改写
    enable_query_rewrite: bool = True
    enable_query_expansion: bool = True
    
    # 混合检索
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    
    # 精排
    enable_cross_encoder: bool = False
    cross_encoder_model: str = "BAAI/bge-reranker-base"
    enable_mmr: bool = True
    mmr_lambda: float = 0.5
    
    # 阈值控制
    enable_adaptive_threshold: bool = True
    base_min_score: float = 0.65
    cliff_drop_threshold: float = 0.20
    
    # Small-to-Big 检索
    enable_small_to_big: bool = True
    max_context_chars: int = 4000
    
    # GraphRAG 多跳检索
    enable_graprag: bool = False
    max_hops: int = 2
    graprag_workspace_id: str = "default"


@dataclass
class AdvancedRetrievalResult:
    """高级检索结果"""
    query: str
    rewritten_query: Optional[str] = None
    expanded_queries: List[str] = field(default_factory=list)
    raw_results: List[RetrievalResult] = field(default_factory=list)
    reranked_results: List[RetrievalResult] = field(default_factory=list)
    final_results: List[RetrievalResult] = field(default_factory=list)
    graprag_results: List[RetrievalResult] = field(default_factory=list)
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedRetrievalPipeline:
    """
    高级检索流水线
    
    整合所有检索增强技术，提供端到端的检索增强能力。
    与 RetrievalService 的区别：
    - AdvancedRetrievalPipeline: 专注高级检索功能，可独立使用
    - RetrievalService: 主服务，封装增强流水线并提供统一 API
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        vector_db_service: IVectorDBService,
        config: Optional[AdvancedRetrievalConfig] = None,
        graprag_engine: Optional[GraphRAGEngine] = None
    ):
        self.config = config or AdvancedRetrievalConfig()
        self.llm_service = llm_service
        self.vector_db = vector_db_service
        
        self._init_components(graprag_engine)
    
    def _init_components(self, graprag_engine: Optional[GraphRAGEngine] = None):
        """初始化所有组件"""
        # 查询改写
        self.query_rewriter = QueryRewriter(self.llm_service)
        self.query_expansion = QueryExpansion(self.llm_service)
        
        # BM25 搜索器
        self.bm25_searcher = BM25KeywordSearch()
        
        # 混合检索 - 使用统一的 HybridSearchService
        self.hybrid_service = HybridSearchService(
            config=HybridSearchConfig(
                vector_weight=self.config.vector_weight,
                keyword_weight=self.config.keyword_weight
            ),
            bm25_searcher=self.bm25_searcher
        )
        
        # 精排
        self.rerank_config = RerankConfig(
            model_name=self.config.cross_encoder_model,
            fusion_weight_vector=0.6,
            fusion_weight_cross=0.4
        )
        self.cross_encoder_reranker = CrossEncoderReranker(self.rerank_config)
        self.mmr_reranker = MMMReranker(
            lambda_param=self.config.mmr_lambda,
            similarity_threshold=0.85
        )
        self.reranking_pipeline = RerankingPipeline(
            primary_reranker=self.cross_encoder_reranker if self.config.enable_cross_encoder else None,
            fallback_reranker=self.cross_encoder_reranker
        )
        
        # 阈值控制
        self.threshold_strategy = AdaptiveThresholdStrategy(
            AdaptiveThresholdConfig(
                base_min_score=self.config.base_min_score,
                cliff_drop_threshold=self.config.cliff_drop_threshold
            )
        )
        
        # Small-to-Big 检索
        self.small_to_big_pipeline = SmallToBigRetrievalPipeline(
            rewriter=self.query_rewriter if self.config.enable_query_rewrite else None
        )
        
        # GraphRAG
        if self.config.enable_graprag:
            self.graprag_engine = graprag_engine or GraphRAGEngine(
                workspace_id=self.config.graprag_workspace_id
            )
        else:
            self.graprag_engine = None
        
        logger.info("Retrieval enhancement pipeline initialized")
    
    async def enhanced_retrieve(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 10,
        embedding: Optional[List[float]] = None,
        filters: Optional[Dict] = None,
        original_results: Optional[List[RetrievalResult]] = None
    ) -> "AdvancedRetrievalResult":
        """
        执行增强检索
        
        Args:
            workspace_id: 工作空间 ID
            query: 查询文本
            top_k: 返回结果数量
            embedding: 预计算的查询嵌入（可选）
            filters: 过滤条件
            original_results: 已有检索结果（用于增强）
            
        Returns:
            增强检索结果
        """
        start_time = datetime.now()
        
        result = AdvancedRetrievalResult(query=query)
        
        try:
            # Step 1: 查询改写
            if self.config.enable_query_rewrite:
                rewrite_result = await self.query_rewriter.rewrite(query)
                result.rewritten_query = rewrite_result.rewritten_query
                result.expanded_queries = [rewrite_result.rewritten_query] + rewrite_result.sub_queries
                logger.info(f"Query rewritten: {query[:50]}... → {result.rewritten_query[:50]}...")
            else:
                result.rewritten_query = query
                result.expanded_queries = [query]
            
            # Step 2: 查询扩展
            if self.config.enable_query_expansion:
                expanded = await self.query_expansion.expand_query(result.rewritten_query)
                result.expanded_queries.extend(expanded[1:])  # 排除已包含的原始查询
                result.expanded_queries = list(set(result.expanded_queries))[:5]  # 去重，限制数量
            
            # Step 3: 检索（或增强现有结果）
            if original_results:
                result.raw_results = original_results
            else:
                # 执行混合检索
                if embedding is None:
                    embeddings = await self.llm_service.embedding([result.rewritten_query])
                    embedding = embeddings[0] if embeddings else None
                
                if embedding:
                    result.raw_results = await self._hybrid_search(
                        workspace_id=workspace_id,
                        query=result.rewritten_query,
                        embedding=embedding,
                        top_k=top_k * 2,
                        filters=filters
                    )
            
            # Step 4: 重排序
            reranked = await self.reranking_pipeline.rerank(
                query=result.rewritten_query,
                results=result.raw_results,
                top_k=top_k * 2,
                use_primary=self.config.enable_cross_encoder
            )
            
            # 转换为 RetrievalResult 列表
            result.reranked_results = [
                RetrievalResult(
                    id=r.id,
                    file_path=r.file_path,
                    content=r.content,
                    similarity=r.fused_score,
                    confidence=r.fused_score,
                    strategy_name="reranked",
                    metadata={**r.metadata, "cross_score": r.cross_score}
                )
                for r in reranked
            ]
            
            # Step 5: MMR 多样性排序
            if self.config.enable_mmr and len(result.reranked_results) > top_k:
                mmr_results = self.mmr_reranker.rerank(
                    query=result.rewritten_query,
                    results=result.reranked_results,
                    top_k=top_k
                )
                result.reranked_results = [
                    RetrievalResult(
                        id=r.id,
                        file_path=r.file_path,
                        content=r.content,
                        similarity=r.fused_score,
                        confidence=r.fused_score,
                        strategy_name="mmr"
                    )
                    for r in mmr_results
                ]
            
            # Step 6: 悬崖截断
            if self.config.enable_adaptive_threshold:
                scores = [r.similarity for r in result.reranked_results]
                complexity = await self.threshold_strategy.estimate_query_complexity(result.rewritten_query)
                indices, analysis = self.threshold_strategy.analyze_and_filter(scores, complexity)
                
                result.final_results = [result.reranked_results[i] for i in indices]
                result.metadata["threshold_analysis"] = {
                    "applied_threshold": analysis.applied_threshold,
                    "retained_count": analysis.retained_count,
                    "cliff_detected": analysis.cliff_detected
                }
            else:
                result.final_results = result.reranked_results[:top_k]
            
            # Step 7: GraphRAG 多跳检索
            if self.config.enable_graprag and self.graprag_engine:
                graprag_results = self.graprag_engine.retrieve_with_hops(
                    query=result.rewritten_query,
                    initial_results=result.final_results[:3],
                    max_hops=self.config.max_hops,
                    max_results_per_hop=top_k // 2
                )
                result.graprag_results = graprag_results
                
                # 合并 GraphRAG 结果
                combined_ids = {r.id for r in result.final_results}
                for graprag_r in graprag_results:
                    if graprag_r.id not in combined_ids:
                        result.final_results.append(RetrievalResult(
                            id=graprag_r.id,
                            file_path=graprag_r.file_path,
                            content=graprag_r.content,
                            similarity=graprag_r.similarity * 0.8,  # 略降低权重
                            confidence=graprag_r.similarity * 0.8,
                            strategy_name="graprag",
                            metadata={"hop_depth": graprag_r.hop_depth}
                        ))
                
                # 重新排序
                result.final_results.sort(key=lambda x: x.similarity, reverse=True)
            
            # 限制最终结果数量
            result.final_results = result.final_results[:top_k]
            
            # 计算处理时间
            result.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(
                f"Enhanced retrieve: query={query[:50]}..., "
                f"results={len(result.final_results)}, "
                f"time={result.processing_time_ms:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {str(e)}")
            result.metadata["error"] = str(e)
            return result
    
    async def _hybrid_search(
        self,
        workspace_id: str,
        query: str,
        embedding: List[float],
        top_k: int,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        执行混合检索
        
        复用 HybridSearchService，避免重复实现
        """
        try:
            # 使用 HybridSearchService 执行混合检索
            fusion_results = await self.hybrid_service.hybrid_search(
                query=query,
                query_embedding=embedding,
                workspace_id=workspace_id,
                top_k=top_k,
                filters=filters
            )
            
            # 将 FusionResult 转换为 RetrievalResult
            results = []
            for r in fusion_results:
                results.append(RetrievalResult(
                    id=r.id,
                    file_path=r.file_path,
                    content=r.content,
                    similarity=r.fused_score,
                    confidence=r.fused_score,
                    strategy_name="hybrid",
                    metadata={
                        "vector_score": r.vector_score,
                        "keyword_score": r.keyword_score,
                        "rrf_score": r.rrf_score
                    }
                ))
            
            logger.debug(f"Hybrid search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            return []
    
    async def index_for_bm25(self, documents: List[Dict[str, str]]):
        """
        索引文档用于 BM25 检索
        
        通过 HybridSearchService 中的 bm25_searcher 进行索引
        """
        if hasattr(self, 'bm25_searcher') and self.bm25_searcher:
            self.bm25_searcher.index_documents(documents)
            logger.info(f"Indexed {len(documents)} documents for BM25")
        else:
            logger.warning("BM25 searcher not initialized")
    
    def index_for_graprag(self, content: str, file_path: str, doc_type: str = "code"):
        """索引文档用于 GraphRAG"""
        if self.graprag_engine:
            self.graprag_engine.index_document(content, file_path, doc_type)
            logger.info(f"Indexed {file_path} for GraphRAG")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取流水线统计信息"""
        return {
            "config": {
                "enable_query_rewrite": self.config.enable_query_rewrite,
                "enable_query_expansion": self.config.enable_query_expansion,
                "enable_cross_encoder": self.config.enable_cross_encoder,
                "enable_mmr": self.config.enable_mmr,
                "enable_adaptive_threshold": self.config.enable_adaptive_threshold,
                "enable_small_to_big": self.config.enable_small_to_big,
                "enable_graprag": self.config.enable_graprag
            },
            "hybrid_search_config": {
                "vector_weight": self.hybrid_service.config.vector_weight,
                "keyword_weight": self.hybrid_service.config.keyword_weight,
                "rrf_k": self.hybrid_service.config.rrf_k
            },
            "graprag_stats": self.graprag_engine.stats() if self.graprag_engine else None
        }


class AdvancedPipelineBuilder:
    """
    高级检索流水线构建器
    
    提供便捷的流水线配置方式
    """
    
    @staticmethod
    def create_basic_pipeline(
        llm_service: LLMService,
        vector_db_service: IVectorDBService
    ) -> AdvancedRetrievalPipeline:
        """创建基础流水线（仅查询改写 + 阈值控制）"""
        config = AdvancedRetrievalConfig(
            enable_query_rewrite=True,
            enable_cross_encoder=False,
            enable_graprag=False
        )
        return AdvancedRetrievalPipeline(llm_service, vector_db_service, config)
    
    @staticmethod
    def create_standard_pipeline(
        llm_service: LLMService,
        vector_db_service: IVectorDBService
    ) -> AdvancedRetrievalPipeline:
        """创建标准流水线（包含所有核心功能）"""
        config = AdvancedRetrievalConfig(
            enable_query_rewrite=True,
            enable_query_expansion=True,
            enable_cross_encoder=False,  # 需要安装依赖
            enable_mmr=True,
            enable_adaptive_threshold=True,
            enable_small_to_big=True
        )
        return AdvancedRetrievalPipeline(llm_service, vector_db_service, config)
    
    @staticmethod
    def create_full_pipeline(
        llm_service: LLMService,
        vector_db_service: IVectorDBService,
        graprag_engine: Optional[GraphRAGEngine] = None
    ) -> AdvancedRetrievalPipeline:
        """创建完整流水线（包含所有功能）"""
        config = AdvancedRetrievalConfig(
            enable_query_rewrite=True,
            enable_query_expansion=True,
            enable_cross_encoder=True,
            enable_mmr=True,
            enable_adaptive_threshold=True,
            enable_small_to_big=True,
            enable_graprag=True
        )
        return AdvancedRetrievalPipeline(
            llm_service, 
            vector_db_service, 
            config,
            graprag_engine
        )


# 向后兼容：保留旧类名
RetrievalEnhancementPipeline = AdvancedRetrievalPipeline
EnhancementConfig = AdvancedRetrievalConfig
EnhancementResult = AdvancedRetrievalResult
RetrievalPipelineBuilder = AdvancedPipelineBuilder
