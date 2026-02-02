"""Retrieval Service - Hybrid retrieval (vector + keyword search)

This service implements the hybrid retrieval strategy combining:
1. Vector similarity search (semantic)
2. Keyword marker search (@deprecated, FIXME, TODO, Security)
3. Weighted merging algorithm
4. Query expansion for improved recall
5. Multi-factor reranking for improved precision
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from .database.vector_db import IVectorDBService, VectorSearchResult
from .llm.service import LLMService
from .query_expansion import QueryExpansion, QueryOptimizer
from .reranker import MultiFactorReranker, DiversityFilter, RetrievalQualityMonitor
from .cache.retrieval_cache import RetrievalCache, CacheConfig
from .monitoring.retrieval_monitor import RetrievalMonitor, MonitoringConfig


logger = logging.getLogger(__name__)


class RetrievalConfig(BaseModel):
    """检索配置"""
    vector_top_k: int = 5
    vector_similarity_threshold: float = 0.7
    keyword_markers: List[str] = ["@deprecated", "FIXME", "TODO", "Security"]
    keyword_boost_factor: float = 1.5
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    dedup_threshold: float = 0.9
    
    # Query expansion settings
    enable_query_expansion: bool = True
    expansion_top_k: int = 10  # Retrieve more results before reranking
    
    # Reranking settings
    enable_reranking: bool = True
    rerank_top_k: int = 5  # Final number of results after reranking
    
    # Diversity settings
    enable_diversity: bool = True
    diversity_threshold: float = 0.85
    
    # Quality monitoring
    enable_quality_monitoring: bool = True


class RetrievalResult(BaseModel):
    """检索结果"""
    id: str
    file_path: str
    content: str
    similarity: float
    confidence: float
    has_deprecated: bool = False
    has_fixme: bool = False
    has_todo: bool = False
    has_security: bool = False
    metadata: Dict[str, Any] = {}
    source: str = "hybrid"  # "vector", "keyword", or "hybrid"


class RetrievalService:
    """
    检索服务
    
    功能：
    - 混合检索（向量 + 关键词）
    - 向量搜索结果排序
    - 关键词搜索结果排序
    - 加权合并算法
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
            llm_service: LLM 服务（用于生成查询嵌入）
            config: 检索配置
            query_expansion: 查询扩展组件（可选）
            reranker: 重排序组件（可选）
            diversity_filter: 多样性过滤组件（可选）
            quality_monitor: 质量监控组件（可选）
            cache: 缓存组件（可选）
            monitor: 监控组件（可选）
        """
        self.vector_db = vector_db_service
        self.llm_service = llm_service
        self.config = config or RetrievalConfig()
        
        # Initialize optional components
        self.query_expansion = query_expansion or QueryExpansion(llm_service)
        self.query_optimizer = QueryOptimizer()
        self.reranker = reranker or MultiFactorReranker()
        self.diversity_filter = diversity_filter or DiversityFilter()
        self.quality_monitor = quality_monitor or RetrievalQualityMonitor()
        
        # Initialize Phase 3 components
        self.cache = cache or RetrievalCache(CacheConfig())
        self.monitor = monitor or RetrievalMonitor(MonitoringConfig())
    
    async def hybrid_retrieve(
        self,
        workspace_id: str,
        query: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        混合检索（向量 + 关键词搜索 + 查询扩展 + 重排序）
        
        Args:
            workspace_id: 工作空间 ID
            query: 查询文本
            top_k: 返回结果数量（默认使用配置值）
            
        Returns:
            检索结果列表
        """
        final_top_k = top_k or self.config.rerank_top_k
        retrieval_top_k = self.config.expansion_top_k if self.config.enable_query_expansion else final_top_k
        
        try:
            # 1. Query optimization and expansion
            optimized_query = query
            expanded_queries = [query]
            
            if self.config.enable_query_expansion:
                logger.info("Optimizing and expanding query...")
                optimized_query = self.query_optimizer.optimize_query(query)
                expanded_queries = await self.query_expansion.expand_query(optimized_query)
                logger.info(f"Expanded to {len(expanded_queries)} queries")
            
            # 2. Generate embeddings for all queries
            logger.info(f"Generating embeddings for {len(expanded_queries)} queries...")
            all_embeddings = await self.llm_service.embedding(expanded_queries)
            
            # 3. Perform searches for each expanded query
            all_vector_results = []
            all_keyword_results = []
            
            for idx, (exp_query, embedding) in enumerate(zip(expanded_queries, all_embeddings)):
                logger.info(f"Searching with query {idx + 1}/{len(expanded_queries)}...")
                
                # Vector search
                vector_results = await self._vector_search(
                    workspace_id=workspace_id,
                    query_embedding=embedding,
                    top_k=retrieval_top_k
                )
                all_vector_results.extend(vector_results)
                
                # Keyword search
                keyword_results = await self._keyword_search(
                    workspace_id=workspace_id,
                    query_embedding=embedding,
                    top_k=retrieval_top_k
                )
                all_keyword_results.extend(keyword_results)
            
            # 4. Merge results from all queries
            logger.info("Merging search results...")
            merged_results = self._merge_results(
                vector_results=all_vector_results,
                keyword_results=all_keyword_results,
                top_k=retrieval_top_k * 2  # Get more results before reranking
            )
            
            # 5. Apply reranking
            if self.config.enable_reranking and len(merged_results) > 0:
                logger.info("Applying multi-factor reranking...")
                merged_results = self.reranker.rerank(
                    query=optimized_query,
                    results=merged_results,
                    top_k=final_top_k * 2  # Get more for diversity filtering
                )
            
            # 6. Apply diversity filtering
            if self.config.enable_diversity and len(merged_results) > 0:
                logger.info("Applying diversity filtering...")
                merged_results = self.diversity_filter.filter(
                    results=merged_results,
                    threshold=self.config.diversity_threshold,
                    top_k=final_top_k
                )
            else:
                # Just take top_k without diversity
                merged_results = merged_results[:final_top_k]
            
            # 7. Monitor quality
            if self.config.enable_quality_monitoring and len(merged_results) > 0:
                metrics = self.quality_monitor.calculate_metrics(
                    query=optimized_query,
                    results=merged_results
                )
                logger.info(f"Retrieval quality metrics: {metrics}")
            
            logger.info(f"Hybrid retrieval returned {len(merged_results)} results")
            return merged_results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}")
            raise
    
    async def _vector_search(
        self,
        workspace_id: str,
        query_embedding: List[float],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        向量搜索
        
        实现需求 3.2:
        - 按相似度分数降序排列
        - 应用相似度阈值过滤（0.7）
        """
        try:
            # 调用向量数据库服务
            db_results = await self.vector_db.vector_search(
                workspace_id=workspace_id,
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=self.config.vector_similarity_threshold
            )
            
            # 转换为 RetrievalResult
            results = []
            for db_result in db_results:
                result = RetrievalResult(
                    id=db_result.id,
                    file_path=db_result.file_path,
                    content=db_result.content,
                    similarity=db_result.similarity,
                    confidence=db_result.similarity,  # 向量搜索的置信度等于相似度
                    has_deprecated=db_result.has_deprecated,
                    has_fixme=db_result.has_fixme,
                    has_todo=db_result.has_todo,
                    has_security=db_result.has_security,
                    metadata=db_result.metadata,
                    source="vector"
                )
                results.append(result)
            
            # 结果已经按相似度降序排列（由数据库函数保证）
            logger.info(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            # 优雅降级：返回空结果
            return []
    
    async def _keyword_search(
        self,
        workspace_id: str,
        query_embedding: List[float],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        关键词搜索
        
        实现需求 3.3, 3.9:
        - 按关键词匹配数量排序
        - 敏感标记命中时应用 1.5 倍加权
        """
        try:
            # 构建关键词过滤器（搜索所有标记）
            keyword_filters = {
                'has_deprecated': True,
                'has_fixme': True,
                'has_todo': True,
                'has_security': True
            }
            
            # 调用混合搜索（实际上是关键词过滤的向量搜索）
            db_results = await self.vector_db.hybrid_search(
                workspace_id=workspace_id,
                query_embedding=query_embedding,
                keyword_filters=keyword_filters,
                top_k=top_k
            )
            
            # 转换为 RetrievalResult 并计算加权分数
            results = []
            for db_result in db_results:
                # 计算关键词匹配数量
                keyword_count = sum([
                    db_result.has_deprecated,
                    db_result.has_fixme,
                    db_result.has_todo,
                    db_result.has_security
                ])
                
                # 应用加权因子（敏感标记命中时 1.5 倍）
                boost_factor = 1.0
                if db_result.has_deprecated or db_result.has_security:
                    boost_factor = self.config.keyword_boost_factor
                elif db_result.has_fixme:
                    boost_factor = 1.3
                elif db_result.has_todo:
                    boost_factor = 1.2
                
                # 计算加权相似度
                weighted_similarity = db_result.similarity * boost_factor
                
                result = RetrievalResult(
                    id=db_result.id,
                    file_path=db_result.file_path,
                    content=db_result.content,
                    similarity=weighted_similarity,
                    confidence=weighted_similarity,
                    has_deprecated=db_result.has_deprecated,
                    has_fixme=db_result.has_fixme,
                    has_todo=db_result.has_todo,
                    has_security=db_result.has_security,
                    metadata={
                        **db_result.metadata,
                        'keyword_count': keyword_count,
                        'boost_factor': boost_factor
                    },
                    source="keyword"
                )
                results.append(result)
            
            # 按加权相似度降序排序
            results.sort(key=lambda x: x.similarity, reverse=True)
            
            logger.info(f"Keyword search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            # 优雅降级：返回空结果
            return []
    
    def _merge_results(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        合并向量搜索和关键词搜索结果
        
        实现需求 3.8, 3.9:
        - 向量搜索权重 0.6，关键词搜索权重 0.4
        - 去重相似度阈值 0.9
        - 返回合并后的 top-k 结果
        """
        # 创建结果字典（用于去重）
        result_dict: Dict[str, RetrievalResult] = {}
        
        # 处理向量搜索结果
        for result in vector_results:
            # 应用向量权重
            weighted_score = result.similarity * self.config.vector_weight
            result.confidence = weighted_score
            result_dict[result.id] = result
        
        # 处理关键词搜索结果
        for result in keyword_results:
            # 应用关键词权重
            weighted_score = result.similarity * self.config.keyword_weight
            
            if result.id in result_dict:
                # 如果已存在，合并分数
                existing = result_dict[result.id]
                # 使用向量搜索的原始相似度 + 关键词加权分数
                combined_score = (
                    existing.similarity * self.config.vector_weight +
                    weighted_score
                )
                existing.confidence = combined_score
                existing.source = "hybrid"
                
                # 合并元数据
                if 'keyword_count' in result.metadata:
                    existing.metadata['keyword_count'] = result.metadata['keyword_count']
                if 'boost_factor' in result.metadata:
                    existing.metadata['boost_factor'] = result.metadata['boost_factor']
            else:
                # 新结果，直接添加
                result.confidence = weighted_score
                result_dict[result.id] = result
        
        # 转换为列表并排序
        merged_results = list(result_dict.values())
        merged_results.sort(key=lambda x: x.confidence, reverse=True)
        
        # 返回 top-k 结果
        final_results = merged_results[:top_k]
        
        logger.info(f"Merged results: {len(final_results)} from {len(vector_results)} vector + {len(keyword_results)} keyword")
        return final_results
