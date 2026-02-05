"""导航器智能体 - 执行 RAG 检索

本模块实现导航器智能体，负责：
1. 混合检索（向量 + 关键词搜索）
2. 从代码中提取元数据
3. 文件大小检查和摘要
4. 带置信度分数的来源溯源跟踪
5. 优雅处理空检索结果
6. 并行检索优化（向量 + 关键词同时搜索）
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any

from ..models import SharedState, RetrievedDocument
from ...services.retrieval_service import RetrievalService
from ...services.parser.tree_sitter_parser import IParserService
from ...services.summarization_service import SummarizationService
from ...services.optimization import (
    AsyncIOOptimizer,
    ParallelExecutor,
    SmartSkipOptimizer,
    CacheBasedSkipper,
    QualityAssessor,
    ComplexityBasedSkipper
)
from ...infrastructure.logging import get_agent_logger


logger = logging.getLogger(__name__)
agent_logger = get_agent_logger(__name__)


async def retrieve_content(
    state: SharedState,
    retrieval_service: RetrievalService,
    parser_service: IParserService,
    summarization_service: SummarizationService,
    workspace_id: str,
    enable_parallel: bool = True
) -> SharedState:
    """
    导航器智能体主函数
    - 集成混合搜索、元数据提取和摘要
    - 添加带置信度分数的来源出处追踪
    - 优雅处理空检索
    - 支持并行检索优化
    
    Args:
        state: 共享状态
        retrieval_service: 检索服务
        parser_service: 解析服务
        summarization_service: 摘要服务
        workspace_id: 工作空间 ID
        enable_parallel: 是否启用并行检索（默认启用）
        
    Returns:
        更新后的共享状态
    """
    try:
        # 获取当前步骤
        if state.current_step_index >= len(state.outline):
            logger.warning("No more steps to process")
            return state
        
        current_step = state.outline[state.current_step_index]
        
        logger.info(f"Navigator: Retrieving content for step {current_step.step_id}: {current_step.description}")
        
        # Log agent transition
        agent_logger.log_agent_transition(
            from_agent="previous",
            to_agent="navigator",
            step_id=current_step.step_id,
            reason="retrieve_content"
        )
        
        query = current_step.description
        
        # 1. 执行检索（支持并行）
        if enable_parallel:
            retrieval_results = await _parallel_retrieve(
                state=state,
                retrieval_service=retrieval_service,
                query=query,
                workspace_id=workspace_id,
                top_k=5
            )
        else:
            retrieval_results = await retrieval_service.hybrid_retrieve(
                workspace_id=workspace_id,
                query=query,
                top_k=5
            )
        
        # 2. 处理检索结果
        retrieved_docs = []
        
        if not retrieval_results:
            # 优雅处理空检索
            logger.warning(f"No retrieval results for step {current_step.step_id}")
            
            # Log empty retrieval
            agent_logger.log_retrieval_result(
                step_id=current_step.step_id,
                doc_count=0,
                sources=[],
                retrieval_method="parallel" if enable_parallel else "hybrid",
                confidence_scores=[]
            )
            
            state.retrieved_docs = []
            return state
        
        # 3. 并行处理文档（解析 + 摘要）
        retrieved_docs = await _parallel_process_results(
            retrieval_results=retrieval_results,
            parser_service=parser_service,
            summarization_service=summarization_service
        )
        
        sources = [doc.source for doc in retrieved_docs]
        confidence_scores = [doc.confidence for doc in retrieved_docs]
        
        # 4. 更新共享状态
        state.retrieved_docs = retrieved_docs
        
        logger.info(f"Navigator: Retrieved {len(retrieved_docs)} documents for step {current_step.step_id}")
        
        # Log retrieval results with sources
        agent_logger.log_retrieval_result(
            step_id=current_step.step_id,
            doc_count=len(retrieved_docs),
            sources=sources,
            retrieval_method="parallel" if enable_parallel else "hybrid",
            confidence_scores=confidence_scores
        )
        
        # 5. 记录执行日志
        state.execution_log.append({
            'agent': 'navigator',
            'action': 'retrieve_content',
            'step_id': current_step.step_id,
            'num_results': len(retrieved_docs),
            'sources': sources,
            'retrieval_method': 'parallel' if enable_parallel else 'hybrid'
        })
        
        return state
        
    except Exception as e:
        logger.error(f"Navigator failed: {str(e)}")
        
        # Log error with context
        agent_logger.log_error_with_context(
            error=e,
            context={
                "step_id": current_step.step_id if 'current_step' in locals() else None,
                "query": query if 'query' in locals() else None
            },
            agent_name="navigator"
        )
        
        # 优雅降级：返回空结果而不是崩溃
        state.retrieved_docs = []
        state.execution_log.append({
            'agent': 'navigator',
            'action': 'retrieve_content_failed',
            'error': str(e)
        })
        return state


async def _parallel_retrieve(
    state: SharedState,
    retrieval_service: RetrievalService,
    query: str,
    workspace_id: str,
    top_k: int = 5
) -> List:
    """
    并行执行向量搜索和关键词搜索

    使用 asyncio.gather 同时执行两种检索策略，提高检索速度。
    缓存命中时跳过重复检索。

    Args:
        state: 共享状态
        retrieval_service: 检索服务
        query: 查询文本
        workspace_id: 工作空间 ID
        top_k: 返回结果数量

    Returns:
        合并的检索结果列表
    """
    cache_skipper = CacheBasedSkipper()
    cache_key = f"retrieval:{workspace_id}:{hash(query)}"

    skip_decision = cache_skipper.should_skip_processing(cache_key, min_hits_for_skip=2)

    if skip_decision.should_skip:
        logger.info(f"Skipping redundant retrieval for: {query[:50]}... "
                   f"(confidence: {skip_decision.confidence:.2f}, reason: {skip_decision.reason})")
        if state.retrieved_docs:
            return []
        return []

    async def vector_search():
        return await retrieval_service.retrieve_with_strategy(
            workspace_id=workspace_id,
            query=query,
            strategy_name="vector_search",
            top_k=top_k
        )

    async def keyword_search():
        return await retrieval_service.retrieve_with_strategy(
            workspace_id=workspace_id,
            query=query,
            strategy_name="keyword_search",
            top_k=top_k
        )

    try:
        vector_results, keyword_results = await asyncio.gather(
            vector_search(),
            keyword_search(),
            return_exceptions=True
        )

        if isinstance(vector_results, Exception):
            logger.warning(f"Vector search failed: {vector_results}")
            vector_results = []

        if isinstance(keyword_results, Exception):
            logger.warning(f"Keyword search failed: {keyword_results}")
            keyword_results = []

        if not vector_results and not keyword_results:
            return []

        merged_results = _merge_retrieval_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
            vector_weight=0.6,
            keyword_weight=0.4,
            top_k=top_k
        )

        return merged_results

    except Exception as e:
        logger.error(f"Parallel retrieval failed: {str(e)}")
        return await retrieval_service.hybrid_retrieve(
            workspace_id=workspace_id,
            query=query,
            top_k=top_k
        )


def _merge_retrieval_results(
    vector_results: List,
    keyword_results: List,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
    top_k: int = 5
) -> List:
    """
    合并向量搜索和关键词搜索的结果

    使用加权融合策略，根据相似度和来源分配权重。

    Args:
        vector_results: 向量搜索结果
        keyword_results: 关键词搜索结果
        vector_weight: 向量结果权重
        keyword_weight: 关键词结果权重
        top_k: 返回结果数量

    Returns:
        合并后的结果列表
    """
    if not vector_results and not keyword_results:
        return []

    if not vector_results:
        return keyword_results[:top_k]

    if not keyword_results:
        return vector_results[:top_k]

    scored_results = []

    for result in vector_results:
        score = getattr(result, 'confidence', 0.5) * vector_weight
        result_copy = result
        result_copy._final_score = score
        result_copy._source = 'vector'
        scored_results.append(result_copy)

    for result in keyword_results:
        score = getattr(result, 'confidence', 0.5) * keyword_weight
        result_copy = result
        result_copy._final_score = score
        result_copy._source = 'keyword'
        scored_results.append(result_copy)

    scored_results.sort(key=lambda x: getattr(x, '_final_score', 0), reverse=True)

    unique_results = []
    seen_sources = set()

    for result in scored_results:
        source = getattr(result, 'source', str(result.file_path))
        if source not in seen_sources:
            unique_results.append(result)
            seen_sources.add(source)

        if len(unique_results) >= top_k:
            break

    return unique_results


async def _parallel_process_results(
    retrieval_results: List,
    parser_service: IParserService,
    summarization_service: SummarizationService
) -> List[RetrievedDocument]:
    """
    并行处理检索结果

    同时执行代码解析和摘要生成，提高处理效率。

    Args:
        retrieval_results: 检索结果列表
        parser_service: 解析服务
        summarization_service: 摘要服务

    Returns:
        处理后的文档列表
    """
    async def process_single_result(result) -> RetrievedDocument:
        try:
            parsed_code = parser_service.parse(
                file_path=result.file_path,
                content=result.content
            )

            content_to_use = result.content
            summary = None

            if summarization_service.check_size(result.content):
                summarized = await summarization_service.summarize_document(
                    content=result.content,
                    file_path=result.file_path,
                    parsed_code=parsed_code
                )
                content_to_use = summarized.summary
                summary = summarized.summary

            doc = RetrievedDocument(
                content=content_to_use,
                source=result.file_path,
                confidence=getattr(result, 'confidence', 0.5),
                metadata={
                    'similarity': getattr(result, 'similarity', 0.0),
                    'search_source': getattr(result, 'strategy_name', 'unknown'),
                    'has_deprecated': parsed_code.has_deprecated,
                    'has_fixme': parsed_code.has_fixme,
                    'has_todo': parsed_code.has_todo,
                    'has_security': parsed_code.has_security,
                    'language': parsed_code.language,
                    'num_elements': len(parsed_code.elements),
                    'was_summarized': summary is not None,
                    'retrieval_source': getattr(result, '_source', 'hybrid')
                },
                summary=summary
            )

            return doc

        except Exception as e:
            logger.error(f"Failed to process result {result.file_path}: {str(e)}")
            return RetrievedDocument(
                content=result.content,
                source=result.file_path,
                confidence=getattr(result, 'confidence', 0.0),
                metadata={'error': str(e)}
            )

    tasks = [process_single_result(result) for result in retrieval_results]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_docs = []
    for result in results:
        if isinstance(result, RetrievedDocument):
            processed_docs.append(result)
        else:
            logger.error(f"Unexpected result type: {type(result)}")

    return processed_docs


async def smart_retrieve_content(
    state: SharedState,
    retrieval_service: RetrievalService,
    parser_service: IParserService,
    summarization_service: SummarizationService,
    workspace_id: str
) -> SharedState:
    """
    智能检索（集成智能跳过优化）

    在 retrieve_content 基础上添加智能跳过功能：
    - 缓存命中时跳过重复检索
    - 高质量内容减少处理步骤
    - 基于复杂度调整处理策略

    Args:
        state: 共享状态
        retrieval_service: 检索服务
        parser_service: 解析服务
        summarization_service: 摘要服务
        workspace_id: 工作空间 ID

    Returns:
        更新后的共享状态
    """
    skip_optimizer = SmartSkipOptimizer(
        enable_quality_skip=True,
        enable_complexity_skip=True,
        enable_cache_skip=True
    )

    if state.retrieved_docs:
        quality_score = skip_optimizer.quality_assessor.assess_quality(
            content=str(state.retrieved_docs[0].content)[:500],
            context=None
        )

        quality_decision = skip_optimizer.quality_assessor.should_skip_fact_check(
            quality_score,
            skip_types=['fact_check']
        )

        if quality_decision.should_skip:
            logger.info(
                f"Skipping redundant retrieval - "
                f"high quality content (quality={quality_score:.2f}, "
                f"confidence={quality_decision.confidence:.2f})"
            )
            state.execution_log.append({
                'agent': 'navigator',
                'action': 'smart_skip_quality',
                'quality_score': quality_score,
                'confidence': quality_decision.confidence
            })
            return state

        complexity_skipper = ComplexityBasedSkipper()
        current_step = state.outline[state.current_step_index] if state.current_step_index < len(state.outline) else None
        query_length = len(current_step.description) if current_step else 0
        complexity_score = min(1.0, query_length / 500 + len(state.retrieved_docs) / 10)

        mode = complexity_skipper.get_processing_mode(complexity_score)
        logger.info(
            f"Complexity assessment: score={complexity_score:.2f}, "
            f"mode={mode}"
        )

        if mode == 'minimal':
            logger.info("Using minimal processing mode for low complexity content")
            state.execution_log.append({
                'agent': 'navigator',
                'action': 'smart_skip_complexity',
                'complexity_score': complexity_score,
                'mode': mode
            })
            return state

    return await retrieve_content(
        state=state,
        retrieval_service=retrieval_service,
        parser_service=parser_service,
        summarization_service=summarization_service,
        workspace_id=workspace_id,
        enable_parallel=True
    )

