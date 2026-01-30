"""Navigator Agent - Performs RAG retrieval

The Navigator agent is responsible for:
1. Hybrid retrieval (vector + keyword search)
2. Metadata extraction from code
3. File size checking and summarization
4. Source provenance tracking with confidence scores
5. Graceful handling of empty retrievals
"""

import logging
from typing import List, Optional

from ..models import SharedState, RetrievedDocument
from ...services.retrieval_service import RetrievalService
from ...services.parser.tree_sitter_parser import IParserService
from ...services.summarization_service import SummarizationService
from ...infrastructure.logging import get_agent_logger


logger = logging.getLogger(__name__)
agent_logger = get_agent_logger(__name__)


async def retrieve_content(
    state: SharedState,
    retrieval_service: RetrievalService,
    parser_service: IParserService,
    summarization_service: SummarizationService,
    workspace_id: str
) -> SharedState:
    """
    导航器智能体主函数
    - 集成混合搜索、元数据提取和摘要
    - 添加带置信度分数的来源出处追踪
    - 优雅处理空检索
    
    Args:
        state: 共享状态
        retrieval_service: 检索服务
        parser_service: 解析服务
        summarization_service: 摘要服务
        workspace_id: 工作空间 ID
        
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
        
        # 1. 执行混合检索
        query = current_step.description
        retrieval_results = await retrieval_service.hybrid_retrieve(
            workspace_id=workspace_id,
            query=query,
            top_k=5
        )
        
        # 2. 处理检索结果
        retrieved_docs = []
        
        if not retrieval_results:
            # 优雅处理空检索（需求 3.7）
            logger.warning(f"No retrieval results for step {current_step.step_id}")
            
            # Log empty retrieval
            agent_logger.log_retrieval_result(
                step_id=current_step.step_id,
                doc_count=0,
                sources=[],
                retrieval_method="hybrid",
                confidence_scores=[]
            )
            
            # 不生成幻觉内容，返回空结果
            state.retrieved_docs = []
            return state
        
        sources = []
        confidence_scores = []
        
        for result in retrieval_results:
            # 3. 解析代码提取元数据
            parsed_code = parser_service.parse(
                file_path=result.source,
                content=result.content
            )
            
            # 4. 检查文件大小并摘要（如果需要）
            content_to_use = result.content
            summary = None
            
            if summarization_service.check_size(result.content):
                logger.info(f"Summarizing large file: {result.file_path}")
                summarized = await summarization_service.summarize_document(
                    content=result.content,
                    file_path=result.file_path,
                    parsed_code=parsed_code
                )
                content_to_use = summarized.summary
                summary = summarized.summary
            
            # 5. 创建 RetrievedDocument 并添加来源出处追踪（需求 3.6）
            doc = RetrievedDocument(
                content=content_to_use,
                source=result.file_path,
                confidence=result.confidence,  # 置信度分数
                metadata={
                    'similarity': result.similarity,
                    'search_source': result.source,  # "vector", "keyword", or "hybrid"
                    'has_deprecated': parsed_code.has_deprecated,
                    'has_fixme': parsed_code.has_fixme,
                    'has_todo': parsed_code.has_todo,
                    'has_security': parsed_code.has_security,
                    'language': parsed_code.language,
                    'num_elements': len(parsed_code.elements),
                    'was_summarized': summary is not None
                },
                summary=summary
            )
            
            retrieved_docs.append(doc)
            sources.append(result.file_path)
            confidence_scores.append(result.confidence)
        
        # 6. 更新共享状态
        state.retrieved_docs = retrieved_docs
        
        logger.info(f"Navigator: Retrieved {len(retrieved_docs)} documents for step {current_step.step_id}")
        
        # Log retrieval results with sources
        agent_logger.log_retrieval_result(
            step_id=current_step.step_id,
            doc_count=len(retrieved_docs),
            sources=sources,
            retrieval_method="hybrid",
            confidence_scores=confidence_scores
        )
        
        # 7. 记录执行日志
        state.execution_log.append({
            'agent': 'navigator',
            'action': 'retrieve_content',
            'step_id': current_step.step_id,
            'num_results': len(retrieved_docs),
            'sources': [doc.source for doc in retrieved_docs]
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

