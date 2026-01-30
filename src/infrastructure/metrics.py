"""Prometheus Metrics Exporter

This module implements Prometheus metrics for monitoring the system.
"""

import logging
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response


logger = logging.getLogger(__name__)


# ==================== Metrics ====================

# Agent execution metrics
agent_executions_total = Counter(
    'agent_executions_total',
    'Total number of agent executions',
    ['agent_name', 'status']
)

agent_execution_duration_seconds = Histogram(
    'agent_execution_duration_seconds',
    'Agent execution duration in seconds',
    ['agent_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# LLM call metrics
llm_calls_total = Counter(
    'llm_calls_total',
    'Total number of LLM calls',
    ['provider', 'model', 'task_type', 'status']
)

llm_call_duration_seconds = Histogram(
    'llm_call_duration_seconds',
    'LLM call duration in seconds',
    ['provider', 'model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total number of tokens used',
    ['provider', 'model', 'token_type']
)

# Retrieval metrics
retrieval_operations_total = Counter(
    'retrieval_operations_total',
    'Total number of retrieval operations',
    ['operation_type', 'status']
)

retrieval_duration_seconds = Histogram(
    'retrieval_duration_seconds',
    'Retrieval operation duration in seconds',
    ['operation_type'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

documents_retrieved_total = Counter(
    'documents_retrieved_total',
    'Total number of documents retrieved',
    ['workspace_id']
)

# Workflow metrics
workflow_executions_total = Counter(
    'workflow_executions_total',
    'Total number of workflow executions',
    ['status']
)

workflow_duration_seconds = Histogram(
    'workflow_duration_seconds',
    'Workflow execution duration in seconds',
    buckets=[10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
)

workflow_steps_total = Histogram(
    'workflow_steps_total',
    'Total number of steps in workflow',
    buckets=[1, 3, 5, 7, 10, 15, 20]
)

workflow_pivots_total = Counter(
    'workflow_pivots_total',
    'Total number of pivots triggered',
    ['pivot_reason']
)

workflow_retries_total = Counter(
    'workflow_retries_total',
    'Total number of retries',
    ['step_id']
)

# Error metrics
errors_total = Counter(
    'errors_total',
    'Total number of errors',
    ['component', 'error_type']
)

# System metrics
active_tasks = Gauge(
    'active_tasks',
    'Number of active tasks'
)

task_queue_size = Gauge(
    'task_queue_size',
    'Size of task queue'
)

# System info
system_info = Info(
    'system',
    'System information'
)


# ==================== Metric Recording Functions ====================

def record_agent_execution(agent_name: str, duration: float, status: str):
    """
    记录智能体执行指标
    
    Args:
        agent_name: 智能体名称
        duration: 执行时长（秒）
        status: 执行状态（success/failed）
    """
    agent_executions_total.labels(agent_name=agent_name, status=status).inc()
    agent_execution_duration_seconds.labels(agent_name=agent_name).observe(duration)


def record_llm_call(
    provider: str,
    model: str,
    task_type: str,
    duration: float,
    status: str,
    tokens: Dict[str, int] = None
):
    """
    记录 LLM 调用指标
    
    Args:
        provider: 提供商名称
        model: 模型名称
        task_type: 任务类型
        duration: 调用时长（秒）
        status: 调用状态（success/failed）
        tokens: Token 使用量字典（prompt_tokens, completion_tokens）
    """
    llm_calls_total.labels(
        provider=provider,
        model=model,
        task_type=task_type,
        status=status
    ).inc()
    
    llm_call_duration_seconds.labels(provider=provider, model=model).observe(duration)
    
    if tokens:
        if 'prompt_tokens' in tokens:
            llm_tokens_total.labels(
                provider=provider,
                model=model,
                token_type='prompt'
            ).inc(tokens['prompt_tokens'])
        
        if 'completion_tokens' in tokens:
            llm_tokens_total.labels(
                provider=provider,
                model=model,
                token_type='completion'
            ).inc(tokens['completion_tokens'])


def record_retrieval_operation(
    operation_type: str,
    duration: float,
    status: str,
    workspace_id: str = None,
    documents_count: int = 0
):
    """
    记录检索操作指标
    
    Args:
        operation_type: 操作类型（vector_search/keyword_search/hybrid）
        duration: 操作时长（秒）
        status: 操作状态（success/failed）
        workspace_id: 工作空间 ID
        documents_count: 检索到的文档数量
    """
    retrieval_operations_total.labels(
        operation_type=operation_type,
        status=status
    ).inc()
    
    retrieval_duration_seconds.labels(operation_type=operation_type).observe(duration)
    
    if workspace_id and documents_count > 0:
        documents_retrieved_total.labels(workspace_id=workspace_id).inc(documents_count)


def record_workflow_execution(
    duration: float,
    status: str,
    steps_count: int,
    pivots_count: int = 0,
    retries_count: int = 0
):
    """
    记录工作流执行指标
    
    Args:
        duration: 执行时长（秒）
        status: 执行状态（success/failed）
        steps_count: 步骤数量
        pivots_count: 转向次数
        retries_count: 重试次数
    """
    workflow_executions_total.labels(status=status).inc()
    workflow_duration_seconds.observe(duration)
    workflow_steps_total.observe(steps_count)
    
    if pivots_count > 0:
        workflow_pivots_total.labels(pivot_reason='total').inc(pivots_count)
    
    if retries_count > 0:
        workflow_retries_total.labels(step_id='total').inc(retries_count)


def record_error(component: str, error_type: str):
    """
    记录错误指标
    
    Args:
        component: 组件名称
        error_type: 错误类型
    """
    errors_total.labels(component=component, error_type=error_type).inc()


def update_active_tasks(count: int):
    """
    更新活动任务数量
    
    Args:
        count: 活动任务数量
    """
    active_tasks.set(count)


def update_task_queue_size(size: int):
    """
    更新任务队列大小
    
    Args:
        size: 队列大小
    """
    task_queue_size.set(size)


def set_system_info(info: Dict[str, str]):
    """
    设置系统信息
    
    Args:
        info: 系统信息字典
    """
    system_info.info(info)


# ==================== Metrics Endpoint ====================

def get_metrics() -> Response:
    """
    获取 Prometheus 指标
    
    Returns:
        Prometheus 格式的指标响应
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
