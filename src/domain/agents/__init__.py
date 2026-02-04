"""Agent implementations

本模块包含两类Agent实现：
1. 原始实现（v1）：基于 SharedState，直接修改状态
2. v2.1 实现：基于 GlobalState (TypedDict)，返回 Diff

新增功能：
- agent_collaboration: Agent 协作优化（协商+并行+追踪+反思）

导出规范：
- v1 节点函数：直接导出，供旧代码使用
- v2.1 节点函数：通过 node_factory 模块访问
- 协作功能：通过 agent_collaboration 模块访问

v2.1 节点函数使用示例：
    from src.domain.agents.node_factory import NodeFactory, create_node_factory
    
    factory = create_node_factory(
        llm_service=llm_service,
        retrieval_service=retrieval_service,
        parser_service=parser_service,
        summarization_service=summarization_service
    )
    
    workflow.add_node("planner", factory.planner_node)
    workflow.add_node("navigator", factory.navigator_node)

协作功能使用示例：
    from src.domain.agents.agent_collaboration import (
        AgentNegotiator,
        ParallelAgentExecutor,
        AgentExecutionTracer,
        AgentReflection,
        CollaborationManager
    )
"""

from .retry_protection import (
    check_retry_limit,
    is_in_infinite_loop,
    reset_retry_counter,
)

from .node_factory import (
    NodeFactory,
    create_node_factory,
    create_success_log,
    create_error_log,
)

from .agent_collaboration import (
    AgentNegotiator,
    ParallelAgentExecutor,
    AgentExecutionTracer,
    AgentReflection,
    CollaborationManager,
    NegotiationStatus,
    DecisionType,
    NegotiationContext,
    ExecutionNode,
    FailureRecord,
    ReflectionResult
)

__all__ = [
    "check_retry_limit",
    "is_in_infinite_loop",
    "reset_retry_counter",
    "NodeFactory",
    "create_node_factory",
    "create_success_log",
    "create_error_log",
    
    # 协作功能
    "AgentNegotiator",
    "ParallelAgentExecutor",
    "AgentExecutionTracer",
    "AgentReflection",
    "CollaborationManager",
    "NegotiationStatus",
    "DecisionType",
    "NegotiationContext",
    "ExecutionNode",
    "FailureRecord",
    "ReflectionResult",
]
