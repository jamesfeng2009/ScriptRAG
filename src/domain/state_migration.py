"""
状态转换工具（SharedState ↔ GlobalState）

本模块提供 SharedState 和 GlobalState 之间的转换函数。
用于 v2.1 迁移期间的渐进式兼容。

使用示例：
    from src.domain.state_migration import (
        to_global_state,
        from_global_state,
    )

迁移策略：
    https://docs/architecture/v2.1_architecture_spec.md#73-迁移指南
"""

import logging
from typing import Dict, Any

from ..domain.state_types import GlobalState


logger = logging.getLogger(__name__)


def to_global_state(state: Dict[str, Any]) -> GlobalState:
    """将字典状态转换为 GlobalState
    
    Args:
        state: 字典类型的状态
        
    Returns:
        GlobalState (TypedDict)
    """
    return {
        "user_topic": state.get("user_topic", ""),
        "project_context": state.get("project_context", ""),
        "outline": state.get("outline", []),
        "current_step_index": state.get("current_step_index", 0),
        "fragments": state.get("fragments", []),
        "retrieved_docs": state.get("retrieved_docs", []),
        "director_feedback": state.get("director_feedback", None),
        "execution_log": state.get("execution_log", []),
        "error_flag": state.get("error_flag", None),
        "retry_count": state.get("retry_count", 0),
    }


def from_global_state(global_state: GlobalState) -> Dict[str, Any]:
    """将 GlobalState 转换回字典状态
    
    Args:
        global_state: GlobalState (TypedDict)
        
    Returns:
        字典类型的状态
    """
    return {
        "user_topic": global_state.get("user_topic", ""),
        "project_context": global_state.get("project_context", ""),
        "outline": global_state.get("outline", []),
        "current_step_index": global_state.get("current_step_index", 0),
        "fragments": global_state.get("fragments", []),
        "retrieved_docs": global_state.get("retrieved_docs", []),
        "director_feedback": global_state.get("director_feedback", None),
        "execution_log": global_state.get("execution_log", []),
        "error_flag": global_state.get("error_flag", None),
        "retry_count": global_state.get("retry_count", 0),
    }
