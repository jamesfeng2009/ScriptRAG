"""
LangGraph 工作流错误处理器

本模块为 LangGraph 工作流提供标准化的错误处理和日志记录功能。

核心原则：
1. 所有节点必须捕获异常，不传播到 LangGraph 层面
2. 使用标准化的错误日志格式
3. 使用 error_flag 标记错误状态
4. 日志使用 audit_log_reducer 自动追加

错误代码体系：
    - boundary_error: 步骤索引越界
    - retrieval_error: 文档检索失败
    - llm_error: LLM 调用失败
    - validation_error: 数据验证失败
    - timeout_error: 节点执行超时
    - unknown_error: 未知错误

使用示例：
    from src.infrastructure.langgraph_error_handler import (
        with_error_handling,
        ErrorCategory,
        ErrorRecovery
    )

    @with_error_handling(agent="planner", action="generate_outline")
    def planner_node(state):
        # 业务逻辑
        pass

规范文档：
    https://docs/architecture/v2.1_architecture_spec.md#51-错误处理规范
"""

import logging
import functools
import asyncio
from typing import Callable, Dict, Any, Optional, TypeVar, Type
from datetime import datetime
from enum import Enum

from ..domain.state_types import (
    GlobalState,
    create_error_log,
    create_success_log,
    ERROR_CODES,
)


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误类别"""
    BOUNDARY = "boundary_error"
    RETRIEVAL = "retrieval_error"
    LLM = "llm_error"
    VALIDATION = "validation_error"
    TIMEOUT = "timeout_error"
    SYSTEM = "system_error"
    UNKNOWN = "unknown_error"


class V2Error(Exception):
    """v2.1 错误基类"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class BoundaryError(V2Error):
    """边界错误（步骤索引越界等）"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.BOUNDARY,
            severity=ErrorSeverity.WARNING,
            details=details
        )


class RetrievalError(V2Error):
    """检索错误"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.RETRIEVAL,
            severity=ErrorSeverity.ERROR,
            details=details
        )


class LLMError(V2Error):
    """LLM 调用错误"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.LLM,
            severity=ErrorSeverity.ERROR,
            details=details
        )


class ValidationError(V2Error):
    """验证错误"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            details=details
        )


class TimeoutError(V2Error):
    """超时错误"""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.WARNING,
            details={"timeout_seconds": timeout_seconds, **(details or {})}
        )


# ============================================================================
# 错误处理装饰器
# ============================================================================


T = TypeVar('T', bound=Callable)


import inspect
from functools import wraps
from typing import Callable, Any, Dict, Optional


def with_error_handling(
    agent_name: str,
    action_name: str,
    error_category: ErrorCategory = ErrorCategory.UNKNOWN,
    fallback_error_code: Optional[str] = None
) -> Callable[[Callable], Callable]:
    """
    节点错误处理装饰器
    
    自动捕获异常，记录错误日志，设置 error_flag。
    
    支持同步和异步函数。
    
    Args:
        agent_name: 智能体名称
        action_name: 操作名称
        error_category: 错误类别
        fallback_error_code: 兜底错误代码
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            state = args[1] if len(args) > 1 else kwargs.get("state")
            try:
                return func(*args, **kwargs)
            except BoundaryError as e:
                logger.warning(f"Boundary error in {agent_name}: {e.message}")
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=e.message,
                        details={"category": e.category.value, **e.details}
                    ),
                    "error_flag": ErrorCategory.BOUNDARY.value
                }
            except RetrievalError as e:
                logger.error(f"Retrieval error in {agent_name}: {e.message}")
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=e.message,
                        details={"category": e.category.value, **e.details}
                    ),
                    "error_flag": ErrorCategory.RETRIEVAL.value
                }
            except LLMError as e:
                logger.error(f"LLM error in {agent_name}: {e.message}")
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=e.message,
                        details={"category": e.category.value, **e.details}
                    ),
                    "error_flag": ErrorCategory.LLM.value
                }
            except ValidationError as e:
                logger.warning(f"Validation error in {agent_name}: {e.message}")
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=e.message,
                        details={"category": e.category.value, **e.details}
                    ),
                    "error_flag": ErrorCategory.VALIDATION.value
                }
            except TimeoutError as e:
                logger.warning(f"Timeout error in {agent_name}: {e.message}")
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=e.message,
                        details={"category": e.category.value, **e.details}
                    ),
                    "error_flag": ErrorCategory.TIMEOUT.value
                }
            except V2Error as e:
                logger.error(f"V2Error in {agent_name}: {e.message}")
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=e.message,
                        details={"category": e.category.value, **e.details}
                    ),
                    "error_flag": e.category.value if e.category != ErrorCategory.UNKNOWN else None
                }
            except Exception as e:
                logger.error(f"Unexpected error in {agent_name}: {str(e)}", exc_info=True)
                error_code = fallback_error_code or ErrorCategory.UNKNOWN.value
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=str(e),
                        details={
                            "error_type": type(e).__name__,
                            "category": error_code
                        }
                    ),
                    "error_flag": error_code
                }
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            state = args[1] if len(args) > 1 else kwargs.get("state")
            try:
                return await func(*args, **kwargs)
            except BoundaryError as e:
                logger.warning(f"Boundary error in {agent_name}: {e.message}")
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=e.message,
                        details={"category": e.category.value, **e.details}
                    ),
                    "error_flag": ErrorCategory.BOUNDARY.value
                }
            except RetrievalError as e:
                logger.error(f"Retrieval error in {agent_name}: {e.message}")
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=e.message,
                        details={"category": e.category.value, **e.details}
                    ),
                    "error_flag": ErrorCategory.RETRIEVAL.value
                }
            except LLMError as e:
                logger.error(f"LLM error in {agent_name}: {e.message}")
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=e.message,
                        details={"category": e.category.value, **e.details}
                    ),
                    "error_flag": ErrorCategory.LLM.value
                }
            except ValidationError as e:
                logger.warning(f"Validation error in {agent_name}: {e.message}")
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=e.message,
                        details={"category": e.category.value, **e.details}
                    ),
                    "error_flag": ErrorCategory.VALIDATION.value
                }
            except TimeoutError as e:
                logger.warning(f"Timeout error in {agent_name}: {e.message}")
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=e.message,
                        details={"category": e.category.value, **e.details}
                    ),
                    "error_flag": ErrorCategory.TIMEOUT.value
                }
            except V2Error as e:
                logger.error(f"V2Error in {agent_name}: {e.message}")
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=e.message,
                        details={"category": e.category.value, **e.details}
                    ),
                    "error_flag": e.category.value if e.category != ErrorCategory.UNKNOWN else None
                }
            except Exception as e:
                logger.error(f"Unexpected error in {agent_name}: {str(e)}", exc_info=True)
                error_code = fallback_error_code or ErrorCategory.UNKNOWN.value
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=str(e),
                        details={
                            "error_type": type(e).__name__,
                            "category": error_code
                        }
                    ),
                    "error_flag": error_code
                }
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


# ============================================================================
# 异步错误处理
# ============================================================================


async def async_with_error_handling(
    agent_name: str,
    action_name: str,
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
) -> Callable[[Callable[..., Any]], Callable[..., Dict[str, Any]]]:
    """
    异步节点错误处理装饰器
    
    Args:
        agent_name: 智能体名称
        action_name: 操作名称
        error_category: 错误类别
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Dict[str, Any]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Dict[str, Any]:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Async error in {agent_name}: {str(e)}", exc_info=True)
                return {
                    "execution_log": create_error_log(
                        agent=agent_name,
                        action=action_name,
                        error_message=str(e),
                        details={"error_type": type(e).__name__}
                    ),
                    "error_flag": error_category.value if error_category != ErrorCategory.UNKNOWN else ErrorCategory.UNKNOWN.value
                }
        return wrapper
    return decorator


# ============================================================================
# 错误恢复工具
# ============================================================================


class ErrorRecovery:
    """错误恢复工具类"""
    
    @staticmethod
    def should_retry(state: GlobalState) -> bool:
        """判断是否应该重试"""
        error_flag = state.get("error_flag")
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        
        if not error_flag:
            return False
        
        recoverable_errors = {
            ErrorCategory.RETRIEVAL.value,
            ErrorCategory.LLM.value,
            ErrorCategory.TIMEOUT.value,
        }
        
        return error_flag in recoverable_errors and retry_count < max_retries
    
    @staticmethod
    def increment_retry(state: GlobalState) -> Dict[str, Any]:
        """增加重试计数"""
        current = state.get("retry_count", 0)
        return {
            "retry_count": current + 1,
            "execution_log": create_success_log(
                agent="retry_protection",
                action="retry_incremented",
                details={"retry_count": current + 1}
            )
        }
    
    @staticmethod
    def clear_error(state: GlobalState) -> Dict[str, Any]:
        """清除错误状态"""
        return {
            "error_flag": None,
            "execution_log": create_success_log(
                agent="retry_protection",
                action="error_cleared",
                details={"previous_error": state.get("error_flag")}
            )
        }
    
    @staticmethod
    def skip_step(state: GlobalState) -> Dict[str, Any]:
        """跳过当前步骤"""
        step_index = state.get("current_step_index", 0)
        outline = state.get("outline", [])
        
        if step_index < len(outline):
            new_outline = [
                {**step, "status": "skipped"} 
                if step.get("id") == step_index else step
                for step in outline
            ]
            
            return {
                "outline": new_outline,
                "execution_log": create_success_log(
                    agent="retry_protection",
                    action="step_skipped",
                    details={"step_index": step_index}
                )
            }
        
        return {}


# ============================================================================
# 日志规范工具
# ============================================================================


class LogFormatter:
    """日志格式化工具"""
    
    @staticmethod
    def format_action(agent: str, action: str) -> str:
        """格式化操作名称"""
        return f"{agent}:{action}"
    
    @staticmethod
    def format_details(**kwargs) -> Dict[str, Any]:
        """格式化详情字典"""
        return {k: v for k, v in kwargs.items() if v is not None}
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100) -> str:
        """截断文本"""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."


# ============================================================================
# 性能监控
# ============================================================================


class PerformanceMonitor:
    """性能监控工具"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.start_time: Optional[datetime] = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            if exc_type:
                logger.warning(
                    f"{self.agent_name} failed in {duration:.2f}s: {exc_val}"
                )
            else:
                logger.info(f"{self.agent_name} completed in {duration:.2f}s")
    
    @staticmethod
    def monitor(agent_name: str) -> Callable:
        """性能监控装饰器"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with PerformanceMonitor(agent_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


# ============================================================================
# 使用示例
# ============================================================================


def example_usage():
    """使用示例"""
    from ..domain.state_types import GlobalState
    
    @with_error_handling(agent="planner", action="generate_outline")
    def planner_node(state: GlobalState) -> Dict[str, Any]:
        """规划器节点（带错误处理）"""
        if not state.get("user_topic"):
            raise ValidationError("user_topic 不能为空")
        
        outline = generate_outline(state["user_topic"])
        
        return {
            "outline": outline,
            "execution_log": create_success_log(
                agent="planner",
                action="outline_generated",
                details={"step_count": len(outline)}
            )
        }
    
    @with_error_handling(
        agent="navigator",
        action="retrieve",
        error_category=ErrorCategory.RETRIEVAL
    )
    def navigator_node(state: GlobalState) -> Dict[str, Any]:
        """导航器节点（带错误处理）"""
        step_index = state.get("current_step_index", 0)
        outline = state.get("outline", [])
        
        if step_index >= len(outline):
            raise BoundaryError(
                f"步骤索引越界: {step_index} >= {len(outline)}",
                details={"step_index": step_index, "outline_length": len(outline)}
            )
        
        docs = retrieve_documents(outline[step_index])
        
        return {
            "retrieved_docs": docs,
            "execution_log": create_success_log(
                agent="navigator",
                action="retrieve_completed",
                details={"doc_count": len(docs)}
            )
        }
    
    async def async_example():
        """异步示例"""
        @async_with_error_handling(
            agent="llm_processor",
            action="generate",
            error_category=ErrorCategory.LLM
        )
        async def llm_generate(prompt: str) -> str:
            return await call_llm(prompt)
        
        result = await llm_generate("Hello")
        return result
    
    return planner_node, navigator_node


if __name__ == "__main__":
    example_usage()
