"""数据访问控制模块

本模块提供数据访问审计和控制机制，用于增强 GlobalState 的数据管理。

核心功能：
1. 数据访问装饰器 - 自动记录 Agent 的数据访问
2. 数据所有权定义 - 明确每个字段的所有者
3. 访问权限检查 - 可选的运行时权限验证
4. 幻觉控制 - 最小化信息获取 + 强制引用验证

设计原则：
- 最小权限原则：Agent 只获取必要信息
- 防止信息膨胀：检索结果每步清空
- 防止幻觉传播：错误历史隔离存储

使用示例：
    @DataAccessControl.agent_access(
        agent_name="planner",
        reads={"user_topic", "project_context"},
        writes={"outline", "execution_log"}
    )
    async def planner_node(self, state: GlobalState) -> Dict[str, Any]:
        pass
"""

import logging
from functools import wraps
from typing import Set, Optional, Callable, Any, Dict, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class DataOwner(Enum):
    """数据所有者枚举"""
    SYSTEM = "system"
    PLANNER = "planner"
    NAVIGATOR = "navigator"
    DIRECTOR = "director"
    PIVOT_MANAGER = "pivot_manager"
    WRITER = "writer"
    FACT_CHECKER = "fact_checker"
    COMPILER = "compiler"
    COLLABORATION_MANAGER = "collaboration_manager"
    SHARED = "shared"


class WritePolicy(Enum):
    """写入策略枚举"""
    READ_ONLY = "read_only"
    OWNER_ONLY = "owner_only"
    APPEND_ONLY = "append_only"
    EXPLICIT_SWITCH = "explicit_switch"
    BOOLEAN_FLAG = "boolean_flag"
    SHARED_WRITE = "shared_write"


class ContextSensitivity(Enum):
    """上下文敏感度（用于幻觉控制）"""
    LOW = "low"       # 基础事实，幻觉风险低
    MEDIUM = "medium"  # 任务数据，需验证
    HIGH = "high"     # 检索结果，直接作为证据，幻觉风险高


@dataclass
class FieldConfig:
    """字段配置"""
    owner: DataOwner
    write_policy: WritePolicy
    required_by: Set[str]
    sensitivity: ContextSensitivity
    description: str
    max_history: Optional[int] = None  # 最大历史长度


DATA_OWNERSHIP_CONFIG: Dict[str, FieldConfig] = {
    "user_topic": FieldConfig(
        owner=DataOwner.SYSTEM,
        write_policy=WritePolicy.READ_ONLY,
        required_by={"*"},
        sensitivity=ContextSensitivity.LOW,
        description="用户输入的主题（基础事实）"
    ),
    "project_context": FieldConfig(
        owner=DataOwner.SYSTEM,
        write_policy=WritePolicy.READ_ONLY,
        required_by={"*"},
        sensitivity=ContextSensitivity.LOW,
        description="项目上下文信息（基础事实）"
    ),
    "outline": FieldConfig(
        owner=DataOwner.PLANNER,
        write_policy=WritePolicy.SHARED_WRITE,
        required_by={"navigator", "director", "writer", "compiler", "retry_protection", "pivot_manager"},
        sensitivity=ContextSensitivity.MEDIUM,
        description="剧本大纲"
    ),
    "retrieved_docs": FieldConfig(
        owner=DataOwner.NAVIGATOR,
        write_policy=WritePolicy.OWNER_ONLY,
        required_by={"director", "writer", "fact_checker", "pivot_manager"},
        sensitivity=ContextSensitivity.HIGH,
        description="检索到的文档（高幻觉风险，需验证来源）"
    ),
    "fragments": FieldConfig(
        owner=DataOwner.WRITER,
        write_policy=WritePolicy.APPEND_ONLY,
        required_by={"director", "fact_checker", "compiler", "pivot_manager"},
        sensitivity=ContextSensitivity.MEDIUM,
        max_history=50,
        description="生成的剧本片段"
    ),
    "current_skill": FieldConfig(
        owner=DataOwner.SHARED,
        write_policy=WritePolicy.EXPLICIT_SWITCH,
        required_by={"director", "pivot_manager", "writer"},
        sensitivity=ContextSensitivity.LOW,
        description="当前使用的技能"
    ),
    "skill_history": FieldConfig(
        owner=DataOwner.SHARED,
        write_policy=WritePolicy.APPEND_ONLY,
        required_by={"compiler"},
        sensitivity=ContextSensitivity.LOW,
        max_history=20,
        description="技能切换历史"
    ),
    "pivot_triggered": FieldConfig(
        owner=DataOwner.DIRECTOR,
        write_policy=WritePolicy.BOOLEAN_FLAG,
        required_by={"pivot_manager"},
        sensitivity=ContextSensitivity.LOW,
        description="转向触发标志"
    ),
    "pivot_reason": FieldConfig(
        owner=DataOwner.DIRECTOR,
        write_policy=WritePolicy.OWNER_ONLY,
        required_by={"pivot_manager"},
        sensitivity=ContextSensitivity.MEDIUM,
        description="转向原因"
    ),
    "current_step_index": FieldConfig(
        owner=DataOwner.SHARED,
        write_policy=WritePolicy.SHARED_WRITE,
        required_by={"navigator", "writer", "compiler", "retry_protection", "director", "pivot_manager"},
        sensitivity=ContextSensitivity.LOW,
        description="当前步骤索引"
    ),
    "fact_check_passed": FieldConfig(
        owner=DataOwner.FACT_CHECKER,
        write_policy=WritePolicy.BOOLEAN_FLAG,
        required_by={"*"},
        sensitivity=ContextSensitivity.LOW,
        description="事实检查通过标志"
    ),
    "awaiting_user_input": FieldConfig(
        owner=DataOwner.WRITER,
        write_policy=WritePolicy.BOOLEAN_FLAG,
        required_by={"*"},
        sensitivity=ContextSensitivity.LOW,
        description="等待用户输入标志"
    ),
    "execution_log": FieldConfig(
        owner=DataOwner.SHARED,
        write_policy=WritePolicy.APPEND_ONLY,
        required_by={"*"},
        sensitivity=ContextSensitivity.LOW,
        max_history=100,
        description="执行日志"
    ),
    "error_flag": FieldConfig(
        owner=DataOwner.SYSTEM,
        write_policy=WritePolicy.SHARED_WRITE,
        required_by={"*"},
        sensitivity=ContextSensitivity.LOW,
        description="错误标记"
    ),
    "retry_count": FieldConfig(
        owner=DataOwner.SYSTEM,
        write_policy=WritePolicy.SHARED_WRITE,
        required_by={"*"},
        sensitivity=ContextSensitivity.LOW,
        description="重试计数"
    ),
    "workflow_complete": FieldConfig(
        owner=DataOwner.SYSTEM,
        write_policy=WritePolicy.BOOLEAN_FLAG,
        required_by={"*"},
        sensitivity=ContextSensitivity.LOW,
        description="工作流完成标志"
    ),
    "final_screenplay": FieldConfig(
        owner=DataOwner.WRITER,
        write_policy=WritePolicy.OWNER_ONLY,
        required_by={"*"},
        sensitivity=ContextSensitivity.MEDIUM,
        description="最终剧本输出"
    ),
    "director_feedback": FieldConfig(
        owner=DataOwner.DIRECTOR,
        write_policy=WritePolicy.OWNER_ONLY,
        required_by={"*"},
        sensitivity=ContextSensitivity.MEDIUM,
        description="导演反馈"
    ),
    "created_at": FieldConfig(
        owner=DataOwner.SYSTEM,
        write_policy=WritePolicy.READ_ONLY,
        required_by={"*"},
        sensitivity=ContextSensitivity.LOW,
        description="创建时间"
    ),
    "resumed_from_session": FieldConfig(
        owner=DataOwner.SYSTEM,
        write_policy=WritePolicy.READ_ONLY,
        required_by={"*"},
        sensitivity=ContextSensitivity.LOW,
        description="恢复自会话 ID"
    ),
}


class AccessDeniedError(Exception):
    """访问被拒绝异常"""
    pass


class DataAccessControl:
    """数据访问控制类

    提供装饰器和工具方法来管理 Agent 对 GlobalState 的访问。
    支持 v2.1 架构的 TypedDict + Reducer 模式。

    幻觉控制特性：
    1. 最小权限：Agent 只读取必要字段
    2. 敏感度标记：高敏感度字段需要额外验证
    3. 历史限制：防止状态无限膨胀
    """

    STRICT_MODE = True

    @staticmethod
    def agent_access(
        agent_name: str,
        reads: Optional[Set[str]] = None,
        writes: Optional[Set[str]] = None,
        description: str = ""
    ) -> Callable:
        """数据访问装饰器（v2.1 GlobalState 版本）

        自动记录 Agent 的数据访问，并进行权限检查。

        Args:
            agent_name: Agent 名称
            reads: 读取的字段集合
            writes: 写入的字段集合
            description: Agent 操作描述

        Returns:
            装饰器函数
        """
        reads = reads or set()
        writes = writes or set()

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(self, state: Dict[str, Any], *args, **kwargs):
                access_start = datetime.now()

                access_log = {
                    "agent": agent_name,
                    "action": "data_access_start",
                    "details": {
                        "reads": list(reads),
                        "writes": list(writes),
                        "description": description,
                        "timestamp": access_start.isoformat()
                    }
                }

                if DataAccessControl.STRICT_MODE:
                    DataAccessControl._validate_access(
                        agent_name=agent_name,
                        reads=reads,
                        writes=writes,
                        state=state
                    )

                try:
                    result = await func(self, state, *args, **kwargs)

                    access_end = datetime.now()
                    duration = (access_end - access_start).total_seconds()

                    if isinstance(result, dict):
                        success_log = {
                            "agent": agent_name,
                            "action": "data_access_end",
                            "details": {
                                "duration_seconds": round(duration, 3),
                                "success": True
                            }
                        }
                        if "execution_log" in result:
                            if isinstance(result["execution_log"], list):
                                result["execution_log"] = result["execution_log"] + [success_log]
                            else:
                                result["execution_log"] = [success_log]
                        else:
                            result["execution_log"] = [success_log]

                    return result

                except AccessDeniedError as e:
                    error_log = {
                        "agent": agent_name,
                        "action": "access_denied",
                        "details": {
                            "error": str(e),
                            "reads": list(reads),
                            "writes": list(writes)
                        }
                    }
                    if isinstance(result := kwargs.get("state", {}), dict):
                        result["execution_log"] = result.get("execution_log", []) + [error_log]
                    raise

            @wraps(func)
            def sync_wrapper(self, state: Dict[str, Any], *args, **kwargs):
                access_start = datetime.now()

                if DataAccessControl.STRICT_MODE:
                    DataAccessControl._validate_access(
                        agent_name=agent_name,
                        reads=reads,
                        writes=writes,
                        state=state
                    )

                try:
                    result = func(self, state, *args, **kwargs)

                    access_end = datetime.now()
                    duration = (access_end - access_start).total_seconds()

                    if isinstance(result, dict):
                        result["execution_log"] = result.get("execution_log", []) + [{
                            "agent": agent_name,
                            "action": "data_access_end",
                            "details": {
                                "duration_seconds": round(duration, 3),
                                "success": True
                            }
                        }]

                    return result

                except AccessDeniedError as e:
                    raise

            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    @staticmethod
    def _validate_access(
        agent_name: str,
        reads: Set[str],
        writes: Set[str],
        state: Dict[str, Any]
    ) -> None:
        """验证访问权限（严格模式）

        Args:
            agent_name: Agent 名称
            reads: 读取的字段
            writes: 写入的字段
            state: 当前状态

        Raises:
            AccessDeniedError: 如果访问违反了所有权规则
        """
        normalized_agent = agent_name.lower()

        for field in reads:
            if field not in DATA_OWNERSHIP_CONFIG:
                logger.debug(f"Field {field} not in config, allowing access")
                continue

            config = DATA_OWNERSHIP_CONFIG[field]

            if "*" not in config.required_by and normalized_agent not in config.required_by:
                raise AccessDeniedError(
                    f"Agent '{agent_name}' is not allowed to read field '{field}'. "
                    f"Required by: {config.required_by}"
                )

        modified_fields = set()
        for key in writes:
            if key in state and key != "execution_log":
                modified_fields.add(key)

        for field in modified_fields:
            if field not in DATA_OWNERSHIP_CONFIG:
                logger.debug(f"Field {field} not in config, allowing write")
                continue

            config = DATA_OWNERSHIP_CONFIG[field]
            write_policy = config.write_policy

            if write_policy == WritePolicy.READ_ONLY:
                raise AccessDeniedError(
                    f"Agent '{agent_name}' cannot write to read-only field '{field}'"
                )

            if write_policy == WritePolicy.OWNER_ONLY:
                if config.owner.value != normalized_agent and config.owner != DataOwner.SHARED:
                    raise AccessDeniedError(
                        f"Agent '{agent_name}' cannot write to field '{field}' "
                        f"owned by '{config.owner.value}'"
                    )

    @staticmethod
    def get_field_info(field_name: str) -> Optional[FieldConfig]:
        """获取字段的所有权和访问策略信息

        Args:
            field_name: 字段名称

        Returns:
            字段配置信息，如果字段不存在则返回 None
        """
        return DATA_OWNERSHIP_CONFIG.get(field_name)

    @staticmethod
    def list_agent_permissions(agent_name: str) -> Dict[str, Set[str]]:
        """列出 Agent 的所有权限

        Args:
            agent_name: Agent 名称

        Returns:
            包含 can_read 和 can_write 集合的字典
        """
        normalized_agent = agent_name.lower()
        can_read = set()
        can_write = set()

        for field_name, config in DATA_OWNERSHIP_CONFIG.items():
            if "*" in config.required_by or normalized_agent in config.required_by:
                can_read.add(field_name)

            write_policy = config.write_policy
            owner_value = config.owner.value

            if write_policy == WritePolicy.READ_ONLY:
                continue
            elif write_policy == WritePolicy.OWNER_ONLY:
                if owner_value == normalized_agent:
                    can_write.add(field_name)
            elif write_policy in [WritePolicy.APPEND_ONLY, WritePolicy.SHARED_WRITE]:
                if owner_value == normalized_agent or config.owner == DataOwner.SHARED:
                    can_write.add(field_name)
            elif write_policy in [WritePolicy.BOOLEAN_FLAG, WritePolicy.EXPLICIT_SWITCH]:
                if owner_value == normalized_agent or config.owner == DataOwner.SHARED:
                    can_write.add(field_name)

        return {
            "can_read": can_read,
            "can_write": can_write
        }

    @staticmethod
    def truncate_history(
        state: Dict[str, Any],
        agent_name: str
    ) -> Dict[str, Any]:
        """截断历史数据（防止状态膨胀）

        Args:
            state: 当前状态
            agent_name: 执行截断的 Agent

        Returns:
            截断后的状态
        """
        result = state.copy()

        for field_name, config in DATA_OWNERSHIP_CONFIG.items():
            if config.max_history is None:
                continue

            if field_name in result and isinstance(result[field_name], list):
                original_len = len(result[field_name])
                if original_len > config.max_history:
                    truncated = result[field_name][-config.max_history:]
                    logger.info(
                        f"Truncated {field_name} from {original_len} to "
                        f"{len(truncated)} items (agent: {agent_name})"
                    )
                    result[field_name] = truncated

        return result

    @staticmethod
    def get_sensitivity_report(state: Dict[str, Any]) -> Dict[str, Any]:
        """生成敏感度报告（用于幻觉风险评估）

        Args:
            state: 当前状态

        Returns:
            敏感度报告
        """
        high_risk_fields = []
        medium_risk_fields = []
        total_docs = 0

        for field_name, value in state.items():
            if field_name not in DATA_OWNERSHIP_CONFIG:
                continue

            config = DATA_OWNERSHIP_CONFIG[field_name]

            if config.sensitivity == ContextSensitivity.HIGH:
                if isinstance(value, list):
                    high_risk_fields.append({
                        "field": field_name,
                        "count": len(value),
                        "sensitivity": "high"
                    })
                    total_docs += len(value)
                else:
                    high_risk_fields.append({
                        "field": field_name,
                        "sensitivity": "high"
                    })

            elif config.sensitivity == ContextSensitivity.MEDIUM:
                if isinstance(value, list):
                    medium_risk_fields.append({
                        "field": field_name,
                        "count": len(value),
                        "sensitivity": "medium"
                    })
                else:
                    medium_risk_fields.append({
                        "field": field_name,
                        "sensitivity": "medium"
                    })

        return {
            "high_risk_fields": high_risk_fields,
            "medium_risk_fields": medium_risk_fields,
            "total_high_risk_docs": total_docs,
            "recommendation": "启用 FactChecker 验证所有高敏感度字段" if high_risk_fields else "状态安全"
        }


def create_audit_log(
    agent: str,
    action: str,
    details: Optional[Dict[str, Any]] = None,
    status: str = "success"
) -> Dict[str, Any]:
    """创建审计日志条目

    Args:
        agent: Agent 名称
        action: 执行的动作
        details: 额外详情
        status: 状态

    Returns:
        审计日志条目
    """
    return {
        "agent": agent,
        "action": action,
        "details": details or {},
        "status": status,
        "timestamp": datetime.now().isoformat()
    }


def create_error_log(
    agent: str,
    action: str,
    error_message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """创建错误日志条目

    Args:
        agent: Agent 名称
        action: 执行的动作
        error_message: 错误信息
        details: 额外详情

    Returns:
        错误日志条目
    """
    return {
        "agent": agent,
        "action": action,
        "details": details or {},
        "status": "error",
        "error_message": error_message,
        "timestamp": datetime.now().isoformat()
    }
