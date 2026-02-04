"""数据访问控制模块

本模块提供数据访问审计和控制机制，用于增强SharedState的数据管理。

核心功能：
1. 数据访问装饰器 - 自动记录Agent的数据访问
2. 数据所有权定义 - 明确每个字段的所有者
3. 访问权限检查 - 可选的运行时权限验证
"""

import logging
from functools import wraps
from typing import Set, Optional, Callable, Any, Dict
from datetime import datetime
from enum import Enum

from .models import SharedState


logger = logging.getLogger(__name__)


class DataOwner(Enum):
    """数据所有者枚举"""
    SYSTEM = "system"           # 系统级数据（只读）
    PLANNER = "planner"         # Planner独占
    NAVIGATOR = "navigator"     # Navigator独占
    DIRECTOR = "director"       # Director独占
    PIVOT_MANAGER = "pivot_manager"  # PivotManager独占
    WRITER = "writer"           # Writer独占
    FACT_CHECKER = "fact_checker"    # FactChecker独占
    COMPILER = "compiler"       # Compiler独占
    SHARED = "shared"           # 多个Agent共享


class WritePolicy(Enum):
    """写入策略枚举"""
    READ_ONLY = "read_only"         # 只读，不允许修改
    OWNER_ONLY = "owner_only"       # 只有所有者可以写入
    APPEND_ONLY = "append_only"     # 只能追加，不能修改
    EXPLICIT_SWITCH = "explicit_switch"  # 显式切换（如技能切换）
    BOOLEAN_FLAG = "boolean_flag"   # 布尔标志（可以设置/清除）
    SHARED_WRITE = "shared_write"   # 多个Agent可以写入


# 数据所有权和访问策略配置
DATA_OWNERSHIP_CONFIG: Dict[str, Dict[str, Any]] = {
    # 系统级只读数据
    "user_topic": {
        "owner": DataOwner.SYSTEM,
        "write_policy": WritePolicy.READ_ONLY,
        "required_by": {DataOwner.PLANNER, DataOwner.COMPILER},
        "description": "用户输入的主题"
    },
    "project_context": {
        "owner": DataOwner.SYSTEM,
        "write_policy": WritePolicy.READ_ONLY,
        "required_by": {DataOwner.PLANNER, DataOwner.COMPILER},
        "description": "项目上下文信息"
    },
    
    # 单一所有者数据
    "outline": {
        "owner": DataOwner.PLANNER,
        "write_policy": WritePolicy.OWNER_ONLY,
        "required_by": {
            DataOwner.NAVIGATOR,
            DataOwner.DIRECTOR,
            DataOwner.WRITER,
            DataOwner.COMPILER
        },
        "description": "剧本大纲"
    },
    "retrieved_docs": {
        "owner": DataOwner.NAVIGATOR,
        "write_policy": WritePolicy.OWNER_ONLY,
        "required_by": {
            DataOwner.DIRECTOR,
            DataOwner.WRITER,
            DataOwner.FACT_CHECKER
        },
        "description": "检索到的文档"
    },
    "fragments": {
        "owner": DataOwner.WRITER,
        "write_policy": WritePolicy.APPEND_ONLY,
        "required_by": {
            DataOwner.FACT_CHECKER,
            DataOwner.COMPILER
        },
        "description": "生成的剧本片段"
    },
    
    # 共享控制数据
    "current_skill": {
        "owner": DataOwner.SHARED,
        "write_policy": WritePolicy.EXPLICIT_SWITCH,
        "required_by": {
            DataOwner.DIRECTOR,
            DataOwner.PIVOT_MANAGER,
            DataOwner.WRITER
        },
        "description": "当前使用的技能"
    },
    "skill_history": {
        "owner": DataOwner.SHARED,
        "write_policy": WritePolicy.APPEND_ONLY,
        "required_by": {DataOwner.COMPILER},
        "description": "技能切换历史"
    },
    "pivot_triggered": {
        "owner": DataOwner.DIRECTOR,
        "write_policy": WritePolicy.BOOLEAN_FLAG,
        "required_by": {DataOwner.PIVOT_MANAGER},
        "description": "转向触发标志"
    },
    "pivot_reason": {
        "owner": DataOwner.DIRECTOR,
        "write_policy": WritePolicy.OWNER_ONLY,
        "required_by": {DataOwner.PIVOT_MANAGER},
        "description": "转向原因"
    },
    
    # 索引和状态
    "current_step_index": {
        "owner": DataOwner.SHARED,
        "write_policy": WritePolicy.SHARED_WRITE,
        "required_by": {
            DataOwner.NAVIGATOR,
            DataOwner.WRITER,
            DataOwner.COMPILER
        },
        "description": "当前步骤索引"
    },
    
    # 控制标志
    "fact_check_passed": {
        "owner": DataOwner.FACT_CHECKER,
        "write_policy": WritePolicy.BOOLEAN_FLAG,
        "required_by": {DataOwner.SHARED},
        "description": "事实检查通过标志"
    },
    "awaiting_user_input": {
        "owner": DataOwner.WRITER,
        "write_policy": WritePolicy.BOOLEAN_FLAG,
        "required_by": {DataOwner.SHARED},
        "description": "等待用户输入标志"
    },
    
    # 日志数据
    "execution_log": {
        "owner": DataOwner.SHARED,
        "write_policy": WritePolicy.APPEND_ONLY,
        "required_by": {DataOwner.SHARED},
        "description": "执行日志"
    }
}


class DataAccessControl:
    """数据访问控制类
    
    提供装饰器和工具方法来管理Agent对GlobalState的访问。
    支持 v2.1 架构的 TypedDict + Reducer 模式。
    """
    
    # 是否启用严格模式（开发时可以启用，生产环境建议关闭）
    # TODO: 逐步完善各Agent的访问权限后，再全局启用
    STRICT_MODE = False
    
    @staticmethod
    def agent_access(
        agent_name: str,
        reads: Optional[Set[str]] = None,
        writes: Optional[Set[str]] = None,
        description: str = ""
    ) -> Callable:
        """数据访问装饰器（v2.1 GlobalState 版本）
        
        自动记录Agent的数据访问，并可选地进行权限检查。
        支持 GlobalState (Dict[str, Any]) 格式。
        
        Args:
            agent_name: Agent名称
            reads: 读取的字段集合
            writes: 写入的字段集合
            description: Agent操作描述
            
        Returns:
            装饰器函数
            
        Example:
            @DataAccessControl.agent_access(
                agent_name="planner",
                reads={"user_topic", "project_context"},
                writes={"outline", "execution_log"}
            )
            async def plan_outline(state: GlobalState, llm_service):
                # ... Agent逻辑
                pass
        """
        reads = reads or set()
        writes = writes or set()
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(self, state: Dict[str, Any], *args, **kwargs):
                access_start = datetime.now()
                
                current_log = state.get("execution_log", [])
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
                                "duration_seconds": duration,
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
                    
                except Exception as e:
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
                                "duration_seconds": duration,
                                "success": True
                            }
                        }]
                    
                    return result
                    
                except Exception as e:
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
            agent_name: Agent名称
            reads: 读取的字段
            writes: 写入的字段
            state: 当前状态
            
        Raises:
            PermissionError: 如果访问违反了所有权规则
        """
        modified_fields = set()
        for key in writes:
            if key in state:
                modified_fields.add(key)
        
        unauthorized = modified_fields - writes - {"execution_log"}
        if unauthorized:
            logger.warning(
                f"Agent {agent_name} modified unauthorized fields: {unauthorized}"
            )
            logger.warning(f"Unknown agent: {agent_name}")
            return
        
        # 检查写入权限
        for field in writes:
            if field not in DATA_OWNERSHIP_CONFIG:
                logger.warning(f"Unknown field: {field}")
                continue
            
            config = DATA_OWNERSHIP_CONFIG[field]
            owner = config["owner"]
            write_policy = config["write_policy"]
            
            # 检查写入策略
            if write_policy == WritePolicy.READ_ONLY:
                raise PermissionError(
                    f"Agent {agent_name} cannot write to read-only field {field}"
                )
            
            if write_policy == WritePolicy.OWNER_ONLY:
                if owner != agent_owner and owner != DataOwner.SHARED:
                    logger.warning(
                        f"Agent {agent_name} is writing to {field} "
                        f"owned by {owner.value}"
                    )
    
    @staticmethod
    def get_field_info(field_name: str) -> Optional[Dict[str, Any]]:
        """获取字段的所有权和访问策略信息
        
        Args:
            field_name: 字段名称
            
        Returns:
            字段配置信息，如果字段不存在则返回None
        """
        return DATA_OWNERSHIP_CONFIG.get(field_name)
    
    @staticmethod
    def list_agent_permissions(agent_name: str) -> Dict[str, Set[str]]:
        """列出Agent的所有权限
        
        Args:
            agent_name: Agent名称
            
        Returns:
            包含can_read和can_write集合的字典
        """
        try:
            agent_owner = DataOwner(agent_name.lower())
        except ValueError:
            return {"can_read": set(), "can_write": set()}
        
        can_read = set()
        can_write = set()
        
        for field_name, config in DATA_OWNERSHIP_CONFIG.items():
            # 检查读取权限
            if agent_owner in config["required_by"] or config["required_by"] == {DataOwner.SHARED}:
                can_read.add(field_name)
            
            # 检查写入权限
            write_policy = config["write_policy"]
            owner = config["owner"]
            
            if write_policy == WritePolicy.READ_ONLY:
                continue
            elif write_policy == WritePolicy.OWNER_ONLY:
                if owner == agent_owner:
                    can_write.add(field_name)
            elif write_policy in [WritePolicy.APPEND_ONLY, WritePolicy.SHARED_WRITE]:
                if owner == agent_owner or owner == DataOwner.SHARED:
                    can_write.add(field_name)
            elif write_policy in [WritePolicy.BOOLEAN_FLAG, WritePolicy.EXPLICIT_SWITCH]:
                if owner == agent_owner or owner == DataOwner.SHARED:
                    can_write.add(field_name)
        
        return {
            "can_read": can_read,
            "can_write": can_write
        }
