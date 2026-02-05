"""
LangGraph 状态类型定义（v2.1）

本模块定义了工作流使用的所有 TypedDict 类型和 Reducer 函数。
遵循 v2.1 架构规范：https://docs/architecture/v2.1_architecture_spec.md

核心设计：
- GlobalState: 工作流全局状态，支持 TypedDict 和 Annotated Reducer
- InitialState: 初始状态类型
- 数据分类：只读上下文、核心控制数据、生产数据、临时数据、审计日志

使用示例：
    from src.domain.state_types import GlobalState, navigator_node
    
    app = workflow.compile()
    result = app.invoke({
        "user_topic": "如何学习 Python",
        "project_context": "从零开始学习编程",
        "outline": [],
        "fragments": [],
        ...
    })
"""

import operator
import datetime
from typing import (
    Annotated,
    List,
    Optional,
    Dict,
    Any,
    TypedDict,
    Union,
)


# ============================================================================
# 自定义 Reducer
# ============================================================================


def audit_log_reducer(current: List[Dict], new: Union[Dict, List[Dict], str]) -> List[Dict]:
    """审计日志追加器
    
    功能：
    1. 支持多种输入格式（Dict、List、str），统一转换为 Dict 列表
    2. 自动添加时间戳（如果尚未添加）
    3. 将新条目追加到当前列表末尾
    
    设计考量：
    - 使用 Union 类型支持灵活输入
    - 自动处理时间戳，避免重复代码
    - 返回 Dict 列表而非字符串，保持类型一致性
    
    Args:
        current: 当前日志列表
        new: 新增的日志条目
        
    Returns:
        合并后的日志列表
    """
    if isinstance(new, str):
        new_entries = [{"raw": new}]
    elif isinstance(new, list):
        new_entries = [
            item if isinstance(item, dict) else {"raw": str(item)}
            for item in new
        ]
    elif isinstance(new, dict):
        new_entries = [new]
    else:
        new_entries = [{"raw": str(new)}]
    
    timestamp = datetime.datetime.now().isoformat()
    for entry in new_entries:
        if "timestamp" not in entry:
            entry["timestamp"] = timestamp
    
    return current + new_entries


def overwrite_reducer(current: Any, new: Any) -> Any:
    """标准覆盖逻辑
    
    功能说明：
    1. 直接返回新值，覆盖当前值
    2. 作为扩展点预留，支持未来添加 Diff 检查、快照等功能
    
    可扩展功能（未来）：
    - Diff 检查：跳过无意义更新，减少状态变更
    - 快照：覆盖前自动保存历史，支持回滚
    - 验证：校验 new 的合法性，防止脏数据
    
    Args:
        current: 当前值
        new: 新值
        
    Returns:
        新值（覆盖当前值）
    """
    return new


def append_only_reducer(current: List[Dict], new: Union[Dict, List[Dict]]) -> List[Dict]:
    """通用追加 Reducer（仅追加，不修改历史）
    
    与 audit_log_reducer 的区别：
    - 不自动添加时间戳
    - 不处理字符串输入
    - 纯粹的追加操作
    
    适用场景：
    - fragments：仅追加新的剧本片段
    - skill_history：仅追加技能切换历史
    
    Args:
        current: 当前列表
        new: 新增条目
        
    Returns:
        合并后的列表
    """
    if isinstance(new, dict):
        return current + [new]
    elif isinstance(new, list):
        return current + new
    return current


# ============================================================================
# 全局状态定义（v2.1）
# ============================================================================


class GlobalState(TypedDict):
    """
    工作流全局状态
    
    数据分类体系：
    ┌─────────────────────────────────────────────────────────────┐
    │ 1. 只读上下文 (Immutable Context)                           │
    │    user_topic, project_context                              │
    │    特点：初始化后不变，所有节点只读                          │
    ├─────────────────────────────────────────────────────────────┤
    │ 2. 核心控制数据 (Control Plane - Overwrite)                 │
    │    outline, current_step_index                              │
    │    特点：Planner 和 PivotManager 需要修改                  │
    │    保护：使用 overwrite_reducer                             │
    ├─────────────────────────────────────────────────────────────┤
    │ 3. 生产数据 (Data Plane - Append Only ⭐)                  │
    │    fragments                                                │
    │    特点：Writer 只能追加，无法修改或删除历史片段             │
    │    保护：operator.add 实现物理级追加保护                     │
    ├─────────────────────────────────────────────────────────────┤
    │ 4. 临时交互数据 (Transient - Overwrite)                     │
    │    last_retrieved_docs, director_feedback                   │
    │    特点：每次节点流转时覆盖，仅供当次使用                    │
    │    保护：overwrite_reducer                                  │
    ├─────────────────────────────────────────────────────────────┤
    │ 5. 审计日志 (Audit - Append Only ⭐)                        │
    │    execution_log                                            │
    │    特点：所有节点追加，记录完整执行历史                      │
    │    保护：audit_log_reducer 自动添加元数据                   │
    ├─────────────────────────────────────────────────────────────┤
    │ 6. 错误处理标记 (Optional)                                  │
    │    error_flag, retry_count                                 │
    │    特点：用于错误处理和重试逻辑                              │
    │    保护：overwrite_reducer                                  │
    └─────────────────────────────────────────────────────────────┘
    
    设计考量：
    - 字段命名使用 snake_case，符合 Python 惯例
    - 使用 Dict[str, Any] 而非具体类型，提高序列化兼容性
    - 明确区分「可覆盖」和「只追加」字段
    """
    
    # ============================================================
    # 1. 只读上下文 (Immutable Context)
    # ============================================================
    user_topic: str
    project_context: str
    
    # ============================================================
    # 2. 核心控制数据 (Control Plane - Overwrite)
    # ============================================================
    outline: Annotated[List[Dict[str, Any]], overwrite_reducer]
    current_step_index: Annotated[int, overwrite_reducer]
    
    # ============================================================
    # 3. 生产数据 (Data Plane - Append Only ⭐)
    # ============================================================
    fragments: Annotated[List[Dict[str, Any]], operator.add]
    
    # ============================================================
    # 4. 临时交互数据 (Transient - Overwrite)
    # ============================================================
    last_retrieved_docs: Annotated[List[Dict[str, Any]], overwrite_reducer]
    director_feedback: Annotated[Optional[Dict[str, Any]], overwrite_reducer]
    fact_check_passed: Annotated[Optional[bool], overwrite_reducer]
    
    # ============================================================
    # 5. 审计日志 (Audit - Append Only ⭐)
    # ============================================================
    execution_log: Annotated[List[Dict[str, Any]], audit_log_reducer]
    
    # ============================================================
    # 6. 错误处理标记 (Optional)
    # ============================================================
    error_flag: Annotated[Optional[str], overwrite_reducer]
    retry_count: Annotated[int, overwrite_reducer]
    workflow_complete: Annotated[Optional[bool], overwrite_reducer]
    
    # ============================================================
    # 7. 转向控制数据 (Control Plane - Overwrite)
    # ============================================================
    pivot_triggered: Annotated[bool, overwrite_reducer]
    pivot_reason: Annotated[Optional[str], overwrite_reducer]
    
    # ============================================================
    # 8. 最终输出 (Output - Overwrite)
    # ============================================================
    final_screenplay: Annotated[Optional[str], overwrite_reducer]


class InitialState(TypedDict):
    """初始状态类型
    
    用于工作流启动时的状态创建。
    只包含必须初始化的字段。
    """
    user_topic: str
    project_context: str


class CheckpointState(TypedDict):
    """Checkpoint 状态类型
    
    用于 LangGraph Checkpoint 机制。
    """
    global_state: GlobalState
    checkpoint_id: str
    parent_checkpoint_id: Optional[str]
    timestamp: str


class ToolCall(TypedDict):
    """工具调用类型
    
    用于 Function Calling 流程中的工具调用。
    """
    id: str
    type: str
    function: Dict[str, str]


class ToolResult(TypedDict):
    """工具执行结果类型
    
    用于 Function Calling 流程中的工具执行结果。
    """
    tool_call_id: str
    name: str
    content: Optional[Dict[str, Any]]
    success: bool
    error: Optional[str]


class HumanIntervention(TypedDict):
    """人工干预请求类型
    
    用于用户干预流程。
    """
    type: str
    prompt: str
    choices: Optional[List[str]]
    requested_at: str
    completed_at: Optional[str]


# ============================================================================
# 辅助函数
# ============================================================================


def create_initial_state(
    user_topic: str,
    project_context: str = ""
) -> GlobalState:
    """创建初始状态
    
    Args:
        user_topic: 用户主题
        project_context: 项目上下文
        
    Returns:
        初始化的 GlobalState
    """
    return {
        "user_topic": user_topic,
        "project_context": project_context,
        "outline": [],
        "current_step_index": 0,
        "fragments": [],
        "last_retrieved_docs": [],
        "director_feedback": None,
        "execution_log": [],
        "error_flag": None,
        "retry_count": 0,
    }


def create_error_log(
    agent: str,
    action: str,
    error_message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """创建错误日志条目
    
    统一错误日志格式。
    
    Args:
        agent: 智能体名称
        action: 执行的动作
        error_message: 错误信息
        details: 额外详情
        
    Returns:
        标准错误日志条目
    """
    from datetime import datetime
    return {
        "agent": agent,
        "action": action,
        "details": details or {},
        "status": "error",
        "error_message": error_message,
        "timestamp": datetime.now().isoformat(),
    }


def create_success_log(
    agent: str,
    action: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """创建成功日志条目
    
    Args:
        agent: 智能体名称
        action: 执行的动作
        details: 额外详情
        
    Returns:
        标准成功日志条目
    """
    from datetime import datetime
    return {
        "agent": agent,
        "action": action,
        "details": details or {},
        "status": "success",
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# 错误代码定义
# ============================================================================

ERROR_CODES = {
    "boundary_error": "步骤索引越界",
    "retrieval_error": "文档检索失败",
    "llm_error": "LLM 调用失败",
    "validation_error": "数据验证失败",
    "timeout_error": "节点执行超时",
    "unknown_error": "未知错误",
}


def validate_state_consistency(state: GlobalState) -> tuple[bool, list[str]]:
    """验证 GlobalState 的一致性
    
    Args:
        state: 要验证的全局状态
        
    Returns:
        tuple: (is_valid, errors) - 是否有效，错误列表
    """
    errors: list[str] = []
    
    outline = state.get("outline", [])
    current_step_index = state.get("current_step_index", 0)
    
    if current_step_index < 0:
        errors.append(f"current_step_index 不能为负数: {current_step_index}")
    
    if current_step_index >= len(outline):
        errors.append(
            f"current_step_index ({current_step_index}) 超出 outline 范围 "
            f"(outline 长度: {len(outline)})"
        )
    
    fragments = state.get("fragments", [])
    if fragments:
        if not outline:
            errors.append("存在 fragments 但 outline 为空")
    
    return len(errors) == 0, errors


def get_error_message(error_code: str) -> str:
    """获取错误代码对应的消息
    
    Args:
        error_code: 错误代码
        
    Returns:
        错误消息
    """
    return ERROR_CODES.get(error_code, ERROR_CODES["unknown_error"])
