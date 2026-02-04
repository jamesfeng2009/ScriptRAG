"""State Types 单元测试

测试 v2.1 架构的状态类型和 Reducer，包括：
- GlobalState TypedDict 定义
- Reducer 函数行为
- 状态合并逻辑
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import Mock
import operator

from src.domain.state_types import (
    GlobalState,
    overwrite_reducer,
    append_only_reducer,
    audit_log_reducer,
    create_success_log,
    create_error_log,
    get_error_message,
    validate_state_consistency,
)


class TestGlobalStateDefinition:
    """GlobalState 定义测试"""

    def test_global_state_required_fields(self):
        """测试必需字段存在"""
        state: GlobalState = {
            "user_topic": "测试主题",
            "project_context": "测试上下文",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        assert state["user_topic"] == "测试主题"
        assert state["outline"] == []
        assert state["fragments"] == []

    def test_global_state_with_optional_fields(self):
        """测试可选字段"""
        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [{"step_id": "1"}],
            "current_step_index": 1,
            "fragments": [{"content": "测试"}],
            "last_retrieved_docs": [{"id": "doc1"}],
            "director_feedback": {"decision": "write"},
            "execution_log": [{"agent": "test", "action": "test"}],
            "error_flag": None,
            "retry_count": 0,
            "skill_history": [{"from": "old", "to": "new"}],
            "current_skill": "screenplay"
        }

        assert "skill_history" in state
        assert state["current_skill"] == "screenplay"


class TestOverwriteReducer:
    """覆盖 Reducer 测试"""

    def test_overwrite_reducer_basic(self):
        """测试基本覆盖行为"""
        old_val = 5
        new_val = 10

        result = overwrite_reducer(old_val, new_val)
        assert result == new_val

    def test_overwrite_reducer_list(self):
        """测试列表覆盖"""
        old_val = [{"id": "1"}]
        new_val = [{"id": "2"}]

        result = overwrite_reducer(old_val, new_val)
        assert result == new_val
        assert result != old_val

    def test_overwrite_reducer_dict(self):
        """测试字典覆盖"""
        old_val = {"key": "old_value"}
        new_val = {"key": "new_value"}

        result = overwrite_reducer(old_val, new_val)
        assert result == new_val


class TestAppendOnlyReducer:
    """追加 Only Reducer 测试"""

    def test_append_only_reducer_empty_list(self):
        """测试空列表追加"""
        old_val: List[Any] = []
        new_val = [{"id": "1"}]

        result = append_only_reducer(old_val, new_val)
        assert result == [{"id": "1"}]

    def test_append_only_reducer_existing_items(self):
        """测试追加到已有列表"""
        old_val = [{"id": "1"}]
        new_val = [{"id": "2"}]

        result = append_only_reducer(old_val, new_val)
        assert len(result) == 2
        assert result[0] == {"id": "1"}
        assert result[1] == {"id": "2"}

    def test_append_only_reducer_multiple_items(self):
        """测试追加多个项目"""
        old_val = [{"id": "1"}]
        new_val = [{"id": "2"}, {"id": "3"}, {"id": "4"}]

        result = append_only_reducer(old_val, new_val)
        assert len(result) == 4
        assert result == [{"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}]

    def test_append_only_reducer_protection(self):
        """测试追加保护（不能修改历史）"""
        old_val = [{"id": "1", "content": "原始"}]
        new_val = [{"id": "2"}]

        result = append_only_reducer(old_val, new_val)

        assert len(old_val) == 1
        assert len(result) == 2
        assert result[0]["id"] == "1"


class TestAuditLogReducer:
    """审计日志 Reducer 测试"""

    def test_audit_log_reducer_empty(self):
        """测试空日志追加"""
        old_val: List[Dict[str, Any]] = []
        new_log = {"agent": "test", "action": "test_action"}

        result = audit_log_reducer(old_val, new_log)
        assert len(result) == 1
        assert result[0]["agent"] == "test"

    def test_audit_log_reducer_append(self):
        """测试日志追加"""
        old_val = [{"agent": "planner", "action": "plan"}]
        new_log = {"agent": "writer", "action": "write"}

        result = audit_log_reducer(old_val, new_log)
        assert len(result) == 2
        assert result[0]["agent"] == "planner"
        assert result[1]["agent"] == "writer"

    def test_audit_log_reducer_preserves_order(self):
        """测试日志顺序保持"""
        old_val = [
            {"agent": "a", "action": "1"},
            {"agent": "b", "action": "2"}
        ]
        new_log = {"agent": "c", "action": "3"}

        result = audit_log_reducer(old_val, new_log)
        assert len(result) == 3
        assert result[0]["agent"] == "a"
        assert result[1]["agent"] == "b"
        assert result[2]["agent"] == "c"


class TestLogCreationFunctions:
    """日志创建函数测试"""

    def test_create_success_log(self):
        """测试成功日志创建"""
        log = create_success_log(
            agent="planner",
            action="generate_outline",
            details={"step_count": 5}
        )

        assert log["agent"] == "planner"
        assert log["action"] == "generate_outline"
        assert log["status"] == "success"
        assert log["details"]["step_count"] == 5
        assert "timestamp" in log
        assert log["timestamp"] is not None

    def test_create_error_log(self):
        """测试错误日志创建"""
        log = create_error_log(
            agent="writer",
            action="generate_fragment",
            error_message="LLM generation failed",
            details={"fragment_id": "frag-123"}
        )

        assert log["agent"] == "writer"
        assert log["action"] == "generate_fragment"
        assert log["status"] == "error"
        assert log["error_message"] == "LLM generation failed"
        assert log["details"]["fragment_id"] == "frag-123"
        assert "timestamp" in log

    def test_create_success_log_with_nested_details(self):
        """测试带嵌套详情成功日志"""
        log = create_success_log(
            agent="navigator",
            action="retrieve",
            details={
                "docs_retrieved": 5,
                "sources": [{"id": "1"}, {"id": "2"}]
            }
        )

        assert log["details"]["docs_retrieved"] == 5
        assert len(log["details"]["sources"]) == 2


class TestStateValidation:
    """状态验证测试"""

    def test_validate_state_consistency_valid(self):
        """测试有效状态验证"""
        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [{"step_id": "1"}, {"step_id": "2"}],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        is_valid, errors = validate_state_consistency(state)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_state_consistency_invalid_index(self):
        """测试无效索引验证"""
        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [{"step_id": "1"}],
            "current_step_index": 5,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        is_valid, errors = validate_state_consistency(state)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_state_consistency_negative_index(self):
        """测试负索引验证"""
        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [{"step_id": "1"}],
            "current_step_index": -1,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        is_valid, errors = validate_state_consistency(state)
        assert is_valid is False


class TestDataClassification:
    """数据分类测试"""

    def test_readonly_context_fields(self):
        """测试只读上下文字段"""
        state: GlobalState = {
            "user_topic": "只读主题",
            "project_context": "只读上下文",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        assert state["user_topic"] == "只读主题"
        assert state["project_context"] == "只读上下文"

    def test_control_plane_fields(self):
        """测试控制平面字段"""
        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [{"step_id": "1"}, {"step_id": "2"}],
            "current_step_index": 1,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": {"decision": "write"},
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        assert state["current_step_index"] == 1
        assert state["director_feedback"]["decision"] == "write"

    def test_data_plane_fields(self):
        """测试数据平面字段"""
        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [],
            "current_step_index": 0,
            "fragments": [
                {"step_id": "1", "content": "第一部分"},
                {"step_id": "2", "content": "第二部分"}
            ],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        assert len(state["fragments"]) == 2


class TestReducerWithOperator:
    """Operator Reducer 测试"""

    def test_operator_add_for_fragments(self):
        """测试 fragments 使用 operator.add"""
        old_val = [{"content": "旧片段"}]
        new_val = [{"content": "新片段"}]

        result = operator.add(old_val, new_val)
        assert len(result) == 2

    def test_operator_add_immutability(self):
        """测试 operator.add 不可变性"""
        old_val = [{"id": "1"}]
        new_val = [{"id": "2"}]

        result = operator.add(old_val, new_val)

        assert len(old_val) == 1
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
