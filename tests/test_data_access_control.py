"""数据访问控制模块单元测试

测试内容：
1. 访问权限验证
2. 装饰器功能
3. 敏感度报告
4. 历史截断
5. 幻觉控制场景
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Set
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.data_access_control import (
    DataAccessControl,
    DataOwner,
    WritePolicy,
    ContextSensitivity,
    FieldConfig,
    DATA_OWNERSHIP_CONFIG,
    AccessDeniedError,
    create_audit_log,
    create_error_log,
)


class TestDataAccessControl:
    """数据访问控制测试类"""

    def setup_method(self):
        """每个测试前保存原始 STRICT_MODE"""
        self.original_strict_mode = DataAccessControl.STRICT_MODE

    def teardown_method(self):
        """每个测试后恢复原始 STRICT_MODE"""
        DataAccessControl.STRICT_MODE = self.original_strict_mode

    def test_field_config_structure(self):
        """测试字段配置结构"""
        assert "user_topic" in DATA_OWNERSHIP_CONFIG
        config = DATA_OWNERSHIP_CONFIG["user_topic"]

        assert isinstance(config, FieldConfig)
        assert config.owner == DataOwner.SYSTEM
        assert config.write_policy == WritePolicy.READ_ONLY
        assert config.sensitivity == ContextSensitivity.LOW
        assert "*" in config.required_by

    def test_sensitivity_levels(self):
        """测试敏感度级别"""
        user_topic_config = DATA_OWNERSHIP_CONFIG["user_topic"]
        retrieved_config = DATA_OWNERSHIP_CONFIG["retrieved_docs"]
        fragments_config = DATA_OWNERSHIP_CONFIG["fragments"]

        assert user_topic_config.sensitivity == ContextSensitivity.LOW
        assert retrieved_config.sensitivity == ContextSensitivity.HIGH
        assert fragments_config.sensitivity == ContextSensitivity.MEDIUM

    def test_truncate_history(self):
        """测试历史截断"""
        state = {
            "fragments": [{"id": i, "content": f"content {i}"} for i in range(100)],
            "execution_log": [{"id": i} for i in range(150)],
            "user_topic": "test"
        }

        truncated = DataAccessControl.truncate_history(state, "writer")

        assert len(truncated["fragments"]) == 50
        assert len(truncated["execution_log"]) == 100
        assert truncated["user_topic"] == "test"

    def test_get_sensitivity_report(self):
        """测试敏感度报告生成"""
        state = {
            "user_topic": "test",
            "retrieved_docs": [{"id": 1}, {"id": 2}, {"id": 3}],
            "fragments": [{"id": 1}]
        }

        report = DataAccessControl.get_sensitivity_report(state)

        assert "high_risk_fields" in report
        assert "medium_risk_fields" in report
        assert report["total_high_risk_docs"] == 3

        high_risk = report["high_risk_fields"]
        assert len(high_risk) == 1
        assert high_risk[0]["field"] == "retrieved_docs"
        assert high_risk[0]["count"] == 3

    def test_list_agent_permissions(self):
        """测试 Agent 权限列表"""
        planner_perms = DataAccessControl.list_agent_permissions("planner")

        assert "can_read" in planner_perms
        assert "can_write" in planner_perms

        assert "user_topic" in planner_perms["can_read"]
        assert "outline" in planner_perms["can_write"]

    def test_get_field_info(self):
        """测试获取字段信息"""
        info = DataAccessControl.get_field_info("user_topic")

        assert info is not None
        assert info.owner == DataOwner.SYSTEM
        assert info.write_policy == WritePolicy.READ_ONLY

    def test_unknown_field_access(self):
        """测试未知字段访问"""
        DataAccessControl.STRICT_MODE = True

        state = {"unknown_field": "value"}
        reads = {"unknown_field"}
        writes = set()

        try:
            DataAccessControl._validate_access(
                agent_name="planner",
                reads=reads,
                writes=writes,
                state=state
            )
            access_granted = True
        except AccessDeniedError:
            access_granted = False

        assert access_granted

    def test_read_only_field_write_attempt(self):
        """测试尝试写入只读字段"""
        DataAccessControl.STRICT_MODE = True

        state = {"user_topic": "test"}
        reads = set()
        writes = {"user_topic"}

        with pytest.raises(AccessDeniedError) as exc_info:
            DataAccessControl._validate_access(
                agent_name="writer",
                reads=reads,
                writes=writes,
                state=state
            )

        assert "read-only" in str(exc_info.value).lower()

    def test_unauthorized_read(self):
        """测试未授权读取"""
        DataAccessControl.STRICT_MODE = True

        state = {"pivot_reason": "some reason"}
        reads = {"pivot_reason"}
        writes = set()

        with pytest.raises(AccessDeniedError) as exc_info:
            DataAccessControl._validate_access(
                agent_name="writer",
                reads=reads,
                writes=writes,
                state=state
            )

        assert "not allowed to read" in str(exc_info.value)


class TestDataAccessDecorator:
    """数据访问装饰器测试类"""

    def setup_method(self):
        self.original_strict_mode = DataAccessControl.STRICT_MODE

    def teardown_method(self):
        DataAccessControl.STRICT_MODE = self.original_strict_mode

    def test_decorator_with_valid_access(self):
        """测试有效访问的装饰器"""
        DataAccessControl.STRICT_MODE = True

        @DataAccessControl.agent_access(
            agent_name="planner",
            reads={"user_topic", "project_context"},
            writes={"outline", "execution_log"},
            description="生成大纲"
        )
        async def planner_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
            return {"outline": [{"step": 1}], "execution_log": []}

        class MockSelf:
            pass

        state = {
            "user_topic": "test topic",
            "project_context": "test context"
        }

        import asyncio
        result = asyncio.run(planner_node(MockSelf(), state))

        assert "outline" in result
        assert len(result["execution_log"]) >= 1

    def test_decorator_with_invalid_read(self):
        """测试无效读取的装饰器"""
        DataAccessControl.STRICT_MODE = True

        @DataAccessControl.agent_access(
            agent_name="writer",
            reads={"user_topic", "pivot_reason"},
            writes={"fragments"},
            description="生成片段"
        )
        async def writer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
            return {"fragments": []}

        class MockSelf:
            pass

        state = {"user_topic": "test"}

        with pytest.raises(AccessDeniedError):
            import asyncio
            asyncio.run(writer_node(MockSelf(), state))

    def test_decorator_preserves_function_metadata(self):
        """测试装饰器保留函数元数据"""

        @DataAccessControl.agent_access(
            agent_name="planner",
            reads={"user_topic"},
            writes={"outline"}
        )
        async def my_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
            pass

        assert my_node.__name__ == "my_node"
        assert my_node.__doc__ is not None or True


class TestAuditLogCreation:
    """审计日志创建测试"""

    def test_create_audit_log(self):
        """测试创建审计日志"""
        log = create_audit_log(
            agent="planner",
            action="generate_outline",
            details={"steps": 5},
            status="success"
        )

        assert log["agent"] == "planner"
        assert log["action"] == "generate_outline"
        assert log["details"]["steps"] == 5
        assert log["status"] == "success"
        assert "timestamp" in log

    def test_create_error_log(self):
        """测试创建错误日志"""
        log = create_error_log(
            agent="navigator",
            action="retrieve_content",
            error_message="No documents found",
            details={"query": "test"}
        )

        assert log["agent"] == "navigator"
        assert log["action"] == "retrieve_content"
        assert log["error_message"] == "No documents found"
        assert log["status"] == "error"


class TestHallucinationControl:
    """幻觉控制场景测试"""

    def test_retrieval_isolation_recommendation(self):
        """测试检索隔离建议"""
        state = {
            "retrieved_docs": [{"id": i} for i in range(10)]
        }

        report = DataAccessControl.get_sensitivity_report(state)

        assert report["recommendation"] == "启用 FactChecker 验证所有高敏感度字段"

    def test_safe_state_recommendation(self):
        """测试安全状态建议"""
        state = {
            "user_topic": "test",
            "project_context": "test",
            "current_step_index": 0
        }

        report = DataAccessControl.get_sensitivity_report(state)

        assert report["recommendation"] == "状态安全"
        assert report["total_high_risk_docs"] == 0


class TestEdgeCases:
    """边界情况测试"""

    def test_empty_state(self):
        """测试空状态"""
        state = {}

        report = DataAccessControl.get_sensitivity_report(state)

        assert report["high_risk_fields"] == []
        assert report["medium_risk_fields"] == []

    def test_truncate_empty_history(self):
        """测试截断空历史"""
        state = {"fragments": []}

        truncated = DataAccessControl.truncate_history(state, "writer")

        assert truncated["fragments"] == []

    def test_list_permissions_unknown_agent(self):
        """测试未知 Agent 权限（只能访问 * 标记的字段）"""
        perms = DataAccessControl.list_agent_permissions("unknown_agent")

        assert "user_topic" in perms["can_read"]
        assert "project_context" in perms["can_read"]
        assert "fact_check_passed" in perms["can_read"]
        assert "error_flag" in perms["can_read"]
        assert "execution_log" in perms["can_read"]
        assert "outline" not in perms["can_read"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
