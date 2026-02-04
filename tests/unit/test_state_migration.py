"""State Migration 单元测试

测试状态迁移工具，包括：
- 基础迁移函数测试
"""

import pytest
from typing import Dict, Any, List

from src.domain.state_types import GlobalState
from src.domain.state_migration import (
    to_global_state,
    from_global_state,
)


class TestMigrationBasics:
    """迁移基础测试"""

    def test_to_global_state_empty(self):
        """测试空状态转换"""
        result = to_global_state({})
        assert isinstance(result, dict)

    def test_from_global_state_empty(self):
        """测试空 GlobalState 转换"""
        empty_state: GlobalState = {
            "user_topic": "",
            "project_context": "",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }
        result = from_global_state(empty_state)
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
