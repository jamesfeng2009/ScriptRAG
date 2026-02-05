"""
Unit tests for Planner Agent

测试规划器智能体的大纲生成功能。
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.domain.agents.task_aware_planner import (
    TaskAwarePlanner,
    create_planner_with_task_stack,
    _build_planning_prompt,
    _parse_outline_response,
    _pad_outline,
    _create_fallback_outline
)


class TestPlannerAgent:
    """测试规划器智能体"""
    
    @pytest.mark.asyncio
    async def test_plan_outline_success(self):
        """测试成功生成大纲"""
        state = {
            "user_topic": "如何使用 Python 实现 REST API",
            "project_context": "使用 FastAPI 框架",
            "outline": [],
            "execution_log": []
        }
        
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
步骤1: 介绍 FastAPI 框架的基本概念 | 关键词: FastAPI, ASGI
步骤2: 安装和配置 FastAPI 环境 | 关键词: pip, uvicorn
步骤3: 创建第一个 API 端点 | 关键词: 路由, 装饰器
步骤4: 请求和响应处理 | 关键词: Pydantic, 验证
步骤5: 数据库集成 | 关键词: SQLAlchemy, ORM
步骤6: 认证和授权 | 关键词: JWT, OAuth2
步骤7: 测试和部署 | 关键词: pytest, Docker
        """)
        
        planner = create_planner_with_task_stack(use_task_stack=False)
        result_state = await planner.plan_outline(state, llm_service)
        
        assert len(result_state["outline"]) == 7
        assert result_state["current_step_index"] == 0
        assert all(step["status"] == "pending" for step in result_state["outline"])
        assert all(step.get("retry_count", 0) == 0 for step in result_state["outline"])
        
        assert "FastAPI" in result_state["outline"][0]["description"]
        assert "安装" in result_state["outline"][1]["description"]
        
        log_entries = [log for log in result_state.get("execution_log", []) if isinstance(log, dict)]
        assert len(log_entries) >= 2
        planner_logs = [log for log in log_entries if log.get("agent_name") == "planner" or log.get("agent") == "planner"]
        assert len(planner_logs) >= 1
    
    @pytest.mark.asyncio
    async def test_plan_outline_minimum_steps(self):
        """测试生成的大纲至少有 5 步"""
        state = {
            "user_topic": "简单主题",
            "project_context": "",
            "outline": [],
            "execution_log": []
        }
        
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
步骤1: 介绍主题
步骤2: 核心概念
步骤3: 实践示例
        """)
        
        planner = create_planner_with_task_stack(use_task_stack=False)
        result_state = await planner.plan_outline(state, llm_service)
        
        assert len(result_state["outline"]) >= 5
    
    @pytest.mark.asyncio
    async def test_plan_outline_maximum_steps(self):
        """测试生成的大纲最多有 10 步"""
        state = {
            "user_topic": "复杂主题",
            "project_context": "",
            "outline": [],
            "execution_log": []
        }
        
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
步骤1: 步骤 1
步骤2: 步骤 2
步骤3: 步骤 3
步骤4: 步骤 4
步骤5: 步骤 5
步骤6: 步骤 6
步骤7: 步骤 7
步骤8: 步骤 8
步骤9: 步骤 9
步骤10: 步骤 10
步骤11: 步骤 11
步骤12: 步骤 12
        """)
        
        planner = create_planner_with_task_stack(use_task_stack=False)
        result_state = await planner.plan_outline(state, llm_service)
        
        assert len(result_state["outline"]) <= 10
    
    @pytest.mark.asyncio
    async def test_plan_outline_fallback_on_error(self):
        """测试 LLM 调用失败时使用回退大纲"""
        state = {
            "user_topic": "测试主题",
            "project_context": "",
            "outline": [],
            "execution_log": []
        }
        
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(side_effect=Exception("LLM service unavailable"))
        
        planner = create_planner_with_task_stack(use_task_stack=False)
        result_state = await planner.plan_outline(state, llm_service)
        
        assert len(result_state["outline"]) == 5
        assert "测试主题" in result_state["outline"][0]["description"]
        
        log_entries = [log for log in result_state.get("execution_log", []) if isinstance(log, dict)]
        error_logs = [log for log in log_entries if log.get("action") == "planning_failed"]
        assert len(error_logs) > 0


class TestPlanningPrompt:
    """测试提示词构建"""
    
    def test_build_planning_prompt_with_context(self):
        """测试带项目上下文的提示词"""
        prompt = _build_planning_prompt(
            user_topic="如何使用 Python 实现 REST API",
            project_context="使用 FastAPI 框架"
        )
        
        assert "如何使用 Python 实现 REST API" in prompt
        assert "使用 FastAPI 框架" in prompt
        assert "5-10 个步骤" in prompt
    
    def test_build_planning_prompt_without_context(self):
        """测试不带项目上下文的提示词"""
        prompt = _build_planning_prompt(
            user_topic="Python 基础",
            project_context=""
        )
        
        assert "Python 基础" in prompt
        assert "无特定上下文" in prompt


class TestOutlineResponseParsing:
    """测试大纲响应解析"""
    
    def test_parse_outline_response_standard_format(self):
        """测试标准格式的响应解析"""
        response = """
步骤1: 介绍 FastAPI 框架 | 关键词: FastAPI, ASGI
步骤2: 安装和配置环境 | 关键词: pip, uvicorn
步骤3: 创建第一个 API | 关键词: 路由, 装饰器
        """
        
        steps = _parse_outline_response(response)
        
        assert len(steps) == 3
        assert steps[0]["step_id"] == 0
        assert "FastAPI" in steps[0]["description"]
        assert steps[1]["step_id"] == 1
        assert "安装" in steps[1]["description"]
    
    def test_parse_outline_response_without_keywords(self):
        """测试不带关键词的响应解析"""
        response = """
步骤1: 介绍主题
步骤2: 核心概念
步骤3: 实践示例
        """
        
        steps = _parse_outline_response(response)
        
        assert len(steps) == 3
        assert steps[0]["description"] == "介绍主题"
        assert steps[1]["description"] == "核心概念"
    
    def test_parse_outline_response_with_noise(self):
        """测试带噪声的响应解析"""
        response = """
这是一些额外的文本
步骤1: 第一步
一些中间文本
步骤2: 第二步
更多噪声
步骤3: 第三步
        """
        
        steps = _parse_outline_response(response)
        
        assert len(steps) == 3
        assert steps[0]["description"] == "第一步"
    
    def test_parse_outline_response_empty(self):
        """测试空响应解析"""
        response = ""
        
        steps = _parse_outline_response(response)
        
        assert len(steps) == 0


class TestOutlinePadding:
    """测试大纲填充"""
    
    def test_pad_outline_to_minimum(self):
        """测试填充到最小步骤数"""
        steps = [
            {"step_id": 0, "description": "步骤 1", "status": "pending"},
            {"step_id": 1, "description": "步骤 2", "status": "pending"}
        ]
        
        padded_steps = _pad_outline(steps, 5)
        
        assert len(padded_steps) == 5
        assert padded_steps[0]["description"] == "步骤 1"
        assert "补充步骤" in padded_steps[2]["description"]
    
    def test_pad_outline_no_padding_needed(self):
        """测试不需要填充的情况"""
        steps = [
            {"step_id": i, "description": f"步骤 {i+1}", "status": "pending"}
            for i in range(7)
        ]
        
        padded_steps = _pad_outline(steps, 5)
        
        assert len(padded_steps) == 7
        assert padded_steps == steps


class TestFallbackOutline:
    """测试回退大纲"""
    
    def test_create_fallback_outline(self):
        """测试创建回退大纲"""
        user_topic = "测试主题"
        
        steps = _create_fallback_outline(user_topic)
        
        assert len(steps) == 5
        assert steps[0]["step_id"] == 0
        assert user_topic in steps[0]["description"]
        assert all(step["status"] == "pending" for step in steps)
        assert all(step.get("retry_count", 0) == 0 for step in steps)
    
    def test_fallback_outline_structure(self):
        """测试回退大纲的结构"""
        steps = _create_fallback_outline("任意主题")
        
        descriptions = [step["description"] for step in steps]
        
        assert any("介绍" in desc for desc in descriptions)
        assert any("概念" in desc for desc in descriptions)
        assert any("实践" in desc or "示例" in desc for desc in descriptions)
        assert any("总结" in desc for desc in descriptions)
