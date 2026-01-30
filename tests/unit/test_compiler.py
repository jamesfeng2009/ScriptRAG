"""
Unit tests for Compiler Agent

测试编译器智能体的片段集成功能。
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.domain.models import SharedState, OutlineStep, ScreenplayFragment
from src.domain.agents.compiler import (
    compile_screenplay,
    _build_compilation_prompt,
    _generate_empty_screenplay,
    _fallback_compilation
)


class TestCompilerAgent:
    """测试编译器智能体"""
    
    @pytest.mark.asyncio
    async def test_compile_screenplay_success(self):
        """测试成功编译剧本"""
        # 准备测试数据
        state = SharedState(
            user_topic="如何使用 Python 实现 REST API",
            project_context="使用 FastAPI 框架",
            outline=[
                OutlineStep(step_id=0, description="介绍 FastAPI", status="completed"),
                OutlineStep(step_id=1, description="创建 API 端点", status="completed"),
                OutlineStep(step_id=2, description="测试和部署", status="completed")
            ],
            fragments=[
                ScreenplayFragment(
                    step_id=0,
                    content="FastAPI 是一个现代、快速的 Web 框架...",
                    skill_used="standard_tutorial",
                    sources=["docs/fastapi.md"]
                ),
                ScreenplayFragment(
                    step_id=1,
                    content="创建 API 端点非常简单，使用装饰器...",
                    skill_used="standard_tutorial",
                    sources=["examples/api.py"]
                ),
                ScreenplayFragment(
                    step_id=2,
                    content="使用 pytest 进行测试，Docker 进行部署...",
                    skill_used="standard_tutorial",
                    sources=["docs/testing.md", "docs/deployment.md"]
                )
            ]
        )
        
        # 模拟 LLM 服务
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
# 如何使用 Python 实现 REST API

## 引言

本剧本将指导您使用 FastAPI 框架实现 REST API。

## 第一部分：介绍 FastAPI

FastAPI 是一个现代、快速的 Web 框架...

## 第二部分：创建 API 端点

创建 API 端点非常简单，使用装饰器...

## 第三部分：测试和部署

使用 pytest 进行测试，Docker 进行部署...

## 结论

通过本剧本，您已经学会了如何使用 FastAPI 创建 REST API。
        """)
        
        # 执行测试
        final_screenplay = await compile_screenplay(state, llm_service)
        
        # 验证结果
        assert final_screenplay is not None
        assert len(final_screenplay) > 0
        assert "如何使用 Python 实现 REST API" in final_screenplay
        assert "FastAPI" in final_screenplay
        
        # 验证 LLM 调用
        llm_service.chat_completion.assert_called_once()
        call_args = llm_service.chat_completion.call_args
        assert call_args.kwargs["task_type"] == "lightweight"
        
        # 验证日志记录
        assert len(state.execution_log) >= 2
        assert state.execution_log[0]["agent_name"] == "compiler"
        assert state.execution_log[0]["action"] == "start_compilation"
    
    @pytest.mark.asyncio
    async def test_compile_screenplay_empty_fragments(self):
        """测试没有片段时的编译"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试上下文",
            outline=[],
            fragments=[]
        )
        
        llm_service = MagicMock()
        
        # 执行测试
        final_screenplay = await compile_screenplay(state, llm_service)
        
        # 验证返回空剧本
        assert final_screenplay is not None
        assert "测试主题" in final_screenplay
        assert "缺少足够的信息" in final_screenplay
        
        # 验证没有调用 LLM
        llm_service.chat_completion.assert_not_called()
        
        # 验证日志记录
        skip_logs = [log for log in state.execution_log if log["action"] == "compilation_skipped"]
        assert len(skip_logs) > 0
    
    @pytest.mark.asyncio
    async def test_compile_screenplay_fallback_on_error(self):
        """测试 LLM 调用失败时使用降级编译"""
        state = SharedState(
            user_topic="测试主题",
            project_context="",
            outline=[
                OutlineStep(step_id=0, description="步骤 1", status="completed"),
                OutlineStep(step_id=1, description="步骤 2", status="completed")
            ],
            fragments=[
                ScreenplayFragment(
                    step_id=0,
                    content="片段 1 内容",
                    skill_used="standard_tutorial",
                    sources=["source1.md"]
                ),
                ScreenplayFragment(
                    step_id=1,
                    content="片段 2 内容",
                    skill_used="standard_tutorial",
                    sources=["source2.md"]
                )
            ]
        )
        
        # 模拟 LLM 服务失败
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(side_effect=Exception("LLM service unavailable"))
        
        # 执行测试
        final_screenplay = await compile_screenplay(state, llm_service)
        
        # 验证使用了降级编译
        assert final_screenplay is not None
        assert "测试主题" in final_screenplay
        assert "片段 1 内容" in final_screenplay
        assert "片段 2 内容" in final_screenplay
        
        # 验证错误日志
        error_logs = [log for log in state.execution_log if log["action"] == "compilation_failed"]
        assert len(error_logs) > 0
    
    @pytest.mark.asyncio
    async def test_compile_screenplay_fragment_ordering(self):
        """测试片段按 step_id 排序"""
        # 准备乱序的片段
        state = SharedState(
            user_topic="测试主题",
            project_context="",
            outline=[
                OutlineStep(step_id=0, description="步骤 1", status="completed"),
                OutlineStep(step_id=1, description="步骤 2", status="completed"),
                OutlineStep(step_id=2, description="步骤 3", status="completed")
            ],
            fragments=[
                ScreenplayFragment(
                    step_id=2,
                    content="第三个片段",
                    skill_used="standard_tutorial",
                    sources=[]
                ),
                ScreenplayFragment(
                    step_id=0,
                    content="第一个片段",
                    skill_used="standard_tutorial",
                    sources=[]
                ),
                ScreenplayFragment(
                    step_id=1,
                    content="第二个片段",
                    skill_used="standard_tutorial",
                    sources=[]
                )
            ]
        )
        
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="编译后的剧本")
        
        # 执行测试
        await compile_screenplay(state, llm_service)
        
        # 验证 LLM 调用的提示词包含正确顺序的片段
        call_args = llm_service.chat_completion.call_args
        prompt = call_args.kwargs["messages"][1]["content"]
        
        # 检查片段顺序
        first_pos = prompt.find("第一个片段")
        second_pos = prompt.find("第二个片段")
        third_pos = prompt.find("第三个片段")
        
        assert first_pos < second_pos < third_pos


class TestCompilationPrompt:
    """测试编译提示词构建"""
    
    def test_build_compilation_prompt_with_context(self):
        """测试带项目上下文的提示词"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试上下文",
            global_tone="professional",
            outline=[
                OutlineStep(step_id=0, description="步骤 1", status="completed")
            ]
        )
        
        fragments = [
            ScreenplayFragment(
                step_id=0,
                content="片段内容",
                skill_used="standard_tutorial",
                sources=["source1.md", "source2.md"]
            )
        ]
        
        prompt = _build_compilation_prompt(state, fragments)
        
        assert "测试主题" in prompt
        assert "测试上下文" in prompt
        assert "professional" in prompt
        assert "片段内容" in prompt
        assert "步骤 1" in prompt
        assert "standard_tutorial" in prompt
    
    def test_build_compilation_prompt_without_context(self):
        """测试不带项目上下文的提示词"""
        state = SharedState(
            user_topic="测试主题",
            project_context="",
            outline=[
                OutlineStep(step_id=0, description="步骤 1", status="completed")
            ]
        )
        
        fragments = [
            ScreenplayFragment(
                step_id=0,
                content="片段内容",
                skill_used="standard_tutorial",
                sources=[]
            )
        ]
        
        prompt = _build_compilation_prompt(state, fragments)
        
        assert "测试主题" in prompt
        assert "片段内容" in prompt
        # 不应包含空的项目上下文部分
        assert "项目上下文\n\n" not in prompt
    
    def test_build_compilation_prompt_multiple_fragments(self):
        """测试多个片段的提示词"""
        state = SharedState(
            user_topic="测试主题",
            project_context="",
            outline=[
                OutlineStep(step_id=0, description="步骤 1", status="completed"),
                OutlineStep(step_id=1, description="步骤 2", status="completed"),
                OutlineStep(step_id=2, description="步骤 3", status="completed")
            ]
        )
        
        fragments = [
            ScreenplayFragment(
                step_id=0,
                content="片段 1",
                skill_used="standard_tutorial",
                sources=["s1.md"]
            ),
            ScreenplayFragment(
                step_id=1,
                content="片段 2",
                skill_used="warning_mode",
                sources=["s2.md", "s3.md"]
            ),
            ScreenplayFragment(
                step_id=2,
                content="片段 3",
                skill_used="visualization_analogy",
                sources=[]
            )
        ]
        
        prompt = _build_compilation_prompt(state, fragments)
        
        # 验证所有片段都包含在提示词中
        assert "片段 1" in prompt
        assert "片段 2" in prompt
        assert "片段 3" in prompt
        
        # 验证片段编号
        assert "片段 1 (步骤 0" in prompt
        assert "片段 2 (步骤 1" in prompt
        assert "片段 3 (步骤 2" in prompt
        
        # 验证 Skill 信息
        assert "standard_tutorial" in prompt
        assert "warning_mode" in prompt
        assert "visualization_analogy" in prompt


class TestEmptyScreenplay:
    """测试空剧本生成"""
    
    def test_generate_empty_screenplay_with_context(self):
        """测试带上下文的空剧本"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试上下文"
        )
        
        screenplay = _generate_empty_screenplay(state)
        
        assert "测试主题" in screenplay
        assert "测试上下文" in screenplay
        assert "缺少足够的信息" in screenplay
    
    def test_generate_empty_screenplay_without_context(self):
        """测试不带上下文的空剧本"""
        state = SharedState(
            user_topic="测试主题",
            project_context=""
        )
        
        screenplay = _generate_empty_screenplay(state)
        
        assert "测试主题" in screenplay
        assert "缺少足够的信息" in screenplay


class TestFallbackCompilation:
    """测试降级编译"""
    
    def test_fallback_compilation_basic(self):
        """测试基本降级编译"""
        state = SharedState(
            user_topic="测试主题",
            project_context="",
            outline=[
                OutlineStep(step_id=0, description="步骤 1", status="completed"),
                OutlineStep(step_id=1, description="步骤 2", status="completed")
            ]
        )
        
        fragments = [
            ScreenplayFragment(
                step_id=0,
                content="片段 1 内容",
                skill_used="standard_tutorial",
                sources=["source1.md"]
            ),
            ScreenplayFragment(
                step_id=1,
                content="片段 2 内容",
                skill_used="standard_tutorial",
                sources=["source2.md", "source3.md"]
            )
        ]
        
        screenplay = _fallback_compilation(state, fragments)
        
        # 验证包含主题
        assert "测试主题" in screenplay
        
        # 验证包含所有片段
        assert "片段 1 内容" in screenplay
        assert "片段 2 内容" in screenplay
        
        # 验证包含步骤描述
        assert "步骤 1" in screenplay
        assert "步骤 2" in screenplay
        
        # 验证包含来源信息
        assert "source1.md" in screenplay
        assert "source2.md" in screenplay
    
    def test_fallback_compilation_with_context(self):
        """测试带上下文的降级编译"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试上下文",
            outline=[
                OutlineStep(step_id=0, description="步骤 1", status="completed")
            ]
        )
        
        fragments = [
            ScreenplayFragment(
                step_id=0,
                content="片段内容",
                skill_used="standard_tutorial",
                sources=[]
            )
        ]
        
        screenplay = _fallback_compilation(state, fragments)
        
        assert "测试主题" in screenplay
        assert "测试上下文" in screenplay
        assert "片段内容" in screenplay
    
    def test_fallback_compilation_many_sources(self):
        """测试多来源的降级编译（只显示前3个）"""
        state = SharedState(
            user_topic="测试主题",
            project_context="",
            outline=[
                OutlineStep(step_id=0, description="步骤 1", status="completed")
            ]
        )
        
        fragments = [
            ScreenplayFragment(
                step_id=0,
                content="片段内容",
                skill_used="standard_tutorial",
                sources=["s1.md", "s2.md", "s3.md", "s4.md", "s5.md"]
            )
        ]
        
        screenplay = _fallback_compilation(state, fragments)
        
        # 验证只显示前3个来源
        assert "s1.md" in screenplay
        assert "s2.md" in screenplay
        assert "s3.md" in screenplay
        
        # 验证有"等 X 个来源"的提示
        assert "等 5 个来源" in screenplay
    
    def test_fallback_compilation_summary(self):
        """测试降级编译的总结部分"""
        state = SharedState(
            user_topic="测试主题",
            project_context="",
            outline=[
                OutlineStep(step_id=0, description="步骤 1", status="completed"),
                OutlineStep(step_id=1, description="步骤 2", status="completed")
            ]
        )
        
        fragments = [
            ScreenplayFragment(
                step_id=0,
                content="片段 1",
                skill_used="standard_tutorial",
                sources=["s1.md", "s2.md"]
            ),
            ScreenplayFragment(
                step_id=1,
                content="片段 2",
                skill_used="standard_tutorial",
                sources=["s3.md"]
            )
        ]
        
        screenplay = _fallback_compilation(state, fragments)
        
        # 验证总结部分
        assert "总结" in screenplay
        assert "2 个主要部分" in screenplay
        assert "3 个来源文档" in screenplay
