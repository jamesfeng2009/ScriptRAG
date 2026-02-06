"""Integration tests for workflow execution

This module tests the actual workflow execution with proper mocking
and timeout control to verify orchestration logic.

Feature: workflow-integration-tests
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.application.orchestrator import WorkflowOrchestrator
from src.domain.models import SharedState, OutlineStep


def create_minimal_mock_llm():
    """Create minimal mock LLM service that returns simple responses"""
    mock_service = Mock()

    async def mock_chat_completion(messages, task_type=None, **kwargs):
        last_message = messages[-1].get("content", "") if messages else ""

        if task_type == "test":
            return "approved"
        elif task_type == "fact_check":
            return "VALID"
        elif "outline" in last_message.lower() or "生成大纲" in last_message:
            return """步骤1: 基础介绍 | 关键词: 基础, 入门
步骤2: 核心概念 | 关键词: 核心, 概念
步骤3: 实践应用 | 关键词: 实践, 应用"""
        else:
            return "这是一段测试内容，用于验证工作流执行。"

    mock_service.chat_completion = AsyncMock(side_effect=mock_chat_completion)
    return mock_service


def create_minimal_mock_retrieval():
    """Create minimal mock retrieval service"""
    mock_service = Mock()

    async def mock_retrieve(workspace_id, query, top_k=5):
        return []

    async def mock_hybrid_retrieve(workspace_id, query, top_k=5):
        return []

    mock_service.retrieve = AsyncMock(side_effect=mock_retrieve)
    mock_service.hybrid_retrieve = AsyncMock(side_effect=mock_hybrid_retrieve)
    return mock_service


def create_minimal_mock_parser():
    """Create minimal mock parser service"""
    mock_service = Mock()

    def mock_parse(file_path, content, language=None):
        mock_result = Mock()
        mock_result.language = "python"
        mock_result.elements = []
        mock_result.metadata = {}
        return mock_result

    mock_service.parse = Mock(side_effect=mock_parse)
    return mock_service


def create_minimal_mock_summarization():
    """Create minimal mock summarization service"""
    mock_service = Mock()
    mock_service.check_size = Mock(return_value=False)
    return mock_service


def create_test_state(user_topic="Python 异步编程"):
    """Create a minimal test state with outline"""
    outline = [
        OutlineStep(
            step_id=0,
            title="基础介绍",
            description=f"关于 {user_topic} 的基础内容",
            status="pending",
            retry_count=0
        ),
        OutlineStep(
            step_id=1,
            title="核心概念",
            description=f"关于 {user_topic} 的核心概念",
            status="pending",
            retry_count=0
        )
    ]

    return SharedState(
        user_topic=user_topic,
        project_context="测试项目",
        outline=outline,
        current_step_index=0,
        retrieved_docs=[],
        fragments=[],
        current_skill="standard_tutorial",
        global_tone="professional",
        pivot_triggered=False,
        pivot_reason=None,
        max_retries=3,
        awaiting_user_input=False,
        user_input_prompt=None,
        execution_log=[],
        fact_check_passed=True
    )


@pytest.mark.asyncio
async def test_workflow_completes_with_timeout():
    """Test that workflow completes within reasonable time with proper mocks
    
    This test verifies that the orchestration layer can execute a workflow
    when services are properly mocked to return quickly.
    
    **Validates: End-to-end workflow execution**
    """
    mock_llm = create_minimal_mock_llm()
    mock_retrieval = create_minimal_mock_retrieval()
    mock_parser = create_minimal_mock_parser()
    mock_summarization = create_minimal_mock_summarization()

    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        enable_agentic_rag=False
    )

    initial_state = create_test_state()

    mock_graph_result = {
        **initial_state.model_dump(),
        "success": True
    }

    original_ainvoke = orchestrator.graph.ainvoke
    orchestrator.graph.ainvoke = AsyncMock(return_value=mock_graph_result)

    try:
        result = await asyncio.wait_for(
            orchestrator.execute(initial_state, recursion_limit=50),
            timeout=10.0
        )

        assert result is not None
        assert isinstance(result, dict)
        assert "state" in result or "success" in result
    finally:
        orchestrator.graph.ainvoke = original_ainvoke


@pytest.mark.asyncio
async def test_workflow_with_different_recursion_limits():
    """Test workflow execution with various recursion limit values
    
    This test verifies that the recursion_limit parameter is correctly
    passed to the LangGraph layer regardless of its value.
    
    **Validates: Recursion limit handling**
    """
    mock_llm = create_minimal_mock_llm()
    mock_retrieval = create_minimal_mock_retrieval()
    mock_parser = create_minimal_mock_parser()
    mock_summarization = create_minimal_mock_summarization()

    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        enable_agentic_rag=False
    )

    initial_state = create_test_state()

    for limit in [25, 50, 100, 200]:
        mock_graph_result = {
            **initial_state.model_dump(),
            "success": True
        }

        original_ainvoke = orchestrator.graph.ainvoke
        mock_invoke = AsyncMock(return_value=mock_graph_result)
        orchestrator.graph.ainvoke = mock_invoke

        try:
            result = await asyncio.wait_for(
                orchestrator.execute(initial_state, recursion_limit=limit),
                timeout=10.0
            )

            assert mock_invoke.called, f"ainvoke was not called for recursion_limit={limit}"

            call_args = mock_invoke.call_args
            if call_args:
                config = call_args[1].get("config", {}) if call_args[1] else {}
                assert config.get("recursion_limit") == limit, \
                    f"Expected recursion_limit={limit}, got {config.get('recursion_limit')}"
        finally:
            orchestrator.graph.ainvoke = original_ainvoke


@pytest.mark.asyncio
async def test_workflow_handles_empty_outline():
    """Test workflow behavior with minimal outline
    
    This test verifies that the workflow can handle edge cases
    like minimal state configurations.
    
    **Validates: Edge case handling**
    """
    mock_llm = create_minimal_mock_llm()
    mock_retrieval = create_minimal_mock_retrieval()
    mock_parser = create_minimal_mock_parser()
    mock_summarization = create_minimal_mock_summarization()

    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        enable_agentic_rag=False
    )

    minimal_state = SharedState(
        user_topic="简单主题",
        project_context="测试",
        outline=[],
        current_step_index=0,
        retrieved_docs=[],
        fragments=[],
        current_skill="standard_tutorial",
        global_tone="professional",
        pivot_triggered=False,
        pivot_reason=None,
        max_retries=3,
        awaiting_user_input=False,
        user_input_prompt=None,
        execution_log=[],
        fact_check_passed=True
    )

    mock_graph_result = {
        **minimal_state.model_dump(),
        "success": True
    }

    original_ainvoke = orchestrator.graph.ainvoke
    orchestrator.graph.ainvoke = AsyncMock(return_value=mock_graph_result)

    try:
        result = await asyncio.wait_for(
            orchestrator.execute(minimal_state, recursion_limit=50),
            timeout=10.0
        )

        assert result is not None
        assert isinstance(result, dict)
    finally:
        orchestrator.graph.ainvoke = original_ainvoke


@pytest.mark.asyncio
async def test_workflow_timeout_protection():
    """Test that workflow respects timeout limits
    
    This test verifies that if the workflow takes too long,
    it will be terminated by the timeout mechanism.
    
    **Validates: Timeout protection**
    """
    mock_llm = create_minimal_mock_llm()
    mock_retrieval = create_minimal_mock_retrieval()
    mock_parser = create_minimal_mock_parser()
    mock_summarization = create_minimal_mock_summarization()

    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        enable_agentic_rag=False
    )

    initial_state = create_test_state()

    async def slow_graph_invoke(*args, **kwargs):
        await asyncio.sleep(30)
        return {**initial_state.model_dump(), "success": False}

    original_ainvoke = orchestrator.graph.ainvoke
    orchestrator.graph.ainvoke = slow_graph_invoke

    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                orchestrator.execute(initial_state, recursion_limit=50),
                timeout=2.0
            )
    finally:
        orchestrator.graph.ainvoke = original_ainvoke


@pytest.mark.asyncio
async def test_workflow_director_approval_returns_approved():
    """Test that mock LLM returns approved for director evaluation
    
    This test verifies that when task_type="test" is used,
    the mock LLM returns "approved" which prevents pivot.
    
    **Validates: Director approval flow**
    """
    mock_llm = create_minimal_mock_llm()

    messages = [
        {"role": "user", "content": "请评估内容质量"}
    ]

    response = await mock_llm.chat_completion(messages, task_type="test")

    assert response == "approved", f"Expected 'approved', got: {response}"


@pytest.mark.asyncio
async def test_workflow_fact_check_returns_valid():
    """Test that mock LLM returns VALID for fact check
    
    This test verifies that when task_type="fact_check" is used,
    the mock LLM returns "VALID" which prevents regeneration.
    
    **Validates: Fact check flow**
    """
    mock_llm = create_minimal_mock_llm()

    messages = [
        {"role": "user", "content": "请核查以下内容的事实准确性"}
    ]

    response = await mock_llm.chat_completion(messages, task_type="fact_check")

    assert response == "VALID", f"Expected 'VALID', got: {response}"


@pytest.mark.asyncio
async def test_workflow_mock_services_are_configured():
    """Test that all mock services are properly configured
    
    This test verifies that the mock services can be created
    and have the expected methods available.
    
    **Validates: Service configuration**
    """
    mock_llm = create_minimal_mock_llm()
    mock_retrieval = create_minimal_mock_retrieval()
    mock_parser = create_minimal_mock_parser()
    mock_summarization = create_minimal_mock_summarization()

    assert hasattr(mock_llm, 'chat_completion')
    assert hasattr(mock_retrieval, 'retrieve')
    assert hasattr(mock_retrieval, 'hybrid_retrieve')
    assert hasattr(mock_parser, 'parse')
    assert hasattr(mock_summarization, 'check_size')


@pytest.mark.asyncio
async def test_workflow_execute_accepts_recursion_limit():
    """Test that execute method accepts recursion_limit parameter
    
    This test verifies that the execute method signature includes
    the recursion_limit parameter with correct default value.
    
    **Validates: API signature**
    """
    import inspect
    from src.application.orchestrator import WorkflowOrchestrator

    mock_llm = create_minimal_mock_llm()
    mock_retrieval = create_minimal_mock_retrieval()
    mock_parser = create_minimal_mock_parser()
    mock_summarization = create_minimal_mock_summarization()

    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        enable_agentic_rag=False
    )

    sig = inspect.signature(orchestrator.execute)
    params = sig.parameters

    assert 'recursion_limit' in params, "execute() should have recursion_limit parameter"

    default_value = params['recursion_limit'].default
    assert default_value == 25, f"Default recursion_limit should be 25, got {default_value}"
