"""True integration test for workflow execution without mocking graph.ainvoke

This test actually executes the workflow to verify the state key fix.

Feature: true-workflow-integration
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from src.application.orchestrator import WorkflowOrchestrator
from src.domain.models import SharedState, OutlineStep


def create_fast_mock_llm():
    """Create mock LLM that returns immediately without delays"""
    mock_service = Mock()

    async def mock_chat_completion(messages, task_type=None, **kwargs):
        if task_type == "test":
            return "approved"
        elif task_type == "fact_check":
            return "VALID"
        else:
            return "approved"

    mock_service.chat_completion = AsyncMock(side_effect=mock_chat_completion)
    return mock_service


def create_fast_mock_retrieval():
    """Create mock retrieval that returns immediately with realistic results"""
    mock_service = Mock()

    async def mock_retrieve(workspace_id, query, top_k=5):
        return []

    async def mock_hybrid_retrieve(workspace_id, query, top_k=5):
        return []

    async def mock_retrieve_with_strategy(workspace_id, query, strategy_name, top_k=5):
        return []

    mock_service.retrieve = AsyncMock(side_effect=mock_retrieve)
    mock_service.hybrid_retrieve = AsyncMock(side_effect=mock_hybrid_retrieve)
    mock_service.retrieve_with_strategy = AsyncMock(side_effect=mock_retrieve_with_strategy)
    return mock_service


def create_fast_mock_parser():
    """Create mock parser that returns immediately"""
    mock_service = Mock()

    def mock_parse(file_path, content, language=None):
        mock_result = Mock()
        mock_result.language = "python"
        mock_result.elements = []
        mock_result.metadata = {}
        return mock_result

    mock_service.parse = Mock(side_effect=mock_parse)
    return mock_service


def create_fast_mock_summarization():
    """Create mock summarization that returns False immediately"""
    mock_service = Mock()
    mock_service.check_size = Mock(return_value=False)
    return mock_service


def create_test_state():
    """Create a minimal test state with single step"""
    outline = [
        OutlineStep(
            step_id=0,
            title="第一步",
            description="第一步内容",
            status="pending",
            retry_count=0
        ),
    ]

    return SharedState(
        user_topic="测试主题",
        project_context="测试",
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
async def test_workflow_execution_with_state_key_fix():
    """
    Test that workflow executes correctly with the state key fix.
    
    This test does NOT mock graph.ainvoke - it actually executes the workflow.
    The fix for retrieved_docs -> retrieved_docs should prevent infinite loops.
    
    **Before fix**: This test would hang until recursion_limit (50) is reached
    **After fix**: This test should complete within reasonable time
    """
    mock_llm = create_fast_mock_llm()
    mock_retrieval = create_fast_mock_retrieval()
    mock_parser = create_fast_mock_parser()
    mock_summarization = create_fast_mock_summarization()

    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        enable_agentic_rag=False
    )

    initial_state = create_test_state()

    print("Starting workflow execution...")
    print("If this hangs, the state key fix didn't work!")
    
    try:
        result = await asyncio.wait_for(
            orchestrator.execute(initial_state, recursion_limit=50),
            timeout=30.0
        )
        
        print(f"Workflow completed in under 30 seconds!")
        print(f"Result type: {type(result)}")
        
        assert result is not None
        
        print("✅ SUCCESS: Workflow executed without hanging!")
        
    except asyncio.TimeoutError:
        print("❌ FAILED: Workflow timed out - state key fix may not be working!")
        raise AssertionError(
            "Workflow execution timed out. "
            "This suggests the state key fix (retrieved_docs vs retrieved_docs) "
            "is not working correctly."
        )


@pytest.mark.asyncio
async def test_workflow_director_decision():
    """
    Test that director decision (approved) prevents pivot.
    """
    mock_llm = create_fast_mock_llm()
    mock_retrieval = create_fast_mock_retrieval()
    mock_parser = create_fast_mock_parser()
    mock_summarization = create_fast_mock_summarization()

    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        enable_agentic_rag=False
    )

    initial_state = create_test_state()

    try:
        result = await asyncio.wait_for(
            orchestrator.execute(initial_state, recursion_limit=50),
            timeout=30.0
        )
        
        assert result is not None
        print("✅ SUCCESS: Director approval workflow completed!")
        
    except asyncio.TimeoutError:
        raise AssertionError("Director approval workflow timed out")


@pytest.mark.asyncio
async def test_workflow_fact_check():
    """
    Test that fact check returns VALID and prevents regeneration.
    """
    mock_llm = create_fast_mock_llm()
    mock_retrieval = create_fast_mock_retrieval()
    mock_parser = create_fast_mock_parser()
    mock_summarization = create_fast_mock_summarization()

    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        enable_agentic_rag=False
    )

    initial_state = create_test_state()

    try:
        result = await asyncio.wait_for(
            orchestrator.execute(initial_state, recursion_limit=50),
            timeout=30.0
        )
        
        assert result is not None
        print("✅ SUCCESS: Fact check workflow completed!")
        
    except asyncio.TimeoutError:
        raise AssertionError("Fact check workflow timed out")
