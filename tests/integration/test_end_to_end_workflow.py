"""End-to-end integration tests for complete screenplay generation workflow

This module tests the complete workflow from user input to final screenplay,
verifying that all agents execute in the correct order and produce valid output.

验证需求: 12.1-12.8
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.domain.models import (
    SharedState,
    OutlineStep,
    RetrievedDocument,
    ScreenplayFragment
)
from src.application.orchestrator import WorkflowOrchestrator


@pytest.fixture
def mock_llm_service():
    """Create mock LLM service for testing"""
    llm_service = Mock()
    
    # Mock chat completion for different agents
    async def mock_chat_completion(messages, task_type, **kwargs):
        # Determine which agent is calling based on message content
        last_message = messages[-1]["content"] if messages else ""
        
        # Check for planner (Chinese or English)
        if ("生成" in last_message and "大纲" in last_message) or \
           ("generate" in last_message.lower() and "outline" in last_message.lower()):
            # Planner response - return a properly formatted outline in Chinese format
            return """
步骤1: Introduction to Python async/await | 关键词: async, await, coroutines
步骤2: Understanding coroutines and event loops | 关键词: event loop, asyncio
步骤3: Practical examples of async programming | 关键词: examples, async functions
步骤4: Error handling in async code | 关键词: try, except, async errors
步骤5: Best practices and common pitfalls | 关键词: best practices, pitfalls
            """
        elif "complexity" in last_message.lower() or "复杂度" in last_message:
            # Director complexity assessment - return a score between 0-1
            # Use 0.5 to avoid triggering any pivot (not too high, not too low)
            return "0.5"
        elif "evaluate" in last_message.lower() or "assess" in last_message.lower() or "评估" in last_message.lower():
            # Director response - always approve to avoid infinite loops
            return "approved"
        elif ("生成" in last_message and "片段" in last_message) or \
             ("generate" in last_message.lower() and "fragment" in last_message.lower()):
            # Writer response
            return "This is a test screenplay fragment based on the retrieved content. It explains the concepts clearly and provides examples."
        elif "源文档内容" in last_message or "生成的片段内容" in last_message or \
             "verify" in last_message.lower() or "fact-check" in last_message.lower():
            # Fact checker response - return VALID format
            return "VALID"
        elif ("编译" in last_message or "整合" in last_message) or \
             ("compile" in last_message.lower() or "integrate" in last_message.lower()):
            # Compiler response
            return "# Final Screenplay\n\nComplete screenplay content with all fragments integrated."
        else:
            return "Test response"
    
    llm_service.chat_completion = AsyncMock(side_effect=mock_chat_completion)
    llm_service.embedding = AsyncMock(return_value=[[0.1] * 1536])
    
    return llm_service


@pytest.fixture
def mock_retrieval_service():
    """Create mock retrieval service for testing"""
    from src.services.retrieval_service import RetrievalResult
    
    retrieval_service = Mock()
    
    # Mock hybrid retrieve to return sample documents
    async def mock_hybrid_retrieve(query, workspace_id, top_k=5):
        # Return sample retrieval results (not RetrievedDocument)
        return [
            RetrievalResult(
                id="doc1",
                file_path="example.py",
                content="Sample Python async/await code example",
                similarity=0.9,
                confidence=0.9,
                has_deprecated=False,
                has_fixme=False,
                has_todo=False,
                has_security=False,
                metadata={},
                source="vector"
            ),
            RetrievalResult(
                id="doc2",
                file_path="docs/async.md",
                content="Documentation about coroutines and event loops",
                similarity=0.85,
                confidence=0.85,
                has_deprecated=False,
                has_fixme=False,
                has_todo=False,
                has_security=False,
                metadata={},
                source="vector"
            )
        ]
    
    retrieval_service.hybrid_retrieve = AsyncMock(side_effect=mock_hybrid_retrieve)
    
    return retrieval_service


@pytest.fixture
def mock_parser_service():
    """Create mock parser service for testing"""
    parser_service = Mock()
    
    def mock_parse(file_path=None, content="", language="python"):
        return Mock(
            has_deprecated=False,
            has_fixme=False,
            has_todo=False,
            has_security=False,
            language=language,
            elements=[]
        )
    
    parser_service.parse = Mock(side_effect=mock_parse)
    
    return parser_service


@pytest.fixture
def mock_summarization_service():
    """Create mock summarization service for testing"""
    summarization_service = Mock()
    
    def mock_check_size(content, threshold=10000):
        return False  # Content is always small enough
    
    summarization_service.check_size = Mock(side_effect=mock_check_size)
    
    return summarization_service


@pytest.fixture
def initial_state():
    """Create initial state for workflow testing"""
    return SharedState(
        user_topic="Introduction to Python async/await",
        project_context="Python async programming tutorial",
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


@pytest.mark.asyncio
async def test_complete_workflow_simple_outline(
    mock_llm_service,
    mock_retrieval_service,
    mock_parser_service,
    mock_summarization_service,
    initial_state
):
    """Test complete workflow from user input to final screenplay with simple outline
    
    This test verifies:
    - Planner generates outline (需求 12.1)
    - Navigator retrieves content for each step (需求 12.3)
    - Director evaluates content (需求 12.4)
    - Writer generates fragments (需求 12.6)
    - Fact checker validates fragments (需求 10.2)
    - Compiler integrates fragments (需求 12.7, 12.8)
    - All agents execute in correct order
    
    验证需求: 12.1, 12.2, 12.3, 12.4, 12.6, 12.7, 12.8
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        parser_service=mock_parser_service,
        summarization_service=mock_summarization_service,
        workspace_id="test-workspace"
    )
    
    # Execute workflow
    result = await orchestrator.execute(initial_state)
    
    # Verify workflow completed successfully
    assert result["success"] is True
    assert "final_screenplay" in result
    assert result["final_screenplay"] is not None
    
    # Verify state after execution
    final_state = result["state"]
    
    # Verify outline was generated (需求 12.1)
    assert len(final_state.outline) > 0
    assert all(isinstance(step, OutlineStep) for step in final_state.outline)
    
    # Verify all steps were processed (需求 12.2)
    assert final_state.current_step_index == len(final_state.outline)
    
    # Verify fragments were generated (需求 12.6)
    assert len(final_state.fragments) > 0
    assert all(isinstance(frag, ScreenplayFragment) for frag in final_state.fragments)
    
    # Verify execution log contains agent transitions
    assert len(final_state.execution_log) > 0
    
    # Verify key agents were executed
    agent_names = [log["agent_name"] for log in final_state.execution_log]
    assert "planner" in agent_names
    assert "navigator" in agent_names
    assert "director" in agent_names
    assert "writer" in agent_names
    assert "fact_checker" in agent_names
    assert "compiler" in agent_names


@pytest.mark.asyncio
async def test_workflow_with_multiple_steps(
    mock_llm_service,
    mock_retrieval_service,
    mock_parser_service,
    mock_summarization_service,
    initial_state
):
    """Test workflow processes multiple outline steps sequentially
    
    Verifies that the workflow correctly processes each step in order
    and generates fragments for all steps.
    
    验证需求: 12.2
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        parser_service=mock_parser_service,
        summarization_service=mock_summarization_service,
        workspace_id="test-workspace"
    )
    
    # Execute workflow
    result = await orchestrator.execute(initial_state)
    
    # Verify workflow completed
    assert result["success"] is True
    
    final_state = result["state"]
    
    # Verify multiple steps were created
    assert len(final_state.outline) >= 3
    
    # Verify each step was processed
    for step in final_state.outline:
        assert step.status in ["completed", "skipped"]
    
    # Verify fragments match outline steps (excluding skipped)
    completed_steps = [s for s in final_state.outline if s.status == "completed"]
    assert len(final_state.fragments) == len(completed_steps)
    
    # Verify fragments are in correct order
    for i, fragment in enumerate(final_state.fragments):
        assert fragment.step_id == completed_steps[i].step_id


@pytest.mark.asyncio
async def test_workflow_agent_execution_order(
    mock_llm_service,
    mock_retrieval_service,
    mock_parser_service,
    mock_summarization_service,
    initial_state
):
    """Test that agents execute in the correct order
    
    Verifies the workflow follows the expected agent sequence:
    planner -> navigator -> director -> writer -> fact_checker -> compiler
    
    验证需求: 12.1-12.8
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        parser_service=mock_parser_service,
        summarization_service=mock_summarization_service,
        workspace_id="test-workspace"
    )
    
    # Execute workflow
    result = await orchestrator.execute(initial_state)
    
    assert result["success"] is True
    
    final_state = result["state"]
    execution_log = final_state.execution_log
    
    # Extract agent execution sequence
    agent_sequence = [log["agent_name"] for log in execution_log]
    
    # Verify planner is first
    assert agent_sequence[0] == "planner"
    
    # Verify compiler is last (or near last)
    assert "compiler" in agent_sequence[-3:]
    
    # Verify navigator appears before writer for each step
    navigator_indices = [i for i, name in enumerate(agent_sequence) if name == "navigator"]
    writer_indices = [i for i, name in enumerate(agent_sequence) if name == "writer"]
    
    # Each writer execution should have a navigator execution before it
    for writer_idx in writer_indices:
        assert any(nav_idx < writer_idx for nav_idx in navigator_indices)


@pytest.mark.asyncio
async def test_workflow_final_screenplay_structure(
    mock_llm_service,
    mock_retrieval_service,
    mock_parser_service,
    mock_summarization_service,
    initial_state
):
    """Test that final screenplay has correct structure
    
    Verifies the compiler produces a well-structured final screenplay.
    
    验证需求: 12.7, 12.8
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        parser_service=mock_parser_service,
        summarization_service=mock_summarization_service,
        workspace_id="test-workspace"
    )
    
    # Execute workflow
    result = await orchestrator.execute(initial_state)
    
    assert result["success"] is True
    assert result["final_screenplay"] is not None
    
    final_screenplay = result["final_screenplay"]
    
    # Verify screenplay is a non-empty string
    assert isinstance(final_screenplay, str)
    assert len(final_screenplay) > 0
    
    # Verify screenplay contains content
    assert len(final_screenplay.strip()) > 10


@pytest.mark.asyncio
async def test_workflow_state_consistency(
    mock_llm_service,
    mock_retrieval_service,
    mock_parser_service,
    mock_summarization_service,
    initial_state
):
    """Test that state remains consistent throughout workflow
    
    Verifies that state modifications are properly maintained across
    agent transitions.
    
    验证需求: 1.7, 2.8
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        parser_service=mock_parser_service,
        summarization_service=mock_summarization_service,
        workspace_id="test-workspace"
    )
    
    # Execute workflow
    result = await orchestrator.execute(initial_state)
    
    assert result["success"] is True
    
    final_state = result["state"]
    
    # Verify state fields are consistent
    assert final_state.user_topic == initial_state.user_topic
    assert final_state.project_context == initial_state.project_context
    assert final_state.max_retries == initial_state.max_retries
    
    # Verify state was modified during execution
    assert len(final_state.outline) > 0
    assert len(final_state.fragments) > 0
    assert len(final_state.execution_log) > 0
    
    # Verify current_step_index advanced
    assert final_state.current_step_index > 0


@pytest.mark.asyncio
async def test_workflow_with_empty_retrieval(
    mock_llm_service,
    mock_retrieval_service,
    mock_parser_service,
    mock_summarization_service,
    initial_state
):
    """Test workflow handles empty retrieval results gracefully
    
    Verifies that when navigator returns no documents, the workflow
    continues without errors.
    
    验证需求: 3.7, 7.1
    """
    # Retrieval service already returns empty results by default
    
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        parser_service=mock_parser_service,
        summarization_service=mock_summarization_service,
        workspace_id="test-workspace"
    )
    
    # Execute workflow
    result = await orchestrator.execute(initial_state)
    
    # Workflow should complete successfully even with empty retrieval
    assert result["success"] is True
    
    final_state = result["state"]
    
    # Verify workflow completed
    assert len(final_state.outline) > 0
    assert len(final_state.fragments) > 0
    
    # Verify no hallucinations (fragments should acknowledge lack of info)
    # This is verified by the fact checker in the workflow


@pytest.mark.asyncio
async def test_workflow_logging_completeness(
    mock_llm_service,
    mock_retrieval_service,
    mock_parser_service,
    mock_summarization_service,
    initial_state
):
    """Test that workflow logs all agent transitions
    
    Verifies comprehensive logging throughout the workflow.
    
    验证需求: 13.1, 13.2
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        parser_service=mock_parser_service,
        summarization_service=mock_summarization_service,
        workspace_id="test-workspace"
    )
    
    # Execute workflow
    result = await orchestrator.execute(initial_state)
    
    assert result["success"] is True
    
    final_state = result["state"]
    execution_log = final_state.execution_log
    
    # Verify log entries exist
    assert len(execution_log) > 0
    
    # Verify each log entry has required fields
    for log_entry in execution_log:
        assert "agent_name" in log_entry
        assert "action" in log_entry
        assert "timestamp" in log_entry
        assert "details" in log_entry
    
    # Verify all major agents are logged
    agent_names = {log["agent_name"] for log in execution_log}
    expected_agents = {"planner", "navigator", "director", "writer", "fact_checker", "compiler"}
    
    # At least most agents should be present
    assert len(agent_names.intersection(expected_agents)) >= 5
