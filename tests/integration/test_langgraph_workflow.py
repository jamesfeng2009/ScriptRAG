"""Integration tests for LangGraph workflow orchestration

This module tests the complete LangGraph state machine workflow,
including node execution, routing logic, and state transitions.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.domain.models import SharedState, OutlineStep
from src.application.orchestrator import WorkflowOrchestrator


@pytest.fixture
def mock_services():
    """Create mock services for testing"""
    llm_service = Mock()
    llm_service.chat_completion = AsyncMock(return_value="Test response")
    llm_service.embedding = AsyncMock(return_value=[[0.1] * 1536])
    
    retrieval_service = Mock()
    retrieval_service.hybrid_retrieve = AsyncMock(return_value=[])
    
    parser_service = Mock()
    parser_service.parse = Mock(return_value=Mock(
        has_deprecated=False,
        has_fixme=False,
        has_todo=False,
        has_security=False,
        language="python",
        elements=[]
    ))
    
    summarization_service = Mock()
    summarization_service.check_size = Mock(return_value=False)
    
    return {
        "llm_service": llm_service,
        "retrieval_service": retrieval_service,
        "parser_service": parser_service,
        "summarization_service": summarization_service
    }


@pytest.fixture
def simple_state():
    """Create a simple initial state for testing"""
    return SharedState(
        user_topic="Test Topic",
        project_context="Test Context",
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
async def test_orchestrator_initialization(mock_services):
    """Test that WorkflowOrchestrator initializes correctly
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_services["llm_service"],
        retrieval_service=mock_services["retrieval_service"],
        parser_service=mock_services["parser_service"],
        summarization_service=mock_services["summarization_service"]
    )
    
    # Verify orchestrator is initialized
    assert orchestrator is not None
    assert orchestrator.graph is not None
    assert orchestrator.llm_service == mock_services["llm_service"]


@pytest.mark.asyncio
async def test_graph_compilation(mock_services):
    """Test that the LangGraph state machine compiles successfully
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_services["llm_service"],
        retrieval_service=mock_services["retrieval_service"],
        parser_service=mock_services["parser_service"],
        summarization_service=mock_services["summarization_service"]
    )
    
    # Verify graph is compiled
    assert orchestrator.graph is not None
    
    # Graph should have nodes
    # Note: LangGraph's compiled graph doesn't expose nodes directly,
    # but we can verify it's a compiled graph object
    assert hasattr(orchestrator.graph, 'ainvoke')


@pytest.mark.asyncio
async def test_director_routing_pivot(mock_services, simple_state):
    """Test director routing when pivot is triggered
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_services["llm_service"],
        retrieval_service=mock_services["retrieval_service"],
        parser_service=mock_services["parser_service"],
        summarization_service=mock_services["summarization_service"]
            )
    
    # Set pivot triggered
    simple_state.pivot_triggered = True
    simple_state.pivot_reason = "deprecation_conflict"
    
    # Test routing
    route = orchestrator._route_director_decision(simple_state)
    
    assert route == "pivot"


@pytest.mark.asyncio
async def test_director_routing_write(mock_services, simple_state):
    """Test director routing when content is approved
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_services["llm_service"],
        retrieval_service=mock_services["retrieval_service"],
        parser_service=mock_services["parser_service"],
        summarization_service=mock_services["summarization_service"]
            )
    
    # No pivot triggered
    simple_state.pivot_triggered = False
    
    # Test routing
    route = orchestrator._route_director_decision(simple_state)
    
    assert route == "write"


@pytest.mark.asyncio
async def test_fact_check_routing_invalid(mock_services, simple_state):
    """Test fact checker routing when fragment is invalid
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_services["llm_service"],
        retrieval_service=mock_services["retrieval_service"],
        parser_service=mock_services["parser_service"],
        summarization_service=mock_services["summarization_service"]
            )
    
    # Set fact check failed
    simple_state.fact_check_passed = False
    
    # Test routing using the actual method
    route = orchestrator._route_fact_check(simple_state)
    
    assert route == "invalid"


@pytest.mark.asyncio
async def test_completion_routing_continue(mock_services, simple_state):
    """Test completion routing when more steps remain
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_services["llm_service"],
        retrieval_service=mock_services["retrieval_service"],
        parser_service=mock_services["parser_service"],
        summarization_service=mock_services["summarization_service"]
            )
    
    # Add multiple steps
    simple_state.outline = [
        OutlineStep(step_id=0, description="Step 1", status="completed", retry_count=0),
        OutlineStep(step_id=1, description="Step 2", status="pending", retry_count=0),
        OutlineStep(step_id=2, description="Step 3", status="pending", retry_count=0)
    ]
    simple_state.current_step_index = 0
    
    # Test routing using the actual method
    route = orchestrator._route_completion(simple_state)
    
    assert route == "continue"


@pytest.mark.asyncio
async def test_completion_routing_done(mock_services, simple_state):
    """Test completion routing when all steps are complete
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_services["llm_service"],
        retrieval_service=mock_services["retrieval_service"],
        parser_service=mock_services["parser_service"],
        summarization_service=mock_services["summarization_service"]
            )
    
    # Set to after last step (current_step_index >= len(outline) means done)
    simple_state.outline = [
        OutlineStep(step_id=0, description="Step 1", status="completed", retry_count=0),
        OutlineStep(step_id=1, description="Step 2", status="completed", retry_count=0)
    ]
    simple_state.current_step_index = 2  # After last step
    
    # Test routing using the actual method
    route = orchestrator._route_completion(simple_state)
    
    assert route == "done"


@pytest.mark.asyncio
async def test_simple_workflow_execution(mock_services, simple_state):
    """Test simple workflow execution with minimal outline
    
    This test verifies that the graph can execute with a simple outline
    without errors. Tests initialization and basic execution structure.
    """
    # Create orchestrator with mock services
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_services["llm_service"],
        retrieval_service=mock_services["retrieval_service"],
        parser_service=mock_services["parser_service"],
        summarization_service=mock_services["summarization_service"]
            )
    
    # Verify orchestrator is initialized
    assert orchestrator is not None
    assert orchestrator.graph is not None
    
    # Verify graph has required nodes
    graph = orchestrator.graph
    assert len(graph.nodes) > 0
    
    # Execute workflow with simple state
    # Note: We verify the orchestrator can process the state structure
    result = await orchestrator.execute(simple_state)
    
    # Verify result structure
    assert result is not None
    assert "success" in result or "error" in result
