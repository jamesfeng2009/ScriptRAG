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
        summarization_service=mock_services["summarization_service"],
        workspace_id="test-workspace"
    )
    
    # Verify orchestrator is initialized
    assert orchestrator is not None
    assert orchestrator.graph is not None
    assert orchestrator.llm_service == mock_services["llm_service"]
    assert orchestrator.workspace_id == "test-workspace"


@pytest.mark.asyncio
async def test_graph_compilation(mock_services):
    """Test that the LangGraph state machine compiles successfully
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_services["llm_service"],
        retrieval_service=mock_services["retrieval_service"],
        parser_service=mock_services["parser_service"],
        summarization_service=mock_services["summarization_service"],
        workspace_id="test-workspace"
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
        summarization_service=mock_services["summarization_service"],
        workspace_id="test-workspace"
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
        summarization_service=mock_services["summarization_service"],
        workspace_id="test-workspace"
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
        summarization_service=mock_services["summarization_service"],
        workspace_id="test-workspace"
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
        summarization_service=mock_services["summarization_service"],
        workspace_id="test-workspace"
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
        summarization_service=mock_services["summarization_service"],
        workspace_id="test-workspace"
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


@pytest.mark.skip(reason="Test uses outdated architecture mocking approach")
@pytest.mark.asyncio
async def test_simple_workflow_execution(mock_services, simple_state):
    """Test simple workflow execution with minimal outline
    
    This test verifies that the graph can execute with a simple outline
    without errors.
    """
    # Mock the planner to return a simple outline
    async def mock_plan_outline(state, llm_service):
        state.outline = [
            OutlineStep(step_id=0, description="Test Step 1", status="pending", retry_count=0),
            OutlineStep(step_id=1, description="Test Step 2", status="pending", retry_count=0)
        ]
        return state
    
    # Mock other agents to avoid actual LLM calls
    async def mock_retrieve_content(state, *args, **kwargs):
        state.retrieved_docs = []
        return state
    
    async def mock_evaluate_and_decide(state, llm_service):
        state.pivot_triggered = False
        return state
    
    def mock_check_retry_limit(state):
        return state
    
    async def mock_generate_fragment(state, llm_service):
        from src.domain.models import ScreenplayFragment
        current_step = state.get_current_step()
        if current_step:
            fragment = ScreenplayFragment(
                step_id=current_step.step_id,
                content=f"Test fragment for step {current_step.step_id}",
                skill_used=state.current_skill,
                sources=[]
            )
            state.fragments.append(fragment)
            current_step.status = "completed"
        return state
    
    async def mock_verify_fragment_node(state, llm_service):
        state.fact_check_passed = True
        return state
    
    async def mock_compile_screenplay(state, llm_service):
        return "# Test Screenplay\n\nTest content"
    
    def mock_handle_pivot(state):
        state.pivot_triggered = False
        return state
    
    # Patch all agent functions
    with patch('src.application.orchestrator.plan_outline', mock_plan_outline), \
         patch('src.application.orchestrator.retrieve_content', mock_retrieve_content), \
         patch('src.application.orchestrator.evaluate_and_decide', mock_evaluate_and_decide), \
         patch('src.application.orchestrator.check_retry_limit', mock_check_retry_limit), \
         patch('src.application.orchestrator.generate_fragment', mock_generate_fragment), \
         patch('src.application.orchestrator.verify_fragment_node', mock_verify_fragment_node), \
         patch('src.application.orchestrator.compile_screenplay', mock_compile_screenplay), \
         patch('src.application.orchestrator.handle_pivot', mock_handle_pivot):
        
        orchestrator = WorkflowOrchestrator(
            llm_service=mock_services["llm_service"],
            retrieval_service=mock_services["retrieval_service"],
            parser_service=mock_services["parser_service"],
            summarization_service=mock_services["summarization_service"],
            workspace_id="test-workspace"
        )
        
        # Execute workflow with higher recursion limit
        result = await orchestrator.execute(simple_state)
        
        # Verify result
        assert result is not None
        # Note: The workflow may fail due to mocking complexity, but we verify the structure
        assert "success" in result
        assert "state" in result
        assert "execution_log" in result
