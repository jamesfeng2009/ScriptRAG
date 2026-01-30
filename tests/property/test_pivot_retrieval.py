"""Property-Based Tests for Pivot Triggering Re-retrieval

属性 20: 转向触发重新检索

Property: When a pivot is triggered (for any reason), the system should clear
retrieved documents to force the Navigator to re-retrieve content with the
updated context (modified outline steps and/or changed skill).
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from src.domain.agents.pivot_manager import handle_pivot
from src.domain.models import SharedState, OutlineStep, RetrievedDocument
from src.domain.skills import SKILLS


# Strategy for generating outline steps
@st.composite
def outline_step_strategy(draw, step_id=None):
    """Generate a valid OutlineStep"""
    if step_id is None:
        step_id = draw(st.integers(min_value=0, max_value=20))
    
    return OutlineStep(
        step_id=step_id,
        description=draw(st.text(min_size=10, max_size=100)),
        status=draw(st.sampled_from(["pending", "in_progress", "completed", "skipped"])),
        retry_count=draw(st.integers(min_value=0, max_value=3))
    )


# Strategy for generating retrieved documents
@st.composite
def retrieved_doc_strategy(draw):
    """Generate a valid RetrievedDocument"""
    return RetrievedDocument(
        content=draw(st.text(min_size=10, max_size=200)),
        source=draw(st.text(min_size=5, max_size=50)),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        metadata=draw(st.dictionaries(
            keys=st.sampled_from(["has_deprecated", "has_fixme", "has_todo", "has_security"]),
            values=st.booleans(),
            min_size=0,
            max_size=4
        ))
    )


# Strategy for generating SharedState with pivot triggered
@st.composite
def pivot_state_strategy(draw, pivot_reason=None):
    """Generate a SharedState with pivot triggered"""
    if pivot_reason is None:
        pivot_reason = draw(st.sampled_from([
            "deprecation_conflict",
            "complexity_trigger",
            "missing_information"
        ]))
    
    num_steps = draw(st.integers(min_value=2, max_value=10))
    current_index = draw(st.integers(min_value=0, max_value=num_steps - 1))
    
    outline = [
        OutlineStep(
            step_id=i,
            description=f"Step {i} description",
            status="pending" if i > current_index else "in_progress" if i == current_index else "completed",
            retry_count=0 if i != current_index else draw(st.integers(min_value=0, max_value=2))
        )
        for i in range(num_steps)
    ]
    
    # Generate some retrieved documents (at least 1)
    num_docs = draw(st.integers(min_value=1, max_value=10))
    retrieved_docs = [draw(retrieved_doc_strategy()) for _ in range(num_docs)]
    
    return SharedState(
        user_topic=draw(st.text(min_size=10, max_size=100)),
        project_context=draw(st.text(min_size=0, max_size=100)),
        outline=outline,
        current_step_index=current_index,
        retrieved_docs=retrieved_docs,
        current_skill=draw(st.sampled_from(list(SKILLS.keys()))),
        global_tone=draw(st.sampled_from(["professional", "cautionary", "engaging"])),
        pivot_triggered=True,
        pivot_reason=pivot_reason,
        max_retries=3
    )


@given(state=pivot_state_strategy())
@settings(max_examples=100)
def test_pivot_clears_retrieved_documents(state: SharedState):
    """
    Property 20: Pivot Triggers Re-retrieval
    
    For any pivot trigger (regardless of reason), the system should:
    1. Clear all retrieved documents
    2. This forces the Navigator to re-retrieve with updated context
    
    **Validates: Requirements 12.5**
    """
    # Ensure we have retrieved documents
    assume(len(state.retrieved_docs) > 0)
    
    original_doc_count = len(state.retrieved_docs)
    original_docs = state.retrieved_docs.copy()
    
    # Handle pivot
    result_state = handle_pivot(state)
    
    # Property 1: Retrieved documents should be cleared
    assert len(result_state.retrieved_docs) == 0, \
        f"Pivot should clear retrieved documents. " \
        f"Had {original_doc_count} docs, now has {len(result_state.retrieved_docs)}"
    
    # Property 2: Original documents should not be in result
    for doc in original_docs:
        assert doc not in result_state.retrieved_docs, \
            f"Original document should not be in result: {doc.source}"


@given(
    pivot_reason=st.sampled_from([
        "deprecation_conflict",
        "complexity_trigger",
        "missing_information"
    ])
)
@settings(max_examples=50)
def test_all_pivot_reasons_trigger_retrieval(pivot_reason: str):
    """
    Property: All pivot reasons should trigger re-retrieval by clearing documents.
    
    **Validates: Requirements 12.5**
    """
    state = SharedState(
        user_topic="Test topic",
        outline=[
            OutlineStep(step_id=0, description="Test step", status="in_progress", retry_count=0)
        ],
        current_step_index=0,
        retrieved_docs=[
            RetrievedDocument(
                content="Test content",
                source="test.py",
                confidence=0.9,
                metadata={}
            )
        ],
        current_skill="standard_tutorial",
        global_tone="professional",
        pivot_triggered=True,
        pivot_reason=pivot_reason,
        max_retries=3
    )
    
    result = handle_pivot(state)
    
    # All pivot reasons should clear documents
    assert len(result.retrieved_docs) == 0, \
        f"Pivot reason '{pivot_reason}' should clear retrieved documents"


@given(state=pivot_state_strategy())
@settings(max_examples=100)
def test_pivot_enables_context_update(state: SharedState):
    """
    Property: Pivot should enable context update by clearing docs AND modifying outline.
    
    When both outline is modified and docs are cleared, the Navigator will
    retrieve with the new context on the next iteration.
    
    **Validates: Requirements 12.5**
    """
    assume(len(state.retrieved_docs) > 0)
    
    original_step = state.get_current_step()
    assume(original_step is not None)
    original_desc = original_step.description
    original_retry = original_step.retry_count
    
    # Handle pivot
    result_state = handle_pivot(state)
    
    # Property 1: Documents cleared
    assert len(result_state.retrieved_docs) == 0
    
    # Property 2: Outline modified (at least current step)
    modified_step = result_state.get_current_step()
    assert modified_step is not None
    assert modified_step.description != original_desc, \
        "Pivot should modify outline to provide new context"
    
    # Property 3: Retry count incremented (shows step will be retried)
    assert modified_step.retry_count == original_retry + 1, \
        f"Retry count should increment from {original_retry} to {original_retry + 1}"


@given(state=pivot_state_strategy())
@settings(max_examples=100)
def test_multiple_pivots_always_clear_docs(state: SharedState):
    """
    Property: Multiple consecutive pivots should always clear documents.
    
    Even if documents were already cleared, pivot should ensure they remain cleared.
    
    **Validates: Requirements 12.5**
    """
    # First pivot
    result1 = handle_pivot(state)
    assert len(result1.retrieved_docs) == 0
    
    # Simulate adding docs back (as Navigator would do)
    result1.retrieved_docs = [
        RetrievedDocument(
            content="New content",
            source="new.py",
            confidence=0.8,
            metadata={}
        )
    ]
    
    # Trigger another pivot
    result1.pivot_triggered = True
    result1.pivot_reason = "complexity_trigger"
    
    result2 = handle_pivot(result1)
    
    # Should clear again
    assert len(result2.retrieved_docs) == 0, \
        "Subsequent pivot should also clear documents"


# Example-based tests
def test_deprecation_pivot_clears_docs():
    """Test that deprecation pivot clears documents"""
    state = SharedState(
        user_topic="Test",
        outline=[OutlineStep(step_id=0, description="Test", status="in_progress", retry_count=0)],
        current_step_index=0,
        retrieved_docs=[
            RetrievedDocument(content="Doc 1", source="a.py", confidence=0.9, metadata={}),
            RetrievedDocument(content="Doc 2", source="b.py", confidence=0.8, metadata={}),
        ],
        pivot_triggered=True,
        pivot_reason="deprecation_conflict",
        max_retries=3
    )
    
    result = handle_pivot(state)
    assert len(result.retrieved_docs) == 0


def test_complexity_pivot_clears_docs():
    """Test that complexity pivot clears documents"""
    state = SharedState(
        user_topic="Test",
        outline=[OutlineStep(step_id=0, description="Test", status="in_progress", retry_count=0)],
        current_step_index=0,
        retrieved_docs=[
            RetrievedDocument(content="Doc 1", source="a.py", confidence=0.9, metadata={})
        ],
        pivot_triggered=True,
        pivot_reason="complexity_trigger",
        max_retries=3
    )
    
    result = handle_pivot(state)
    assert len(result.retrieved_docs) == 0


def test_missing_info_pivot_clears_docs():
    """Test that missing information pivot clears documents"""
    state = SharedState(
        user_topic="Test",
        outline=[OutlineStep(step_id=0, description="Test", status="in_progress", retry_count=0)],
        current_step_index=0,
        retrieved_docs=[
            RetrievedDocument(content="Doc 1", source="a.py", confidence=0.9, metadata={})
        ],
        pivot_triggered=True,
        pivot_reason="missing_information",
        max_retries=3
    )
    
    result = handle_pivot(state)
    assert len(result.retrieved_docs) == 0


def test_empty_docs_remain_empty():
    """Test that pivot with no docs keeps docs empty"""
    state = SharedState(
        user_topic="Test",
        outline=[OutlineStep(step_id=0, description="Test", status="in_progress", retry_count=0)],
        current_step_index=0,
        retrieved_docs=[],  # Already empty
        pivot_triggered=True,
        pivot_reason="deprecation_conflict",
        max_retries=3
    )
    
    result = handle_pivot(state)
    assert len(result.retrieved_docs) == 0


def test_pivot_log_mentions_retrieval():
    """Test that pivot log entry mentions re-retrieval"""
    state = SharedState(
        user_topic="Test",
        outline=[OutlineStep(step_id=0, description="Test", status="in_progress", retry_count=0)],
        current_step_index=0,
        retrieved_docs=[
            RetrievedDocument(content="Doc", source="a.py", confidence=0.9, metadata={})
        ],
        pivot_triggered=True,
        pivot_reason="deprecation_conflict",
        max_retries=3
    )
    
    result = handle_pivot(state)
    
    # Check that documents were cleared (implicit re-retrieval trigger)
    assert len(result.retrieved_docs) == 0
    
    # Check log entry exists
    assert len(result.execution_log) > 0
    last_log = result.execution_log[-1]
    assert last_log["agent_name"] == "pivot_manager"
