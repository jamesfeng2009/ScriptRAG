"""Property-Based Tests for Approval Triggering Generation

Feature: rag-screenplay-multi-agent
Property 21: 批准触发生成

属性描述:
当导演批准内容时（没有触发转向），编剧应生成剧本片段。
生成的片段应包含正确的 step_id、使用的 Skill 和来源列表。
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from src.domain.models import SharedState, OutlineStep, RetrievedDocument, ScreenplayFragment
from src.domain.agents.writer import generate_fragment
from src.services.llm.service import LLMService
from unittest.mock import AsyncMock, MagicMock, patch
import logging

logger = logging.getLogger(__name__)


# Strategy for generating valid skill names
@st.composite
def skill_name_strategy(draw):
    """Generate a valid skill name"""
    valid_skills = [
        "standard_tutorial",
        "warning_mode",
        "visualization_analogy",
        "research_mode",
        "meme_style",
        "fallback_summary"
    ]
    return draw(st.sampled_from(valid_skills))


# Strategy for generating SharedState with approval conditions
@st.composite
def approved_state_strategy(draw):
    """Generate a SharedState that represents an approved state (no pivot)"""
    user_topic = draw(st.text(min_size=10, max_size=100))
    project_context = draw(st.text(min_size=0, max_size=100))
    
    # Generate outline with at least one step
    num_steps = draw(st.integers(min_value=1, max_value=5))
    outline = []
    for i in range(num_steps):
        step = OutlineStep(
            step_id=i,
            description=draw(st.text(min_size=10, max_size=100)),
            status="in_progress",
            retry_count=draw(st.integers(min_value=0, max_value=2))
        )
        outline.append(step)
    
    current_step_index = draw(st.integers(min_value=0, max_value=num_steps-1))
    
    # Generate retrieved documents (non-empty for approval)
    num_docs = draw(st.integers(min_value=1, max_value=3))
    retrieved_docs = []
    for _ in range(num_docs):
        doc = RetrievedDocument(
            content=draw(st.text(min_size=50, max_size=200)),
            source=draw(st.text(min_size=5, max_size=30, alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='/_.-'))) or "test.py",
            confidence=draw(st.floats(min_value=0.5, max_value=1.0)),
            metadata={}
        )
        retrieved_docs.append(doc)
    
    current_skill = draw(skill_name_strategy())
    
    state = SharedState(
        user_topic=user_topic,
        project_context=project_context,
        outline=outline,
        current_step_index=current_step_index,
        retrieved_docs=retrieved_docs,
        current_skill=current_skill,
        pivot_triggered=False,  # No pivot - approved
        pivot_reason=None
    )
    
    return state


@pytest.mark.asyncio
@pytest.mark.property
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(state=approved_state_strategy())
async def test_property_approval_triggers_generation(state):
    """
    属性 21: 批准触发生成
    
    验证: 需求 12.6
    
    当导演批准内容时（pivot_triggered=False），编剧应生成剧本片段。
    生成的片段应包含正确的 step_id、使用的 Skill 和来源列表。
    """
    # Get current step before generation
    current_step = state.get_current_step()
    assert current_step is not None, "State should have a current step"
    
    original_fragment_count = len(state.fragments)
    expected_step_id = current_step.step_id
    expected_skill = state.current_skill
    expected_sources = [doc.source for doc in state.retrieved_docs]
    
    # Create mock LLM service
    mock_llm_service = MagicMock(spec=LLMService)
    mock_llm_service.chat_completion = AsyncMock(
        return_value=f"Generated content for step {expected_step_id} using {expected_skill}"
    )
    
    # Generate fragment
    updated_state = await generate_fragment(state, mock_llm_service)
    
    # Property 1: A new fragment should be added
    assert len(updated_state.fragments) == original_fragment_count + 1, (
        f"Expected {original_fragment_count + 1} fragments, "
        f"got {len(updated_state.fragments)}"
    )
    
    # Property 2: The new fragment should have the correct step_id
    new_fragment = updated_state.fragments[-1]
    assert new_fragment.step_id == expected_step_id, (
        f"Fragment step_id ({new_fragment.step_id}) "
        f"does not match current step ({expected_step_id})"
    )
    
    # Property 3: The fragment should use the correct skill
    assert new_fragment.skill_used == expected_skill, (
        f"Fragment skill_used ({new_fragment.skill_used}) "
        f"does not match current_skill ({expected_skill})"
    )
    
    # Property 4: The fragment should have sources from retrieved documents
    assert new_fragment.sources == expected_sources, (
        f"Fragment sources ({new_fragment.sources}) "
        f"do not match retrieved document sources ({expected_sources})"
    )
    
    # Property 5: The fragment should have non-empty content
    assert len(new_fragment.content) > 0, "Fragment content should not be empty"
    
    # Property 6: The current step status should be updated to completed
    assert current_step.status == "completed", (
        f"Current step status should be 'completed', got '{current_step.status}'"
    )
    
    # Property 7: LLM should have been called
    mock_llm_service.chat_completion.assert_called_once()
    
    logger.info(
        f"✓ Property 21 verified: Approval triggered generation for step {expected_step_id} "
        f"with skill '{expected_skill}' and {len(expected_sources)} sources"
    )


@pytest.mark.asyncio
@pytest.mark.property
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(
    user_topic=st.text(min_size=10, max_size=100),
    step_description=st.text(min_size=10, max_size=100),
    skill_name=skill_name_strategy()
)
async def test_property_generation_with_empty_retrieval(user_topic, step_description, skill_name):
    """
    属性 21.1: 空检索时的生成行为
    
    验证: 需求 12.6, 7.2, 7.3
    
    当检索内容为空时，编剧应切换到 research_mode 并生成承认信息缺口的片段。
    """
    # Create state with empty retrieved_docs
    state = SharedState(
        user_topic=user_topic,
        project_context="",
        outline=[
            OutlineStep(
                step_id=0,
                description=step_description,
                status="in_progress",
                retry_count=0
            )
        ],
        current_step_index=0,
        retrieved_docs=[],  # Empty retrieval
        current_skill=skill_name,
        pivot_triggered=False
    )
    
    # Create mock LLM service (should not be called for empty retrieval)
    mock_llm_service = MagicMock(spec=LLMService)
    mock_llm_service.chat_completion = AsyncMock(
        return_value="This should not be called"
    )
    
    # Generate fragment
    updated_state = await generate_fragment(state, mock_llm_service)
    
    # Property 1: Should switch to research_mode
    assert updated_state.current_skill == "research_mode", (
        f"Should switch to research_mode for empty retrieval, "
        f"got '{updated_state.current_skill}'"
    )
    
    # Property 2: Should set awaiting_user_input flag
    assert updated_state.awaiting_user_input is True, (
        "Should set awaiting_user_input flag for empty retrieval"
    )
    
    # Property 3: Should provide user_input_prompt
    assert updated_state.user_input_prompt is not None, (
        "Should provide user_input_prompt for empty retrieval"
    )
    assert len(updated_state.user_input_prompt) > 0, (
        "user_input_prompt should not be empty"
    )
    
    # Property 4: Should generate a fragment acknowledging information gap
    assert len(updated_state.fragments) == 1, (
        "Should generate one fragment for empty retrieval"
    )
    
    fragment = updated_state.fragments[0]
    
    # Property 5: Fragment should use research_mode
    assert fragment.skill_used == "research_mode", (
        f"Fragment should use research_mode, got '{fragment.skill_used}'"
    )
    
    # Property 6: Fragment content should acknowledge information gap
    assert "信息不足" in fragment.content or "需要进一步研究" in fragment.content, (
        "Fragment should acknowledge information gap"
    )
    
    # Property 7: Fragment should have empty sources
    assert fragment.sources == [], (
        "Fragment should have empty sources for empty retrieval"
    )
    
    # Property 8: LLM should NOT have been called (direct generation)
    mock_llm_service.chat_completion.assert_not_called()
    
    logger.info(
        f"✓ Property 21.1 verified: Empty retrieval correctly handled with research_mode"
    )


@pytest.mark.asyncio
@pytest.mark.property
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(state=approved_state_strategy())
async def test_property_generation_failure_fallback(state):
    """
    属性 21.2: 生成失败时的回退行为
    
    验证: 需求 12.6, 18.2
    
    当 LLM 调用失败时，编剧应回退到 fallback_summary 并生成简单的回退片段。
    """
    # Get current step
    current_step = state.get_current_step()
    assert current_step is not None
    
    original_fragment_count = len(state.fragments)
    
    # Create mock LLM service that raises an exception
    mock_llm_service = MagicMock(spec=LLMService)
    mock_llm_service.chat_completion = AsyncMock(
        side_effect=Exception("LLM service unavailable")
    )
    
    # Generate fragment (should handle exception gracefully)
    updated_state = await generate_fragment(state, mock_llm_service)
    
    # Property 1: Should still generate a fragment (fallback)
    assert len(updated_state.fragments) == original_fragment_count + 1, (
        "Should generate fallback fragment even when LLM fails"
    )
    
    # Property 2: Should switch to fallback_summary
    assert updated_state.current_skill == "fallback_summary", (
        f"Should switch to fallback_summary on failure, "
        f"got '{updated_state.current_skill}'"
    )
    
    # Property 3: Fallback fragment should use fallback_summary
    fallback_fragment = updated_state.fragments[-1]
    assert fallback_fragment.skill_used == "fallback_summary", (
        f"Fallback fragment should use fallback_summary, "
        f"got '{fallback_fragment.skill_used}'"
    )
    
    # Property 4: Fallback fragment should have content
    assert len(fallback_fragment.content) > 0, (
        "Fallback fragment should have content"
    )
    
    # Property 5: Current step should be marked as completed
    assert current_step.status == "completed", (
        "Current step should be completed even with fallback"
    )
    
    # Property 6: Should log the error
    assert len(updated_state.execution_log) > 0, (
        "Should log the error"
    )
    
    logger.info(
        f"✓ Property 21.2 verified: Generation failure correctly handled with fallback"
    )


@pytest.mark.asyncio
@pytest.mark.property
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(
    user_topic=st.text(min_size=10, max_size=100),
    num_steps=st.integers(min_value=2, max_value=5)
)
async def test_property_generation_at_end_of_outline(user_topic, num_steps):
    """
    属性 21.3: 大纲末尾的生成行为
    
    验证: 需求 12.6
    
    当 current_step_index 指向最后一个步骤且所有步骤已完成时，
    编剧应优雅地处理而不崩溃。
    """
    # Create state with all steps completed
    outline = [
        OutlineStep(
            step_id=i,
            description=f"Step {i}",
            status="completed",
            retry_count=0
        )
        for i in range(num_steps)
    ]
    
    # Use last valid index
    state = SharedState(
        user_topic=user_topic,
        project_context="",
        outline=outline,
        current_step_index=num_steps - 1,  # Last valid index
        retrieved_docs=[],
        current_skill="standard_tutorial"
    )
    
    # Create mock LLM service
    mock_llm_service = MagicMock(spec=LLMService)
    mock_llm_service.chat_completion = AsyncMock(
        return_value="Should not be called for empty retrieval"
    )
    
    # Generate fragment (should handle gracefully with empty retrieval)
    updated_state = await generate_fragment(state, mock_llm_service)
    
    # Property 1: Should not crash
    assert updated_state is not None, "Should return a state object"
    
    # Property 2: Should generate a research_mode fragment for empty retrieval
    assert len(updated_state.fragments) == 1, (
        "Should generate research_mode fragment for empty retrieval"
    )
    
    # Property 3: Should switch to research_mode
    assert updated_state.current_skill == "research_mode", (
        "Should switch to research_mode for empty retrieval"
    )
    
    # Property 4: Should set awaiting_user_input
    assert updated_state.awaiting_user_input is True, (
        "Should set awaiting_user_input for empty retrieval"
    )
    
    logger.info(
        f"✓ Property 21.3 verified: Gracefully handled generation at end of outline"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
