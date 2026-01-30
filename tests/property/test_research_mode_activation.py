"""Property-Based Tests for Research Mode Activation

Feature: rag-screenplay-multi-agent
Property 12: 研究模式激活

属性描述:
对于任何 RAG 检索返回无结果的大纲步骤，系统应切换到 research_mode Skill
并生成明确指出信息缺口的片段。
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from src.domain.models import SharedState, OutlineStep, RetrievedDocument, ScreenplayFragment
from src.domain.agents.writer import generate_fragment, handle_user_input_resume
from src.services.llm.service import LLMService
from unittest.mock import AsyncMock, MagicMock
import logging

logger = logging.getLogger(__name__)


# Strategy for generating valid skill names (excluding research_mode)
@st.composite
def non_research_skill_strategy(draw):
    """Generate a valid skill name that is not research_mode"""
    valid_skills = [
        "standard_tutorial",
        "warning_mode",
        "visualization_analogy",
        "meme_style",
        "fallback_summary"
    ]
    return draw(st.sampled_from(valid_skills))


# Strategy for generating SharedState with empty retrieval
@st.composite
def empty_retrieval_state_strategy(draw):
    """Generate a SharedState with empty retrieved_docs"""
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
    current_skill = draw(non_research_skill_strategy())
    
    state = SharedState(
        user_topic=user_topic,
        project_context=project_context,
        outline=outline,
        current_step_index=current_step_index,
        retrieved_docs=[],  # Empty retrieval
        current_skill=current_skill,
        pivot_triggered=False
    )
    
    return state


@pytest.mark.asyncio
@pytest.mark.property
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(state=empty_retrieval_state_strategy())
async def test_property_research_mode_activation(state):
    """
    属性 12: 研究模式激活
    
    验证: 需求 7.2, 7.3
    
    对于任何 RAG 检索返回无结果的大纲步骤，系统应切换到 research_mode Skill
    并生成明确指出信息缺口的片段。
    """
    # Get current step
    current_step = state.get_current_step()
    assert current_step is not None, "State should have a current step"
    
    original_skill = state.current_skill
    original_fragment_count = len(state.fragments)
    
    # Verify precondition: empty retrieval
    assert len(state.retrieved_docs) == 0, "Precondition: retrieved_docs should be empty"
    assert original_skill != "research_mode", "Precondition: should not already be in research_mode"
    
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
        f"but current_skill is '{updated_state.current_skill}'"
    )
    
    # Property 2: Should set awaiting_user_input flag
    assert updated_state.awaiting_user_input is True, (
        "Should set awaiting_user_input flag when switching to research_mode"
    )
    
    # Property 3: Should provide user_input_prompt
    assert updated_state.user_input_prompt is not None, (
        "Should provide user_input_prompt when awaiting user input"
    )
    assert len(updated_state.user_input_prompt) > 0, (
        "user_input_prompt should not be empty"
    )
    
    # Property 4: user_input_prompt should mention the step
    assert str(current_step.step_id) in updated_state.user_input_prompt, (
        "user_input_prompt should mention the step_id"
    )
    
    # Property 5: Should generate exactly one fragment
    assert len(updated_state.fragments) == original_fragment_count + 1, (
        f"Should generate one fragment, but got {len(updated_state.fragments) - original_fragment_count}"
    )
    
    # Property 6: Fragment should use research_mode
    fragment = updated_state.fragments[-1]
    assert fragment.skill_used == "research_mode", (
        f"Fragment should use research_mode, but uses '{fragment.skill_used}'"
    )
    
    # Property 7: Fragment should have correct step_id
    assert fragment.step_id == current_step.step_id, (
        f"Fragment step_id ({fragment.step_id}) should match current step ({current_step.step_id})"
    )
    
    # Property 8: Fragment content should acknowledge information gap
    content_lower = fragment.content.lower()
    gap_indicators = ["信息不足", "需要进一步研究", "research needed", "information gap"]
    has_gap_indicator = any(indicator in content_lower for indicator in gap_indicators)
    assert has_gap_indicator, (
        f"Fragment content should acknowledge information gap. Content: {fragment.content[:200]}"
    )
    
    # Property 9: Fragment should have empty sources (no retrieval)
    assert fragment.sources == [], (
        f"Fragment should have empty sources for empty retrieval, but got {fragment.sources}"
    )
    
    # Property 10: LLM should NOT be called (direct generation)
    mock_llm_service.chat_completion.assert_not_called()
    
    # Property 11: Should log the skill switch
    log_entries = [log for log in updated_state.execution_log if log.get("action") == "empty_retrieval_research_mode"]
    assert len(log_entries) > 0, (
        "Should log the skill switch to research_mode"
    )
    
    logger.info(
        f"✓ Property 12 verified: Empty retrieval triggered research_mode "
        f"(original skill: {original_skill})"
    )


@pytest.mark.asyncio
@pytest.mark.property
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(
    user_topic=st.text(min_size=10, max_size=100),
    step_description=st.text(min_size=10, max_size=100),
    original_skill=non_research_skill_strategy()
)
async def test_property_research_mode_content_requirements(user_topic, step_description, original_skill):
    """
    属性 12.1: 研究模式内容要求
    
    验证: 需求 7.3
    
    当切换到 research_mode 时，生成的片段应明确指出信息缺口，
    并提供研究建议，而不是编造内容。
    """
    # Create state with empty retrieval
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
        retrieved_docs=[],
        current_skill=original_skill
    )
    
    # Create mock LLM service
    mock_llm_service = MagicMock(spec=LLMService)
    mock_llm_service.chat_completion = AsyncMock(
        return_value="Should not be called"
    )
    
    # Generate fragment
    updated_state = await generate_fragment(state, mock_llm_service)
    
    fragment = updated_state.fragments[0]
    content = fragment.content
    
    # Property 1: Content should acknowledge information gap
    gap_phrases = ["信息不足", "需要进一步研究", "缺少相关信息"]
    has_gap_acknowledgment = any(phrase in content for phrase in gap_phrases)
    assert has_gap_acknowledgment, (
        f"Content should acknowledge information gap. Content: {content[:200]}"
    )
    
    # Property 2: Content should provide research suggestions
    suggestion_phrases = ["建议", "查找", "检查", "咨询", "提供更多"]
    has_suggestions = any(phrase in content for phrase in suggestion_phrases)
    assert has_suggestions, (
        f"Content should provide research suggestions. Content: {content[:200]}"
    )
    
    # Property 3: Content should NOT contain specific code examples
    # (since there's no retrieval, it shouldn't fabricate code)
    code_indicators = ["def ", "class ", "function ", "import ", "const ", "let "]
    has_code = any(indicator in content for indicator in code_indicators)
    assert not has_code, (
        f"Content should not contain fabricated code examples. Content: {content[:200]}"
    )
    
    # Property 4: Content should mention the step description
    assert step_description in content or any(word in content for word in step_description.split()[:3]), (
        f"Content should reference the step description. Content: {content[:200]}"
    )
    
    logger.info(
        f"✓ Property 12.1 verified: Research mode content meets requirements"
    )


@pytest.mark.asyncio
@pytest.mark.property
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(
    user_topic=st.text(min_size=10, max_size=100),
    step_description=st.text(min_size=10, max_size=100),
    user_input=st.text(min_size=20, max_size=200)
)
async def test_property_research_mode_user_input_resume(user_topic, step_description, user_input):
    """
    属性 12.2: 研究模式用户输入恢复
    
    验证: 需求 7.5, 7.6, 7.7
    
    当用户提供输入后，系统应恢复执行，清除 awaiting_user_input 标志，
    并使用用户输入作为额外上下文重新生成片段。
    """
    # Create state in research_mode awaiting user input
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
        retrieved_docs=[],
        current_skill="research_mode",
        awaiting_user_input=True,
        user_input_prompt="Please provide more information"
    )
    
    # Add initial research_mode fragment
    state.fragments.append(
        ScreenplayFragment(
            step_id=0,
            content="Information gap acknowledged",
            skill_used="research_mode",
            sources=[]
        )
    )
    
    # Create mock LLM service
    mock_llm_service = MagicMock(spec=LLMService)
    mock_llm_service.chat_completion = AsyncMock(
        return_value=f"Generated content based on user input: {user_input[:50]}"
    )
    
    # Handle user input and resume
    updated_state = await handle_user_input_resume(state, user_input, mock_llm_service)
    
    # Property 1: Should clear awaiting_user_input flag
    assert updated_state.awaiting_user_input is False, (
        "Should clear awaiting_user_input flag after user input"
    )
    
    # Property 2: Should clear user_input_prompt
    assert updated_state.user_input_prompt is None, (
        "Should clear user_input_prompt after user input"
    )
    
    # Property 3: Should add user input as a retrieved document
    user_docs = [doc for doc in updated_state.retrieved_docs if doc.source == "user_input"]
    assert len(user_docs) == 1, (
        f"Should add user input as retrieved document, found {len(user_docs)}"
    )
    
    # Property 4: User document should contain the user input
    user_doc = user_docs[0]
    assert user_doc.content == user_input, (
        "User document content should match user input"
    )
    
    # Property 5: User document should have confidence 1.0
    assert user_doc.confidence == 1.0, (
        f"User document should have confidence 1.0, got {user_doc.confidence}"
    )
    
    # Property 6: Should switch away from research_mode
    assert updated_state.current_skill != "research_mode", (
        f"Should switch away from research_mode, but current_skill is '{updated_state.current_skill}'"
    )
    
    # Property 7: Should generate a new fragment
    assert len(updated_state.fragments) == 2, (
        f"Should have 2 fragments (initial + new), got {len(updated_state.fragments)}"
    )
    
    # Property 8: New fragment should not use research_mode
    new_fragment = updated_state.fragments[-1]
    assert new_fragment.skill_used != "research_mode", (
        f"New fragment should not use research_mode, got '{new_fragment.skill_used}'"
    )
    
    # Property 9: LLM should be called with user input context
    mock_llm_service.chat_completion.assert_called_once()
    
    # Property 10: Should log the resume action
    log_entries = [log for log in updated_state.execution_log if log.get("action") == "resume_with_user_input"]
    assert len(log_entries) > 0, (
        "Should log the resume action"
    )
    
    logger.info(
        f"✓ Property 12.2 verified: User input correctly resumed execution"
    )


@pytest.mark.asyncio
@pytest.mark.property
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(
    user_topic=st.text(min_size=10, max_size=100),
    num_empty_steps=st.integers(min_value=2, max_value=5)
)
async def test_property_research_mode_multiple_empty_steps(user_topic, num_empty_steps):
    """
    属性 12.3: 多个空检索步骤的研究模式
    
    验证: 需求 7.2, 7.3
    
    当连续多个步骤都有空检索时，每个步骤都应独立触发 research_mode。
    """
    # Create state with multiple steps
    outline = [
        OutlineStep(
            step_id=i,
            description=f"Step {i} description",
            status="in_progress",
            retry_count=0
        )
        for i in range(num_empty_steps)
    ]
    
    # Create mock LLM service
    mock_llm_service = MagicMock(spec=LLMService)
    mock_llm_service.chat_completion = AsyncMock(
        return_value="Should not be called"
    )
    
    # Process each step
    for step_index in range(num_empty_steps):
        state = SharedState(
            user_topic=user_topic,
            project_context="",
            outline=outline,
            current_step_index=step_index,
            retrieved_docs=[],  # Empty for each step
            current_skill="standard_tutorial"
        )
        
        # Generate fragment
        updated_state = await generate_fragment(state, mock_llm_service)
        
        # Property 1: Each step should trigger research_mode
        assert updated_state.current_skill == "research_mode", (
            f"Step {step_index} should trigger research_mode"
        )
        
        # Property 2: Each step should set awaiting_user_input
        assert updated_state.awaiting_user_input is True, (
            f"Step {step_index} should set awaiting_user_input"
        )
        
        # Property 3: Each step should generate a fragment
        assert len(updated_state.fragments) == 1, (
            f"Step {step_index} should generate one fragment"
        )
        
        # Property 4: Each fragment should use research_mode
        fragment = updated_state.fragments[0]
        assert fragment.skill_used == "research_mode", (
            f"Step {step_index} fragment should use research_mode"
        )
        
        # Property 5: Each fragment should have correct step_id
        assert fragment.step_id == step_index, (
            f"Fragment step_id should be {step_index}, got {fragment.step_id}"
        )
    
    logger.info(
        f"✓ Property 12.3 verified: {num_empty_steps} empty steps each triggered research_mode"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
