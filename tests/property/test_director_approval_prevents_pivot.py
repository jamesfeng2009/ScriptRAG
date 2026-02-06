"""Unit tests for director approval behavior

This module tests that the director agent correctly returns "approved" decisions,
preventing pivot triggers.

Feature: fix-integration-test-mock-data
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, AsyncMock, patch
from src.domain.models import (
    SharedState,
    OutlineStep
)


@pytest.mark.asyncio
@settings(max_examples=10, deadline=None)
@given(
    quality_score=st.floats(min_value=0.0, max_value=1.0),
    confidence=st.floats(min_value=0.0, max_value=1.0)
)
async def test_director_always_returns_approved(quality_score, confidence):
    """Property 11: Director mock always returns 'approved'
    
    For any quality evaluation result, the mock LLM should always return
    "approved" when asked about director decisions.
    
    **Validates: Requirements 4.3**
    """
    from tests.fixtures.realistic_mock_data import create_mock_llm_service

    mock_llm = create_mock_llm_service()

    test_messages = [
        {"role": "system", "content": "你是一个专业的写作指导专家"},
        {"role": "user", "content": f"请评估内容质量并决定下一步操作。质量分数: {quality_score}, 置信度: {confidence}"}
    ]

    response = await mock_llm.chat_completion(test_messages, task_type="test")

    assert response == "approved", \
        f"Mock director should always return 'approved', but got: {response}"


@pytest.mark.asyncio
async def test_director_decision_prevents_pivot():
    """Test that approved decisions prevent pivot triggers
    
    When the director returns 'approved', the workflow should continue
    linearly without triggering any pivot events.
    
    **Validates: Requirements 4.3**
    """
    from tests.fixtures.realistic_mock_data import create_mock_llm_service

    mock_llm = create_mock_llm_service()

    test_messages = [
        {"role": "user", "content": "请评估以下内容质量：这是一个测试内容，质量良好。"}
    ]

    response = await mock_llm.chat_completion(test_messages, task_type="test")

    assert response == "approved", \
        f"Expected 'approved' decision to prevent pivot, got: {response}"


@pytest.mark.asyncio
@settings(max_examples=5, deadline=None)
@given(
    step_title=st.text(min_size=5, max_size=50),
    quality_level=st.sampled_from(["excellent", "good", "acceptable"])
)
async def test_director_approval_regardless_of_quality(step_title, quality_level):
    """Property: Director mock returns approved regardless of input quality
    
    The mock director should always return 'approved' to prevent pivot loops,
    regardless of the input quality level or step title.
    
    **Validates: Requirements 4.3**
    """
    from tests.fixtures.realistic_mock_data import create_mock_llm_service

    mock_llm = create_mock_llm_service()

    messages = [
        {"role": "system", "content": f"评估步骤：{step_title}"},
        {"role": "user", "content": f"质量等级：{quality_level}。请决定是否批准继续写作。"}
    ]

    response = await mock_llm.chat_completion(messages, task_type="test")

    assert response == "approved", (
        f"Mock director must always return 'approved' to prevent pivot loops. "
        f"Got: {response} for quality_level={quality_level}, step_title={step_title[:20]}"
    )


@pytest.mark.asyncio
async def test_mock_director_response_format():
    """Test that mock director returns clean 'approved' string
    
    The response should be exactly 'approved' without any extra whitespace
    or formatting that might cause parsing issues.
    
    **Validates: Requirements 4.3**
    """
    from tests.fixtures.realistic_mock_data import create_mock_llm_service

    mock_llm = create_mock_llm_service()

    messages = [
        {"role": "user", "content": "评估并决定：批准还是拒绝？"}
    ]

    response = await mock_llm.chat_completion(messages, task_type="test")

    assert response.strip() == "approved", \
        f"Response should be exactly 'approved', got: '{response}'"

    assert response == "approved", \
        f"No extra formatting expected, got: '{response}'"


@pytest.mark.asyncio
async def test_no_pivot_events_when_always_approved():
    """Test workflow pattern: no pivot events when director always approves
    
    This simulates the pattern where if the director always returns 'approved',
    the workflow should complete without triggering any pivot logic.
    
    **Validates: Requirements 4.3**
    """
    from tests.fixtures.realistic_mock_data import create_mock_llm_service

    mock_llm = create_mock_llm_service()

    decision_count = 0
    pivot_triggered = False

    for i in range(5):
        messages = [{"role": "user", "content": f"第{i+1}次质量评估"}]
        response = await mock_llm.chat_completion(messages, task_type="test")

        if response == "approved":
            decision_count += 1
        else:
            pivot_triggered = True
            break

    assert decision_count == 5, \
        f"Expected 5 approved decisions, got: {decision_count}"

    assert pivot_triggered is False, \
        "Pivot should never be triggered when mock always approves"
