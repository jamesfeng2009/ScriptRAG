"""Unit tests for fact check validation

This module tests that the mock fact checker correctly returns "VALID",
preventing regeneration loops.

Feature: fix-integration-test-mock-data
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, AsyncMock


@pytest.mark.asyncio
@settings(max_examples=10, deadline=None)
@given(
    content=st.text(min_size=50, max_size=500),
    fragment_id=st.integers(min_value=0, max_value=100)
)
async def test_mock_fact_checker_always_returns_valid(content, fragment_id):
    """Property 12: Mock fact checker always returns 'VALID'
    
    For any content fragment, the mock LLM should always return "VALID"
    when asked about fact checking.
    
    **Validates: Requirements 4.4**
    """
    from tests.fixtures.realistic_mock_data import create_mock_llm_service

    mock_llm = create_mock_llm_service()

    messages = [
        {"role": "system", "content": "你是一个事实核查专家，负责验证内容的准确性和一致性。"},
        {"role": "user", "content": f"请对以下内容进行事实核查，返回 VALID 或 INVALID：\n\n{content[:200]}..."}
    ]

    response = await mock_llm.chat_completion(messages, task_type="fact_check")

    assert response == "VALID", \
        f"Mock fact checker should always return 'VALID', but got: {response}"


@pytest.mark.asyncio
async def test_fact_check_response_prevents_regeneration():
    """Test that 'VALID' fact check responses prevent regeneration
    
    When the fact checker returns 'VALID', the workflow should not
    trigger any regeneration logic.
    
    **Validates: Requirements 4.4**
    """
    from tests.fixtures.realistic_mock_data import create_mock_llm_service

    mock_llm = create_mock_llm_service()

    messages = [
        {"role": "user", "content": "请核查以下内容：Python 是一种高级编程语言。"}
    ]

    response = await mock_llm.chat_completion(messages, task_type="fact_check")

    assert response == "VALID", \
        f"Expected 'VALID' to prevent regeneration, got: {response}"


@pytest.mark.asyncio
@settings(max_examples=5, deadline=None)
@given(
    topic=st.text(min_size=10, max_size=100)
)
async def test_fact_check_valid_regardless_of_topic(topic):
    """Property: Fact checker returns VALID regardless of content topic
    
    The mock fact checker should always return 'VALID' to prevent
    regeneration loops, regardless of the content topic.
    
    **Validates: Requirements 4.4**
    """
    from tests.fixtures.realistic_mock_data import create_mock_llm_service

    mock_llm = create_mock_llm_service()

    messages = [
        {"role": "user", "content": f"请核查关于 {topic} 的以下内容是否准确：这是关于 {topic} 的一段描述。"}
    ]

    response = await mock_llm.chat_completion(messages, task_type="fact_check")

    assert response == "VALID", (
        f"Mock fact checker must always return 'VALID' to prevent regeneration loops. "
        f"Got: {response} for topic={topic[:30]}"
    )


@pytest.mark.asyncio
async def test_mock_fact_check_response_format():
    """Test that mock fact checker returns clean 'VALID' string
    
    The response should be exactly 'VALID' without any extra whitespace
    or formatting that might cause parsing issues.
    
    **Validates: Requirements 4.4**
    """
    from tests.fixtures.realistic_mock_data import create_mock_llm_service

    mock_llm = create_mock_llm_service()

    messages = [
        {"role": "user", "content": "请核查：1 + 1 = 2"}
    ]

    response = await mock_llm.chat_completion(messages, task_type="fact_check")

    assert response.strip() == "VALID", \
        f"Response should be exactly 'VALID', got: '{response}'"

    assert response == "VALID", \
        f"No extra formatting expected, got: '{response}'"


@pytest.mark.asyncio
async def test_no_regeneration_when_fact_check_passes():
    """Test workflow pattern: no regeneration when fact check always passes
    
    This simulates the pattern where if the fact checker always returns 'VALID',
    the workflow should complete without triggering any regeneration logic.
    
    **Validates: Requirements 4.4**
    """
    from tests.fixtures.realistic_mock_data import create_mock_llm_service

    mock_llm = create_mock_llm_service()

    valid_count = 0
    regeneration_triggered = False

    for i in range(5):
        messages = [{"role": "user", "content": f"第{i+1}次事实核查：验证内容的一致性。"}]
        response = await mock_llm.chat_completion(messages, task_type="fact_check")

        if response == "VALID":
            valid_count += 1
        else:
            regeneration_triggered = True
            break

    assert valid_count == 5, \
        f"Expected 5 VALID responses, got: {valid_count}"

    assert regeneration_triggered is False, \
        "Regeneration should never be triggered when mock always returns VALID"


@pytest.mark.asyncio
@settings(max_examples=5, deadline=None)
@given(
    quality_score=st.floats(min_value=0.5, max_value=1.0),
    coherence_score=st.floats(min_value=0.5, max_value=1.0)
)
async def test_fact_check_response_with_various_scores(quality_score, coherence_score):
    """Property: Fact checker returns VALID regardless of evaluation scores
    
    The mock fact checker should return 'VALID' regardless of
    quality or coherence scores to prevent regeneration loops.
    
    **Validates: Requirements 4.4**
    """
    from tests.fixtures.realistic_mock_data import create_mock_llm_service

    mock_llm = create_mock_llm_service()

    messages = [
        {
            "role": "user",
            "content": f"质量分数: {quality_score:.2f}, 一致性分数: {coherence_score:.2f}。内容是否通过事实核查？"
        }
    ]

    response = await mock_llm.chat_completion(messages, task_type="fact_check")

    assert response == "VALID", (
        f"Mock fact checker must always return 'VALID'. "
        f"Got: {response} for scores (quality={quality_score:.2f}, coherence={coherence_score:.2f})"
    )
