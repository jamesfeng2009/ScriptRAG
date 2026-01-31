"""Property-based tests for mock LLM director response format
属性 5: Director 复杂度 Mock 响应是数字
属性 6: Director 评估 Mock 响应是精确的
"""

import pytest
from hypothesis import given, strategies as st, settings
from tests.fixtures.realistic_mock_data import create_mock_llm_service


@settings(max_examples=100)
@given(
    query_content=st.text(min_size=10, max_size=200)
)
@pytest.mark.asyncio
async def test_director_complexity_response_is_numeric(query_content):
    """
    Property 5: Director 复杂度 Mock 响应是数字
    
    For any director complexity assessment request, the Mock_LLM response
    should be a numeric string representing a float between 0.0 and 1.0.
    """
    # Create mock LLM service
    mock_llm = create_mock_llm_service()
    
    # Create director complexity assessment message
    message_content = f"请评估以下查询的复杂度：{query_content}"
    messages = [
        {"role": "system", "content": "你是一个复杂度评估助手"},
        {"role": "user", "content": message_content}
    ]
    
    # Get response from mock LLM
    response = await mock_llm.chat_completion(messages, task_type="test")
    
    # Verify response is a string
    assert isinstance(response, str), "Response must be a string"
    assert len(response) > 0, "Response must not be empty"
    
    # Verify response can be converted to float
    try:
        complexity_value = float(response)
    except ValueError:
        pytest.fail(f"Director complexity response must be numeric, but got: {response}")
    
    # Verify value is between 0.0 and 1.0
    assert 0.0 <= complexity_value <= 1.0, (
        f"Director complexity must be between 0.0 and 1.0, but got: {complexity_value}"
    )


@settings(max_examples=100)
@given(
    outline_content=st.text(min_size=10, max_size=200),
    fragments_content=st.text(min_size=10, max_size=200)
)
@pytest.mark.asyncio
async def test_director_evaluation_response_is_exact(outline_content, fragments_content):
    """
    Property 6: Director 评估 Mock 响应是精确的
    
    For any director evaluation request, the Mock_LLM response should be
    exactly "approved" or "pivot_needed".
    """
    # Create mock LLM service
    mock_llm = create_mock_llm_service()
    
    # Create director evaluation message
    message_content = f"请评估大纲和片段的质量。大纲：{outline_content[:50]} 片段：{fragments_content[:50]}"
    messages = [
        {"role": "system", "content": "你是一个质量评估助手"},
        {"role": "user", "content": message_content}
    ]
    
    # Get response from mock LLM
    response = await mock_llm.chat_completion(messages, task_type="test")
    
    # Verify response is a string
    assert isinstance(response, str), "Response must be a string"
    assert len(response) > 0, "Response must not be empty"
    
    # Verify response is exactly "approved" or "pivot_needed"
    valid_responses = ["approved", "pivot_needed"]
    assert response in valid_responses, (
        f"Director evaluation response must be 'approved' or 'pivot_needed', "
        f"but got: {response}"
    )
