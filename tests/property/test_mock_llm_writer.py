"""Property-based tests for mock LLM writer response format
属性 8: Writer Mock 响应具有最小长度
"""

import pytest
from hypothesis import given, strategies as st, settings
from tests.fixtures.realistic_mock_data import create_mock_llm_service


@settings(max_examples=100)
@given(
    step_title=st.text(min_size=5, max_size=100),
    keywords=st.text(min_size=5, max_size=50)
)
@pytest.mark.asyncio
async def test_writer_response_has_minimum_length(step_title, keywords):
    """
    Property 8: Writer Mock 响应具有最小长度
    
    For any writer fragment generation request, the Mock_LLM response should
    be non-empty text with at least 50 characters.
    """
    # Create mock LLM service
    mock_llm = create_mock_llm_service()
    
    # Create writer fragment generation message
    # Use a message that doesn't trigger other agent patterns
    message_content = f"请为以下步骤生成内容片段。标题：{step_title} 关键词：{keywords}"
    messages = [
        {"role": "system", "content": "你是一个内容生成助手"},
        {"role": "user", "content": message_content}
    ]
    
    # Get response from mock LLM
    response = await mock_llm.chat_completion(messages, task_type="test")
    
    # Verify response is a string
    assert isinstance(response, str), "Response must be a string"
    
    # Verify response is not empty
    assert len(response) > 0, "Response must not be empty"
    
    # Verify response has at least 50 characters
    assert len(response) >= 50, (
        f"Writer response must have at least 50 characters, "
        f"but got {len(response)} characters. Response: {response[:100]}"
    )
    
    # Verify response contains actual content (not just whitespace)
    assert response.strip(), "Response must contain non-whitespace content"
    
    # Verify response length after stripping is still >= 50
    assert len(response.strip()) >= 50, (
        f"Writer response (after stripping) must have at least 50 characters, "
        f"but got {len(response.strip())} characters"
    )
