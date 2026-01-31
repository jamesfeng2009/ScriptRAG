"""Property-based tests for mock LLM fact_checker response format
属性 4: Fact Checker Mock 响应匹配格式
"""

import pytest
from hypothesis import given, strategies as st, settings
from tests.fixtures.realistic_mock_data import create_mock_llm_service


@settings(max_examples=100)
@given(
    fragment_content=st.text(min_size=10, max_size=200),
    has_function=st.booleans()
)
@pytest.mark.asyncio
async def test_fact_checker_response_format(fragment_content, has_function):
    """
    Property 4: Fact Checker Mock 响应匹配格式
    
    For any fact_checker verification request, the Mock_LLM response should be
    exactly "VALID" or start with "INVALID\n-" followed by hallucination descriptions.
    """
    # Create mock LLM service
    mock_llm = create_mock_llm_service()
    
    # Create fact_checker message with verification request
    if has_function:
        # Include a known function to get VALID response
        message_content = f"请验证以下片段是否存在幻觉：{fragment_content} run_with_timeout() AsyncContextManager"
    else:
        # Don't include known functions to potentially get INVALID response
        message_content = f"请验证以下片段是否存在幻觉：{fragment_content}"
    
    messages = [
        {"role": "system", "content": "你是一个事实检查助手"},
        {"role": "user", "content": message_content}
    ]
    
    # Get response from mock LLM
    response = await mock_llm.chat_completion(messages, task_type="test")
    
    # Verify response format
    assert isinstance(response, str), "Response must be a string"
    assert len(response) > 0, "Response must not be empty"
    
    # Response must be either "VALID" or start with "INVALID\n-"
    is_valid_format = (
        response == "VALID" or
        response.startswith("INVALID\n-")
    )
    
    assert is_valid_format, (
        f"Fact checker response must be 'VALID' or start with 'INVALID\\n-', "
        f"but got: {response[:50]}"
    )
    
    # If INVALID, verify it has hallucination description
    if response.startswith("INVALID"):
        assert "\n-" in response, "INVALID response must contain hallucination list with '\\n-'"
        lines = response.split("\n")
        assert len(lines) >= 2, "INVALID response must have at least 2 lines"
        assert lines[0] == "INVALID", "First line must be exactly 'INVALID'"
        assert lines[1].startswith("-"), "Second line must start with '-'"
