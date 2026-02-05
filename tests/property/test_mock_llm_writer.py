"""Property-based tests for mock LLM writer response format
属性 8: Writer Mock 响应具有最小长度
"""

import pytest
from hypothesis import given, strategies as st, settings
from tests.fixtures.realistic_mock_data import create_mock_llm_service


@settings(max_examples=100)
@given(
    step_title=st.text(min_size=5, max_size=100).filter(lambda x: "hallucination" not in x.lower()),
    keywords=st.text(min_size=5, max_size=50).filter(lambda x: "hallucination" not in x.lower())
)
@pytest.mark.asyncio
async def test_writer_response_has_minimum_length(step_title, keywords):
    """
    Property 8: Writer Mock 响应具有最小长度
    
    For any writer fragment generation request, the Mock_LLM response should
    be non-empty text with at least 50 characters.
    """
    mock_llm = create_mock_llm_service()
    
    message_content = f"生成内容片段。标题：{step_title}。关键词：{keywords}。"
    messages = [
        {"role": "system", "content": "你是一个专业的技术内容写作者"},
        {"role": "user", "content": message_content}
    ]
    
    response = await mock_llm.chat_completion(messages, task_type="writer")
    
    assert isinstance(response, str), "Response must be a string"
    assert len(response) > 0, "Response must not be empty"
    
    assert len(response) >= 50, (
        f"Writer response must have at least 50 characters, "
        f"but got {len(response)} characters. Response: {response[:100]}"
    )
    
    assert response.strip(), "Response must contain non-whitespace content"
    
    assert len(response.strip()) >= 50, (
        f"Writer response (after stripping) must have at least 50 characters, "
        f"but got {len(response.strip())} characters"
    )
