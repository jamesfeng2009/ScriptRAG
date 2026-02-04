"""Property-based tests for mock LLM planner response format
属性 7: Planner Mock 响应匹配中文格式
"""

import pytest
import re
import json
from hypothesis import given, strategies as st, settings
from tests.fixtures.realistic_mock_data import create_mock_llm_service


@settings(max_examples=100)
@given(
    query_content=st.text(min_size=10, max_size=200)
)
@pytest.mark.asyncio
async def test_planner_response_matches_chinese_format(query_content):
    """
    Property 7: Planner Mock 响应匹配中文格式
    
    For any planner outline generation request, the Mock_LLM response should
    contain at least 3 steps in either:
    1. JSON format: {"steps": [{"step_id": 0, "title": "...", "description": "..."}, ...]}
    2. Line format: "步骤N: Title | 关键词: keywords"
    """
    # Create mock LLM service
    mock_llm = create_mock_llm_service()
    
    # Create planner outline generation message
    message_content = f"请为以下查询生成大纲：{query_content}"
    messages = [
        {"role": "system", "content": "你是一个大纲生成助手"},
        {"role": "user", "content": message_content}
    ]
    
    # Get response from mock LLM
    response = await mock_llm.chat_completion(messages, task_type="test")
    
    # Verify response is a string
    assert isinstance(response, str), "Response must be a string"
    assert len(response) > 0, "Response must not be empty"
    
    # Try to parse as JSON
    try:
        result = json.loads(response)
        if isinstance(result, dict) and "steps" in result:
            # JSON format
            steps = result["steps"]
            assert len(steps) >= 3, (
                f"Planner response must have at least 3 steps, but got {len(steps)} steps in JSON"
            )
            # Verify each step has required fields
            for i, step in enumerate(steps):
                assert "title" in step, f"Step {i} missing 'title' field"
                assert "description" in step or "description" in str(type(step)), f"Step {i} missing 'description' field"
            return
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    
    # Fall back to line format parsing
    lines = response.strip().split("\n")
    
    # Verify at least 3 lines
    assert len(lines) >= 3, (
        f"Planner response must have at least 3 steps, but got {len(lines)} lines. "
        f"Response: {response[:200]}"
    )
    
    # Verify each line matches the pattern "步骤N: Title | 关键词: keywords"
    pattern = r"步骤\d+:\s*.+\s*\|\s*关键词:\s*.+"
    
    matching_lines = 0
    for line in lines:
        if re.match(pattern, line):
            matching_lines += 1
    
    assert matching_lines >= 3, (
        f"Planner response must have at least 3 lines matching pattern "
        f"'步骤N: Title | 关键词: keywords', but only {matching_lines} lines matched. "
        f"Response: {response[:200]}"
    )
    
    # Verify each matching line has the required components
    for i, line in enumerate(lines[:5], 1):  # Check first 5 lines
        if re.match(pattern, line):
            # Verify line starts with "步骤" followed by a number
            assert line.startswith("步骤"), f"Line {i} must start with '步骤'"
            
            # Verify line contains "|" separator
            assert "|" in line, f"Line {i} must contain '|' separator"
            
            # Verify line contains "关键词:"
            assert "关键词:" in line, f"Line {i} must contain '关键词:'"
            
            # Split by "|" and verify both parts are non-empty
            parts = line.split("|")
            assert len(parts) == 2, f"Line {i} must have exactly 2 parts separated by '|'"
            
            title_part = parts[0].strip()
            keywords_part = parts[1].strip()
            
            assert len(title_part) > 0, f"Line {i} title part must not be empty"
            assert len(keywords_part) > 0, f"Line {i} keywords part must not be empty"
            assert keywords_part.startswith("关键词:"), f"Line {i} second part must start with '关键词:'"
