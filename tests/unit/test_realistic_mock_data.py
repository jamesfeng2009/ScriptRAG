"""Unit tests for realistic mock data generators

Tests the optimized mock data generators to ensure they produce
appropriate responses for simple and complex test scenarios.
"""

import pytest
from tests.fixtures.realistic_mock_data import (
    create_mock_llm_service,
    create_realistic_retrieval_results
)


@pytest.mark.asyncio
async def test_planner_simple_scenario_generates_3_steps():
    """Test that planner generates exactly 3 steps for simple scenarios
    """
    mock_llm = create_mock_llm_service()
    
    # Simple scenario message
    messages = [
        {"role": "user", "content": "生成大纲：简单的Python async/await介绍"}
    ]
    
    response = await mock_llm.chat_completion(messages, task_type="test")
    
    # Count steps in response
    lines = response.strip().split('\n')
    step_lines = [line for line in lines if line.strip().startswith('步骤')]
    
    # Should have exactly 3 steps for simple scenarios
    assert len(step_lines) == 3, f"Expected 3 steps for simple scenario, got {len(step_lines)}"
    
    # Verify format
    for line in step_lines:
        assert '|' in line, "Step should contain | separator"
        assert '关键词:' in line, "Step should contain keywords"


@pytest.mark.asyncio
async def test_planner_complex_scenario_generates_max_5_steps():
    """Test that planner generates at most 5 steps for complex scenarios
    """
    mock_llm = create_mock_llm_service()
    
    # Complex scenario message
    messages = [
        {"role": "user", "content": "生成大纲：完整的Python异步编程教程，包括基础、进阶和最佳实践"}
    ]
    
    response = await mock_llm.chat_completion(messages, task_type="test")
    
    # Count steps in response
    lines = response.strip().split('\n')
    step_lines = [line for line in lines if line.strip().startswith('步骤')]
    
    # Should have at most 5 steps for complex scenarios
    assert len(step_lines) <= 5, f"Expected at most 5 steps for complex scenario, got {len(step_lines)}"
    assert len(step_lines) >= 3, f"Expected at least 3 steps, got {len(step_lines)}"
    
    # Verify format
    for line in step_lines:
        assert '|' in line, "Step should contain | separator"
        assert '关键词:' in line, "Step should contain keywords"


@pytest.mark.asyncio
async def test_director_always_approves():
    """Test that director always returns 'approved' to prevent pivot loops
    """
    mock_llm = create_mock_llm_service()
    
    # Director evaluation message
    messages = [
        {"role": "user", "content": "评估大纲质量并决定是否批准"}
    ]
    
    response = await mock_llm.chat_completion(messages, task_type="test")
    
    # Should always return approved
    assert response == "approved", f"Expected 'approved', got '{response}'"


@pytest.mark.asyncio
async def test_fact_checker_returns_valid_for_proper_fragments():
    """Test that fact_checker returns VALID for properly formatted fragments
    """
    mock_llm = create_mock_llm_service()
    
    # Fact checker message with proper fragment referencing real functions
    messages = [
        {"role": "user", "content": """验证以下片段是否包含幻觉：

本节介绍run_with_timeout()函数，它使用asyncio.wait_for()来执行带超时保护的协程。
AsyncContextManager类展示了如何实现异步上下文管理器。"""}
    ]
    
    response = await mock_llm.chat_completion(messages, task_type="test")
    
    # Should return VALID for fragments with real function references
    assert response == "VALID", f"Expected 'VALID', got '{response}'"


@pytest.mark.asyncio
async def test_fact_checker_returns_invalid_for_nonexistent_functions():
    """Test that fact_checker returns INVALID for fragments with nonexistent functions
    """
    mock_llm = create_mock_llm_service()
    
    # Fact checker message with nonexistent function
    messages = [
        {"role": "user", "content": """验证以下片段是否包含幻觉：

本节介绍nonexistent_function()函数，它不存在于源文档中。"""}
    ]
    
    response = await mock_llm.chat_completion(messages, task_type="test")
    
    # Should return INVALID for fragments with nonexistent functions
    assert response.startswith("INVALID"), f"Expected response to start with 'INVALID', got '{response}'"
    assert "函数" in response or "未在源文档中找到" in response, "Should mention missing function"


@pytest.mark.asyncio
async def test_director_complexity_returns_numeric():
    """Test that director complexity assessment returns numeric string
    """
    mock_llm = create_mock_llm_service()
    
    # Director complexity message
    messages = [
        {"role": "user", "content": "评估主题复杂度"}
    ]
    
    response = await mock_llm.chat_completion(messages, task_type="test")
    
    # Should return numeric string
    assert response == "0.5", f"Expected '0.5', got '{response}'"
    
    # Should be parseable as float
    complexity = float(response)
    assert 0.0 <= complexity <= 1.0, f"Complexity should be between 0.0 and 1.0, got {complexity}"


def test_retrieval_results_contain_realistic_code():
    """Test that retrieval results contain realistic Python code patterns
    """
    results = create_realistic_retrieval_results("async/await", num_results=5)
    
    # Should return 5 results
    assert len(results) == 5
    
    # All results should have content
    for result in results:
        assert result.content is not None
        assert len(result.content) > 0
        
        # Should contain at least one realistic Python pattern
        content = result.content
        has_pattern = (
            'def ' in content or
            'class ' in content or
            'import ' in content or
            'async def' in content or
            '@' in content
        )
        assert has_pattern, f"Result should contain realistic Python code pattern"


def test_retrieval_results_have_realistic_file_paths():
    """Test that retrieval results have realistic file paths
    """
    results = create_realistic_retrieval_results("async/await", num_results=5)
    
    for result in results:
        file_path = result.file_path
        
        # Should have realistic path pattern
        has_realistic_path = (
            file_path.startswith('src/') or
            file_path.startswith('tests/') or
            file_path.startswith('docs/') or
            file_path.endswith('.py') or
            file_path.endswith('.md')
        )
        assert has_realistic_path, f"File path '{file_path}' should match realistic pattern"
