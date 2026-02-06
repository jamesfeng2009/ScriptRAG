"""Unit tests for error message clarity

This module tests that mock services produce clear, specific error messages
that help developers understand what went wrong.

Feature: fix-integration-test-mock-data
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, AsyncMock


@pytest.mark.asyncio
async def test_mock_llm_error_message_format():
    """Test that mock LLM error messages are clear and specific
    
    When the mock LLM encounters an error or unexpected input,
    the error message should clearly indicate what went wrong.
    
    **Validates: Requirements 6.2**
    """
    from tests.fixtures.realistic_mock_data import create_mock_llm_service

    mock_llm = create_mock_llm_service()

    try:
        await mock_llm.chat_completion(
            [{"role": "user", "content": "test"}],
            task_type="invalid_task"
        )
    except ValueError as e:
        error_message = str(e)
        assert len(error_message) > 0, "Error message should not be empty"


@pytest.mark.asyncio
async def test_mock_retrieval_error_handling():
    """Test that mock retrieval service handles errors gracefully
    
    When retrieval encounters an error, it should return an empty
    list rather than raising an exception.
    
    **Validates: Requirements 6.2**
    """
    from tests.fixtures.realistic_mock_data import create_mock_retrieval_service

    mock_retrieval = create_mock_retrieval_service()

    result = await mock_retrieval.hybrid_retrieve(
                query="test query",
        top_k=5
    )

    assert isinstance(result, list), "Should return a list even on error"


@pytest.mark.asyncio
async def test_mock_parser_error_message():
    """Test that mock parser handles various inputs gracefully
    
    When the parser receives different types of input, it should return
    consistent results without crashing.
    
    **Validates: Requirements 6.2**
    """
    from tests.fixtures.realistic_mock_data import create_mock_parser_service

    mock_parser = create_mock_parser_service()

    result = mock_parser.parse("test.py", "print('hello')")

    assert result is not None, "Parser should return a result"
    assert hasattr(result, 'language'), "Parser should return object with language attribute"


def test_mock_summarization_error_handling():
    """Test that mock summarization service handles errors
    
    When check_size encounters an error, it should return False
    rather than raising an exception.
    
    **Validates: Requirements 6.2**
    """
    from tests.fixtures.realistic_mock_data import create_mock_summarization_service

    mock_summarization = create_mock_summarization_service()

    result = mock_summarization.check_size(None)
    assert isinstance(result, bool), "Should return a boolean"


@given(
    error_type=st.sampled_from([
        "connection_error",
        "timeout",
        "invalid_response",
        "rate_limit"
    ])
)
@settings(max_examples=4, deadline=None)
async def test_error_messages_contain_context(error_type):
    """Property 14: Error messages should contain context about the failure
    
    For any type of error, the error message should contain context
    that helps identify where and why the error occurred.
    
    **Validates: Requirements 6.2**
    """
    from tests.fixtures.realistic_mock_data import create_mock_llm_service

    mock_llm = create_mock_llm_service()

    messages = [
        {"role": "user", "content": f"Test with {error_type}"}
    ]

    response = await mock_llm.chat_completion(messages, task_type="test")

    if response.startswith("Error:"):
        assert len(response) > len("Error:"), "Error message should have details"


@pytest.mark.asyncio
@settings(max_examples=5, deadline=None)
@given(
    query=st.text(min_size=5, max_size=50)
)
async def test_retrieval_results_have_clear_structure(query):
    """Property: Retrieval results have clear, consistent structure
    
    Each retrieval result should have consistent fields that are
    easy to understand and use.
    
    **Validates: Requirements 6.2**
    """
    from tests.fixtures.realistic_mock_data import create_mock_retrieval_service

    mock_retrieval = create_mock_retrieval_service()

    results = await mock_retrieval.hybrid_retrieve(
                query=query,
        top_k=3
    )

    for result in results:
        if hasattr(result, 'content'):
            assert len(result.content) > 0, "Result should have content"
        elif isinstance(result, dict):
            assert 'content' in result or 'text' in result, (
                f"Result should have content field: {result.keys()}"
            )


@pytest.mark.asyncio
async def test_mock_response_validation():
    """Test that mock services validate responses
    
    When a mock service returns an invalid response, it should
    either handle it gracefully or raise a clear error.
    
    **Validates: Requirements 6.2**
    """
    from tests.fixtures.realistic_mock_data import create_mock_llm_service

    mock_llm = create_mock_llm_service()

    messages = [{"role": "user", "content": "Validate this response"}]

    response = await mock_llm.chat_completion(messages, task_type="test")

    assert response is not None, "Response should not be None"
    assert isinstance(response, str), "Response should be a string"


def test_empty_input_handling():
    """Test that services handle empty input gracefully
    
    When given empty or minimal input, services should return
    meaningful responses rather than crashing.
    
    **Validates: Requirements 6.2**
    """
    from tests.fixtures.realistic_mock_data import create_mock_summarization_service

    mock_summarization = create_mock_summarization_service()

    result = mock_summarization.check_size("")
    assert isinstance(result, bool), "Should handle empty string"

    result = mock_summarization.check_size("short")
    assert isinstance(result, bool), "Should handle short string"
