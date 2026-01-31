"""Property-based tests for realistic mock data generators

This module tests the correctness properties of the mock data generation
functions to ensure they produce high-fidelity data for integration tests.
"""

import pytest
from hypothesis import given, strategies as st, settings
import re
from tests.fixtures.realistic_mock_data import (
    create_realistic_retrieval_results,
    create_realistic_code_examples
)


@given(query=st.text(min_size=1, max_size=100))
@settings(max_examples=100)
def test_mock_data_contains_realistic_python_patterns(query):
    """Property 1: Mock data contains realistic Python code patterns
    
    Feature: fix-integration-test-mock-data
    
    For any mock retrieval document generated, the content should contain
    at least one realistic Python pattern: function definitions (def),
    class definitions (class), import statements (import/from),
    async functions (async def), or decorators (@).
    
    **Validates: Requirements 1.1, 1.3**
    """
    # Generate mock data
    mock_results = create_realistic_retrieval_results(query, num_results=5)
    
    # Verify we got results
    assert len(mock_results) > 0
    
    # Check each result for realistic patterns
    for result in mock_results:
        content = result.content
        
        # Check for at least one realistic Python pattern
        has_function_def = bool(re.search(r'\bdef\s+\w+\s*\(', content))
        has_class_def = bool(re.search(r'\bclass\s+\w+', content))
        has_import = bool(re.search(r'\b(import|from)\s+\w+', content))
        has_async_def = bool(re.search(r'\basync\s+def\s+\w+', content))
        has_decorator = bool(re.search(r'@\w+', content))
        
        # At least one pattern should be present
        assert (
            has_function_def or
            has_class_def or
            has_import or
            has_async_def or
            has_decorator
        ), f"Mock data lacks realistic Python patterns: {content[:100]}"


@given(
    query=st.text(min_size=1, max_size=100),
    num_results=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100)
def test_mock_data_contains_multiple_pattern_types(query, num_results):
    """Property 1 (Extended): Mock data contains multiple pattern types
    
    Feature: fix-integration-test-mock-data
    
    Across all mock retrieval documents, at least 3 different pattern types
    should be present (functions, classes, imports, async, decorators).
    
    **Validates: Requirements 1.1, 1.3**
    """
    # Generate mock data
    mock_results = create_realistic_retrieval_results(query, num_results=num_results)
    
    # Aggregate all content
    all_content = " ".join([r.content for r in mock_results])
    
    # Count pattern types
    pattern_types = {
        "function_def": bool(re.search(r'\bdef\s+\w+\s*\(', all_content)),
        "class_def": bool(re.search(r'\bclass\s+\w+', all_content)),
        "import": bool(re.search(r'\b(import|from)\s+\w+', all_content)),
        "async_def": bool(re.search(r'\basync\s+def\s+\w+', all_content)),
        "decorator": bool(re.search(r'@\w+', all_content))
    }
    
    # Count how many pattern types are present
    present_patterns = sum(pattern_types.values())
    
    # Should have at least 3 different pattern types
    assert present_patterns >= 3, (
        f"Mock data should contain at least 3 pattern types, "
        f"found {present_patterns}: {pattern_types}"
    )


@given(query=st.text(min_size=1, max_size=100))
@settings(max_examples=100)
def test_mock_data_has_realistic_file_paths(query):
    """Property 3: Mock data has realistic file paths
    
    Feature: fix-integration-test-mock-data
    
    For any mock retrieval result, the file_path should match realistic
    Python project patterns (e.g., starts with "src/", "tests/", "docs/"
    or ends with ".py", ".md").
    
    **Validates: Requirements 1.5**
    """
    # Generate mock data
    mock_results = create_realistic_retrieval_results(query, num_results=5)
    
    # Verify we got results
    assert len(mock_results) > 0
    
    # Check each result for realistic file paths
    for result in mock_results:
        file_path = result.file_path
        
        # Check for realistic path patterns
        starts_with_common_dir = (
            file_path.startswith("src/") or
            file_path.startswith("tests/") or
            file_path.startswith("docs/") or
            file_path.startswith("lib/") or
            file_path.startswith("app/")
        )
        
        ends_with_common_ext = (
            file_path.endswith(".py") or
            file_path.endswith(".md") or
            file_path.endswith(".txt") or
            file_path.endswith(".rst")
        )
        
        # At least one pattern should match
        assert starts_with_common_dir or ends_with_common_ext, (
            f"File path '{file_path}' does not match realistic Python project patterns"
        )


@given(
    query=st.text(min_size=1, max_size=100),
    num_results=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100)
def test_mock_data_file_paths_are_valid(query, num_results):
    """Property 3 (Extended): Mock data file paths are valid
    
    Feature: fix-integration-test-mock-data
    
    All file paths should be non-empty strings without invalid characters.
    
    **Validates: Requirements 1.5**
    """
    # Generate mock data
    mock_results = create_realistic_retrieval_results(query, num_results=num_results)
    
    # Check each file path
    for result in mock_results:
        file_path = result.file_path
        
        # Should be non-empty
        assert len(file_path) > 0, "File path should not be empty"
        
        # Should not contain invalid characters
        invalid_chars = ['\0', '\n', '\r', '\t']
        for char in invalid_chars:
            assert char not in file_path, (
                f"File path contains invalid character: {repr(char)}"
            )
        
        # Should contain at least one path separator or extension
        assert '/' in file_path or '.' in file_path, (
            f"File path '{file_path}' should contain path separator or extension"
        )


@given(
    function_names=st.lists(
        st.sampled_from([
            'run_with_timeout',
            'gather_with_concurrency',
            'connect',
            'disconnect',
            'retry_async',
            'fetch_data_with_retry',
            'safe_execute',
            'execute_query',
            'AsyncContextManager',
            'DatabaseConnection',
            'TimeoutError',
            'AsyncOperationError'
        ]),
        min_size=1,
        max_size=5,
        unique=True
    )
)
@settings(max_examples=100, deadline=None)
def test_mock_data_contains_referenced_functions(function_names):
    """Property 2: Mock data contains referenced functions
    
    Feature: fix-integration-test-mock-data
    
    For any test fragment that references function names, class names, or
    method names, the mock retrieval documents should contain definitions
    of those entities with matching name patterns.
    
    **Validates: Requirements 1.2**
    """
    # Generate mock data
    mock_results = create_realistic_retrieval_results("test query", num_results=5)
    
    # Aggregate all content from mock data
    all_content = " ".join([r.content for r in mock_results])
    
    # For each function/class name in our test set
    for name in function_names:
        # Check if the name exists in mock data (using word boundary)
        pattern = r'\b' + re.escape(name) + r'\b'
        assert re.search(pattern, all_content), (
            f"Referenced function/class '{name}' not found in mock data. "
            f"Mock data should contain all referenced entities."
        )


@given(
    fragment_content=st.sampled_from([
        # Fragment referencing run_with_timeout
        "本节介绍Python的async/await语法。`run_with_timeout()`函数展示了如何使用超时保护。",
        # Fragment referencing AsyncContextManager
        "`AsyncContextManager`类展示了如何正确实现异步上下文管理器。",
        # Fragment referencing multiple functions
        "使用`gather_with_concurrency()`和`run_with_timeout()`可以控制并发。",
        # Fragment referencing retry_async decorator
        "装饰器`retry_async`提供了自动重试机制。",
        # Fragment referencing safe_execute
        "`safe_execute()`函数展示了如何安全地执行协程。",
        # Fragment referencing DatabaseConnection
        "`DatabaseConnection`类继承自`AsyncContextManager`。",
        # Fragment referencing execute_query
        "使用`execute_query()`方法可以执行数据库查询。"
    ])
)
@settings(max_examples=100, deadline=None)
def test_heuristic_verification_passes_with_complete_mock_data(fragment_content):
    """Property 13: Heuristic verification passes with complete mock data
    
    Feature: fix-integration-test-mock-data
    
    For any fragment that references only functions/classes present in the
    mock retrieval documents, the fact_checker's heuristic verification
    should return (True, []) indicating no hallucinations detected.
    
    **Validates: Requirements 1.2**
    """
    from src.domain.models import ScreenplayFragment, RetrievedDocument
    from src.domain.agents.fact_checker import _heuristic_verification
    
    # Generate mock retrieval results
    mock_results = create_realistic_retrieval_results("test query", num_results=5)
    
    # Convert to RetrievedDocument objects
    retrieved_docs = [
        RetrievedDocument(
            content=result.content,
            source=result.file_path,
            confidence=result.confidence,
            metadata=result.metadata
        )
        for result in mock_results
    ]
    
    # Create a fragment with the test content
    fragment = ScreenplayFragment(
        step_id=1,
        content=fragment_content,
        skill_used="standard_tutorial",
        sources=["doc1"]
    )
    
    # Run heuristic verification
    is_valid, hallucinations = _heuristic_verification(fragment, retrieved_docs)
    
    # Verification should pass (no hallucinations)
    assert is_valid, (
        f"Heuristic verification failed for fragment: {fragment_content}\n"
        f"Detected hallucinations: {hallucinations}\n"
        f"Mock data should contain all referenced functions/classes."
    )
    
    # Should have no hallucinations
    assert len(hallucinations) == 0, (
        f"Expected no hallucinations, but found: {hallucinations}"
    )


@given(
    fragment_content=st.sampled_from([
        # Fragment with non-existent function
        "本节介绍`nonexistent_function()`函数的使用方法。",
        # Fragment with non-existent class
        "`FakeClass`类提供了虚构的功能。",
        # Fragment with non-existent method
        "使用`imaginary_method()`可以实现不存在的功能。",
        # Fragment with multiple non-existent references
        "`fake_func()`和`another_fake()`都是编造的函数。"
    ])
)
@settings(max_examples=100, deadline=None)
def test_heuristic_verification_detects_hallucinations(fragment_content):
    """Property 13 (Negative): Heuristic verification detects hallucinations
    
    Feature: fix-integration-test-mock-data
    
    For any fragment that references functions/classes NOT present in the
    mock retrieval documents, the fact_checker's heuristic verification
    should return (False, hallucinations) with a non-empty list.
    
    **Validates: Requirements 1.2**
    """
    from src.domain.models import ScreenplayFragment, RetrievedDocument
    from src.domain.agents.fact_checker import _heuristic_verification
    
    # Generate mock retrieval results
    mock_results = create_realistic_retrieval_results("test query", num_results=5)
    
    # Convert to RetrievedDocument objects
    retrieved_docs = [
        RetrievedDocument(
            content=result.content,
            source=result.file_path,
            confidence=result.confidence,
            metadata=result.metadata
        )
        for result in mock_results
    ]
    
    # Create a fragment with non-existent references
    fragment = ScreenplayFragment(
        step_id=1,
        content=fragment_content,
        skill_used="standard_tutorial",
        sources=["doc1"]
    )
    
    # Run heuristic verification
    is_valid, hallucinations = _heuristic_verification(fragment, retrieved_docs)
    
    # Verification should fail (hallucinations detected)
    assert not is_valid, (
        f"Heuristic verification should detect hallucinations in: {fragment_content}\n"
        f"But returned valid=True"
    )
    
    # Should have detected hallucinations
    assert len(hallucinations) > 0, (
        f"Expected hallucinations to be detected, but list is empty"
    )
