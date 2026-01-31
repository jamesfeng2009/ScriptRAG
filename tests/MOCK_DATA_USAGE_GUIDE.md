# Mock Data Usage Guide

## Overview

This guide explains how to use the realistic mock data fixtures in your integration tests. The mock data system provides high-fidelity simulations of LLM responses, retrieval results, and parser outputs.

## Quick Start

```python
from tests.fixtures.realistic_mock_data import (
    create_mock_llm_service,
    create_mock_retrieval_service,
    create_mock_parser_service,
    create_realistic_retrieval_results
)

# In your test
async def test_my_workflow():
    # Create mock services
    mock_llm = create_mock_llm_service()
    mock_retrieval = create_mock_retrieval_service()
    mock_parser = create_mock_parser_service()
    
    # Use in your test...
```

## Mock Data Components

### 1. Mock LLM Service

The mock LLM service provides realistic, format-compliant responses for all agent types.

#### Basic Usage

```python
from tests.fixtures.realistic_mock_data import create_mock_llm_service

mock_llm = create_mock_llm_service()

# The mock automatically detects agent type from message content
response = await mock_llm.chat_completion(
    messages=[{"role": "user", "content": "Generate outline..."}],
    task_type="test"
)
```

#### Response Formats by Agent

**Planner Agent:**
```python
# Input: Message containing "outline" or "步骤"
# Output: Chinese format with 3-5 steps
"""
步骤1: Introduction to async/await basics | 关键词: async, await, coroutines
步骤2: Understanding event loops | 关键词: event loop, asyncio
步骤3: Practical async examples | 关键词: examples, timeout
"""
```

**Director Agent (Complexity):**
```python
# Input: Message containing "complexity" or "复杂度"
# Output: Numeric string between 0.0 and 1.0
"0.5"
```

**Director Agent (Evaluation):**
```python
# Input: Message containing "evaluate" or "评估"
# Output: Exactly "approved"
"approved"
```

**Writer Agent:**
```python
# Input: Message containing "write" or "生成"
# Output: Realistic text content (50+ characters)
"""
This section introduces Python's async/await syntax. The `run_with_timeout()` 
function demonstrates how to execute coroutines with timeout protection using 
`asyncio.wait_for()`. The `AsyncContextManager` class shows proper implementation 
of async context managers with `__aenter__` and `__aexit__` methods.
"""
```

**Fact Checker Agent:**
```python
# Input: Message containing "verify" or "验证"
# Output: "VALID" or "INVALID\n- hallucination: description"
"VALID"
# or
"""
INVALID
- 函数 'nonexistent_function()' 未在源文档中找到
- 类 'FakeClass' 未在源文档中找到
"""
```

**Compiler Agent:**
```python
# Input: Message containing "compile" or "编译"
# Output: Formatted screenplay with title
"""
# Final Screenplay

## Introduction to async/await basics

This section introduces Python's async/await syntax...

## Understanding event loops

The event loop is the core of asyncio...
"""
```

### 2. Mock Retrieval Service

The mock retrieval service returns realistic code examples and documentation.

#### Basic Usage

```python
from tests.fixtures.realistic_mock_data import create_mock_retrieval_service

mock_retrieval = create_mock_retrieval_service()

# Returns realistic retrieval results
results = await mock_retrieval.retrieve(
    query="async await examples",
    top_k=5
)
```

#### Retrieval Result Structure

```python
{
    "id": "doc1",
    "file_path": "src/utils/async_helpers.py",
    "content": """
import asyncio
from typing import Awaitable, TypeVar

async def run_with_timeout(coro: Awaitable[T], timeout: float) -> T:
    '''Run coroutine with timeout.'''
    return await asyncio.wait_for(coro, timeout=timeout)
""",
    "similarity": 0.9,
    "confidence": 0.9,
    "has_deprecated": False,
    "has_fixme": False,
    "has_todo": False,
    "has_security": False,
    "metadata": {},
    "source": "vector"
}
```

#### Code Patterns Included

The mock retrieval results include realistic Python code patterns:

1. **Async Functions:**
```python
async def run_with_timeout(coro: Awaitable[T], timeout: float) -> T:
    return await asyncio.wait_for(coro, timeout=timeout)
```

2. **Class Definitions:**
```python
class AsyncContextManager:
    async def __aenter__(self):
        await self.connect()
        return self
```

3. **Decorators:**
```python
@retry(max_attempts=3)
async def fetch_data(url: str) -> dict:
    pass
```

4. **Error Handling:**
```python
try:
    result = await operation()
except TimeoutError:
    logger.error("Operation timed out")
```

5. **Import Statements:**
```python
import asyncio
from typing import Awaitable, TypeVar
```

### 3. Mock Parser Service

The mock parser service provides realistic parse results with language detection.

#### Basic Usage

```python
from tests.fixtures.realistic_mock_data import create_mock_parser_service

mock_parser = create_mock_parser_service()

# Returns parse results with detected language
result = await mock_parser.parse(
    content="def hello(): pass",
    file_path="example.py"
)
```

#### Parse Result Structure

```python
{
    "language": "python",
    "functions": ["run_with_timeout", "connect", "disconnect"],
    "classes": ["AsyncContextManager"],
    "imports": ["asyncio", "typing"],
    "has_syntax_errors": False
}
```

### 4. Custom Retrieval Results

For specific test scenarios, you can create custom retrieval results:

```python
from tests.fixtures.realistic_mock_data import create_realistic_retrieval_results

# Create results for a specific query
results = create_realistic_retrieval_results(
    query="database connection pooling",
    num_results=3
)

# Results will contain realistic code examples related to the query
```

## Integration Test Examples

### Example 1: Basic Workflow Test

```python
import pytest
from tests.fixtures.realistic_mock_data import (
    create_mock_llm_service,
    create_mock_retrieval_service,
    create_mock_parser_service
)
from src.application.orchestrator import WorkflowOrchestrator
from src.domain.models import SharedState

@pytest.mark.asyncio
async def test_complete_workflow():
    # Setup mock services
    mock_llm = create_mock_llm_service()
    mock_retrieval = create_mock_retrieval_service()
    mock_parser = create_mock_parser_service()
    
    # Create orchestrator with mocks
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser
    )
    
    # Create initial state
    state = SharedState(
        topic="Python async/await patterns",
        context="Educational tutorial"
    )
    
    # Execute workflow with increased recursion limit
    result = await orchestrator.execute(
        state=state,
        recursion_limit=50  # Increased for complex workflows
    )
    
    # Verify results
    assert result["success"] is True
    assert "final_screenplay" in result
    assert len(result["state"].fragments) > 0
```

### Example 2: Testing Agent Execution Order

```python
@pytest.mark.asyncio
async def test_agent_execution_order():
    # Setup mocks
    mock_llm = create_mock_llm_service()
    mock_retrieval = create_mock_retrieval_service()
    mock_parser = create_mock_parser_service()
    
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser
    )
    
    state = SharedState(topic="Test topic")
    result = await orchestrator.execute(state, recursion_limit=50)
    
    # Verify agent execution order
    execution_log = result["state"].execution_log
    agent_sequence = [entry["agent"] for entry in execution_log]
    
    # Expected order: planner → navigator → director → writer → fact_checker → compiler
    assert "planner" in agent_sequence
    assert "navigator" in agent_sequence
    assert "director" in agent_sequence
    assert "writer" in agent_sequence
    assert "fact_checker" in agent_sequence
    assert "compiler" in agent_sequence
    
    # Verify planner comes before writer
    planner_idx = agent_sequence.index("planner")
    writer_idx = agent_sequence.index("writer")
    assert planner_idx < writer_idx
```

### Example 3: Testing with Empty Retrieval (Research Mode)

```python
@pytest.mark.asyncio
async def test_workflow_with_empty_retrieval():
    # Create mock that returns empty results
    mock_retrieval = create_mock_retrieval_service()
    mock_retrieval.retrieve = AsyncMock(return_value=[])
    
    mock_llm = create_mock_llm_service()
    mock_parser = create_mock_parser_service()
    
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser
    )
    
    state = SharedState(topic="Obscure topic with no docs")
    result = await orchestrator.execute(state, recursion_limit=50)
    
    # Should complete successfully in research mode
    assert result["success"] is True
    assert result["state"].current_skill == "research_mode"
```

### Example 4: Testing State Consistency

```python
@pytest.mark.asyncio
async def test_workflow_state_consistency():
    mock_llm = create_mock_llm_service()
    mock_retrieval = create_mock_retrieval_service()
    mock_parser = create_mock_parser_service()
    
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser
    )
    
    initial_state = SharedState(
        topic="Test topic",
        context="Test context"
    )
    
    result = await orchestrator.execute(initial_state, recursion_limit=50)
    final_state = result["state"]
    
    # Verify state consistency
    assert final_state.topic == initial_state.topic
    assert final_state.context == initial_state.context
    assert len(final_state.execution_log) > 0
    assert final_state.updated_at > initial_state.updated_at
```

## Best Practices

### 1. Use Appropriate Recursion Limits

```python
# Simple workflows (3 steps)
result = await orchestrator.execute(state, recursion_limit=25)

# Complex workflows (5+ steps)
result = await orchestrator.execute(state, recursion_limit=50)

# Very complex workflows
result = await orchestrator.execute(state, recursion_limit=100)
```

### 2. Verify Mock Behavior

```python
# Verify mock was called
mock_llm.chat_completion.assert_called()

# Verify specific call count
assert mock_retrieval.retrieve.call_count == 3

# Verify call arguments
mock_parser.parse.assert_called_with(
    content=expected_content,
    file_path=expected_path
)
```

### 3. Test Error Handling

```python
# Simulate LLM failure
mock_llm.chat_completion = AsyncMock(side_effect=Exception("LLM error"))

result = await orchestrator.execute(state, recursion_limit=50)

# Verify graceful degradation
assert result["success"] is False
assert "error" in result
```

### 4. Validate Output Structure

```python
result = await orchestrator.execute(state, recursion_limit=50)

# Verify result structure
assert "success" in result
assert "state" in result
assert "final_screenplay" in result or "error" in result

# Verify state structure
state = result["state"]
assert hasattr(state, "outline")
assert hasattr(state, "fragments")
assert hasattr(state, "execution_log")
```

## Common Patterns

### Pattern 1: Testing Specific Agent Behavior

```python
@pytest.mark.asyncio
async def test_fact_checker_validation():
    mock_llm = create_mock_llm_service()
    
    # Fact checker should return VALID for realistic fragments
    response = await mock_llm.chat_completion(
        messages=[{
            "role": "user",
            "content": "Verify this fragment against source documents..."
        }],
        task_type="fact_check"
    )
    
    assert response == "VALID"
```

### Pattern 2: Testing State Transitions

```python
@pytest.mark.asyncio
async def test_state_transitions():
    orchestrator = create_test_orchestrator()
    
    state = SharedState(topic="Test")
    result = await orchestrator.execute(state, recursion_limit=50)
    
    # Verify state progressed through expected stages
    log = result["state"].execution_log
    stages = [entry["stage"] for entry in log]
    
    assert "planning" in stages
    assert "navigation" in stages
    assert "evaluation" in stages
    assert "generation" in stages
    assert "verification" in stages
    assert "compilation" in stages
```

### Pattern 3: Testing with Custom Mock Data

```python
@pytest.mark.asyncio
async def test_with_custom_mock_data():
    # Create custom retrieval results
    custom_results = [
        {
            "id": "custom1",
            "file_path": "custom/path.py",
            "content": "def custom_function(): pass",
            "similarity": 0.95,
            "confidence": 0.95,
            "source": "vector"
        }
    ]
    
    mock_retrieval = create_mock_retrieval_service()
    mock_retrieval.retrieve = AsyncMock(return_value=custom_results)
    
    # Use in test...
```

## Troubleshooting

### Issue: Mock LLM Returns Wrong Format

**Problem:** Mock LLM response doesn't match expected format.

**Solution:** Ensure message content contains the right keywords:
- Planner: "outline", "步骤"
- Director: "complexity", "evaluate", "复杂度", "评估"
- Writer: "write", "generate", "生成"
- Fact Checker: "verify", "验证"
- Compiler: "compile", "编译"

### Issue: Recursion Limit Exceeded

**Problem:** Test fails with recursion limit error.

**Solution:** Increase recursion limit:
```python
result = await orchestrator.execute(state, recursion_limit=100)
```

### Issue: Mock Not Being Called

**Problem:** Mock service methods aren't being invoked.

**Solution:** Verify mock is properly injected:
```python
# Correct
orchestrator = WorkflowOrchestrator(
    llm_service=mock_llm,  # Pass mock
    retrieval_service=mock_retrieval,
    parser_service=mock_parser
)

# Incorrect
orchestrator = WorkflowOrchestrator()  # Uses real services
```

### Issue: Test Timeout

**Problem:** Test takes too long to complete.

**Solution:** 
1. Reduce outline steps (use simple mock)
2. Increase recursion limit
3. Check for infinite loops in workflow

## Reference

### Mock Service Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `create_mock_llm_service()` | Create format-aware mock LLM | Mock LLM service |
| `create_mock_retrieval_service()` | Create mock retrieval service | Mock retrieval service |
| `create_mock_parser_service()` | Create mock parser service | Mock parser service |
| `create_realistic_retrieval_results(query, num_results)` | Generate custom retrieval results | List of retrieval results |
| `create_realistic_code_examples()` | Generate code examples | List of code strings |

### Response Format Reference

See the "Response Formats by Agent" section above for detailed format specifications.

---

**Last Updated:** January 31, 2026  
**Related Files:**
- `tests/fixtures/realistic_mock_data.py` - Mock data implementation
- `tests/integration/test_end_to_end_workflow.py` - Usage examples
- `INTEGRATION_TEST_IMPROVEMENT_SUMMARY.md` - Implementation summary
