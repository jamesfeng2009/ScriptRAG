"""Realistic mock data generators for integration tests

This module provides high-fidelity mock data that closely resembles production data,
including realistic Python code examples, properly formatted LLM responses, and
complete retrieval results.
"""

from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock


def create_realistic_code_examples() -> List[Dict[str, str]]:
    """Create realistic Python code examples with various patterns
    
    Returns at least 5 different code patterns:
    1. Async functions with await
    2. Class definitions with methods
    3. Decorators
    4. Error handling
    5. Import statements
    """
    return [
        {
            "file_path": "src/utils/async_helpers.py",
            "content": """import asyncio
from typing import Awaitable, TypeVar, Optional

T = TypeVar('T')

async def run_with_timeout(
    coro: Awaitable[T],
    timeout: float
) -> T:
    '''Run coroutine with timeout.
    
    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        
    Returns:
        Result of coroutine
        
    Raises:
        asyncio.TimeoutError: If timeout exceeded
    '''
    return await asyncio.wait_for(coro, timeout=timeout)

async def gather_with_concurrency(
    *tasks: Awaitable,
    max_concurrency: int = 10
) -> List[Any]:
    '''Execute multiple tasks with concurrency limit.
    
    Args:
        tasks: Coroutines to execute
        max_concurrency: Maximum concurrent tasks
        
    Returns:
        List of results
    '''
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def bounded_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*[bounded_task(t) for t in tasks])
"""
        },
        {
            "file_path": "src/utils/context_managers.py",
            "content": """from typing import Optional
import asyncio

class AsyncContextManager:
    '''Example async context manager for resource management.'''
    
    def __init__(self, resource_name: str):
        self.resource_name = resource_name
        self.connection: Optional[Any] = None
    
    async def __aenter__(self):
        '''Establish connection on enter.'''
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        '''Close connection on exit.'''
        await self.disconnect()
    
    async def connect(self):
        '''Establish connection to resource.'''
        await asyncio.sleep(0.1)  # Simulate connection
        self.connection = f"Connected to {self.resource_name}"
    
    async def disconnect(self):
        '''Close connection to resource.'''
        await asyncio.sleep(0.1)  # Simulate disconnection
        self.connection = None

class DatabaseConnection(AsyncContextManager):
    '''Database connection context manager.'''
    
    async def execute_query(self, query: str) -> List[Dict]:
        '''Execute database query.
        
        Args:
            query: SQL query string
            
        Returns:
            Query results
        '''
        if not self.connection:
            raise RuntimeError("Not connected to database")
        # Simulate query execution
        return [{"id": 1, "data": "example"}]
"""
        },
        {
            "file_path": "src/decorators/retry.py",
            "content": """import asyncio
import functools
from typing import Callable, TypeVar, Any

T = TypeVar('T')

def retry_async(max_attempts: int = 3, delay: float = 1.0):
    '''Decorator for retrying async functions.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Decorated function
    '''
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay * (attempt + 1))
            
            raise last_exception
        
        return wrapper
    return decorator

@retry_async(max_attempts=5, delay=2.0)
async def fetch_data_with_retry(url: str) -> Dict[str, Any]:
    '''Fetch data from URL with automatic retry.
    
    Args:
        url: URL to fetch from
        
    Returns:
        Response data
    '''
    # Simulate HTTP request
    await asyncio.sleep(0.1)
    return {"status": "success", "data": []}
"""
        },
        {
            "file_path": "src/error_handling/exceptions.py",
            "content": """class AsyncOperationError(Exception):
    '''Base exception for async operations.'''
    
    def __init__(self, message: str, operation: str):
        self.message = message
        self.operation = operation
        super().__init__(f"{operation}: {message}")

class TimeoutError(AsyncOperationError):
    '''Raised when async operation times out.'''
    
    def __init__(self, operation: str, timeout: float):
        super().__init__(
            f"Operation timed out after {timeout}s",
            operation
        )
        self.timeout = timeout

async def safe_execute(coro: Awaitable[T], timeout: float = 30.0) -> Optional[T]:
    '''Safely execute coroutine with error handling.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        
    Returns:
        Result or None if error occurred
    '''
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError("safe_execute", timeout)
    except Exception as e:
        # Log error and return None
        print(f"Error executing coroutine: {e}")
        return None
"""
        },
        {
            "file_path": "docs/async_patterns.md",
            "content": """# Python Async/Await Patterns

## Introduction

Python's async/await syntax provides a clean way to write asynchronous code.
This guide covers common patterns and best practices.

## Basic Async Function

```python
async def fetch_user(user_id: int) -> User:
    '''Fetch user from database.'''
    async with database.connection() as conn:
        result = await conn.execute(
            "SELECT * FROM users WHERE id = $1",
            user_id
        )
        return User(**result)
```

## Event Loop Management

The event loop is the core of asyncio. Use `asyncio.run()` for simple cases:

```python
import asyncio

async def main():
    result = await fetch_user(123)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Concurrency Control

Use semaphores to limit concurrent operations:

```python
semaphore = asyncio.Semaphore(10)

async def limited_fetch(url: str):
    async with semaphore:
        return await fetch(url)
```
"""
        }
    ]


def create_realistic_retrieval_results(
    query: str,
    num_results: int = 5
) -> List[Any]:
    """Create realistic retrieval results with actual code examples
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of RetrievalResult objects with realistic content
    """
    from src.services.retrieval_service import RetrievalResult
    
    code_examples = create_realistic_code_examples()
    results = []
    
    for i, example in enumerate(code_examples[:num_results]):
        result = RetrievalResult(
            id=f"doc{i+1}",
            file_path=example["file_path"],
            content=example["content"],
            similarity=0.9 - (i * 0.05),
            confidence=0.9 - (i * 0.05),
            strategy_name="vector_search",
            has_deprecated=False,
            has_fixme=False,
            has_todo=False,
            has_security=False,
            metadata={"language": "python"}
        )
        results.append(result)
    
    return results


def create_mock_llm_service() -> Mock:
    """Create mock LLM service with properly formatted responses
    
    Returns mock LLM service that detects agent type from message content
    and returns appropriately formatted responses:
    
    - Planner: "步骤N: Title | 关键词: keywords" (3-5 steps)
    - Director (complexity): "0.5" (numeric string between 0.0-1.0)
    - Director (evaluation): "approved" (always approve to prevent pivots)
    - Writer: Realistic fragment text (50+ characters)
    - Fact Checker: "VALID" or "INVALID\n- hallucination: desc"
    - Compiler: "# Final Screenplay\n\n..." (formatted output)
    """
    mock_service = Mock()
    
    async def mock_chat_completion(
        messages: List[Dict[str, str]],
        task_type: str,
        **kwargs
    ) -> str:
        """Mock chat completion that returns format-appropriate responses"""
        # Get the last message content to detect agent type
        last_message = messages[-1]["content"] if messages else ""
        last_message_lower = last_message.lower()
        
        # Detect agent type from message content (order matters - check more specific patterns first)
        
        # Check for planner first (most specific with "生成大纲")
        if "生成大纲" in last_message or ("outline" in last_message_lower and "generate" in last_message_lower):
            # Planner agent - return Chinese format outline
            # Simple scenarios (default): 3 steps to reduce iterations
            # Complex scenarios: up to 5 steps
            return """步骤1: Python异步编程基础介绍 | 关键词: async, await, 协程
步骤2: 事件循环和并发控制 | 关键词: event loop, asyncio, 并发
步骤3: 实用异步模式和最佳实践 | 关键词: 模式, 最佳实践, 错误处理"""
        
        # Check for fact checker (check for verification context, not just keywords)
        elif ("验证" in last_message and "片段" in last_message) or ("verify" in last_message_lower and "fragment" in last_message_lower):
            # Fact checker - return VALID for properly formatted fragments
            # Check for obvious hallucinations (nonexistent functions explicitly mentioned)
            if "nonexistent_function" in last_message.lower() or "fake_function" in last_message.lower():
                return """INVALID
- 函数 'nonexistent_function()' 未在源文档中找到"""
            else:
                # For all other cases, return VALID to prevent regeneration loops (需求 4.4)
                return "VALID"
        
        # Check for director complexity FIRST (more specific than evaluation)
        elif "复杂度" in last_message or ("complexity" in last_message_lower and "assess" in last_message_lower):
            # Director complexity assessment - return numeric string
            # Only match if it's specifically asking to assess complexity
            return "0.5"
        
        # Check for director evaluation AFTER complexity (evaluation is more general)
        elif ("评估" in last_message or "evaluation" in last_message_lower or 
              "批准" in last_message or "approve" in last_message_lower or
              "质量" in last_message or "quality" in last_message_lower):
            # Director evaluation - always return approved to prevent pivot loops
            return "approved"
        
        # Check for compiler
        elif "编译" in last_message or "compile" in last_message_lower or "最终" in last_message or "final" in last_message_lower:
            # Compiler - return formatted screenplay
            return """# Python异步编程完整指南

## Python异步编程基础介绍

Python的async/await语法提供了一种简洁的方式来编写异步代码。`run_with_timeout()`函数展示了如何使用`asyncio.wait_for()`来执行带超时保护的协程。`AsyncContextManager`类展示了如何正确实现异步上下文管理器，包括`__aenter__`和`__aexit__`方法。

## 事件循环和并发控制

事件循环是asyncio的核心。使用`asyncio.run()`可以简化事件循环的管理。通过信号量可以限制并发操作的数量，`gather_with_concurrency()`函数展示了如何使用Semaphore来控制并发任务数量。

## 实用异步模式和最佳实践

装饰器`retry_async`提供了自动重试机制，可以处理临时性错误。异步上下文管理器确保资源的正确获取和释放。错误处理应该使用try-except块来捕获特定的异常类型，如`asyncio.TimeoutError`。"""
        
        # Check for planner with less specific patterns (fallback)
        elif "步骤" in last_message or "outline" in last_message_lower:
            # Planner agent - return Chinese format outline
            # Simple scenarios (default): 3 steps to reduce iterations
            # Complex scenarios: up to 5 steps
            
            # Detect complexity from message content
            is_complex = any(keyword in last_message for keyword in [
                "完整", "详细", "深入", "全面", "进阶", "高级",
                "complete", "detailed", "comprehensive", "advanced"
            ])
            
            if is_complex:
                # Complex scenario - return 5 steps
                return """步骤1: Python异步编程基础介绍 | 关键词: async, await, 协程
步骤2: 事件循环和并发控制 | 关键词: event loop, asyncio, 并发
步骤3: 实用异步模式和最佳实践 | 关键词: 上下文管理器, 重试机制, 错误处理
步骤4: 异步错误处理和调试 | 关键词: 异常处理, 调试, 日志
步骤5: 性能优化和最佳实践总结 | 关键词: 性能, 优化, 总结"""
            else:
                # Simple scenario - return exactly 3 steps to reduce iterations
                return """步骤1: Python异步编程基础介绍 | 关键词: async, await, 协程
步骤2: 事件循环和并发控制 | 关键词: event loop, asyncio, 并发
步骤3: 实用异步模式和最佳实践 | 关键词: 上下文管理器, 重试机制, 错误处理"""
        
        else:
            # Writer agent - return realistic fragment text (50+ characters)
            return """本节介绍Python的async/await语法。`run_with_timeout()`函数展示了如何使用`asyncio.wait_for()`来执行带超时保护的协程。`AsyncContextManager`类展示了如何正确实现异步上下文管理器，包括`__aenter__`和`__aexit__`方法。这些模式在实际开发中非常有用，可以帮助我们编写更加健壮的异步代码。"""
    
    # Set up the mock
    mock_service.chat_completion = AsyncMock(side_effect=mock_chat_completion)
    
    return mock_service


def create_mock_retrieval_service() -> Mock:
    """Create mock retrieval service returning realistic documents
    
    Returns mock retrieval service that returns realistic code examples
    from create_realistic_retrieval_results().
    """
    mock_service = Mock()
    
    async def mock_hybrid_retrieve(
        workspace_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Any]:
        """Mock hybrid retrieval that returns realistic code examples"""
        return create_realistic_retrieval_results(query, top_k)
    
    # Set up the mock
    mock_service.hybrid_retrieve = AsyncMock(side_effect=mock_hybrid_retrieve)
    
    return mock_service


def create_mock_parser_service() -> Mock:
    """Create mock parser service with realistic parse results
    
    Returns mock parser service that returns ParsedCode objects with
    realistic language detection and code element extraction.
    """
    from src.services.parser.tree_sitter_parser import ParsedCode, CodeElement, CodeElementType
    
    mock_service = Mock()
    
    def mock_parse(
        file_path: str,
        content: str,
        language: str = None
    ) -> ParsedCode:
        """Mock parse that returns realistic parsed code"""
        # Detect language from file path
        if not language:
            if file_path.endswith('.py'):
                language = 'python'
            elif file_path.endswith('.js'):
                language = 'javascript'
            elif file_path.endswith('.ts'):
                language = 'typescript'
            elif file_path.endswith('.md'):
                language = 'markdown'
            else:
                language = 'unknown'
        
        # Extract code elements (simple pattern matching)
        elements = []
        lines = content.split('\n')
        
        # Extract functions
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Python async functions
            if 'async def ' in stripped:
                func_name = stripped.split('async def ')[1].split('(')[0].strip()
                elements.append(CodeElement(
                    type=CodeElementType.FUNCTION,
                    name=func_name,
                    content=line,
                    line_start=i + 1,
                    line_end=i + 1
                ))
            # Python regular functions
            elif stripped.startswith('def '):
                func_name = stripped.split('def ')[1].split('(')[0].strip()
                elements.append(CodeElement(
                    type=CodeElementType.FUNCTION,
                    name=func_name,
                    content=line,
                    line_start=i + 1,
                    line_end=i + 1
                ))
            # Python classes
            elif stripped.startswith('class '):
                class_name = stripped.split('class ')[1].split('(')[0].split(':')[0].strip()
                elements.append(CodeElement(
                    type=CodeElementType.CLASS,
                    name=class_name,
                    content=line,
                    line_start=i + 1,
                    line_end=i + 1
                ))
        
        # Detect markers
        content_lower = content.lower()
        has_deprecated = '@deprecated' in content_lower or 'deprecated' in content_lower
        has_fixme = 'fixme' in content_lower
        has_todo = 'todo' in content_lower
        has_security = 'security' in content_lower
        
        return ParsedCode(
            file_path=file_path,
            language=language,
            elements=elements,
            has_deprecated=has_deprecated,
            has_fixme=has_fixme,
            has_todo=has_todo,
            has_security=has_security,
            raw_content=content,
            metadata={"parser": "mock"}
        )
    
    # Set up the mock
    mock_service.parse = Mock(side_effect=mock_parse)
    
    return mock_service
