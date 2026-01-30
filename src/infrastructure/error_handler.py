"""Error Handler - Error handling and recovery mechanisms"""

import logging
import asyncio
import time
from typing import Optional, Callable, Any, TypeVar, List
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetrievalError(Exception):
    """检索错误基类"""
    pass


class PostgreSQLConnectionError(RetrievalError):
    """PostgreSQL 连接失败"""
    pass


class EmbeddingGenerationError(RetrievalError):
    """嵌入生成失败"""
    pass


class LLMError(Exception):
    """LLM 错误基类"""
    pass


class RateLimitError(LLMError):
    """速率限制错误"""
    pass


class MalformedResponseError(LLMError):
    """格式错误的响应"""
    pass


class ProviderUnavailableError(LLMError):
    """提供商不可用"""
    pass


class ComponentError(Exception):
    """组件错误基类"""
    pass


class FactCheckerError(ComponentError):
    """事实检查器失败"""
    pass


class SummarizerError(ComponentError):
    """摘要器失败"""
    pass


class WriterError(ComponentError):
    """编剧失败"""
    pass


class TimeoutError(Exception):
    """超时错误"""
    pass


class ErrorHandler:
    """
    统一的错误处理器
    
    功能：
    - 检索错误处理（PostgreSQL 连接失败、嵌入生成失败）
    - LLM 错误处理（速率限制、格式错误、提供商切换）
    - 组件错误处理（事实检查器、摘要器、编剧）
    - 超时保护
    """
    
    @staticmethod
    async def handle_retrieval_error(
        func: Callable,
        *args,
        fallback_to_keyword_only: bool = True,
        **kwargs
    ) -> Any:
        """
        处理检索错误
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            fallback_to_keyword_only: 是否回退到仅关键词搜索
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果或回退结果
        """
        try:
            return await func(*args, **kwargs)
        except (PostgreSQLConnectionError, EmbeddingGenerationError) as e:
            error_type = type(e).__name__
            logger.error(f"{error_type}: {str(e)}")
            
            if fallback_to_keyword_only:
                logger.info("Falling back to keyword-only search")
                # 回退到仅关键词搜索
                if 'use_vector_search' in kwargs:
                    kwargs['use_vector_search'] = False
                    try:
                        return await func(*args, **kwargs)
                    except Exception as retry_error:
                        logger.error(f"Fallback also failed: {str(retry_error)}")
                        # 返回空结果而不是抛出异常
                        logger.warning("Returning empty results due to fallback failure")
                        return []
            
            # 返回空结果而不是抛出异常
            logger.warning("Returning empty results due to retrieval error")
            return []
        except Exception as e:
            logger.error(f"Unexpected retrieval error: {str(e)}")
            # 优雅降级：返回空结果
            return []
    
    @staticmethod
    async def retry_with_exponential_backoff(
        func: Callable,
        *args,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        **kwargs
    ) -> Any:
        """
        使用指数退避重试
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            max_retries: 最大重试次数
            initial_delay: 初始延迟（秒）
            max_delay: 最大延迟（秒）
            exponential_base: 指数基数
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果
        """
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except RateLimitError as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
                else:
                    logger.error(f"Max retries reached for rate limit")
                    raise
            except MalformedResponseError as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(f"Malformed response, retrying (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
                else:
                    logger.error(f"Max retries reached for malformed response")
                    raise
            except Exception as e:
                # 对于其他错误，不重试
                logger.error(f"Non-retryable error: {str(e)}")
                raise
        
        # 如果所有重试都失败
        if last_exception:
            raise last_exception
    
    @staticmethod
    async def handle_component_failure(
        func: Callable,
        component_name: str,
        *args,
        fallback_value: Any = None,
        log_level: str = "warning",
        **kwargs
    ) -> Any:
        """
        处理组件失败
        
        Args:
            func: 要执行的函数
            component_name: 组件名称
            *args: 位置参数
            fallback_value: 回退值
            log_level: 日志级别（warning/error）
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果或回退值
        """
        try:
            return await func(*args, **kwargs)
        except FactCheckerError as e:
            log_func = getattr(logger, log_level)
            log_func(f"Fact checker failed: {str(e)}, continuing with warning")
            return fallback_value
        except SummarizerError as e:
            logger.warning(f"Summarizer failed: {str(e)}, using truncation instead")
            # 如果摘要失败，使用截断
            if 'content' in kwargs:
                content = kwargs['content']
                max_length = kwargs.get('max_length', 10000)
                return content[:max_length] + "..." if len(content) > max_length else content
            return fallback_value
        except WriterError as e:
            logger.warning(f"Writer failed: {str(e)}, retrying with simpler skill")
            # 如果编剧失败，尝试使用更简单的 Skill
            # 检查 state 参数并修改其 current_skill
            if args and hasattr(args[0], 'current_skill'):
                state = args[0]
                if state.current_skill != 'fallback_summary':
                    state.current_skill = 'fallback_summary'
                    state.add_log_entry(
                        agent_name="error_handler",
                        action="skill_switch",
                        details={
                            "reason": "writer_failure",
                            "new_skill": "fallback_summary"
                        }
                    )
                    try:
                        return await func(*args, **kwargs)
                    except Exception as retry_error:
                        logger.error(f"Retry with simpler skill also failed: {str(retry_error)}")
                        return fallback_value
            return fallback_value
        except Exception as e:
            logger.error(f"{component_name} failed with unexpected error: {str(e)}")
            return fallback_value
    
    @staticmethod
    async def with_timeout(
        func: Callable,
        timeout_seconds: float,
        *args,
        **kwargs
    ) -> Any:
        """
        为函数添加超时保护
        
        Args:
            func: 要执行的函数
            timeout_seconds: 超时时间（秒）
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果
            
        Raises:
            TimeoutError: 如果超时
        """
        try:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"Function {func.__name__} timed out after {timeout_seconds}s")
            raise TimeoutError(f"Operation timed out after {timeout_seconds}s")


def handle_retrieval_errors(fallback_to_keyword_only: bool = True):
    """
    装饰器：处理检索错误
    
    Args:
        fallback_to_keyword_only: 是否回退到仅关键词搜索
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await ErrorHandler.handle_retrieval_error(
                func,
                *args,
                fallback_to_keyword_only=fallback_to_keyword_only,
                **kwargs
            )
        return wrapper
    return decorator


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
):
    """
    装饰器：使用指数退避重试
    
    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        exponential_base: 指数基数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await ErrorHandler.retry_with_exponential_backoff(
                func,
                *args,
                max_retries=max_retries,
                initial_delay=initial_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                **kwargs
            )
        return wrapper
    return decorator


def handle_component_errors(
    component_name: str,
    fallback_value: Any = None,
    log_level: str = "warning"
):
    """
    装饰器：处理组件失败
    
    Args:
        component_name: 组件名称
        fallback_value: 回退值
        log_level: 日志级别
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await ErrorHandler.handle_component_failure(
                func,
                component_name,
                *args,
                fallback_value=fallback_value,
                log_level=log_level,
                **kwargs
            )
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float):
    """
    装饰器：添加超时保护
    
    Args:
        timeout_seconds: 超时时间（秒）
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await ErrorHandler.with_timeout(
                func,
                timeout_seconds,
                *args,
                **kwargs
            )
        return wrapper
    return decorator
