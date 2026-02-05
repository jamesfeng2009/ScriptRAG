"""异步I/O优化 - 连接池、Redis管道和并行处理

本模块实现异步I/O优化功能：
1. 连接池管理：复用数据库连接
2. Redis管道：批量操作减少网络往返
3. 并行检索：同时查询多个数据源
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """连接池配置"""
    min_size: int = 5
    max_size: int = 20
    max_idle_time: float = 300.0
    timeout: float = 30.0


@dataclass
class ParallelTaskResult:
    """并行任务结果"""
    task_name: str
    result: Any
    execution_time: float
    error: Optional[str] = None


class ConnectionPoolManager:
    """
    连接池管理器

    功能：
    - 管理数据库连接池
    - 连接复用和自动重连
    - 连接健康检查
    """

    def __init__(
        self,
        pool_config: Optional[PoolConfig] = None,
        health_check_interval: float = 60.0
    ):
        self.pool_config = pool_config or PoolConfig()
        self.health_check_interval = health_check_interval

        self._pools: Dict[str, Any] = {}
        self._connection_stats: Dict[str, Dict[str, int]] = {}

        self._health_check_task: Optional[asyncio.Task] = None

    async def get_connection(self, pool_name: str) -> Any:
        """
        获取连接

        Args:
            pool_name: 连接池名称

        Returns:
            数据库连接
        """
        pool = self._pools.get(pool_name)

        if pool is None:
            logger.warning(f"Connection pool {pool_name} not found")
            return None

        if hasattr(pool, 'acquire'):
            return await pool.acquire()

        return pool

    async def release_connection(
        self,
        pool_name: str,
        connection: Any
    ) -> None:
        """
        释放连接

        Args:
            pool_name: 连接池名称
            connection: 要释放的连接
        """
        pool = self._pools.get(pool_name)

        if pool is None:
            logger.warning(f"Connection pool {pool_name} not found")
            return

        if hasattr(pool, 'release'):
            await pool.release(connection)

    async def execute_with_connection(
        self,
        pool_name: str,
        query_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        使用连接执行查询

        Args:
            pool_name: 连接池名称
            query_func: 查询函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            查询结果
        """
        connection = await self.get_connection(pool_name)

        if connection is None:
            raise ConnectionError(f"Failed to get connection from pool: {pool_name}")

        try:
            if asyncio.iscoroutinefunction(query_func):
                result = await query_func(connection, *args, **kwargs)
            else:
                result = query_func(connection, *args, **kwargs)
            return result
        finally:
            await self.release_connection(pool_name, connection)

    def get_pool_stats(self, pool_name: str) -> Dict[str, Any]:
        """
        获取连接池统计

        Args:
            pool_name: 连接池名称

        Returns:
            连接池统计信息
        """
        stats = self._connection_stats.get(pool_name, {})

        return {
            'pool_name': pool_name,
            'active_connections': stats.get('active', 0),
            'idle_connections': stats.get('idle', 0),
            'total_acquisitions': stats.get('acquisitions', 0),
            'total_releases': stats.get('releases', 0)
        }


class RedisPipelineManager:
    """
    Redis管道管理器

    功能：
    - 批量操作减少网络往返
    - 管道命令缓冲
    - 自动重试和错误恢复
    """

    def __init__(
        self,
        max_commands_buffered: int = 1000,
        flush_interval: float = 0.05,
        max_retries: int = 3
    ):
        self.max_commands_buffered = max_commands_buffered
        self.flush_interval = flush_interval
        self.max_retries = max_retries

        self._pipelines: Dict[str, Any] = {}
        self._command_buffer: Dict[str, List[Callable]] = {}

        self._flush_tasks: Dict[str, asyncio.Task] = {}

    def get_pipeline(self, redis_client: Any) -> Any:
        """
        获取Redis管道

        Args:
            redis_client: Redis客户端

        Returns:
            Redis管道
        """
        client_id = id(redis_client)

        if client_id in self._pipelines:
            return self._pipelines[client_id]

        pipeline = redis_client.pipeline()
        self._pipelines[client_id] = pipeline
        self._command_buffer[client_id] = []

        return pipeline

    def queue_command(
        self,
        pipeline: Any,
        command_func: Callable,
        *args,
        **kwargs
    ) -> None:
        """
        队列命令

        Args:
            pipeline: Redis管道
            command_func: 命令函数
            *args: 位置参数
            **kwargs: 关键字参数
        """
        client_id = id(pipeline)

        if client_id not in self._command_buffer:
            self._command_buffer[client_id] = []

        self._command_buffer[client_id].append(
            lambda: command_func(pipeline, *args, **kwargs)
        )

        if len(self._command_buffer[client_id]) >= self.max_commands_buffered:
            asyncio.create_task(self.flush_pipeline(pipeline))

    async def flush_pipeline(self, pipeline: Any) -> int:
        """
        刷新管道

        Args:
            pipeline: Redis管道

        Returns:
            执行的命令数量
        """
        client_id = id(pipeline)

        if client_id not in self._command_buffer:
            return 0

        commands = self._command_buffer[client_id]
        if not commands:
            return 0

        self._command_buffer[client_id] = []

        for retry in range(self.max_retries):
            try:
                for command in commands:
                    command()

                results = await pipeline.execute()

                logger.debug(
                    f"Flushed {len(commands)} commands for pipeline {client_id}"
                )

                return len(results)

            except Exception as e:
                logger.error(
                    f"Failed to flush pipeline (attempt {retry + 1}): {str(e)}"
                )

                if retry < self.max_retries - 1:
                    await asyncio.sleep(0.1 * (retry + 1))
                else:
                    raise

        return 0

    async def batch_set(
        self,
        redis_client: Any,
        key_values: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> int:
        """
        批量设置

        Args:
            redis_client: Redis客户端
            key_values: 键值对字典
            ttl: 过期时间

        Returns:
            成功设置的数量
        """
        pipeline = self.get_pipeline(redis_client)

        for key, value in key_values.items():
            if ttl:
                pipeline.setex(key, ttl, value)
            else:
                pipeline.set(key, value)

        return await self.flush_pipeline(pipeline)

    async def batch_get(
        self,
        redis_client: Any,
        keys: List[str]
    ) -> Dict[str, Any]:
        """
        批量获取

        Args:
            redis_client: Redis客户端
            keys: 键列表

        Returns:
            键值对字典
        """
        pipeline = self.get_pipeline(redis_client)

        for key in keys:
            pipeline.get(key)

        results = await self.flush_pipeline(pipeline)

        return {
            key: results[i] for i, key in enumerate(keys)
            if results[i] is not None
        }


class ParallelExecutor:
    """
    并行执行器

    功能：
    - 并行执行多个任务
    - 控制并发数量
    - 错误处理和超时控制
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        timeout: float = 30.0
    ):
        self.max_concurrency = max_concurrency
        self.timeout = timeout

        self._semaphore: Optional[asyncio.Semaphore] = None
        self._running_tasks: Dict[str, asyncio.Task] = {}

    def _get_semaphore(self) -> asyncio.Semaphore:
        """获取信号量"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def execute_parallel(
        self,
        tasks: Dict[str, Callable]
    ) -> Dict[str, ParallelTaskResult]:
        """
        并行执行任务

        Args:
            tasks: 任务字典 {任务名: 任务函数}

        Returns:
            结果字典 {任务名: 结果}
        """
        semaphore = self._get_semaphore()

        async def run_with_semaphore(
            task_name: str,
            task_func: Callable
        ) -> ParallelTaskResult:
            start_time = asyncio.get_event_loop().time()

            async with semaphore:
                try:
                    if asyncio.iscoroutinefunction(task_func):
                        result = await task_func()
                    else:
                        result = task_func()

                    execution_time = asyncio.get_event_loop().time() - start_time

                    return ParallelTaskResult(
                        task_name=task_name,
                        result=result,
                        execution_time=execution_time
                    )

                except Exception as e:
                    execution_time = asyncio.get_event_loop().time() - start_time

                    logger.error(f"Task {task_name} failed: {str(e)}")

                    return ParallelTaskResult(
                        task_name=task_name,
                        result=None,
                        execution_time=execution_time,
                        error=str(e)
                    )

        task_list = [
            run_with_semaphore(name, func)
            for name, func in tasks.items()
        ]

        results = await asyncio.gather(*task_list, return_exceptions=True)

        result_dict = {}
        for result in results:
            if isinstance(result, ParallelTaskResult):
                result_dict[result.task_name] = result
            elif isinstance(result, Exception):
                logger.error(f"Unexpected exception: {str(result)}")

        return result_dict

    async def execute_with_timeout(
        self,
        task_func: Callable,
        timeout: Optional[float] = None
    ) -> Any:
        """
        带超时执行任务

        Args:
            task_func: 任务函数
            timeout: 超时时间

        Returns:
            任务结果
        """
        timeout_duration = timeout or self.timeout

        try:
            if asyncio.iscoroutinefunction(task_func):
                return await asyncio.wait_for(task_func(), timeout=timeout_duration)
            else:
                return await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, task_func),
                    timeout=timeout_duration
                )
        except asyncio.TimeoutError:
            logger.warning(f"Task timed out after {timeout_duration}s")
            raise
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            raise


class AsyncIOOptimizer:
    """
    异步I/O优化器

    综合管理连接池、Redis管道和并行执行。
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        pool_config: Optional[PoolConfig] = None
    ):
        self.pool_manager = ConnectionPoolManager(pool_config)
        self.pipeline_manager = RedisPipelineManager()
        self.parallel_executor = ParallelExecutor(max_concurrency)

    async def parallel_retrieve(
        self,
        vector_search_func: Callable,
        keyword_search_func: Callable,
        query: str,
        workspace_id: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        并行检索

        同时执行向量搜索和关键词搜索。

        Args:
            vector_search_func: 向量搜索函数
            keyword_search_func: 关键词搜索函数
            query: 查询文本
            workspace_id: 工作空间ID
            top_k: 返回结果数量

        Returns:
            合并的检索结果
        """
        tasks = {
            'vector_search': lambda: vector_search_func(
                query=query,
                workspace_id=workspace_id,
                top_k=top_k
            ),
            'keyword_search': lambda: keyword_search_func(
                query=query,
                workspace_id=workspace_id,
                top_k=top_k
            )
        }

        results = await self.parallel_executor.execute_parallel(tasks)

        merged_results = {
            'vector_results': None,
            'keyword_results': None,
            'execution_times': {}
        }

        for task_name, result in results.items():
            if result.error:
                logger.warning(f"Task {task_name} failed: {result.error}")
                merged_results[task_name] = []
            else:
                merged_results[task_name] = result.result

            merged_results['execution_times'][task_name] = result.execution_time

        return merged_results

    async def batch_embedding(
        self,
        embedding_func: Callable,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Any]:
        """
        批量嵌入处理

        Args:
            embedding_func: 嵌入函数
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            嵌入向量列表
        """
        semaphore = self.parallel_executor._get_semaphore()
        all_results = []

        async def process_batch(batch: List[str]) -> List[Any]:
            async with semaphore:
                if asyncio.iscoroutinefunction(embedding_func):
                    return await embedding_func(batch)
                else:
                    return embedding_func(batch)

        batches = [
            texts[i:i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]

        batch_tasks = [process_batch(batch) for batch in batches]

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch embedding failed: {str(result)}")
                all_results.extend([None] * batch_size)
            else:
                all_results.extend(result)

        return all_results

    def get_optimizer_stats(self) -> Dict[str, Any]:
        """获取优化器统计"""
        return {
            'pool_stats': self.pool_manager._connection_stats,
            'pipeline_stats': {
                'active_pipelines': len(self.pipeline_manager._pipelines),
                'buffered_commands': sum(
                    len(commands)
                    for commands in self.pipeline_manager._command_buffer.values()
                )
            },
            'parallel_stats': {
                'max_concurrency': self.max_concurrency
            }
        }
