"""Redis 缓存服务 - 缓存层实现

本模块提供 Redis 缓存功能：
1. LLM 响应缓存（24 小时 TTL）
2. 嵌入向量缓存（7 天 TTL）
3. 检索结果缓存（1 小时 TTL）
4. 会话状态缓存（30 分钟 TTL）
"""

import logging
import json
import hashlib
import sys
from pathlib import Path
from typing import Optional, Any, Dict, List
import redis.asyncio as redis

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


logger = logging.getLogger(__name__)


# 缓存键前缀
CACHE_PREFIX = {
    "llm_response": "llm",
    "embedding": "emb",
    "retrieval": "ret",
    "session": "sess",
    "quota": "quota",
    "config": "cfg"
}

# 缓存 TTL（秒）
CACHE_TTL = {
    "llm_response": 86400,      # 24 小时
    "embedding": 604800,        # 7 天
    "retrieval": 3600,          # 1 小时
    "session": 1800,            # 30 分钟
    "quota": 300,               # 5 分钟
    "config": 3600              # 1 小时
}


class RedisCacheService:
    """Redis 缓存操作服务"""
    
    def __init__(
        self,
        config: Optional[RedisConfig] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        max_connections: Optional[int] = None
    ):
        """
        初始化 Redis 缓存服务
        
        Args:
            config: Redis 配置对象（优先使用）
            host: Redis 主机（当 config 为空时使用）
            port: Redis 端口（当 config 为空时使用）
            db: Redis 数据库编号（当 config 为空时使用）
            password: Redis 密码（当 config 为空时使用）
            max_connections: 最大连接池大小（当 config 为空时使用）
        """
        if config is not None:
            self.redis_url = config.url
            self.max_connections = config.max_connections
        else:
            redis_host = host or "localhost"
            redis_port = port or 6379
            redis_db = db or 0
            redis_password = password
            
            if redis_password:
                self.redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
            else:
                self.redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
            self.max_connections = max_connections or 50
        
        self.pool = redis.ConnectionPool.from_url(
            self.redis_url,
            max_connections=self.max_connections,
            decode_responses=True
        )
        self.client: Optional[redis.Redis] = None
    
    async def connect(self):
        """建立 Redis 连接"""
        try:
            self.client = redis.Redis(connection_pool=self.pool)
            await self.client.ping()
            logger.info("Redis cache service connected")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
    
    async def disconnect(self):
        """关闭 Redis 连接"""
        if self.client:
            await self.client.close()
            await self.pool.disconnect()
            logger.info("Redis cache service disconnected")
    
    def _generate_key(self, prefix: str, *args) -> str:
        """
        生成缓存键
        
        Args:
            prefix: 键前缀
            *args: 键组件
            
        Returns:
            缓存键字符串
        """
        key_parts = [prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)
    
    def _hash_content(self, content: str) -> str:
        """
        生成内容的哈希值
        
        Args:
            content: 要哈希的内容
            
        Returns:
            哈希字符串
        """
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def get(self, key: str) -> Optional[Any]:
        """
        从缓存获取值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的值，未找到返回 None
        """
        try:
            if not self.client:
                return None
            
            value = await self.client.get(key)
            if value:
                logger.debug(f"Cache hit: {key}")
                return json.loads(value)
            
            logger.debug(f"Cache miss: {key}")
            return None
        except Exception as e:
            logger.error(f"Failed to get from cache: {str(e)}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 要缓存的值
            ttl: 生存时间（秒）（可选）
            
        Returns:
            成功返回 True，否则返回 False
        """
        try:
            if not self.client:
                return False
            
            serialized = json.dumps(value, default=str)
            
            if ttl:
                await self.client.setex(key, ttl, serialized)
            else:
                await self.client.set(key, serialized)
            
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Failed to set cache: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        从缓存删除值
        
        Args:
            key: 缓存键
            
        Returns:
            成功返回 True，否则返回 False
        """
        try:
            if not self.client:
                return False
            
            await self.client.delete(key)
            logger.debug(f"Cache deleted: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from cache: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        检查键是否存在于缓存中
        
        Args:
            key: 缓存键
            
        Returns:
            存在返回 True，否则返回 False
        """
        try:
            if not self.client:
                return False
            
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Failed to check cache existence: {str(e)}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        清除所有匹配模式的键
        
        Args:
            pattern: 键模式（例如："llm:*"）
            
        Returns:
            删除的键数量
        """
        try:
            if not self.client:
                return 0
            
            keys = []
            async for key in self.client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self.client.delete(*keys)
                logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted
            
            return 0
        except Exception as e:
            logger.error(f"Failed to clear cache pattern: {str(e)}")
            return 0
    
    # LLM 响应缓存
    
    async def cache_llm_response(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        response: str,
        temperature: float = 0.7
    ) -> bool:
        """
        缓存 LLM 响应
        
        Args:
            provider: LLM 提供商
            model: 模型名称
            messages: 输入消息
            response: LLM 响应
            temperature: 温度参数
            
        Returns:
            成功返回 True，否则返回 False
        """
        try:
            # 从消息生成哈希
            messages_str = json.dumps(messages, sort_keys=True)
            content_hash = self._hash_content(messages_str + str(temperature))
            
            key = self._generate_key(
                CACHE_PREFIX["llm_response"],
                provider,
                model,
                content_hash
            )
            
            return await self.set(
                key,
                {"response": response, "messages": messages},
                CACHE_TTL["llm_response"]
            )
        except Exception as e:
            logger.error(f"Failed to cache LLM response: {str(e)}")
            return False
    
    async def get_cached_llm_response(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        获取缓存的 LLM 响应
        
        Args:
            provider: LLM 提供商
            model: 模型名称
            messages: 输入消息
            temperature: 温度参数
            
        Returns:
            缓存的响应，未找到返回 None
        """
        try:
            messages_str = json.dumps(messages, sort_keys=True)
            content_hash = self._hash_content(messages_str + str(temperature))
            
            key = self._generate_key(
                CACHE_PREFIX["llm_response"],
                provider,
                model,
                content_hash
            )
            
            cached = await self.get(key)
            if cached:
                return cached.get("response")
            
            return None
        except Exception as e:
            logger.error(f"Failed to get cached LLM response: {str(e)}")
            return None
    
    # 嵌入缓存
    
    async def cache_embedding(
        self,
        provider: str,
        model: str,
        text: str,
        embedding: List[float]
    ) -> bool:
        """
        缓存嵌入向量
        
        Args:
            provider: 嵌入提供商
            model: 模型名称
            text: 输入文本
            embedding: 嵌入向量
            
        Returns:
            成功返回 True，否则返回 False
        """
        try:
            text_hash = self._hash_content(text)
            
            key = self._generate_key(
                CACHE_PREFIX["embedding"],
                provider,
                model,
                text_hash
            )
            
            return await self.set(
                key,
                {"text": text, "embedding": embedding},
                CACHE_TTL["embedding"]
            )
        except Exception as e:
            logger.error(f"Failed to cache embedding: {str(e)}")
            return False
    
    async def get_cached_embedding(
        self,
        provider: str,
        model: str,
        text: str
    ) -> Optional[List[float]]:
        """
        获取缓存的嵌入向量
        
        Args:
            provider: 嵌入提供商
            model: 模型名称
            text: 输入文本
            
        Returns:
            缓存的嵌入，未找到返回 None
        """
        try:
            text_hash = self._hash_content(text)
            
            key = self._generate_key(
                CACHE_PREFIX["embedding"],
                provider,
                model,
                text_hash
            )
            
            cached = await self.get(key)
            if cached:
                return cached.get("embedding")
            
            return None
        except Exception as e:
            logger.error(f"Failed to get cached embedding: {str(e)}")
            return None
    
    # 检索结果缓存
    
    async def cache_retrieval_result(
        self,
        workspace_id: str,
        query: str,
        results: List[Dict[str, Any]]
    ) -> bool:
        """
        缓存检索结果
        
        Args:
            workspace_id: 工作空间 ID
            query: 搜索查询
            results: 检索结果
            
        Returns:
            成功返回 True，否则返回 False
        """
        try:
            query_hash = self._hash_content(query)
            
            key = self._generate_key(
                CACHE_PREFIX["retrieval"],
                workspace_id,
                query_hash
            )
            
            return await self.set(
                key,
                {"query": query, "results": results},
                CACHE_TTL["retrieval"]
            )
        except Exception as e:
            logger.error(f"Failed to cache retrieval result: {str(e)}")
            return False
    
    async def get_cached_retrieval_result(
        self,
        workspace_id: str,
        query: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        获取缓存的检索结果
        
        Args:
            workspace_id: 工作空间 ID
            query: 搜索查询
            
        Returns:
            缓存的结果，未找到返回 None
        """
        try:
            query_hash = self._hash_content(query)
            
            key = self._generate_key(
                CACHE_PREFIX["retrieval"],
                workspace_id,
                query_hash
            )
            
            cached = await self.get(key)
            if cached:
                return cached.get("results")
            
            return None
        except Exception as e:
            logger.error(f"Failed to get cached retrieval result: {str(e)}")
            return None
    
    # 会话状态缓存
    
    async def cache_session_state(
        self,
        session_id: str,
        state: Dict[str, Any]
    ) -> bool:
        """
        缓存会话状态
        
        Args:
            session_id: 会话 ID
            state: 会话状态
            
        Returns:
            成功返回 True，否则返回 False
        """
        try:
            key = self._generate_key(CACHE_PREFIX["session"], session_id)
            return await self.set(key, state, CACHE_TTL["session"])
        except Exception as e:
            logger.error(f"Failed to cache session state: {str(e)}")
            return False
    
    async def get_cached_session_state(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取缓存的会话状态
        
        Args:
            session_id: 会话 ID
            
        Returns:
            缓存的状态，未找到返回 None
        """
        try:
            key = self._generate_key(CACHE_PREFIX["session"], session_id)
            return await self.get(key)
        except Exception as e:
            logger.error(f"Failed to get cached session state: {str(e)}")
            return None
    
    async def delete_session_state(self, session_id: str) -> bool:
        """
        删除缓存的会话状态
        
        Args:
            session_id: 会话 ID
            
        Returns:
            成功返回 True，否则返回 False
        """
        try:
            key = self._generate_key(CACHE_PREFIX["session"], session_id)
            return await self.delete(key)
        except Exception as e:
            logger.error(f"Failed to delete session state: {str(e)}")
            return False
    
    # 统计
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计
        
        Returns:
            包含缓存统计信息的字典
        """
        try:
            if not self.client:
                return {}
            
            info = await self.client.info("stats")
            
            return {
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                ),
                "total_keys": await self.client.dbsize(),
                "memory_used": info.get("used_memory_human", "0"),
                "connected_clients": info.get("connected_clients", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """计算缓存命中率"""
        total = hits + misses
        if total == 0:
            return 0.0
        return hits / total
    
    async def clear_all(self) -> bool:
        """
        清除所有缓存（谨慎使用）
        
        Returns:
            成功返回 True，否则返回 False
        """
        try:
            if not self.client:
                return False
            
            await self.client.flushdb()
            logger.warning("All cache cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear all cache: {str(e)}")
            return False
