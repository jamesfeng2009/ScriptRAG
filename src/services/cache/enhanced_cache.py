"""增强检索缓存 - 支持多级缓存和查询改写缓存

本模块扩展原有检索缓存功能：
1. 查询改写结果缓存（LLM 生成）
2. 嵌入结果缓存（LLM 生成）
3. 检索结果缓存（完整流程）
4. 缓存统计和监控
5. 缓存预热功能
6. Redis 后端支持（生产环境推荐）

使用示例：
    # 内存缓存（开发环境）
    cache = EnhancedRetrievalCache()

    # Redis 缓存（生产环境）
    from src.services.database.redis_cache import RedisCacheService
    redis_service = RedisCacheService()
    await redis_service.connect()

    redis_cache = EnhancedRetrievalCache(
        storage_backend={
            "type": "redis",
            "service": redis_service
        }
    )
"""

import hashlib
import logging
import time
import json
import struct
from typing import (
    Dict, Any, Optional, List, Tuple, Callable, Union
)
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


# Redis 键前缀
REDIS_KEY_PREFIX = "enhcache"


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    hit_rate: float = 0.0

    def record_hit(self):
        self.hits += 1
        self._update_hit_rate()

    def record_miss(self):
        self.misses += 1
        self._update_hit_rate()

    def record_set(self):
        self.sets += 1

    def record_eviction(self):
        self.evictions += 1

    def _update_hit_rate(self):
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "total_requests": self.hits + self.misses
        }


@dataclass
class RewriteResultCacheEntry:
    """查询改写结果缓存条目"""
    rewritten_query: str
    sub_queries: List[str]
    confidence: float
    query_type: str
    created_at: float
    ttl: int

    def is_expired(self, current_time: float) -> bool:
        return current_time - self.created_at > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rewritten_query": self.rewritten_query,
            "sub_queries": self.sub_queries,
            "confidence": self.confidence,
            "query_type": self.query_type,
            "created_at": self.created_at,
            "ttl": self.ttl
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewriteResultCacheEntry":
        return cls(
            rewritten_query=data["rewritten_query"],
            sub_queries=data["sub_queries"],
            confidence=data["confidence"],
            query_type=data["query_type"],
            created_at=data["created_at"],
            ttl=data["ttl"]
        )


@dataclass
class EnhancedCacheConfig:
    """增强缓存配置"""
    enabled: bool = True

    query_rewrite_enabled: bool = True
    query_rewrite_ttl: int = 3600
    query_rewrite_max_size: int = 1000

    embedding_enabled: bool = True
    embedding_ttl: int = 86400
    embedding_max_size: int = 10000

    result_enabled: bool = True
    result_ttl: int = 300
    result_max_size: int = 500

    preheat_enabled: bool = False
    preheat_batch_size: int = 10


class MemoryCacheBackend:
    """内存缓存后端（开发环境）"""

    def __init__(self, config: EnhancedCacheConfig):
        self.config = config
        self._query_rewrite_cache: Dict[str, RewriteResultCacheEntry] = {}
        self._embedding_cache: Dict[str, Tuple[List[float], float]] = {}
        self._result_cache: Dict[str, Tuple[Any, float]] = {}

    def _generate_key(self, cache_type: str, *args) -> str:
        content = ":".join(str(arg) for arg in args)
        return f"{REDIS_KEY_PREFIX}:{cache_type}:{hashlib.md5(content.encode()).hexdigest()}"

    def _is_expired(self, timestamp: float, ttl: int) -> bool:
        return time.time() - timestamp > ttl

    def _evict_lru(self, cache: Dict, ttl: int, max_size: int) -> None:
        if len(cache) < max_size:
            return

        current_time = time.time()
        valid_items = [
            (key, ts) for key, (_, ts) in cache.items()
            if not self._is_expired(ts, ttl)
        ]
        valid_items.sort(key=lambda x: x[1])

        for _ in range(len(cache) - max_size + 10):
            if valid_items:
                key, _ = valid_items.pop(0)
                if key in cache:
                    del cache[key]


class RedisCacheBackend:
    """Redis 缓存后端（生产环境）

    数据结构设计：
    - 查询改写结果：String (JSON) - 键: {prefix}:rewrite:{hash}
    - 嵌入向量：Hash - 键: {prefix}:emb:{hash}, 字段: 0,1,2..., 值: float32
    - 检索结果：String (JSON) - 键: {prefix}:result:{hash}
    - LRU 追踪：Sorted Set - 键: {prefix}:lru:{type}, 分数: timestamp
    """

    def __init__(
        self,
        config: EnhancedCacheConfig,
        redis_service: Any
    ):
        self.config = config
        self.redis = redis_service

        self._ttls = {
            "query_rewrite": config.query_rewrite_ttl,
            "embedding": config.embedding_ttl,
            "result": config.result_ttl
        }

        self._max_sizes = {
            "query_rewrite": config.query_rewrite_max_size,
            "embedding": config.embedding_max_size,
            "result": config.result_max_size
        }

    def _generate_key(self, cache_type: str, *args) -> str:
        content = ":".join(str(arg) for arg in args)
        return f"{REDIS_KEY_PREFIX}:{cache_type}:{hashlib.md5(content.encode()).hexdigest()}"

    async def get_rewritten_query(self, query: str) -> Optional[Dict[str, Any]]:
        key = self._generate_key("rewrite", query)
        data = await self.redis.get(key)
        if data:
            entry = RewriteResultCacheEntry.from_dict(data)
            if not entry.is_expired(time.time()):
                return entry.to_dict()
            else:
                await self.redis.delete(key)
        return None

    async def set_rewritten_query(
        self,
        query: str,
        rewritten_query: str,
        sub_queries: List[str],
        confidence: float,
        query_type: str
    ) -> None:
        key = self._generate_key("rewrite", query)
        entry = RewriteResultCacheEntry(
            rewritten_query=rewritten_query,
            sub_queries=sub_queries,
            confidence=confidence,
            query_type=query_type,
            created_at=time.time(),
            ttl=self._ttls["query_rewrite"]
        )
        await self.redis.set(key, entry.to_dict(), self._ttls["query_rewrite"])

        lru_key = f"{REDIS_KEY_PREFIX}:lru:rewrite"
        await self.redis.client.zadd(lru_key, {key: time.time()})

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        key = self._generate_key("emb", text)

        exists = await self.redis.exists(key)
        if not exists:
            return None

        fields = await self.redis.client.hgetall(key)
        if not fields:
            await self.redis.delete(key)
            return None

        embedding = []
        for field, value in fields.items():
            if isinstance(value, bytes):
                embedding.append(struct.unpack('!f', value)[0])
            else:
                try:
                    embedding.append(float(value))
                except (ValueError, TypeError):
                    embedding.append(float.fromhex(value) if isinstance(value, str) else 0.0)

        return embedding

    async def _is_embedding_expired(self, key: str) -> bool:
        lru_key = f"{REDIS_KEY_PREFIX}:lru:emb"
        score = await self.redis.client.zscore(lru_key, key)
        if score is None:
            ttl = await self.redis.client.ttl(key)
            if ttl == -2:
                return True
            return False
        return time.time() - score > self._ttls["embedding"]

    async def set_embedding(self, text: str, embedding: List[float]) -> None:
        key = self._generate_key("emb", text)

        pipe = self.redis.client.pipeline()
        pipe.delete(key)

        for i, val in enumerate(embedding):
            pipe.hset(key, str(i), struct.pack('!f', val))

        pipe.expire(key, self._ttls["embedding"])
        pipe.zadd(f"{REDIS_KEY_PREFIX}:lru:emb", {key: time.time()})
        await pipe.execute()

    async def get_results(
        self,
        workspace_id: str,
        query: str,
        strategy: str,
        top_k: int
    ) -> Optional[List[Any]]:
        key = self._generate_key("result", workspace_id, query, strategy, top_k)
        return await self.redis.get(key)

    async def set_results(
        self,
        workspace_id: str,
        query: str,
        strategy: str,
        top_k: int,
        results: Any
    ) -> None:
        key = self._generate_key("result", workspace_id, query, strategy, top_k)
        await self.redis.set(key, results, self._ttls["result"])

        lru_key = f"{REDIS_KEY_PREFIX}:lru:result"
        await self.redis.client.zadd(lru_key, {key: time.time()})

    async def clear_cache(self, cache_type: str) -> None:
        pattern = f"{REDIS_KEY_PREFIX}:{cache_type}:*"
        await self.redis.clear_pattern(pattern)

        lru_key = f"{REDIS_KEY_PREFIX}:lru:{cache_type}"
        await self.redis.client.delete(lru_key)

    async def get_stats(self) -> Dict[str, Any]:
        stats = {"query_rewrite": {}, "embedding": {}, "result": {}}

        for cache_type in ["rewrite", "emb", "result"]:
            lru_key = f"{REDIS_KEY_PREFIX}:lru:{cache_type}"
            total = await self.redis.client.zcard(lru_key)

            current_time = time.time()
            expired = 0
            async for key in self.redis.client.scan_iter(match=f"{REDIS_KEY_PREFIX}:{cache_type}:*"):
                score = await self.redis.client.zscore(lru_key, key)
                if score and current_time - score > self._ttls.get(cache_type, 3600):
                    expired += 1

            stats[cache_type] = {
                "total_keys": total,
                "expired_keys": expired,
                "ttl_seconds": self._ttls.get(cache_type, 3600)
            }

        return stats


class EnhancedRetrievalCache:
    """
    增强检索缓存

    功能：
    1. 查询改写结果缓存
    2. 嵌入结果缓存
    3. 检索结果缓存
    4. 多级缓存统计
    5. 缓存预热（可选）
    6. Redis 后端支持

    支持两种后端：
    - MemoryCacheBackend: 内存缓存，适合开发测试
    - RedisCacheBackend: Redis 缓存，适合生产环境

    设计考量：
    - 使用 LRU + TTL 策略
    - 分离不同类型的缓存，避免互相影响
    - 提供完整的统计信息
    - Redis 后端使用 Hash 存储向量，分字段压缩
    """

    def __init__(
        self,
        config: Optional[EnhancedCacheConfig] = None,
        storage_backend: Optional[Dict[str, Any]] = None
    ):
        """
        初始化增强检索缓存

        Args:
            config: 缓存配置
            storage_backend: 存储后端（可选，用于持久化）
        """
        self.config = config or EnhancedCacheConfig()
        self.storage = storage_backend

        self._query_rewrite_cache: Dict[str, RewriteResultCacheEntry] = {}
        self._embedding_cache: Dict[str, Tuple[List[float], float]] = {}
        self._result_cache: Dict[str, Tuple[Any, float]] = {}

        self._stats = {
            "query_rewrite": CacheStats(),
            "embedding": CacheStats(),
            "result": CacheStats()
        }

        self._max_sizes = {
            "query_rewrite": self.config.query_rewrite_max_size,
            "embedding": self.config.embedding_max_size,
            "result": self.config.result_max_size
        }

        self._ttls = {
            "query_rewrite": self.config.query_rewrite_ttl,
            "embedding": self.config.embedding_ttl,
            "result": self.config.result_ttl
        }

        logger.info(f"EnhancedRetrievalCache initialized (enabled={self.config.enabled})")

    def _generate_key(self, cache_type: str, *args) -> str:
        """生成缓存键"""
        content = ":".join(str(arg) for arg in args)
        return f"{cache_type}:{hashlib.md5(content.encode()).hexdigest()}"

    def _is_expired(self, timestamp: float, cache_type: str) -> bool:
        """检查是否过期"""
        ttl = self._ttls.get(cache_type, 300)
        return time.time() - timestamp > ttl

    def _evict_lru(self, cache_type: str, count: int = 1) -> None:
        """淘汰最近最少使用的条目"""
        cache = self._get_cache(cache_type)
        if not cache:
            return

        current_time = time.time()
        sorted_items = [
            (key, timestamp) for key, (_, timestamp) in cache.items()
            if not self._is_expired(timestamp, cache_type)
        ]
        sorted_items.sort(key=lambda x: x[1])

        for _ in range(count):
            if sorted_items:
                key, _ = sorted_items.pop(0)
                if key in cache:
                    del cache[key]
                    self._stats[cache_type].record_eviction()

    def _get_cache(self, cache_type: str) -> Optional[Dict]:
        """获取指定类型的缓存"""
        cache_map = {
            "query_rewrite": self._query_rewrite_cache,
            "embedding": self._embedding_cache,
            "result": self._result_cache
        }
        return cache_map.get(cache_type)

    def _check_capacity(self, cache_type: str) -> None:
        """检查容量，超出则淘汰"""
        cache = self._get_cache(cache_type)
        if not cache:
            return

        max_size = self._max_sizes.get(cache_type, 1000)
        if len(cache) >= max_size:
            self._evict_lru(cache_type, len(cache) - max_size + 10)

    # =========================================================================
    # 查询改写缓存
    # =========================================================================

    def get_rewritten_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存的查询改写结果

        Args:
            query: 原始查询

        Returns:
            改写结果字典，未缓存返回 None
        """
        if not self.config.enabled or not self.config.query_rewrite_enabled:
            return None

        cache_key = self._generate_key("rewrite", query)
        cache = self._query_rewrite_cache

        if cache_key in cache:
            entry = cache[cache_key]
            if not self._is_expired(entry.created_at, "query_rewrite"):
                self._stats["query_rewrite"].record_hit()
                logger.debug(f"Query rewrite cache hit: {query[:50]}...")
                return entry.to_dict()
            else:
                del cache[cache_key]

        self._stats["query_rewrite"].record_miss()
        return None

    def set_rewritten_query(
        self,
        query: str,
        rewritten_query: str,
        sub_queries: List[str],
        confidence: float = 1.0,
        query_type: str = "hybrid"
    ) -> None:
        """
        缓存查询改写结果

        Args:
            query: 原始查询
            rewritten_query: 改写后的查询
            sub_queries: 子查询列表
            confidence: 置信度
            query_type: 查询类型
        """
        if not self.config.enabled or not self.config.query_rewrite_enabled:
            return

        self._check_capacity("query_rewrite")

        cache_key = self._generate_key("rewrite", query)
        entry = RewriteResultCacheEntry(
            rewritten_query=rewritten_query,
            sub_queries=sub_queries,
            confidence=confidence,
            query_type=query_type,
            created_at=time.time(),
            ttl=self.config.query_rewrite_ttl
        )

        self._query_rewrite_cache[cache_key] = entry
        self._stats["query_rewrite"].record_set()
        logger.debug(f"Cached query rewrite: {query[:50]}...")

    # =========================================================================
    # 嵌入缓存
    # =========================================================================

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        获取缓存的文本嵌入

        Args:
            text: 文本

        Returns:
            嵌入向量，未缓存返回 None
        """
        if not self.config.enabled or not self.config.embedding_enabled:
            return None

        cache_key = self._generate_key("embedding", text)
        cache = self._embedding_cache

        if cache_key in cache:
            embedding, timestamp = cache[cache_key]
            if not self._is_expired(timestamp, "embedding"):
                self._stats["embedding"].record_hit()
                logger.debug(f"Embedding cache hit: {text[:50]}...")
                return embedding
            else:
                del cache[cache_key]

        self._stats["embedding"].record_miss()
        return None

    def set_embedding(self, text: str, embedding: List[float]) -> None:
        """
        缓存文本嵌入

        Args:
            text: 文本
            embedding: 嵌入向量
        """
        if not self.config.enabled or not self.config.embedding_enabled:
            return

        self._check_capacity("embedding")

        cache_key = self._generate_key("embedding", text)
        self._embedding_cache[cache_key] = (embedding, time.time())
        self._stats["embedding"].record_set()
        logger.debug(f"Cached embedding: {text[:50]}...")

    def get_embeddings_batch(
        self,
        texts: List[str]
    ) -> Tuple[List[Optional[List[float]]], List[int]]:
        """
        获取多个文本的缓存嵌入

        Args:
            texts: 文本列表

        Returns:
            元组（嵌入列表，未命中的为 None，未命中的索引）
        """
        embeddings = []
        miss_indices = []

        for i, text in enumerate(texts):
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
            if embedding is None:
                miss_indices.append(i)

        return embeddings, miss_indices

    def set_embeddings_batch(
        self,
        texts: List[str],
        embeddings: List[List[float]]
    ) -> None:
        """
        缓存多个文本的嵌入

        Args:
            texts: 文本列表
            embeddings: 嵌入向量列表
        """
        for text, embedding in zip(texts, embeddings):
            self.set_embedding(text, embedding)

    # =========================================================================
    # 结果缓存
    # =========================================================================

    def get_results(
        self,
        workspace_id: str,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 10
    ) -> Optional[List[Any]]:
        """
        获取缓存的检索结果

        Args:
            workspace_id: 工作空间 ID
            query: 查询文本
            strategy: 检索策略
            top_k: 返回结果数量

        Returns:
            缓存结果，未缓存返回 None
        """
        if not self.config.enabled or not self.config.result_enabled:
            return None

        cache_key = self._generate_key("result", workspace_id, query, strategy, top_k)
        cache = self._result_cache

        if cache_key in cache:
            results, timestamp = cache[cache_key]
            if not self._is_expired(timestamp, "result"):
                self._stats["result"].record_hit()
                logger.debug(f"Result cache hit: {query[:50]}...")
                return results
            else:
                del cache[cache_key]

        self._stats["result"].record_miss()
        return None

    def set_results(
        self,
        workspace_id: str,
        query: str,
        results: List[Any],
        strategy: str = "hybrid",
        top_k: int = 10
    ) -> None:
        """
        缓存检索结果

        Args:
            workspace_id: 工作空间 ID
            query: 查询文本
            results: 检索结果
            strategy: 检索策略
            top_k: 返回结果数量
        """
        if not self.config.enabled or not self.config.result_enabled:
            return

        self._check_capacity("result")

        cache_key = self._generate_key("result", workspace_id, query, strategy, top_k)
        self._result_cache[cache_key] = (results, time.time())
        self._stats["result"].record_set()
        logger.debug(f"Cached results: {query[:50]}...")

    # =========================================================================
    # 缓存管理
    # =========================================================================

    def invalidate(self, cache_type: str, key: str) -> bool:
        """
        使单个缓存条目失效

        Args:
            cache_type: 缓存类型
            key: 缓存键

        Returns:
            是否删除成功
        """
        cache = self._get_cache(cache_type)
        if cache and key in cache:
            del cache[key]
            return True
        return False

    def invalidate_query_rewrite(self, query: str) -> bool:
        """使查询改写缓存失效"""
        cache_key = self._generate_key("rewrite", query)
        return self.invalidate("query_rewrite", cache_key)

    def invalidate_embedding(self, text: str) -> bool:
        """使嵌入缓存失效"""
        cache_key = self._generate_key("embedding", text)
        return self.invalidate("embedding", cache_key)

    def invalidate_results(
        self,
        workspace_id: str,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 10
    ) -> bool:
        """使检索结果缓存失效"""
        cache_key = self._generate_key("result", workspace_id, query, strategy, top_k)
        return self.invalidate("result", cache_key)

    def invalidate_workspace(self, workspace_id: str) -> int:
        """
        使工作空间的所有缓存失效

        Args:
            workspace_id: 工作空间 ID

        Returns:
            失效的条目数量
        """
        count = 0

        for key in list(self._result_cache.keys()):
            if key.startswith(f"result:{workspace_id}"):
                del self._result_cache[key]
                count += 1

        logger.info(f"Invalidated {count} cache entries for workspace {workspace_id}")
        return count

    def clear(self, cache_type: Optional[str] = None) -> Dict[str, int]:
        """
        清除缓存

        Args:
            cache_type: 缓存类型，None 表示所有

        Returns:
            清除的条目数量
        """
        counts = {}

        if cache_type is None or cache_type == "query_rewrite":
            count = len(self._query_rewrite_cache)
            self._query_rewrite_cache.clear()
            counts["query_rewrite"] = count

        if cache_type is None or cache_type == "embedding":
            count = len(self._embedding_cache)
            self._embedding_cache.clear()
            counts["embedding"] = count

        if cache_type is None or cache_type == "result":
            count = len(self._result_cache)
            self._result_cache.clear()
            counts["result"] = count

        logger.info(f"Cleared caches: {counts}")
        return counts

    def get_stats(self, cache_type: Optional[str] = None) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Args:
            cache_type: 缓存类型，None 表示所有

        Returns:
            统计信息字典
        """
        if cache_type:
            return self._stats.get(cache_type, CacheStats()).to_dict()

        return {
            "query_rewrite": self._stats["query_rewrite"].to_dict(),
            "embedding": self._stats["embedding"].to_dict(),
            "result": self._stats["result"].to_dict()
        }

    def get_sizes(self) -> Dict[str, int]:
        """获取各缓存的大小"""
        return {
            "query_rewrite": len(self._query_rewrite_cache),
            "embedding": len(self._embedding_cache),
            "result": len(self._result_cache)
        }

    def get_capacity_info(self) -> Dict[str, Any]:
        """获取容量信息"""
        sizes = self.get_sizes()
        return {
            "query_rewrite": {
                "current": sizes["query_rewrite"],
                "max": self._max_sizes["query_rewrite"],
                "utilization": sizes["query_rewrite"] / self._max_sizes["query_rewrite"]
                if self._max_sizes["query_rewrite"] > 0 else 0
            },
            "embedding": {
                "current": sizes["embedding"],
                "max": self._max_sizes["embedding"],
                "utilization": sizes["embedding"] / self._max_sizes["embedding"]
                if self._max_sizes["embedding"] > 0 else 0
            },
            "result": {
                "current": sizes["result"],
                "max": self._max_sizes["result"],
                "utilization": sizes["result"] / self._max_sizes["result"]
                if self._max_sizes["result"] > 0 else 0
            }
        }

    # =========================================================================
    # 缓存预热
    # =========================================================================

    async def preheat(
        self,
        queries: List[str],
        embedding_func: Optional[Callable] = None,
        rewrite_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        缓存预热

        Args:
            queries: 查询列表
            embedding_func: 嵌入函数（可选）
            rewrite_func: 改写函数（可选）

        Returns:
            预热结果统计
        """
        if not self.config.preheat_enabled:
            logger.info("Preheat disabled, skipping")
            return {"skipped": True}

        results = {
            "total_queries": len(queries),
            "rewritten": 0,
            "embedded": 0,
            "errors": []
        }

        for i, query in enumerate(queries):
            try:
                if rewrite_func:
                    rewritten = await rewrite_func(query)
                    if rewritten:
                        self.set_rewritten_query(
                            query,
                            rewritten.get("rewritten_query", query),
                            rewritten.get("sub_queries", []),
                            rewritten.get("confidence", 1.0)
                        )
                        results["rewritten"] += 1

                if embedding_func:
                    embedding = await embedding_func(query)
                    if embedding:
                        self.set_embedding(query, embedding)
                        results["embedded"] += 1

            except Exception as e:
                results["errors"].append({"query": query[:50], "error": str(e)})

            if (i + 1) % self.config.preheat_batch_size == 0:
                logger.info(f"Preheat progress: {i + 1}/{len(queries)}")

        logger.info(
            f"Preheat completed: {results['rewritten']} rewritten, "
            f"{results['embedded']} embedded"
        )

        return results

    # =========================================================================
    # 持久化支持
    # =========================================================================

    async def save_to_storage(self, path: str) -> bool:
        """
        保存缓存到文件

        Args:
            path: 文件路径

        Returns:
            是否保存成功
        """
        try:
            data = {
                "query_rewrite": {
                    k: v.to_dict() for k, v in self._query_rewrite_cache.items()
                },
                "embedding": {
                    k: {"embedding": emb, "timestamp": ts}
                    for k, (emb, ts) in self._embedding_cache.items()
                },
                "result": {
                    k: {"results": res, "timestamp": ts}
                    for k, (res, ts) in self._result_cache.items()
                }
            }

            with open(path, 'w') as f:
                json.dump(data, f)

            logger.info(f"Cache saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save cache: {str(e)}")
            return False

    async def load_from_storage(self, path: str) -> bool:
        """
        从文件加载缓存

        Args:
            path: 文件路径

        Returns:
            是否加载成功
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            for k, v in data.get("query_rewrite", {}).items():
                entry = RewriteResultCacheEntry.from_dict(v)
                if not self._is_expired(entry.created_at, "query_rewrite"):
                    self._query_rewrite_cache[k] = entry

            for k, v in data.get("embedding", {}).items():
                if not self._is_expired(v["timestamp"], "embedding"):
                    self._embedding_cache[k] = (v["embedding"], v["timestamp"])

            for k, v in data.get("result", {}).items():
                if not self._is_expired(v["timestamp"], "result"):
                    self._result_cache[k] = (v["results"], v["timestamp"])

            logger.info(f"Cache loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load cache: {str(e)}")
            return False
