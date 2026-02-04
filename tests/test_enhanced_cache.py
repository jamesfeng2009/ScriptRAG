"""增强检索缓存测试用例

测试模块：
1. MemoryCacheBackend - 内存缓存后端
2. RedisCacheBackend - Redis 缓存后端
3. EnhancedRetrievalCache - 增强检索缓存
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any, Optional

from src.services.cache.enhanced_cache import (
    EnhancedRetrievalCache,
    EnhancedCacheConfig,
    RewriteResultCacheEntry,
    CacheStats,
    MemoryCacheBackend,
    RedisCacheBackend,
    REDIS_KEY_PREFIX
)


class TestRewriteResultCacheEntry:
    """查询改写缓存条目测试"""

    def test_create_entry(self):
        """测试创建缓存条目"""
        entry = RewriteResultCacheEntry(
            rewritten_query="Python 是一种高级编程语言",
            sub_queries=["Python 特点", "Python 用途"],
            confidence=0.95,
            query_type="hybrid",
            created_at=time.time(),
            ttl=3600
        )

        assert entry.rewritten_query == "Python 是一种高级编程语言"
        assert entry.sub_queries == ["Python 特点", "Python 用途"]
        assert entry.confidence == 0.95
        assert entry.query_type == "hybrid"

    def test_is_expired(self):
        """测试过期判断"""
        entry = RewriteResultCacheEntry(
            rewritten_query="test",
            sub_queries=[],
            confidence=1.0,
            query_type="simple",
            created_at=time.time() - 7200,
            ttl=3600
        )

        assert entry.is_expired(time.time()) is True

        fresh_entry = RewriteResultCacheEntry(
            rewritten_query="test",
            sub_queries=[],
            confidence=1.0,
            query_type="simple",
            created_at=time.time(),
            ttl=3600
        )

        assert fresh_entry.is_expired(time.time()) is False

    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化"""
        original = RewriteResultCacheEntry(
            rewritten_query="测试查询",
            sub_queries=["子查询1", "子查询2"],
            confidence=0.88,
            query_type="complex",
            created_at=1234567890.0,
            ttl=1800
        )

        data = original.to_dict()
        restored = RewriteResultCacheEntry.from_dict(data)

        assert restored.rewritten_query == original.rewritten_query
        assert restored.sub_queries == original.sub_queries
        assert restored.confidence == original.confidence
        assert restored.query_type == original.query_type
        assert restored.created_at == original.created_at
        assert restored.ttl == original.ttl


class TestEnhancedCacheConfig:
    """增强缓存配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = EnhancedCacheConfig()

        assert config.enabled is True
        assert config.query_rewrite_enabled is True
        assert config.query_rewrite_ttl == 3600
        assert config.query_rewrite_max_size == 1000
        assert config.embedding_enabled is True
        assert config.embedding_ttl == 86400
        assert config.embedding_max_size == 10000
        assert config.result_enabled is True
        assert config.result_ttl == 300
        assert config.result_max_size == 500

    def test_custom_config(self):
        """测试自定义配置"""
        config = EnhancedCacheConfig(
            enabled=False,
            query_rewrite_ttl=7200,
            embedding_max_size=5000,
            result_ttl=600
        )

        assert config.enabled is False
        assert config.query_rewrite_ttl == 7200
        assert config.embedding_max_size == 5000
        assert config.result_ttl == 600


class TestCacheStats:
    """缓存统计测试"""

    def test_initial_stats(self):
        """测试初始统计"""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.evictions == 0
        assert stats.hit_rate == 0.0

    def test_record_hit_and_miss(self):
        """测试命中和未命中记录"""
        stats = CacheStats()

        for _ in range(3):
            stats.record_hit()
        for _ in range(7):
            stats.record_miss()

        assert stats.hits == 3
        assert stats.misses == 7
        assert stats.hit_rate == pytest.approx(0.3, rel=1e-2)

    def test_record_set_and_eviction(self):
        """测试设置和淘汰记录"""
        stats = CacheStats()

        stats.record_set()
        stats.record_set()
        stats.record_eviction()

        assert stats.sets == 2
        assert stats.evictions == 1

    def test_to_dict(self):
        """测试转换为字典"""
        stats = CacheStats()
        stats.hits = 10
        stats.misses = 10
        stats.sets = 5
        stats.evictions = 2

        data = stats.to_dict()

        assert data["hits"] == 10
        assert data["misses"] == 10
        assert data["sets"] == 5
        assert data["evictions"] == 2
        assert data["total_requests"] == 20
        assert "hit_rate" in data


class TestMemoryCacheBackend:
    """内存缓存后端测试"""

    @pytest.fixture
    def config(self):
        return EnhancedCacheConfig(
            query_rewrite_max_size=100,
            embedding_max_size=1000,
            result_max_size=50
        )

    @pytest.fixture
    def backend(self, config):
        return MemoryCacheBackend(config)

    def test_init(self, backend, config):
        """测试初始化"""
        assert backend.config == config
        assert hasattr(backend, "_query_rewrite_cache")
        assert hasattr(backend, "_embedding_cache")
        assert hasattr(backend, "_result_cache")

    def test_generate_key(self, backend):
        """测试键生成"""
        key1 = backend._generate_key("rewrite", "test query")
        key2 = backend._generate_key("rewrite", "test query")
        key3 = backend._generate_key("rewrite", "different query")

        assert key1 == key2
        assert key1 != key3
        assert key1.startswith(f"{REDIS_KEY_PREFIX}:rewrite:")

    def test_is_expired(self, backend):
        """测试过期检查"""
        assert backend._is_expired(time.time() - 7200, 3600) is True
        assert backend._is_expired(time.time() - 1800, 3600) is False

    def test_rewrite_query_cache(self, backend):
        """测试查询改写缓存"""
        query = "Python 是什么"
        rewritten = "Python 是一种编程语言"
        sub_queries = ["Python 特点", "Python 用途"]

        result = backend._query_rewrite_cache.get(query)
        assert result is None

        backend._query_rewrite_cache[query] = RewriteResultCacheEntry(
            rewritten_query=rewritten,
            sub_queries=sub_queries,
            confidence=0.95,
            query_type="hybrid",
            created_at=time.time(),
            ttl=3600
        )

        cached = backend._query_rewrite_cache.get(query)
        assert cached is not None
        assert cached.rewritten_query == rewritten
        assert cached.sub_queries == sub_queries

    def test_embedding_cache(self, backend):
        """测试嵌入缓存"""
        text = "测试文本"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        result = backend._embedding_cache.get(text)
        assert result is None

        backend._embedding_cache[text] = (embedding, time.time())

        cached = backend._embedding_cache.get(text)
        assert cached is not None
        assert cached[0] == embedding


class MockRedisClient:
    """Mock Redis 客户端"""

    def __init__(self):
        self.data = {}
        self.sorted_sets = {}
        self.hash_fields = {}
        self.expiry = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ttl=None):
        self.data[key] = value
        if ttl:
            self.expiry[key] = time.time() + ttl
        return True

    async def setex(self, key, ttl, value):
        self.data[key] = value
        self.expiry[key] = time.time() + ttl
        return True

    async def delete(self, *keys):
        deleted = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                deleted += 1
            if key in self.hash_fields:
                del self.hash_fields[key]
                deleted += 1
        return deleted

    async def exists(self, key):
        if key in self.data:
            return 1
        if key in self.hash_fields:
            return 1
        return 0

    async def hset(self, key, field, value):
        if key not in self.hash_fields:
            self.hash_fields[key] = {}
        self.hash_fields[key][field] = value
        return 1

    async def hgetall(self, key):
        return self.hash_fields.get(key, {})

    async def expire(self, key, seconds):
        self.expiry[key] = time.time() + seconds
        return 1

    async def zadd(self, key, mapping):
        if key not in self.sorted_sets:
            self.sorted_sets[key] = {}
        self.sorted_sets[key].update(mapping)
        return len(mapping)

    async def zscore(self, key, member):
        return self.sorted_sets.get(key, {}).get(member)

    async def zcard(self, key):
        return len(self.sorted_sets.get(key, {}))

    async def clear_pattern(self, pattern):
        keys_to_delete = [k for k in self.data if k.startswith(pattern.replace("*", ""))]
        for key in keys_to_delete:
            del self.data[key]
        return len(keys_to_delete)

    async def ttl(self, key):
        if key in self.expiry:
            remaining = self.expiry[key] - time.time()
            if remaining > 0:
                return int(remaining)
            else:
                if key in self.expiry:
                    del self.expiry[key]
                return -2
        return -2

    def pipeline(self):
        return MockPipeline(self)


class MockPipeline:
    """Mock Redis Pipeline"""

    def __init__(self, client):
        self.client = client
        self.commands = []

    def delete(self, key):
        self.commands.append(("delete", key))
        return self

    def hset(self, key, field, value):
        self.commands.append(("hset", key, field, value))
        return self

    def expire(self, key, seconds):
        self.commands.append(("expire", key, seconds))
        return self

    def zadd(self, key, mapping):
        self.commands.append(("zadd", key, mapping))
        return self

    async def execute(self):
        results = []
        for cmd in self.commands:
            if cmd[0] == "delete":
                results.append(await self.client.delete(cmd[1]))
            elif cmd[0] == "hset":
                results.append(await self.client.hset(cmd[1], cmd[2], cmd[3]))
            elif cmd[0] == "expire":
                results.append(await self.client.expire(cmd[1], cmd[2]))
            elif cmd[0] == "zadd":
                results.append(await self.client.zadd(cmd[1], cmd[2]))
        return results


class MockRedisService:
    """Mock Redis 服务"""

    def __init__(self):
        self.client = MockRedisClient()

    async def get(self, key):
        value = await self.client.get(key)
        if value:
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return None

    async def set(self, key, value, ttl=None):
        if isinstance(value, dict):
            value = json.dumps(value)
        await self.client.set(key, value, ttl)
        return True

    async def delete(self, key):
        return await self.client.delete(key)

    async def exists(self, key):
        return await self.client.exists(key)

    async def clear_pattern(self, pattern):
        return await self.client.clear_pattern(pattern)


class TestRedisCacheBackend:
    """Redis 缓存后端测试"""

    @pytest.fixture
    def config(self):
        return EnhancedCacheConfig(
            query_rewrite_max_size=100,
            embedding_max_size=1000,
            result_max_size=50
        )

    @pytest.fixture
    def redis_service(self):
        return MockRedisService()

    @pytest.fixture
    def backend(self, config, redis_service):
        return RedisCacheBackend(config, redis_service)

    @pytest.mark.asyncio
    async def test_get_rewritten_query_miss(self, backend, redis_service):
        """测试查询改写缓存未命中"""
        result = await backend.get_rewritten_query("不存在查询")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get_rewritten_query(self, backend, redis_service):
        """测试查询改写缓存设置和获取"""
        query = "Python 教程"
        rewritten = "Python 编程教程"
        sub_queries = ["Python 入门", "Python 基础"]

        await backend.set_rewritten_query(
            query=query,
            rewritten_query=rewritten,
            sub_queries=sub_queries,
            confidence=0.92,
            query_type="hybrid"
        )

        result = await backend.get_rewritten_query(query)

        assert result is not None
        assert result["rewritten_query"] == rewritten
        assert result["sub_queries"] == sub_queries
        assert result["confidence"] == 0.92
        assert result["query_type"] == "hybrid"

    @pytest.mark.asyncio
    async def test_embedding_cache(self, backend, redis_service):
        """测试嵌入缓存"""
        import struct

        text = "测试嵌入文本"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        result = await backend.get_embedding(text)
        assert result is None

        await backend.set_embedding(text, embedding)

        cached = await backend.get_embedding(text)
        assert cached is not None
        assert len(cached) == len(embedding)
        assert all(abs(a - b) < 1e-6 for a, b in zip(cached, embedding))

    @pytest.mark.asyncio
    async def test_results_cache(self, backend, redis_service):
        """测试检索结果缓存"""
        workspace_id = "ws_001"
        query = "测试查询"
        strategy = "hybrid"
        top_k = 10
        results = [{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.8}]

        result = await backend.get_results(workspace_id, query, strategy, top_k)
        assert result is None

        await backend.set_results(workspace_id, query, strategy, top_k, results)

        cached = await backend.get_results(workspace_id, query, strategy, top_k)
        assert cached is not None
        assert len(cached) == 2


class TestEnhancedRetrievalCache:
    """增强检索缓存测试"""

    @pytest.fixture
    def config(self):
        return EnhancedCacheConfig(
            query_rewrite_max_size=10,
            embedding_max_size=100,
            result_max_size=20
        )

    @pytest.fixture
    def cache(self, config):
        return EnhancedRetrievalCache(config=config)

    def test_init_with_memory_backend(self, cache, config):
        """测试使用内存后端初始化"""
        assert cache.config == config
        assert cache.storage is None
        assert hasattr(cache, "_query_rewrite_cache")
        assert hasattr(cache, "_embedding_cache")
        assert hasattr(cache, "_result_cache")

    def test_get_rewritten_query_memory(self, cache):
        """测试内存后端获取查询改写结果"""
        query = "测试查询"

        result = cache.get_rewritten_query(query)
        assert result is None

        cache.set_rewritten_query(
            query=query,
            rewritten_query="改写后的查询",
            sub_queries=["子查询1", "子查询2"],
            confidence=0.9,
            query_type="simple"
        )

        cached = cache.get_rewritten_query(query)
        assert cached is not None
        assert cached["rewritten_query"] == "改写后的查询"
        assert cached["sub_queries"] == ["子查询1", "子查询2"]

    def test_embedding_cache_memory(self, cache):
        """测试内存后端嵌入缓存"""
        text = "测试文本"
        embedding = [0.1, 0.2, 0.3]

        result = cache.get_embedding(text)
        assert result is None

        cache.set_embedding(text, embedding)

        cached = cache.get_embedding(text)
        assert cached is not None
        assert cached == embedding

    def test_results_cache_memory(self, cache):
        """测试内存后端检索结果缓存"""
        workspace_id = "ws_001"
        query = "测试查询"
        strategy = "hybrid"
        top_k = 5
        results = [{"id": "doc1"}]

        cache.set_results(
            workspace_id=workspace_id,
            query=query,
            results=results,
            strategy=strategy,
            top_k=top_k
        )

        cached = cache.get_results(workspace_id, query, strategy, top_k)
        assert cached is not None
        assert len(cached) == 1

    def test_cache_expiration(self, cache):
        """测试缓存过期"""
        query = "过期测试"

        cache.set_rewritten_query(
            query=query,
            rewritten_query="结果",
            sub_queries=[],
            confidence=1.0,
            query_type="simple"
        )

        result = cache.get_rewritten_query(query)
        assert result is not None

    def test_cache_stats(self, cache):
        """测试缓存统计"""
        query = "统计测试"

        cache.get_rewritten_query(query)
        assert cache._stats["query_rewrite"].misses == 1

        cache.set_rewritten_query(
            query=query,
            rewritten_query="结果",
            sub_queries=[],
            confidence=1.0,
            query_type="simple"
        )

        cache.get_rewritten_query(query)
        assert cache._stats["query_rewrite"].hits == 1

        stats = cache.get_stats()
        assert "query_rewrite" in stats

    def test_clear_cache(self, cache):
        """测试清除缓存"""
        cache.set_rewritten_query(
            query="查询1",
            rewritten_query="结果1",
            sub_queries=[],
            confidence=1.0,
            query_type="simple"
        )
        cache.set_rewritten_query(
            query="查询2",
            rewritten_query="结果2",
            sub_queries=[],
            confidence=1.0,
            query_type="simple"
        )

        assert len(cache._query_rewrite_cache) == 2

        cache.clear("query_rewrite")
        assert len(cache._query_rewrite_cache) == 0

    def test_clear_all_caches(self, cache):
        """测试清除所有缓存"""
        cache.set_rewritten_query(
            query="查询1",
            rewritten_query="结果1",
            sub_queries=[],
            confidence=1.0,
            query_type="simple"
        )
        cache.set_embedding("文本1", [0.1, 0.2])
        cache._result_cache["key"] = ([], time.time())

        assert len(cache._query_rewrite_cache) == 1
        assert len(cache._embedding_cache) == 1
        assert len(cache._result_cache) == 1

        cache.clear()

        assert len(cache._query_rewrite_cache) == 0
        assert len(cache._embedding_cache) == 0
        assert len(cache._result_cache) == 0


class TestRedisIntegration:
    """Redis 集成测试"""

    @pytest.mark.asyncio
    async def test_full_redis_workflow(self):
        """测试完整的 Redis 工作流程"""
        from src.services.cache.enhanced_cache import (
            EnhancedRetrievalCache,
            EnhancedCacheConfig,
            RedisCacheBackend
        )

        redis_service = MockRedisService()
        config = EnhancedCacheConfig()

        backend = RedisCacheBackend(config=config, redis_service=redis_service)

        query = "集成测试查询"
        rewritten = "集成测试改写结果"
        sub_queries = ["子查询1", "子查询2"]

        await backend.set_rewritten_query(
            query=query,
            rewritten_query=rewritten,
            sub_queries=sub_queries,
            confidence=0.95,
            query_type="hybrid"
        )

        result = await backend.get_rewritten_query(query)
        assert result is not None
        assert result["rewritten_query"] == rewritten

        embedding = [0.1 * i for i in range(1, 101)]
        await backend.set_embedding("嵌入文本", embedding)
        cached_emb = await backend.get_embedding("嵌入文本")
        assert cached_emb is not None
        assert len(cached_emb) == 100
        assert all(abs(a - b) < 1e-6 for a, b in zip(cached_emb, embedding))


class TestCacheKeyGeneration:
    """缓存键生成测试"""

    @pytest.fixture
    def cache(self):
        return EnhancedRetrievalCache()

    def test_key_format(self, cache):
        """测试键格式"""
        key = cache._generate_key("rewrite", "test query")
        assert key.startswith("rewrite:")
        assert len(key) > len("rewrite:")

    def test_same_query_same_key(self, cache):
        """测试相同查询生成相同键"""
        key1 = cache._generate_key("rewrite", "查询")
        key2 = cache._generate_key("rewrite", "查询")
        assert key1 == key2

    def test_different_query_different_key(self, cache):
        """测试不同查询生成不同键"""
        key1 = cache._generate_key("rewrite", "查询1")
        key2 = cache._generate_key("rewrite", "查询2")
        assert key1 != key2

    def test_different_type_different_key(self, cache):
        """测试不同类型生成不同键"""
        key1 = cache._generate_key("rewrite", "查询")
        key2 = cache._generate_key("embedding", "查询")
        assert key1 != key2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
