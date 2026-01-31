"""Unit tests for Retrieval Cache"""

import pytest
import time
from src.services.cache.retrieval_cache import (
    LRUCache,
    RetrievalCache,
    CacheConfig
)


class TestLRUCache:
    """Test LRUCache class"""
    
    def test_basic_get_set(self):
        """Test basic get/set operations"""
        cache = LRUCache(max_size=10, ttl=60)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
    
    def test_cache_miss(self):
        """Test cache miss returns None"""
        cache = LRUCache(max_size=10, ttl=60)
        
        assert cache.get("nonexistent") is None
    
    def test_ttl_expiration(self):
        """Test TTL expiration"""
        cache = LRUCache(max_size=10, ttl=1)  # 1 second TTL
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None
    
    def test_lru_eviction(self):
        """Test LRU eviction when at capacity"""
        cache = LRUCache(max_size=3, ttl=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add key4, should evict key2 (least recently used)
        cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"  # Still there
        assert cache.get("key4") == "value4"  # New entry
    
    def test_invalidate(self):
        """Test cache invalidation"""
        cache = LRUCache(max_size=10, ttl=60)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        assert cache.invalidate("key1") is True
        assert cache.get("key1") is None
        
        assert cache.invalidate("nonexistent") is False
    
    def test_clear(self):
        """Test cache clear"""
        cache = LRUCache(max_size=10, ttl=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_stats(self):
        """Test cache statistics"""
        cache = LRUCache(max_size=10, ttl=60)
        
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        
        assert stats['size'] == 1
        assert stats['max_size'] == 10
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5


class TestRetrievalCache:
    """Test RetrievalCache class"""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance"""
        config = CacheConfig(
            enabled=True,
            query_expansion_ttl=60,
            embedding_ttl=60,
            result_ttl=60
        )
        return RetrievalCache(config)
    
    def test_query_expansion_cache(self, cache):
        """Test query expansion caching"""
        query = "test query"
        expansions = ["query 1", "query 2", "query 3"]
        
        # Cache miss
        assert cache.get_expanded_query(query) is None
        
        # Set cache
        cache.set_expanded_query(query, expansions)
        
        # Cache hit
        cached = cache.get_expanded_query(query)
        assert cached == expansions
    
    def test_embedding_cache(self, cache):
        """Test embedding caching"""
        text = "test text"
        embedding = [0.1, 0.2, 0.3]
        
        # Cache miss
        assert cache.get_embedding(text) is None
        
        # Set cache
        cache.set_embedding(text, embedding)
        
        # Cache hit
        cached = cache.get_embedding(text)
        assert cached == embedding
    
    def test_embeddings_batch(self, cache):
        """Test batch embedding operations"""
        texts = ["text1", "text2", "text3"]
        embeddings = [[0.1], [0.2], [0.3]]
        
        # Cache some embeddings
        cache.set_embedding("text1", [0.1])
        cache.set_embedding("text3", [0.3])
        
        # Get batch (text2 should be a miss)
        cached_embeddings, miss_indices = cache.get_embeddings_batch(texts)
        
        assert cached_embeddings[0] == [0.1]
        assert cached_embeddings[1] is None
        assert cached_embeddings[2] == [0.3]
        assert miss_indices == [1]
        
        # Set batch
        cache.set_embeddings_batch(texts, embeddings)
        
        # All should be cached now
        cached_embeddings, miss_indices = cache.get_embeddings_batch(texts)
        assert all(e is not None for e in cached_embeddings)
        assert miss_indices == []
    
    def test_result_cache(self, cache):
        """Test result caching"""
        workspace_id = "workspace1"
        query = "test query"
        config_hash = "abc123"
        results = [{"id": "1"}, {"id": "2"}]
        
        # Cache miss
        assert cache.get_results(workspace_id, query, config_hash) is None
        
        # Set cache
        cache.set_results(workspace_id, query, config_hash, results)
        
        # Cache hit
        cached = cache.get_results(workspace_id, query, config_hash)
        assert cached == results
    
    def test_invalidate_workspace(self, cache):
        """Test workspace invalidation"""
        workspace_id = "workspace1"
        
        # Cache some results
        cache.set_results(workspace_id, "query1", "hash1", [{"id": "1"}])
        cache.set_results(workspace_id, "query2", "hash2", [{"id": "2"}])
        cache.set_results("workspace2", "query3", "hash3", [{"id": "3"}])
        
        # Invalidate workspace1
        count = cache.invalidate_workspace(workspace_id)
        
        assert count == 2
        assert cache.get_results(workspace_id, "query1", "hash1") is None
        assert cache.get_results(workspace_id, "query2", "hash2") is None
        assert cache.get_results("workspace2", "query3", "hash3") is not None
    
    def test_clear_all(self, cache):
        """Test clearing all caches"""
        cache.set_expanded_query("query", ["exp1"])
        cache.set_embedding("text", [0.1])
        cache.set_results("ws1", "q1", "h1", [{"id": "1"}])
        
        cache.clear_all()
        
        assert cache.get_expanded_query("query") is None
        assert cache.get_embedding("text") is None
        assert cache.get_results("ws1", "q1", "h1") is None
    
    def test_get_stats(self, cache):
        """Test cache statistics"""
        cache.set_expanded_query("query", ["exp1"])
        cache.get_expanded_query("query")  # Hit
        cache.get_expanded_query("other")  # Miss
        
        stats = cache.get_stats()
        
        assert stats['enabled'] is True
        assert 'query_expansion' in stats
        assert stats['query_expansion']['hits'] == 1
        assert stats['query_expansion']['misses'] == 1
    
    def test_disabled_cache(self):
        """Test that disabled cache doesn't cache"""
        config = CacheConfig(enabled=False)
        cache = RetrievalCache(config)
        
        cache.set_expanded_query("query", ["exp1"])
        assert cache.get_expanded_query("query") is None
    
    def test_config_hash_generation(self):
        """Test configuration hash generation"""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}  # Same but different order
        config3 = {"a": 1, "b": 3}  # Different value
        
        hash1 = RetrievalCache.generate_config_hash(config1)
        hash2 = RetrievalCache.generate_config_hash(config2)
        hash3 = RetrievalCache.generate_config_hash(config3)
        
        assert hash1 == hash2  # Order shouldn't matter
        assert hash1 != hash3  # Different values should produce different hashes
