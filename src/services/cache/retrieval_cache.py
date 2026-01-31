"""Retrieval Cache - Cache expensive operations for better performance

This module implements intelligent caching for:
1. Query expansions (LLM-generated)
2. Text embeddings (LLM-generated)
3. Retrieval results (full pipeline)
"""

import hashlib
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Cache configuration"""
    enabled: bool = True
    
    # Query expansion cache
    query_expansion_enabled: bool = True
    query_expansion_ttl: int = 3600  # 1 hour
    query_expansion_max_size: int = 1000
    
    # Embedding cache
    embedding_enabled: bool = True
    embedding_ttl: int = 86400  # 24 hours
    embedding_max_size: int = 10000
    
    # Result cache
    result_enabled: bool = True
    result_ttl: int = 300  # 5 minutes
    result_max_size: int = 500


class LRUCache:
    """
    LRU (Least Recently Used) Cache with TTL support
    
    Features:
    - Automatic eviction of least recently used items
    - Time-to-live (TTL) expiration
    - Thread-safe operations
    - Memory-efficient
    """
    
    def __init__(self, max_size: int, ttl: int):
        """
        Initialize LRU cache
        
        Args:
            max_size: Maximum number of items
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self.cache:
            self.misses += 1
            return None
        
        # Check TTL
        if self._is_expired(key):
            self._remove(key)
            self.misses += 1
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove if exists (to update timestamp)
        if key in self.cache:
            self._remove(key)
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        # Add new entry
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry
        
        Args:
            key: Cache key
            
        Returns:
            True if key was found and removed
        """
        if key in self.cache:
            self._remove(key)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        self.timestamps.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.timestamps:
            return True
        
        age = time.time() - self.timestamps[key]
        return age > self.ttl
    
    def _remove(self, key: str) -> None:
        """Remove entry from cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
    
    def _evict_oldest(self) -> None:
        """Evict least recently used entry"""
        if self.cache:
            oldest_key = next(iter(self.cache))
            self._remove(oldest_key)
            self.evictions += 1
            logger.debug(f"Evicted cache entry: {oldest_key}")


class RetrievalCache:
    """
    Retrieval cache manager
    
    Manages multiple caches for different retrieval operations:
    - Query expansion cache
    - Embedding cache
    - Result cache
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize retrieval cache
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        
        # Initialize caches
        self.query_expansion_cache = LRUCache(
            max_size=self.config.query_expansion_max_size,
            ttl=self.config.query_expansion_ttl
        ) if self.config.query_expansion_enabled else None
        
        self.embedding_cache = LRUCache(
            max_size=self.config.embedding_max_size,
            ttl=self.config.embedding_ttl
        ) if self.config.embedding_enabled else None
        
        self.result_cache = LRUCache(
            max_size=self.config.result_max_size,
            ttl=self.config.result_ttl
        ) if self.config.result_enabled else None
        
        logger.info("RetrievalCache initialized")
    
    # Query Expansion Cache
    
    def get_expanded_query(self, query: str) -> Optional[List[str]]:
        """
        Get cached query expansions
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries or None if not cached
        """
        if not self.config.enabled or not self.query_expansion_cache:
            return None
        
        cache_key = self._hash_text(query)
        result = self.query_expansion_cache.get(cache_key)
        
        if result:
            logger.debug(f"Query expansion cache hit: {query[:50]}...")
        
        return result
    
    def set_expanded_query(self, query: str, expansions: List[str]) -> None:
        """
        Cache query expansions
        
        Args:
            query: Original query
            expansions: List of expanded queries
        """
        if not self.config.enabled or not self.query_expansion_cache:
            return
        
        cache_key = self._hash_text(query)
        self.query_expansion_cache.set(cache_key, expansions)
        logger.debug(f"Cached query expansion: {query[:50]}...")
    
    # Embedding Cache
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached text embedding
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Embedding vector or None if not cached
        """
        if not self.config.enabled or not self.embedding_cache:
            return None
        
        cache_key = self._hash_text(text)
        result = self.embedding_cache.get(cache_key)
        
        if result:
            logger.debug(f"Embedding cache hit: {text[:50]}...")
        
        return result
    
    def set_embedding(self, text: str, embedding: List[float]) -> None:
        """
        Cache text embedding
        
        Args:
            text: Text
            embedding: Embedding vector
        """
        if not self.config.enabled or not self.embedding_cache:
            return
        
        cache_key = self._hash_text(text)
        self.embedding_cache.set(cache_key, embedding)
        logger.debug(f"Cached embedding: {text[:50]}...")
    
    def get_embeddings_batch(self, texts: List[str]) -> Tuple[List[Optional[List[float]]], List[int]]:
        """
        Get cached embeddings for multiple texts
        
        Args:
            texts: List of texts
            
        Returns:
            Tuple of (embeddings list with None for cache misses, indices of cache misses)
        """
        embeddings = []
        miss_indices = []
        
        for i, text in enumerate(texts):
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
            if embedding is None:
                miss_indices.append(i)
        
        return embeddings, miss_indices
    
    def set_embeddings_batch(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """
        Cache embeddings for multiple texts
        
        Args:
            texts: List of texts
            embeddings: List of embedding vectors
        """
        for text, embedding in zip(texts, embeddings):
            self.set_embedding(text, embedding)
    
    # Result Cache
    
    def get_results(self, workspace_id: str, query: str, config_hash: str) -> Optional[List[Any]]:
        """
        Get cached retrieval results
        
        Args:
            workspace_id: Workspace ID
            query: Query text
            config_hash: Hash of retrieval configuration
            
        Returns:
            Cached results or None if not cached
        """
        if not self.config.enabled or not self.result_cache:
            return None
        
        cache_key = self._generate_result_key(workspace_id, query, config_hash)
        result = self.result_cache.get(cache_key)
        
        if result:
            logger.debug(f"Result cache hit: {query[:50]}...")
        
        return result
    
    def set_results(
        self,
        workspace_id: str,
        query: str,
        config_hash: str,
        results: List[Any]
    ) -> None:
        """
        Cache retrieval results
        
        Args:
            workspace_id: Workspace ID
            query: Query text
            config_hash: Hash of retrieval configuration
            results: Retrieval results
        """
        if not self.config.enabled or not self.result_cache:
            return
        
        cache_key = self._generate_result_key(workspace_id, query, config_hash)
        self.result_cache.set(cache_key, results)
        logger.debug(f"Cached results: {query[:50]}...")
    
    # Cache Management
    
    def invalidate_workspace(self, workspace_id: str) -> int:
        """
        Invalidate all cache entries for a workspace
        
        Args:
            workspace_id: Workspace ID
            
        Returns:
            Number of entries invalidated
        """
        if not self.config.enabled or not self.result_cache:
            return 0
        
        # Find and remove all entries with this workspace_id
        count = 0
        keys_to_remove = []
        
        for key in self.result_cache.cache.keys():
            if key.startswith(f"{workspace_id}:"):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.result_cache.invalidate(key)
            count += 1
        
        logger.info(f"Invalidated {count} cache entries for workspace {workspace_id}")
        return count
    
    def clear_all(self) -> None:
        """Clear all caches"""
        if self.query_expansion_cache:
            self.query_expansion_cache.clear()
        if self.embedding_cache:
            self.embedding_cache.clear()
        if self.result_cache:
            self.result_cache.clear()
        
        logger.info("All caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all caches
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'enabled': self.config.enabled,
            'query_expansion': None,
            'embedding': None,
            'result': None
        }
        
        if self.query_expansion_cache:
            stats['query_expansion'] = self.query_expansion_cache.get_stats()
        
        if self.embedding_cache:
            stats['embedding'] = self.embedding_cache.get_stats()
        
        if self.result_cache:
            stats['result'] = self.result_cache.get_stats()
        
        return stats
    
    # Helper Methods
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _generate_result_key(self, workspace_id: str, query: str, config_hash: str) -> str:
        """Generate cache key for results"""
        query_hash = self._hash_text(query)
        return f"{workspace_id}:{query_hash}:{config_hash}"
    
    @staticmethod
    def generate_config_hash(config: Dict[str, Any]) -> str:
        """
        Generate hash for configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration hash
        """
        # Sort keys for consistent hashing
        config_str = str(sorted(config.items()))
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]
