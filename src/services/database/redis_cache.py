"""Redis Cache Service - Caching layer implementation

This module provides Redis caching functionality:
1. LLM response caching (24 hour TTL)
2. Embedding vector caching (7 day TTL)
3. Retrieval result caching (1 hour TTL)
4. Session state caching (30 minute TTL)
"""

import logging
import json
import hashlib
from typing import Optional, Any, Dict, List
from datetime import timedelta
import redis.asyncio as redis


logger = logging.getLogger(__name__)


# Cache key prefixes
CACHE_PREFIX = {
    "llm_response": "llm",
    "embedding": "emb",
    "retrieval": "ret",
    "session": "sess",
    "quota": "quota",
    "config": "cfg"
}

# Cache TTL in seconds
CACHE_TTL = {
    "llm_response": 86400,      # 24 hours
    "embedding": 604800,         # 7 days
    "retrieval": 3600,           # 1 hour
    "session": 1800,             # 30 minutes
    "quota": 300,                # 5 minutes
    "config": 3600               # 1 hour
}


class RedisCacheService:
    """Service for Redis caching operations"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50
    ):
        """
        Initialize Redis cache service
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            max_connections: Maximum connection pool size
        """
        self.redis_url = f"redis://{host}:{port}/{db}"
        if password:
            self.redis_url = f"redis://:{password}@{host}:{port}/{db}"
        
        self.pool = redis.ConnectionPool.from_url(
            self.redis_url,
            max_connections=max_connections,
            decode_responses=True
        )
        self.client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Establish Redis connection"""
        try:
            self.client = redis.Redis(connection_pool=self.pool)
            await self.client.ping()
            logger.info("Redis cache service connected")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            await self.pool.disconnect()
            logger.info("Redis cache service disconnected")
    
    def _generate_key(self, prefix: str, *args) -> str:
        """
        Generate cache key
        
        Args:
            prefix: Key prefix
            *args: Key components
            
        Returns:
            Cache key string
        """
        key_parts = [prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)
    
    def _hash_content(self, content: str) -> str:
        """
        Generate hash for content
        
        Args:
            content: Content to hash
            
        Returns:
            Hash string
        """
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
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
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successful, False otherwise
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
        Delete value from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
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
        Check if key exists in cache
        
        Args:
            key: Cache key
            
        Returns:
            True if exists, False otherwise
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
        Clear all keys matching pattern
        
        Args:
            pattern: Key pattern (e.g., "llm:*")
            
        Returns:
            Number of keys deleted
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
    
    # LLM Response Caching
    
    async def cache_llm_response(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        response: str,
        temperature: float = 0.7
    ) -> bool:
        """
        Cache LLM response
        
        Args:
            provider: LLM provider
            model: Model name
            messages: Input messages
            response: LLM response
            temperature: Temperature parameter
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate hash from messages
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
        Get cached LLM response
        
        Args:
            provider: LLM provider
            model: Model name
            messages: Input messages
            temperature: Temperature parameter
            
        Returns:
            Cached response or None if not found
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
    
    # Embedding Caching
    
    async def cache_embedding(
        self,
        provider: str,
        model: str,
        text: str,
        embedding: List[float]
    ) -> bool:
        """
        Cache embedding vector
        
        Args:
            provider: Embedding provider
            model: Model name
            text: Input text
            embedding: Embedding vector
            
        Returns:
            True if successful, False otherwise
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
        Get cached embedding vector
        
        Args:
            provider: Embedding provider
            model: Model name
            text: Input text
            
        Returns:
            Cached embedding or None if not found
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
    
    # Retrieval Result Caching
    
    async def cache_retrieval_result(
        self,
        workspace_id: str,
        query: str,
        results: List[Dict[str, Any]]
    ) -> bool:
        """
        Cache retrieval results
        
        Args:
            workspace_id: Workspace ID
            query: Search query
            results: Retrieval results
            
        Returns:
            True if successful, False otherwise
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
        Get cached retrieval results
        
        Args:
            workspace_id: Workspace ID
            query: Search query
            
        Returns:
            Cached results or None if not found
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
    
    # Session State Caching
    
    async def cache_session_state(
        self,
        session_id: str,
        state: Dict[str, Any]
    ) -> bool:
        """
        Cache session state
        
        Args:
            session_id: Session ID
            state: Session state
            
        Returns:
            True if successful, False otherwise
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
        Get cached session state
        
        Args:
            session_id: Session ID
            
        Returns:
            Cached state or None if not found
        """
        try:
            key = self._generate_key(CACHE_PREFIX["session"], session_id)
            return await self.get(key)
        except Exception as e:
            logger.error(f"Failed to get cached session state: {str(e)}")
            return None
    
    async def delete_session_state(self, session_id: str) -> bool:
        """
        Delete cached session state
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._generate_key(CACHE_PREFIX["session"], session_id)
            return await self.delete(key)
        except Exception as e:
            logger.error(f"Failed to delete session state: {str(e)}")
            return False
    
    # Statistics
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
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
        """Calculate cache hit rate"""
        total = hits + misses
        if total == 0:
            return 0.0
        return hits / total
    
    async def clear_all(self) -> bool:
        """
        Clear all cache (use with caution)
        
        Returns:
            True if successful, False otherwise
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
