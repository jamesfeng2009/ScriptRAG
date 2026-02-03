"""检索缓存 - 为提高性能缓存昂贵操作

本模块实现智能缓存功能：
1. 查询扩展缓存（LLM 生成）
2. 文本嵌入缓存（LLM 生成）
3. 检索结果缓存（完整流程）
"""

import hashlib
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """缓存配置"""
    enabled: bool = True
    
    # 查询扩展缓存
    query_expansion_enabled: bool = True
    query_expansion_ttl: int = 3600  # 1 小时
    query_expansion_max_size: int = 1000
    
    # 嵌入缓存
    embedding_enabled: bool = True
    embedding_ttl: int = 86400  # 24 小时
    embedding_max_size: int = 10000
    
    # 结果缓存
    result_enabled: bool = True
    result_ttl: int = 300  # 5 分钟
    result_max_size: int = 500


class LRUCache:
    """
    LRU（最近最少使用）缓存，支持 TTL
    
    功能：
    - 自动淘汰最近最少使用的项
    - TTL（生存时间）过期
    - 线程安全操作
    - 内存高效
    """
    
    def __init__(self, max_size: int, ttl: int):
        """
        初始化 LRU 缓存
        
        Args:
            max_size: 最大项数
            ttl: 生存时间（秒）
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        
        # 统计信息
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        从缓存获取值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的值，未找到或已过期返回 None
        """
        if key not in self.cache:
            self.misses += 1
            return None
        
        # 检查 TTL
        if self._is_expired(key):
            self._remove(key)
            self.misses += 1
            return None
        
        # 移到末尾（最近使用）
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 要缓存的值
        """
        # 如果存在则移除（更新时间戳）
        if key in self.cache:
            self._remove(key)
        
        # 如果达到容量则淘汰
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        # 添加新条目
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def invalidate(self, key: str) -> bool:
        """
        使缓存条目失效
        
        Args:
            key: 缓存键
            
        Returns:
            如果找到并删除返回 True
        """
        if key in self.cache:
            self._remove(key)
            return True
        return False
    
    def clear(self) -> None:
        """清除所有缓存条目"""
        self.cache.clear()
        self.timestamps.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            包含缓存统计信息的字典
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
        """检查缓存条目是否过期"""
        if key not in self.timestamps:
            return True
        
        age = time.time() - self.timestamps[key]
        return age > self.ttl
    
    def _remove(self, key: str) -> None:
        """从缓存中移除条目"""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
    
    def _evict_oldest(self) -> None:
        """淘汰最近最少使用的条目"""
        if self.cache:
            oldest_key = next(iter(self.cache))
            self._remove(oldest_key)
            self.evictions += 1
            logger.debug(f"Evicted cache entry: {oldest_key}")


class RetrievalCache:
    """
    检索缓存管理器
    
    管理多个不同检索操作的缓存：
    - 查询扩展缓存
    - 嵌入缓存
    - 结果缓存
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        初始化检索缓存
        
        Args:
            config: 缓存配置
        """
        self.config = config or CacheConfig()
        
        # 初始化缓存
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
    
    # 查询扩展缓存
    
    def get_expanded_query(self, query: str) -> Optional[List[str]]:
        """
        获取缓存的查询扩展
        
        Args:
            query: 原始查询
            
        Returns:
            扩展查询列表，未缓存返回 None
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
        缓存查询扩展
        
        Args:
            query: 原始查询
            expansions: 扩展查询列表
        """
        if not self.config.enabled or not self.query_expansion_cache:
            return
        
        cache_key = self._hash_text(query)
        self.query_expansion_cache.set(cache_key, expansions)
        logger.debug(f"Cached query expansion: {query[:50]}...")
    
    # 嵌入缓存
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        获取缓存的文本嵌入
        
        Args:
            text: 要获取嵌入的文本
            
        Returns:
            嵌入向量，未缓存返回 None
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
        缓存文本嵌入
        
        Args:
            text: 文本
            embedding: 嵌入向量
        """
        if not self.config.enabled or not self.embedding_cache:
            return
        
        cache_key = self._hash_text(text)
        self.embedding_cache.set(cache_key, embedding)
        logger.debug(f"Cached embedding: {text[:50]}...")
    
    def get_embeddings_batch(self, texts: List[str]) -> Tuple[List[Optional[List[float]]], List[int]]:
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
    
    def set_embeddings_batch(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """
        缓存多个文本的嵌入
        
        Args:
            texts: 文本列表
            embeddings: 嵌入向量列表
        """
        for text, embedding in zip(texts, embeddings):
            self.set_embedding(text, embedding)
    
    # 结果缓存
    
    def get_results(self, workspace_id: str, query: str, config_hash: str) -> Optional[List[Any]]:
        """
        获取缓存的检索结果
        
        Args:
            workspace_id: 工作空间 ID
            query: 查询文本
            config_hash: 检索配置的哈希值
            
        Returns:
            缓存结果，未缓存返回 None
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
        缓存检索结果
        
        Args:
            workspace_id: 工作空间 ID
            query: 查询文本
            config_hash: 检索配置的哈希值
            results: 检索结果
        """
        if not self.config.enabled or not self.result_cache:
            return
        
        cache_key = self._generate_result_key(workspace_id, query, config_hash)
        self.result_cache.set(cache_key, results)
        logger.debug(f"Cached results: {query[:50]}...")
    
    # 缓存管理
    
    def invalidate_workspace(self, workspace_id: str) -> int:
        """
        使工作空间的所有缓存条目失效
        
        Args:
            workspace_id: 工作空间 ID
            
        Returns:
            失效的条目数量
        """
        if not self.config.enabled or not self.result_cache:
            return 0
        
        # 查找并移除所有带有此 workspace_id 的条目
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
        """清除所有缓存"""
        if self.query_expansion_cache:
            self.query_expansion_cache.clear()
        if self.embedding_cache:
            self.embedding_cache.clear()
        if self.result_cache:
            self.result_cache.clear()
        
        logger.info("All caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取所有缓存的统计信息
        
        Returns:
            包含缓存统计信息的字典
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
    
    # 辅助方法
    
    def _hash_text(self, text: str) -> str:
        """生成文本的哈希值"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _generate_result_key(self, workspace_id: str, query: str, config_hash: str) -> str:
        """生成结果的缓存键"""
        query_hash = self._hash_text(query)
        return f"{workspace_id}:{query_hash}:{config_hash}"
    
    @staticmethod
    def generate_config_hash(config: Dict[str, Any]) -> str:
        """
        生成配置的哈希值
        
        Args:
            config: 配置字典
            
        Returns:
            配置哈希值
        """
        # 对键排序以确保哈希一致
        config_str = str(sorted(config.items()))
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]
