"""PostgreSQL Service - Relational database operations"""

import logging
from typing import Optional, Dict, Any
import asyncpg


logger = logging.getLogger(__name__)


class PostgresService:
    """
    PostgreSQL 数据库服务
    
    职责：
    1. 管理数据库连接
    2. 提供基本的数据库操作接口
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 PostgreSQL 服务
        
        Args:
            config: 数据库配置字典
        """
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        logger.info("PostgresService initialized")
    
    async def connect(self):
        """建立数据库连接池"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                min_size=2,
                max_size=10
            )
            logger.info("PostgreSQL connection pool created")
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL connection pool: {e}")
            raise
    
    async def disconnect(self):
        """关闭数据库连接池"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
    
    async def execute(self, query: str, *args):
        """
        执行 SQL 查询
        
        Args:
            query: SQL 查询语句
            *args: 查询参数
            
        Returns:
            查询结果
        """
        if not self.pool:
            raise RuntimeError("Database connection pool not initialized")
        
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args):
        """
        获取查询结果
        
        Args:
            query: SQL 查询语句
            *args: 查询参数
            
        Returns:
            查询结果列表
        """
        if not self.pool:
            raise RuntimeError("Database connection pool not initialized")
        
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args):
        """
        获取单行查询结果
        
        Args:
            query: SQL 查询语句
            *args: 查询参数
            
        Returns:
            单行查询结果
        """
        if not self.pool:
            raise RuntimeError("Database connection pool not initialized")
        
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

