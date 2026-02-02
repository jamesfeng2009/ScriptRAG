"""
ORM 数据库服务

本模块提供基于 SQLAlchemy 的数据库服务，整合实体类和仓储模式。
"""

import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from ...domain.entities import Base
from ...domain.repositories import (
    ScreenplaySessionRepository, CodeDocumentRepository, ExecutionLogRepository,
    LLMCallLogRepository, WorkspaceRepository, UserRepository
)

logger = logging.getLogger(__name__)


class ORMDatabaseService:
    """ORM 数据库服务"""
    
    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        echo: bool = False
    ):
        """
        初始化 ORM 数据库服务
        
        Args:
            host: 数据库主机
            port: 数据库端口
            database: 数据库名称
            user: 用户名
            password: 密码
            pool_size: 连接池大小
            max_overflow: 连接池最大溢出
            echo: 是否打印 SQL 语句
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        
        # 构建数据库 URL
        self.database_url = (
            f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        )
        
        # 创建异步引擎
        self.engine = create_async_engine(
            self.database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            echo=echo,
            poolclass=NullPool if echo else None  # 开发模式使用 NullPool
        )
        
        # 创建会话工厂
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info(f"ORMDatabaseService initialized for {host}:{port}/{database}")
    
    async def create_tables(self):
        """创建所有表（仅用于测试，生产环境使用 Alembic）"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("All tables created")
    
    async def drop_tables(self):
        """删除所有表（仅用于测试）"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("All tables dropped")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库会话（上下文管理器）"""
        async with self.async_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def get_screenplay_session_repository(
        self, 
        session: AsyncSession
    ) -> ScreenplaySessionRepository:
        """获取剧本会话仓储"""
        return ScreenplaySessionRepository(session)
    
    async def get_code_document_repository(
        self, 
        session: AsyncSession
    ) -> CodeDocumentRepository:
        """获取代码文档仓储"""
        return CodeDocumentRepository(session)
    
    async def get_execution_log_repository(
        self, 
        session: AsyncSession
    ) -> ExecutionLogRepository:
        """获取执行日志仓储"""
        return ExecutionLogRepository(session)
    
    async def get_llm_call_log_repository(
        self, 
        session: AsyncSession
    ) -> LLMCallLogRepository:
        """获取 LLM 调用日志仓储"""
        return LLMCallLogRepository(session)
    
    async def get_workspace_repository(
        self, 
        session: AsyncSession
    ) -> WorkspaceRepository:
        """获取工作空间仓储"""
        return WorkspaceRepository(session)
    
    async def get_user_repository(
        self, 
        session: AsyncSession
    ) -> UserRepository:
        """获取用户仓储"""
        return UserRepository(session)
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            from sqlalchemy import text
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def close(self):
        """关闭数据库连接"""
        await self.engine.dispose()
        logger.info("Database connections closed")


class DatabaseServiceFactory:
    """数据库服务工厂"""
    
    @staticmethod
    def create_orm_service(
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        **kwargs
    ) -> ORMDatabaseService:
        """创建 ORM 数据库服务"""
        return ORMDatabaseService(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            **kwargs
        )
    
    @staticmethod
    def create_from_env() -> ORMDatabaseService:
        """从环境变量创建 ORM 数据库服务"""
        import os
        
        return ORMDatabaseService(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', 5433)),
            database=os.getenv('POSTGRES_DB', 'Screenplay'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', '123456'),
            echo=os.getenv('DATABASE_ECHO', 'false').lower() == 'true'
        )