"""
Task 持久化服务

本模块提供 Task 数据的数据库持久化功能，替代内存存储以支持服务重启后的数据保留。
使用 SQLAlchemy ORM 进行数据库操作。
"""

import logging
from typing import Optional, Dict, Any, List, AsyncGenerator
from datetime import datetime
from contextlib import asynccontextmanager

from sqlalchemy import Column, String, Text, DateTime, JSON, select, update, delete, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects.postgresql import insert

from ..domain.entities import Base, Task

logger = logging.getLogger(__name__)


class TaskRecord:
    """Task 记录类（内存表示）"""
    
    def __init__(
        self,
        task_id: str,
        status: str,
        topic: str,
        context: str = "",
        current_skill: str = "standard_tutorial",
        screenplay: Optional[str] = None,
        outline: Optional[List[Dict[str, Any]]] = None,
        skill_history: Optional[List[Dict[str, Any]]] = None,
        direction_changes: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        chat_session_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        self.task_id = task_id
        self.status = status
        self.topic = topic
        self.context = context
        self.current_skill = current_skill
        self.screenplay = screenplay
        self.outline = outline or []
        self.skill_history = skill_history or []
        self.direction_changes = direction_changes or []
        self.error = error
        self.request_data = request_data or {}
        self.chat_session_id = chat_session_id
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "topic": self.topic,
            "context": self.context,
            "current_skill": self.current_skill,
            "screenplay": self.screenplay,
            "outline": self.outline,
            "skill_history": self.skill_history,
            "direction_changes": self.direction_changes,
            "error": self.error,
            "request_data": self.request_data,
            "chat_session_id": self.chat_session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_entity(cls, entity: Task) -> "TaskRecord":
        """从实体创建"""
        return cls(
            task_id=entity.task_id,
            status=entity.status,
            topic=entity.topic,
            context=entity.context or "",
            current_skill=entity.current_skill or "standard_tutorial",
            screenplay=entity.screenplay,
            outline=entity.outline or [],
            skill_history=entity.skill_history or [],
            direction_changes=entity.direction_changes or [],
            error=entity.error,
            request_data=entity.request_data or {},
            created_at=entity.created_at,
            updated_at=entity.updated_at
        )
    
    def to_entity(self) -> Task:
        """转换为实体"""
        return Task(
            task_id=self.task_id,
            status=self.status,
            topic=self.topic,
            context=self.context,
            current_skill=self.current_skill,
            screenplay=self.screenplay,
            outline=self.outline,
            skill_history=self.skill_history,
            direction_changes=self.direction_changes,
            error=self.error,
            request_data=self.request_data,
            chat_session_id=self.chat_session_id
        )


class TaskDatabaseService:
    """Task 数据库服务（使用 SQLAlchemy ORM）"""
    
    _instance = None
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5433,
        database: str = "Screenplay",
        user: str = "postgres",
        password: str = "123456",
        echo: bool = False
    ):
        """
        初始化 Task 数据库服务
        
        Args:
            host: 数据库主机
            port: 数据库端口
            database: 数据库名称
            user: 用户名
            password: 密码
            echo: 是否打印 SQL 语句
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        
        database_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        
        self.engine = create_async_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            echo=echo,
            poolclass=NullPool if echo else None
        )
        
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info(f"TaskDatabaseService initialized for {host}:{port}/{database}")
    
    @classmethod
    async def create_instance(cls, host: str, port: int, database: str, user: str, password: str) -> "TaskDatabaseService":
        """异步创建实例"""
        instance = cls(host, port, database, user, password)
        await instance.create_tables()
        return instance
    
    @classmethod
    def create_from_env(cls) -> "TaskDatabaseService":
        """从环境变量创建服务"""
        from sqlalchemy import create_engine, text
        from ..config import get_database_config
        
        db_config = get_database_config()
        
        sync_engine = create_engine(db_config.url, echo=False)
        
        with sync_engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS screenplay.tasks (
                    task_id VARCHAR(36) PRIMARY KEY,
                    status VARCHAR(50) NOT NULL DEFAULT 'pending',
                    topic TEXT NOT NULL,
                    context TEXT DEFAULT '',
                    current_skill VARCHAR(100) DEFAULT 'standard_tutorial',
                    screenplay TEXT,
                    outline JSON DEFAULT '[]',
                    skill_history JSON DEFAULT '[]',
                    direction_changes JSON DEFAULT '[]',
                    error TEXT,
                    request_data JSON DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))
            conn.commit()
        
        sync_engine.dispose()
        
        return cls(
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            user=db_config.user,
            password=db_config.password,
            echo=db_config.echo
        )
    
    async def create_tables(self):
        """创建 Task 表"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Task tables created")
    
    async def drop_tables(self):
        """删除 Task 表"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Task tables dropped")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库会话"""
        async with self.async_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def create(self, task: TaskRecord) -> TaskRecord:
        """创建 Task 记录"""
        async with self.get_session() as session:
            entity = task.to_entity()
            session.add(entity)
            await session.commit()
        
        logger.info(f"Task created: {task.task_id}")
        return task
    
    async def get(self, task_id: str) -> Optional[TaskRecord]:
        """获取 Task 记录"""
        async with self.get_session() as session:
            result = await session.execute(
                select(Task).where(Task.task_id == task_id)
            )
            entity = result.scalar_one_or_none()
            
            if not entity:
                return None
            
            return TaskRecord.from_entity(entity)
    
    async def update(self, task_id: str, **kwargs) -> Optional[TaskRecord]:
        """更新 Task 记录"""
        update_data = {k: v for k, v in kwargs.items() if v is not None}
        update_data['updated_at'] = datetime.now()
        
        async with self.get_session() as session:
            await session.execute(
                update(Task)
                .where(Task.task_id == task_id)
                .values(**update_data)
            )
            await session.commit()
        
        logger.info(f"Task updated: {task_id}")
        return await self.get(task_id)
    
    async def delete(self, task_id: str) -> bool:
        """删除 Task 记录"""
        async with self.get_session() as session:
            result = await session.execute(
                delete(Task).where(Task.task_id == task_id)
            )
            await session.commit()
        
        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"Task deleted: {task_id}")
        return deleted
    
    async def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[TaskRecord]:
        """列出 Task 记录"""
        async with self.get_session() as session:
            query = select(Task).order_by(Task.created_at.desc()).limit(limit).offset(offset)
            
            if status:
                query = query.where(Task.status == status)
            
            result = await session.execute(query)
            entities = result.scalars().all()
            
            return [TaskRecord.from_entity(entity) for entity in entities]
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with self.get_session() as session:
                await session.execute(select(func.count()).select_from(Task))
            return True
        except Exception as e:
            logger.error(f"Task database health check failed: {e}")
            return False
    
    async def close(self):
        """关闭连接"""
        await self.engine.dispose()
        logger.info("Task database connections closed")


class TaskService:
    """Task 服务（内存 + 数据库双写）"""
    
    def __init__(self, db_service: TaskDatabaseService, enable_cache: bool = True):
        """
        初始化 Task 服务
        
        Args:
            db_service: 数据库服务
            enable_cache: 是否启用内存缓存（提高性能）
        """
        self.db_service = db_service
        self.enable_cache = enable_cache
        self._cache: Dict[str, TaskRecord] = {}
    
    async def create(self, task: TaskRecord) -> TaskRecord:
        """创建 Task（双写缓存和数据库）"""
        self._cache[task.task_id] = task
        await self.db_service.create(task)
        return task
    
    async def get(self, task_id: str) -> Optional[TaskRecord]:
        """获取 Task（先查缓存，再查数据库）"""
        if task_id in self._cache:
            return self._cache[task_id]
        
        task = await self.db_service.get(task_id)
        if task and self.enable_cache:
            self._cache[task_id] = task
        return task
    
    async def update(self, task_id: str, **kwargs) -> Optional[TaskRecord]:
        """更新 Task（双写）"""
        if task_id in self._cache:
            cached_task = self._cache[task_id]
            for key, value in kwargs.items():
                if hasattr(cached_task, key):
                    setattr(cached_task, key, value)
            cached_task.updated_at = datetime.now()
        
        task = await self.db_service.update(task_id, **kwargs)
        if task and self.enable_cache:
            self._cache[task_id] = task
        return task
    
    async def delete(self, task_id: str) -> bool:
        """删除 Task（双删）"""
        self._cache.pop(task_id, None)
        return await self.db_service.delete(task_id)
    
    async def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[TaskRecord]:
        """列出 Task"""
        return await self.db_service.list_tasks(status, limit, offset)
    
    async def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("Task cache cleared")
