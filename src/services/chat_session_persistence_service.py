"""Chat Session 持久化服务"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncpg

logger = logging.getLogger(__name__)


class ChatSessionRecord:
    """Chat Session 记录类"""
    
    def __init__(
        self,
        id: str,
        topic: str = "",
        mode: str = "agent",
        config: Optional[Dict[str, Any]] = None,
        message_history: Optional[List[Dict[str, Any]]] = None,
        related_task_id: Optional[str] = None,
        status: str = "active",
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        self.id = id
        self.topic = topic
        self.mode = mode
        self.config = config or {}
        self.message_history = message_history or []
        self.related_task_id = related_task_id
        self.status = status
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "topic": self.topic,
            "mode": self.mode,
            "config": self.config,
            "message_history": self.message_history,
            "related_task_id": self.related_task_id,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class ChatSessionPersistenceService:
    """Chat Session 数据库服务"""
    
    _instance = None
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5433,
        database: str = "Screenplay",
        user: str = "postgres",
        password: str = "123456"
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._pool: Optional[asyncpg.Pool] = None
    
    @classmethod
    def get_instance(cls) -> "ChatSessionPersistenceService":
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def connect(self):
        """建立数据库连接"""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=2,
                max_size=10
            )
            logger.info(f"ChatSessionPersistenceService connected to {self.host}:{self.port}/{self.database}")
    
    async def close(self):
        """关闭数据库连接"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("ChatSessionPersistenceService disconnected")
    
    async def create(self, record: ChatSessionRecord) -> ChatSessionRecord:
        """创建 Chat Session"""
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO screenplay.chat_sessions (id, topic, mode, config, message_history, related_task_id, status, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                record.id,
                record.topic,
                record.mode,
                json.dumps(record.config) if record.config else '{}',
                json.dumps(record.message_history) if record.message_history else '[]',
                record.related_task_id,
                record.status,
                record.created_at or datetime.now(),
                record.updated_at or datetime.now()
            )
        
        logger.info(f"ChatSession created: {record.id}")
        return record
    
    async def get(self, session_id: str) -> Optional[ChatSessionRecord]:
        """获取 Chat Session"""
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM screenplay.chat_sessions WHERE id = $1",
                session_id
            )
        
        if row is None:
            return None
        
        return ChatSessionRecord(
            id=row["id"],
            topic=row["topic"] or "",
            mode=row["mode"],
            config=row["config"] if isinstance(row["config"], dict) else json.loads(row["config"]) if row["config"] else {},
            message_history=row["message_history"] if isinstance(row["message_history"], list) else json.loads(row["message_history"]) if row["message_history"] else [],
            related_task_id=row["related_task_id"],
            status=row["status"] or "active",
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )
    
    async def update_message_history(
        self,
        session_id: str,
        message_history: List[Dict[str, Any]]
    ) -> bool:
        """更新消息历史"""
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE screenplay.chat_sessions
                SET message_history = $1, updated_at = $2
                WHERE id = $3
                """,
                json.dumps(message_history),
                datetime.now(),
                session_id
            )
        
        logger.debug(f"ChatSession message history updated: {session_id}")
        return True
    
    async def link_task(self, session_id: str, task_id: str) -> bool:
        """关联 Task"""
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE screenplay.chat_sessions
                SET related_task_id = $1, updated_at = $2
                WHERE id = $3
                """,
                task_id,
                datetime.now(),
                session_id
            )
        
        logger.info(f"ChatSession {session_id} linked to Task {task_id}")
        return True
    
    async def update_config(self, session_id: str, config: Dict[str, Any]) -> bool:
        """更新会话配置"""
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE screenplay.chat_sessions
                SET config = $1, updated_at = $2
                WHERE id = $3
                """,
                json.dumps(config),
                datetime.now(),
                session_id
            )
        
        return True
    
    async def close_session(self, session_id: str) -> bool:
        """关闭会话"""
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE screenplay.chat_sessions
                SET status = 'closed', updated_at = $1
                WHERE id = $2
                """,
                datetime.now(),
                session_id
            )
        
        logger.info(f"ChatSession closed: {session_id}")
        return True
    
    async def delete(self, session_id: str) -> bool:
        """删除会话"""
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM screenplay.chat_sessions WHERE id = $1",
                session_id
            )
        
        logger.info(f"ChatSession deleted: {session_id}")
        return True
    
    async def get_by_task(self, task_id: str) -> Optional[ChatSessionRecord]:
        """通过 Task ID 获取关联的 Chat Session"""
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM screenplay.chat_sessions WHERE related_task_id = $1",
                task_id
            )
        
        if row is None:
            return None
        
        return ChatSessionRecord(
            id=row["id"],
            topic=row["topic"] or "",
            mode=row["mode"],
            config=row["config"] if isinstance(row["config"], dict) else json.loads(row["config"]) if row["config"] else {},
            message_history=row["message_history"] if isinstance(row["message_history"], list) else json.loads(row["message_history"]) if row["message_history"] else [],
            related_task_id=row["related_task_id"],
            status=row["status"] or "active",
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )
