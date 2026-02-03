"""Document Repository - 文件管理 Repository（幂等控制）"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import uuid

from src.config import get_database_config
from src.services.database.postgres import PostgresService


logger = logging.getLogger(__name__)


class FileStatus(Enum):
    """文件状态枚举"""
    UPLOADING = "UPLOADING"
    INDEXING = "INDEXING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


@dataclass
class DocumentFile:
    """文件记录实体"""
    id: str
    file_name: str
    file_hash: str
    status: FileStatus
    chunk_count: int = 0
    error_msg: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DocumentRepository:
    """文件管理 Repository - 幂等控制"""

    def __init__(self, postgres_service: PostgresService = None):
        """
        初始化 Document Repository

        Args:
            postgres_service: PostgreSQL 服务实例，如果为 None 则自动创建
        """
        if postgres_service:
            self.pg = postgres_service
        else:
            config = get_database_config()
            self.pg = PostgresService({
                'host': config.host,
                'port': config.port,
                'database': config.database,
                'user': config.user,
                'password': config.password
            })
        self._initialized = False

    async def initialize(self) -> None:
        """初始化表结构"""
        if self._initialized:
            return

        await self.pg.connect()

        await self.pg.execute("""
            CREATE TABLE IF NOT EXISTS document_files (
                id VARCHAR(36) PRIMARY KEY,
                file_name VARCHAR(500) NOT NULL,
                file_hash VARCHAR(64) NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'UPLOADING',
                chunk_count INT DEFAULT 0,
                error_msg TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await self.pg.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_hash_status
            ON document_files(file_hash, status)
        """)

        await self.pg.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_file_hash_unique
            ON document_files(file_hash)
        """)

        self._initialized = True
        logger.info("DocumentRepository initialized")

    async def close(self) -> None:
        """关闭数据库连接"""
        if self._initialized:
            await self.pg.disconnect()
            self._initialized = False

    def get_by_hash(self, file_hash: str) -> Optional[DocumentFile]:
        """根据 Hash 查重 - 秒传机制"""
        import asyncio

        try:
            row = asyncio.get_event_loop().run_until_complete(
                self.pg.fetchrow(
                    """
                    SELECT id, file_name, file_hash, status, chunk_count,
                           error_msg, created_at, updated_at
                    FROM document_files
                    WHERE file_hash = $1
                    """,
                    file_hash
                )
            )

            if row:
                return DocumentFile(
                    id=row['id'],
                    file_name=row['file_name'],
                    file_hash=row['file_hash'],
                    status=FileStatus(row['status']),
                    chunk_count=row['chunk_count'],
                    error_msg=row['error_msg'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            return None
        except Exception as e:
            logger.error(f"查询文件失败: {e}")
            return None

    def create(self, file: DocumentFile) -> DocumentFile:
        """创建文件记录"""
        import asyncio

        now = datetime.utcnow()

        asyncio.get_event_loop().run_until_complete(
            self.pg.execute(
                """
                INSERT INTO document_files
                (id, file_name, file_hash, status, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                file.id,
                file.file_name,
                file.file_hash,
                file.status.value,
                now,
                now
            )
        )

        file.created_at = now
        file.updated_at = now
        return file

    def update_status(
        self,
        file_id: str,
        status: FileStatus,
        chunk_count: int = None,
        error_msg: str = None
    ) -> None:
        """更新状态 - 幂等更新"""
        import asyncio

        now = datetime.utcnow()

        if chunk_count is not None:
            asyncio.get_event_loop().run_until_complete(
                self.pg.execute(
                    """
                    UPDATE document_files
                    SET status = $1, chunk_count = $2, updated_at = $3
                    WHERE id = $4
                    """,
                    status.value,
                    chunk_count,
                    now,
                    file_id
                )
            )
        elif error_msg is not None:
            asyncio.get_event_loop().run_until_complete(
                self.pg.execute(
                    """
                    UPDATE document_files
                    SET status = $1, error_msg = $2, updated_at = $3
                    WHERE id = $4
                    """,
                    status.value,
                    error_msg,
                    now,
                    file_id
                )
            )
        else:
            asyncio.get_event_loop().run_until_complete(
                self.pg.execute(
                    """
                    UPDATE document_files
                    SET status = $1, updated_at = $2
                    WHERE id = $3
                    """,
                    status.value,
                    now,
                    file_id
                )
            )

    def get_by_id(self, file_id: str) -> Optional[DocumentFile]:
        """根据 ID 查询文件"""
        import asyncio

        row = asyncio.get_event_loop().run_until_complete(
            self.pg.fetchrow(
                """
                SELECT id, file_name, file_hash, status, chunk_count,
                       error_msg, created_at, updated_at
                FROM document_files
                WHERE id = $1
                """,
                file_id
            )
        )

        if row:
            return DocumentFile(
                id=row['id'],
                file_name=row['file_name'],
                file_hash=row['file_hash'],
                status=FileStatus(row['status']),
                chunk_count=row['chunk_count'],
                error_msg=row['error_msg'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
        return None

    def get_all(self, limit: int = 100, offset: int = 0) -> List[DocumentFile]:
        """获取所有文件记录"""
        import asyncio

        rows = asyncio.get_event_loop().run_until_complete(
            self.pg.fetch(
                """
                SELECT id, file_name, file_hash, status, chunk_count,
                       error_msg, created_at, updated_at
                FROM document_files
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset
            )
        )

        return [
            DocumentFile(
                id=row['id'],
                file_name=row['file_name'],
                file_hash=row['file_hash'],
                status=FileStatus(row['status']),
                chunk_count=row['chunk_count'],
                error_msg=row['error_msg'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            for row in rows
        ]

    def delete(self, file_id: str) -> bool:
        """删除文件记录"""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            self.pg.execute(
                "DELETE FROM document_files WHERE id = $1",
                file_id
            )
        )

        return result.split()[-1] == '1'

    def create_new(self, file_name: str, file_hash: str) -> DocumentFile:
        """创建新文件记录（快捷方法）"""
        file = DocumentFile(
            id=str(uuid.uuid4()),
            file_name=file_name,
            file_hash=file_hash,
            status=FileStatus.UPLOADING
        )
        return self.create(file)
