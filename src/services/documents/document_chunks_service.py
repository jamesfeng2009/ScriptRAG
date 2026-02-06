"""Document Chunks Service - 文档分块管理服务"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.utils.database_utils import build_db_url

logger = logging.getLogger(__name__)


class DocumentChunksService:
    """文档分块服务"""
    
    def __init__(self, config=None):
        self._config = config
        self._engine = None
        self._session_factory = None
        self._initialized = False
    
    def _get_db_url(self) -> str:
        """获取数据库连接 URL"""
        if self._config is not None:
            db_config = self._config
        else:
            db_config = None
        return build_db_url(db_config)
    
    async def initialize(self):
        try:
            self._engine = create_async_engine(
                self._get_db_url(),
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            
            self._session_factory = sessionmaker(
                self._engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            self._initialized = True
            logger.info("DocumentChunksService initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DocumentChunksService: {e}")
            raise
    
    async def close(self):
        if self._engine:
            await self._engine.dispose()
            self._initialized = False
            logger.info("DocumentChunksService closed")
    
    async def health_check(self) -> bool:
        if not self._initialized:
            return False
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    
    async def create_chunk(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        char_count: Optional[int] = None,
        embedding: Optional[List[float]] = None
    ) -> str:
        """创建文档分块"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        chunk_id = str(uuid4())
        embedding_str = f"[{','.join(map(str, embedding))}]" if embedding else None
        char_count = char_count or len(content)
        
        async with self._session_factory() as session:
            query = text("""
                INSERT INTO screenplay.document_chunks (
                    id, document_id, chunk_index, content, embedding,
                    start_line, end_line, char_count
                ) VALUES (
                    :id, :document_id, :chunk_index, :content, :embedding,
                    :start_line, :end_line, :char_count
                )
                RETURNING id
            """)
            
            result = await session.execute(query, {
                'id': chunk_id,
                'document_id': document_id,
                'chunk_index': chunk_index,
                'content': content,
                'embedding': embedding_str,
                'start_line': start_line,
                'end_line': end_line,
                'char_count': char_count
            })
            await session.commit()
            
            return str(result.scalar())
    
    async def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """获取分块"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            query = text("""
                SELECT id, document_id, chunk_index, content, embedding,
                       start_line, end_line, char_count, created_at
                FROM screenplay.document_chunks
                WHERE id = :id
            """)
            
            result = await session.execute(query, {'id': chunk_id})
            row = result.fetchone()
            
            if row:
                return {
                    'id': str(row[0]),
                    'document_id': str(row[1]),
                    'chunk_index': row[2],
                    'content': row[3],
                    'embedding': row[4],
                    'start_line': row[5],
                    'end_line': row[6],
                    'char_count': row[7],
                    'created_at': row[8].isoformat() if row[8] else None
                }
            return None
    
    async def list_chunks(
        self,
        document_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """列出分块"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            if document_id:
                query = text("""
                    SELECT id, document_id, chunk_index, content, embedding,
                           start_line, end_line, char_count, created_at
                    FROM screenplay.document_chunks
                    WHERE document_id = :document_id
                    ORDER BY chunk_index
                    LIMIT :limit OFFSET :offset
                """)
                params = {'document_id': document_id, 'limit': limit, 'offset': offset}
            else:
                query = text("""
                    SELECT id, document_id, chunk_index, content, embedding,
                           start_line, end_line, char_count, created_at
                    FROM screenplay.document_chunks
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """)
                params = {'limit': limit, 'offset': offset}
            
            result = await session.execute(query, params)
            rows = result.fetchall()
            
            chunks = []
            for row in rows:
                chunks.append({
                    'id': str(row[0]),
                    'document_id': str(row[1]),
                    'chunk_index': row[2],
                    'content': row[3],
                    'embedding': row[4],
                    'start_line': row[5],
                    'end_line': row[6],
                    'char_count': row[7],
                    'created_at': row[8].isoformat() if row[8] else None
                })
            
            return chunks
    
    async def delete_chunk(self, chunk_id: str) -> bool:
        """删除分块"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            query = text("""
                DELETE FROM screenplay.document_chunks
                WHERE id = :id
                RETURNING id
            """)
            
            result = await session.execute(query, {'id': chunk_id})
            await session.commit()
            
            return result.scalar() is not None
    
    async def delete_chunks_by_document(self, document_id: str) -> int:
        """删除文档的所有分块"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            query = text("""
                DELETE FROM screenplay.document_chunks
                WHERE document_id = :document_id
                RETURNING id
            """)
            
            result = await session.execute(query, {'document_id': document_id})
            await session.commit()
            
            deleted = result.fetchall()
            return len(deleted)
    
    async def get_chunks_count(self, document_id: Optional[str] = None) -> int:
        """获取分块数量"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            if document_id:
                query = text("""
                    SELECT COUNT(*) FROM screenplay.document_chunks
                    WHERE document_id = :document_id
                """)
                result = await session.execute(query, {'document_id': document_id})
            else:
                query = text("SELECT COUNT(*) FROM screenplay.document_chunks")
                result = await session.execute(query)
            
            return result.scalar() or 0
