"""Document Persistence Service - RAG文档管理服务"""

import logging
import hashlib
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, select, update, delete
from sqlalchemy.dialects.postgresql import insert

import asyncio

from src.utils.database_utils import build_db_url

logger = logging.getLogger(__name__)


def get_db_url():
    """构建数据库连接 URL（已废弃，请使用 database_utils.build_db_url）"""
    return build_db_url()


class DocumentRecord:
    """文档记录类（内存表示）"""
    
    def __init__(
        self,
        id: str,
        title: str,
        file_name: str,
        content: str,
        content_hash: str,
        file_size: int,
        file_path: Optional[str] = None,
        category: Optional[str] = None,
        language: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        indexed_at: Optional[datetime] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        self.id = id
        self.title = title
        self.file_name = file_name
        self.file_path = file_path
        self.content = content
        self.content_hash = content_hash
        self.category = category
        self.language = language
        self.file_size = file_size
        self.metadata = metadata or {}
        self.indexed_at = indexed_at or datetime.now()
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()


class DocumentService:
    """文档管理服务"""
    
    def __init__(self):
        self.engine = None
        self.async_session = None
        self._initialized = False
    
    async def init(self):
        """初始化数据库连接"""
        if self._initialized:
            return
        
        try:
            self.engine = create_async_engine(get_db_url(), echo=False)
            self.async_session = sessionmaker(
                self.engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            async with self.engine.begin() as conn:
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS screenplay.documents (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        title VARCHAR(500) NOT NULL,
                        file_name VARCHAR(500) NOT NULL,
                        file_path VARCHAR(1000),
                        content TEXT NOT NULL,
                        content_hash VARCHAR(64) NOT NULL,
                        category VARCHAR(100),
                        language VARCHAR(50),
                        file_size INTEGER NOT NULL,
                        doc_metadata JSONB DEFAULT '{}',
                        indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """))
            
            self._initialized = True
            logger.info("Document service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize document service: {e}")
            raise
    
    async def create(
        self,
        title: str,
        file_name: str,
        content: str,
        file_path: Optional[str] = None,
        category: Optional[str] = None,
        language: Optional[str] = None,
        file_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentRecord:
        """创建文档记录"""
        await self.init()
        
        doc_id = str(uuid4())
        content_hash = hashlib.md5(content.encode()).hexdigest()
        file_size = file_size or len(content.encode())
        
        async with self.async_session() as session:
            await session.execute(
                text("""
                    INSERT INTO screenplay.documents 
                    (id, title, file_name, file_path, content, content_hash, 
                     category, language, file_size, doc_metadata)
                    VALUES (:id, :title, :file_name, :file_path, :content, :content_hash,
                             :category, :language, :file_size, :metadata)
                """),
                {
                    "id": doc_id,
                    "title": title,
                    "file_name": file_name,
                    "file_path": file_path,
                    "content": content,
                    "content_hash": content_hash,
                    "category": category,
                    "language": language,
                    "file_size": file_size,
                    "metadata": json.dumps(metadata or {})
                }
            )
            await session.commit()
        
        return DocumentRecord(
            id=doc_id,
            title=title,
            file_name=file_name,
            file_path=file_path,
            content=content,
            content_hash=content_hash,
            category=category,
            language=language,
            file_size=file_size,
            metadata=metadata
        )
    
    async def get_by_id(self, doc_id: str) -> Optional[DocumentRecord]:
        """根据 ID 获取文档"""
        await self.init()
        
        async with self.async_session() as session:
            result = await session.execute(
                text("""
                    SELECT id, title, file_name, file_path, content, content_hash,
                           category, language, file_size, doc_metadata,
                           indexed_at, created_at, updated_at
                    FROM screenplay.documents
                    WHERE id = :id
                """),
                {"id": doc_id}
            )
            row = result.fetchone()
            
            if not row:
                return None
            
            return DocumentRecord(
                id=row[0],
                title=row[1],
                file_name=row[2],
                file_path=row[3],
                content=row[4],
                content_hash=row[5],
                category=row[6],
                language=row[7],
                file_size=row[8],
                metadata=row[9] or {},
                indexed_at=row[10],
                created_at=row[11],
                updated_at=row[12]
            )
    
    async def list_all(
        self,
        page: int = 1,
        page_size: int = 20,
        category: Optional[str] = None
    ) -> tuple[List[DocumentRecord], int]:
        """列出文档"""
        await self.init()
        
        offset = (page - 1) * page_size
        
        async with self.async_session() as session:
            if category:
                count_result = await session.execute(
                    text("""
                        SELECT COUNT(*) FROM screenplay.documents
                        WHERE category = :category
                    """),
                    {"category": category}
                )
                result = await session.execute(
                    text("""
                        SELECT id, title, file_name, file_path, content, content_hash,
                               category, language, file_size, doc_metadata,
                               indexed_at, created_at, updated_at
                        FROM screenplay.documents
                        WHERE category = :category
                        ORDER BY created_at DESC
                        LIMIT :limit OFFSET :offset
                    """),
                    {
                        "category": category,
                        "limit": page_size,
                        "offset": offset
                    }
                )
            else:
                count_result = await session.execute(
                    text("SELECT COUNT(*) FROM screenplay.documents")
                )
                result = await session.execute(
                    text("""
                        SELECT id, title, file_name, file_path, content, content_hash,
                               category, language, file_size, doc_metadata,
                               indexed_at, created_at, updated_at
                        FROM screenplay.documents
                        ORDER BY created_at DESC
                        LIMIT :limit OFFSET :offset
                    """),
                    {
                        "limit": page_size,
                        "offset": offset
                    }
                )
            
            total = count_result.scalar()
            rows = result.fetchall()
        
        docs = [
            DocumentRecord(
                id=row[0],
                title=row[1],
                file_name=row[2],
                file_path=row[3],
                content=row[4],
                content_hash=row[5],
                category=row[6],
                language=row[7],
                file_size=row[8],
                metadata=row[9] or {},
                indexed_at=row[10],
                created_at=row[11],
                updated_at=row[12]
            )
            for row in rows
        ]
        
        return docs, total
    
    async def delete(self, doc_id: str) -> bool:
        """删除文档"""
        await self.init()
        
        async with self.async_session() as session:
            result = await session.execute(
                text("DELETE FROM screenplay.documents WHERE id = :id"),
                {"id": doc_id}
            )
            await session.commit()
        
        return result.rowcount > 0
    
    async def search_by_content(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """简单的内容搜索（关键词匹配）"""
        await self.init()
        
        async with self.async_session() as session:
            result = await session.execute(
                text("""
                    SELECT id, title, file_name, content,
                           created_at, file_size, category
                    FROM screenplay.documents
                    WHERE content ILIKE :query
                    ORDER BY created_at DESC
                    LIMIT :limit
                """),
                {
                    "query": f"%{query}%",
                    "limit": top_k
                }
            )
            rows = result.fetchall()
        
        return [
            {
                "doc_id": row[0],
                "title": row[1],
                "file_name": row[2],
                "content_preview": row[3][:200] + "..." if len(row[3]) > 200 else row[3],
                "created_at": row[4].isoformat() if row[4] else None,
                "file_size": row[5],
                "category": row[6]
            }
            for row in rows
        ]
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取文档统计信息"""
        await self.init()
        
        async with self.async_session() as session:
            result = await session.execute(
                text("""
                    SELECT COUNT(*), SUM(file_size)
                    FROM screenplay.documents
                """)
            )
            row = result.fetchone()
        
        return {
            "total_documents": row[0] or 0,
            "total_size_bytes": row[1] or 0
        }
    
    async def close(self):
        """关闭数据库连接"""
        if self.engine:
            await self.engine.dispose()


def _create_chunker(llm_service=None):
    """创建智能分块器

    Args:
        llm_service: LLM 服务实例（用于上下文增强）

    Returns:
        SmartChunker 实例
    """
    from .document_chunker import SmartChunker

    return SmartChunker(
        llm_service=llm_service,
        enable_contextual_enrichment=llm_service is not None,
        enable_atomic_chunking=llm_service is not None,
        enable_strategy_cache=True
    )


async def process_and_index_document(
    content: str,
    file_path: str,
    title: str = None,
    category: str = None,
    document_service: DocumentService = None,
    chunks_service: "DocumentChunksService" = None,
    llm_service: None = None
) -> Dict[str, Any]:
    """一站式文档处理：创建文档记录 + 分块 + 存储分块

    Args:
        content: 文档内容
        file_path: 文件路径
        title: 文档标题（默认从文件名提取）
        category: 文档分类
        document_service: 文档服务实例
        chunks_service: 分块服务实例
        llm_service: LLM 服务实例（用于上下文增强）

    Returns:
        处理结果字典
    """
    from pathlib import Path
    from .document_chunker import SmartChunker

    title = title or Path(file_path).stem
    file_name = Path(file_path).name
    file_size = len(content.encode())

    if document_service is None:
        document_service = document_service or DocumentService()
    if chunks_service is None:
        from .document_chunks_service import DocumentChunksService
        chunks_service = DocumentChunksService()

    await document_service.init()
    await chunks_service.initialize()

    doc_record = await document_service.create(
        title=title,
        file_name=file_name,
        content=content,
        file_path=file_path,
        category=category,
        file_size=file_size
    )

    chunker = _create_chunker(llm_service)
    chunks = chunker(content, file_path)

    created_chunks = []
    for chunk in chunks:
        chunk_id = await chunks_service.create_chunk(
            document_id=str(doc_record.id),
            chunk_index=int(chunk.id.split('_')[-1]) if '_' in chunk.id else 0,
            content=chunk.content,
            start_line=chunk.metadata.start_line,
            end_line=chunk.metadata.end_line,
            char_count=chunk.metadata.char_count
        )
        created_chunks.append({
            "chunk_id": chunk_id,
            "content": chunk.content,
            "start_line": chunk.metadata.start_line,
            "end_line": chunk.metadata.end_line
        })

    await chunks_service.close()

    return {
        "document_id": str(doc_record.id),
        "title": doc_record.title,
        "file_name": doc_record.file_name,
        "chunks_count": len(created_chunks),
        "chunks": created_chunks
    }


document_service = DocumentService()
