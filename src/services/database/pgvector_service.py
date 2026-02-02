"""
pgvector PostgreSQL 向量数据库服务实现

功能：
1. 文档索引与嵌入存储
2. 向量相似度搜索（HNSW 索引）
3. 混合搜索（向量 + 标量过滤）
4. 连接池和错误重试
5. 成本跟踪
6. 自动分块

使用示例：
```python
from src.services.database.pgvector_service import PgVectorDBService

# 初始化服务
db = PgVectorDBService(
    host="localhost",
    port=5432,
    database="rag_system",
    user="postgres",
    password="your_password",
    min_pool_size=5,
    max_pool_size=20
)

# 初始化连接
await db.initialize()

# 索引文档
doc_id = await db.index_document(
    workspace_id="project-alpha",
    file_path="src/utils/async_helpers.py",
    content=code_content,
    embedding=embedding_vector,
    language="python"
)

# 混合搜索
results = await db.hybrid_search(
    workspace_id="project-alpha",
    query_embedding=query_embedding,
    keyword_filters={"has_deprecated": True},
    top_k=10
)

# 关闭连接
await db.close()
```
"""

import logging
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import text, select, update, delete, func, Column, String, Integer, Boolean, Text, DateTime, JSON
from sqlalchemy.dialects.postgresql import insert, UUID

from src.services.database.vector_db import (
    IVectorDBService,
    VectorSearchResult
)

logger = logging.getLogger(__name__)

Base = declarative_base()


class DocumentEmbeddingModel(Base):
    """文档嵌入模型（SQLAlchemy ORM - 仅用于查询，不用于插入/更新）"""
    __tablename__ = 'document_embeddings'
    __table_args__ = {'schema': 'screenplay'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(String(255), nullable=False, index=True)
    file_path = Column(Text, nullable=False)
    file_hash = Column(String(64), nullable=False)
    content = Column(Text, nullable=False)
    content_summary = Column(Text)
    embedding = Column(Text)
    language = Column(String(50))
    file_type = Column(String(50))
    file_size = Column(Integer)
    line_count = Column(Integer)
    has_deprecated = Column(Boolean, default=False)
    has_fixme = Column(Boolean, default=False)
    has_todo = Column(Boolean, default=False)
    has_security = Column(Boolean, default=False)
    doc_metadata = Column('metadata', JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DocumentChunkModel(Base):
    """文档分块模型（SQLAlchemy ORM）"""
    __tablename__ = 'document_chunks'
    __table_args__ = {'schema': 'screenplay'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Text)  # pgvector 需要用文本存储
    start_line = Column(Integer)
    end_line = Column(Integer)
    char_count = Column(Integer)
    token_count = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


@dataclass
class DocumentRecord:
    """文档记录"""
    id: str
    workspace_id: str
    file_path: str
    file_hash: str
    content: str
    content_summary: str = None
    embedding: List[float] = None
    language: str = None
    file_type: str = None
    file_size: int = None
    line_count: int = None
    has_deprecated: bool = False
    has_fixme: bool = False
    has_todo: bool = False
    has_security: bool = False
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class RetrievalMetrics:
    """检索指标"""
    search_time_ms: float
    result_count: int
    avg_score: float
    cache_hit: bool = False


class PgVectorDBService(IVectorDBService):
    """
    PostgreSQL + pgvector 向量数据库服务实现
    
    功能：
    - 文档索引与嵌入存储
    - 向量相似度搜索（HNSW 索引）
    - 混合搜索（向量 + 标量过滤）
    - 连接池和错误重试
    - 自动分块长文档
    - 成本跟踪
    """
    
    def __init__(
        self,
        embedding_dim: int = 1024,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        config = None
    ):
        """
        初始化 PostgreSQL 向量数据库服务
        
        Args:
            embedding_dim: 嵌入向量维度
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
            config: 数据库配置对象（可选，默认从 config.py 加载）
        """
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self._config = config
        self.engine = None
        self.async_session = None
        self._initialized = False
        
        self._query_count = 0
        self._total_query_time = 0.0
        self._cache_hits = 0
    
    def _get_db_url(self) -> str:
        """获取数据库连接 URL"""
        if self._config is not None:
            db_config = self._config
        else:
            from ..config import get_database_config
            db_config = get_database_config()
        return f"postgresql+asyncpg://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
    
    async def initialize(self):
        """初始化连接池"""
        try:
            db_config = self._config
            if db_config is None:
                from ..config import get_database_config
                db_config = get_database_config()
            
            self.engine = create_async_engine(
                self._get_db_url(),
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            
            self.async_session = sessionmaker(
                self.engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            self._initialized = True
            logger.info(f"PostgreSQL connection pool initialized: {db_config.host}:{db_config.port}/{db_config.database}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {str(e)}")
            raise
    
    async def close(self):
        """关闭连接池"""
        if self.engine:
            await self.engine.dispose()
            self._initialized = False
            logger.info("PostgreSQL connection pool closed")
    
    async def health_check(self) -> bool:
        """健康检查"""
        if not self._initialized:
            return False
        
        try:
            async with self.async_session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def _calculate_file_hash(self, content: str) -> str:
        """计算文件内容哈希"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _embedding_to_pgvector(self, embedding: List[float]) -> str:
        """将嵌入向量转换为 pgvector 格式"""
        return "[" + ", ".join(str(x) for x in embedding) + "]"
    
    def _embedding_from_pgvector(self, pgvector_str: str) -> List[float]:
        """从 pgvector 格式转换回嵌入向量"""
        if pgvector_str is None:
            return []
        cleaned = pgvector_str.strip("[]")
        if not cleaned:
            return []
        return [float(x.strip()) for x in cleaned.split(",")]
    
    def _detect_markers(self, content: str) -> Dict[str, bool]:
        """检测内容中的标记"""
        content_lower = content.lower()
        return {
            "has_deprecated": "@deprecated" in content or "deprecated" in content_lower,
            "has_fixme": "fixme" in content_lower,
            "has_todo": "todo" in content_lower or "TODO" in content,
            "has_security": "security" in content_lower or "SECURITY" in content
        }
    
    def _detect_language(self, file_path: str, content: str) -> Tuple[str, str]:
        """检测编程语言和文件类型"""
        import os
        extension_map = {
            ".py": ("python", "python"),
            ".js": ("javascript", "javascript"),
            ".ts": ("typescript", "typescript"),
            ".java": ("java", "java"),
            ".go": ("go", "go"),
            ".rs": ("rust", "rust"),
            ".cpp": ("cpp", "cpp"),
            ".c": ("c", "c"),
            ".h": ("c", "c"),
            ".cs": ("csharp", "csharp"),
            ".rb": ("ruby", "ruby"),
            ".php": ("php", "php"),
            ".swift": ("swift", "swift"),
            ".kt": ("kotlin", "kotlin"),
            ".scala": ("scala", "scala"),
            ".md": ("markdown", "markdown"),
            ".txt": ("text", "text"),
            ".json": ("json", "json"),
            ".yaml": ("yaml", "yaml"),
            ".yml": ("yaml", "yaml"),
            ".xml": ("xml", "xml"),
            ".html": ("html", "html"),
            ".css": ("css", "css"),
            ".sql": ("sql", "sql"),
            ".sh": ("shell", "shell"),
            ".bash": ("shell", "shell"),
            ".zsh": ("shell", "shell"),
        }
        
        ext = os.path.splitext(file_path)[1].lower()
        language = extension_map.get(ext, ("unknown", "unknown"))
        
        if language[0] == "unknown":
            content_lower = content[:1000].lower()
            if "def " in content or "import " in content or "class " in content:
                language = ("python", "python")
            elif "function " in content or "const " in content or "let " in content:
                language = ("javascript", "javascript")
        
        return language
    
    def _chunk_document(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """将文档分割成块"""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_size = 0
        chunk_index = 0
        start_line = 1
        
        for line_num, line in enumerate(lines, 1):
            line_size = len(line) + 1
            
            if current_size + line_size > self.chunk_size:
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk)
                    chunks.append({
                        "chunk_index": chunk_index,
                        "content": chunk_content,
                        "start_line": start_line,
                        "end_line": line_num - 1,
                        "char_count": len(chunk_content),
                        "line_count": len(current_chunk)
                    })
                    chunk_index += 1
                
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    current_chunk = current_chunk[-self.chunk_overlap:]
                    current_size = sum(len(l) + 1 for l in current_chunk)
                    start_line = line_num - len(current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
                    start_line = line_num
            
            current_chunk.append(line)
            current_size += line_size
        
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                "chunk_index": chunk_index,
                "content": chunk_content,
                "start_line": start_line,
                "end_line": len(lines),
                "char_count": len(chunk_content),
                "line_count": len(current_chunk)
            })
        
        return chunks
    
    async def index_document(
        self,
        workspace_id: str,
        file_path: str,
        content: str,
        embedding: List[float],
        language: str = None,
        has_deprecated: bool = False,
        has_fixme: bool = False,
        has_todo: bool = False,
        has_security: bool = False,
        metadata: Dict[str, Any] = None,
        auto_chunk: bool = True
    ) -> str:
        """
        索引文档
        
        Args:
            workspace_id: 工作空间 ID
            file_path: 文件路径
            content: 文档内容
            embedding: 嵌入向量
            language: 编程语言
            has_deprecated: 是否包含废弃标记
            has_fixme: 是否包含 FIXME 标记
            has_todo: 是否包含 TODO 标记
            has_security: 是否包含安全标记
            metadata: 元数据
            auto_chunk: 是否自动分块
            
        Returns:
            文档 ID
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        import os
        file_hash = self._calculate_file_hash(content)
        file_size = len(content.encode('utf-8'))
        line_count = content.count('\n') + 1
        
        if not language:
            language, file_type = self._detect_language(file_path, content)
        else:
            file_type = language
        
        markers = self._detect_markers(content)
        has_deprecated = has_deprecated or markers["has_deprecated"]
        has_fixme = has_fixme or markers["has_fixme"]
        has_todo = has_todo or markers["has_todo"]
        has_security = has_security or markers["has_security"]
        
        pgvector_embedding = self._embedding_to_pgvector(embedding)
        
        async with self.async_session() as session:
            metadata_json = json.dumps(metadata or {})
            
            insert_query = text("""
                INSERT INTO screenplay.document_embeddings (
                    workspace_id, file_path, file_hash, content,
                    embedding, language, file_type, file_size, line_count,
                    has_deprecated, has_fixme, has_todo, has_security,
                    metadata
                ) VALUES (
                    :workspace_id, :file_path, :file_hash, :content,
                    CAST(:embedding AS vector), :language, :file_type, :file_size, :line_count,
                    :has_deprecated, :has_fixme, :has_todo, :has_security,
                    CAST(:metadata AS jsonb)
                )
                ON CONFLICT (workspace_id, file_path)
                DO UPDATE SET
                    file_hash = EXCLUDED.file_hash,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    language = EXCLUDED.language,
                    file_type = EXCLUDED.file_type,
                    file_size = EXCLUDED.file_size,
                    line_count = EXCLUDED.line_count,
                    has_deprecated = EXCLUDED.has_deprecated,
                    has_fixme = EXCLUDED.has_fixme,
                    has_todo = EXCLUDED.has_todo,
                    has_security = EXCLUDED.has_security,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                RETURNING id
            """)
            
            result = await session.execute(insert_query, {
                'workspace_id': workspace_id,
                'file_path': file_path,
                'file_hash': file_hash,
                'content': content,
                'embedding': pgvector_embedding,
                'language': language,
                'file_type': file_type,
                'file_size': file_size,
                'line_count': line_count,
                'has_deprecated': has_deprecated,
                'has_fixme': has_fixme,
                'has_todo': has_todo,
                'has_security': has_security,
                'metadata': metadata_json
            })
            doc_id = result.scalar()
            await session.commit()
            
            if auto_chunk and len(content) > self.chunk_size:
                await self._delete_chunks(session, doc_id)
                await self._create_chunks(session, doc_id, content, embedding)
            
            logger.debug(f"Indexed document: {file_path} (ID: {doc_id})")
            return str(doc_id)
    
    async def _delete_chunks(self, session: AsyncSession, document_id: str):
        """删除文档分块"""
        await session.execute(
            delete(DocumentChunkModel).where(
                DocumentChunkModel.document_id == document_id
            )
        )
    
    async def _create_chunks(
        self,
        session: AsyncSession,
        document_id: str,
        content: str,
        embedding: List[float]
    ):
        """创建文档分块"""
        chunks = self._chunk_document(content, "")
        
        for chunk in chunks:
            pgvector_chunk = self._embedding_to_pgvector(embedding)
            session.add(DocumentChunkModel(
                document_id=document_id,
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                embedding=pgvector_chunk,
                start_line=chunk["start_line"],
                end_line=chunk["end_line"],
                char_count=chunk["char_count"],
                token_count=chunk["char_count"] // 4
            ))
        
        await session.commit()
    
    async def index_documents_batch(
        self,
        workspace_id: str,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> Dict[str, str]:
        """
        批量索引文档
        
        Args:
            workspace_id: 工作空间 ID
            documents: 文档列表
            embeddings: 嵌入向量列表
            
        Returns:
            文件路径到文档 ID 的映射
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        results = {}
        
        async with self.async_session() as session:
            for doc, embedding in zip(documents, embeddings):
                doc_id = await self.index_document(
                    workspace_id=workspace_id,
                    file_path=doc["file_path"],
                    content=doc["content"],
                    embedding=embedding,
                    language=doc.get("language"),
                    metadata=doc.get("metadata"),
                    auto_chunk=doc.get("auto_chunk", True)
                )
                results[doc["file_path"]] = doc_id
        
        logger.info(f"Batch indexed {len(results)} documents")
        return results
    
    async def get_document(
        self,
        workspace_id: str,
        file_path: str
    ) -> Optional[DocumentRecord]:
        """
        获取文档记录
        
        Args:
            workspace_id: 工作空间 ID
            file_path: 文件路径
            
        Returns:
            文档记录，不存在则返回 None
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        async with self.async_session() as session:
            stmt = select(DocumentEmbeddingModel).where(
                DocumentEmbeddingModel.workspace_id == workspace_id,
                DocumentEmbeddingModel.file_path == file_path
            )
            result = await session.execute(stmt)
            doc = result.scalar_one_or_none()
            
            if doc is None:
                return None
            
            return DocumentRecord(
                id=str(doc.id),
                workspace_id=doc.workspace_id,
                file_path=doc.file_path,
                file_hash=doc.file_hash,
                content=doc.content,
                content_summary=doc.content_summary,
                embedding=self._embedding_from_pgvector(doc.embedding) if doc.embedding else None,
                language=doc.language,
                file_type=doc.file_type,
                file_size=doc.file_size,
                line_count=doc.line_count,
                has_deprecated=doc.has_deprecated,
                has_fixme=doc.has_fixme,
                has_todo=doc.has_todo,
                has_security=doc.has_security,
                metadata=doc.metadata or {},
                created_at=doc.created_at,
                updated_at=doc.updated_at
            )
    
    async def delete_document(
        self,
        workspace_id: str,
        file_path: str
    ) -> bool:
        """
        删除文档
        
        Args:
            workspace_id: 工作空间 ID
            file_path: 文件路径
            
        Returns:
            是否成功删除
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        async with self.async_session() as session:
            stmt = delete(DocumentEmbeddingModel).where(
                DocumentEmbeddingModel.workspace_id == workspace_id,
                DocumentEmbeddingModel.file_path == file_path
            )
            result = await session.execute(stmt)
            await session.commit()
            
            return result.rowcount > 0
    
    async def vector_search(
        self,
        workspace_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        filters: Dict[str, bool] = None
    ) -> List[VectorSearchResult]:
        """
        向量相似度搜索（使用 raw SQL，因为 pgvector 有特殊语法）
        
        Args:
            workspace_id: 工作空间 ID
            query_embedding: 查询嵌入向量
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            filters: 标量过滤器
            
        Returns:
            搜索结果列表
        """
        import time
        start_time = time.time()
        
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        pgvector_embedding = self._embedding_to_pgvector(query_embedding)
        
        async with self.async_session() as session:
            query = text(f"""
                SELECT 
                    id, file_path, content, 
                    1 - (embedding <=> CAST(:query_vector AS vector)) as similarity,
                    has_deprecated, has_fixme, has_todo, has_security,
                    metadata, created_at
                FROM screenplay.document_embeddings
                WHERE workspace_id = :workspace_id
                AND (embedding <=> CAST(:query_vector AS vector)) < :threshold
            """)
            
            params = {
                "query_vector": pgvector_embedding,
                "workspace_id": workspace_id,
                "threshold": 1 - similarity_threshold
            }
            
            if filters:
                if filters.get("has_deprecated"):
                    query = text(str(query) + " AND has_deprecated = true")
                if filters.get("has_fixme"):
                    query = text(str(query) + " AND has_fixme = true")
                if filters.get("has_todo"):
                    query = text(str(query) + " AND has_todo = true")
                if filters.get("has_security"):
                    query = text(str(query) + " AND has_security = true")
            
            query = text(str(query) + f" ORDER BY embedding <=> CAST(:query_vector AS vector) LIMIT {top_k}")
            
            result = await session.execute(query, params)
            rows = result.fetchall()
        
        results = []
        for row in rows:
            metadata = row.metadata
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            results.append(VectorSearchResult(
                id=str(row.id),
                file_path=row.file_path,
                content=row.content,
                similarity=float(row.similarity),
                has_deprecated=row.has_deprecated,
                has_fixme=row.has_fixme,
                has_todo=row.has_todo,
                has_security=row.has_security,
                metadata=metadata if metadata else {}
            ))
        
        search_time = (time.time() - start_time) * 1000
        self._query_count += 1
        self._total_query_time += search_time
        
        logger.debug(f"Vector search completed in {search_time:.2f}ms, found {len(results)} results")
        
        return results
    
    async def keyword_search(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 5,
        markers: List[str] = None
    ) -> List[VectorSearchResult]:
        """
        关键词搜索（使用 raw SQL，因为需要 pg_trgm 扩展）
        
        Args:
            workspace_id: 工作空间 ID
            query: 查询关键词
            top_k: 返回结果数量
            markers: 标记过滤器列表
            
        Returns:
            搜索结果列表
        """
        import time
        start_time = time.time()
        
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        async with self.async_session() as session:
            sql_query = f"""
                SELECT 
                    id, file_path, content, 
                    CASE 
                        WHEN content LIKE :exact_match THEN 1.0
                        ELSE similarity(content, :query)
                    END as similarity,
                    has_deprecated, has_fixme, has_todo, has_security,
                    metadata, created_at
                FROM screenplay.document_embeddings
                WHERE workspace_id = :workspace_id
                AND (
                    content ILIKE :pattern
                    OR content % :query
                )
            """
            
            params = {
                "exact_match": f"%{query}%",
                "query": query,
                "workspace_id": workspace_id,
                "pattern": f"%{query}%"
            }
            
            if markers:
                marker_conditions = []
                if "deprecated" in markers:
                    marker_conditions.append("has_deprecated = true")
                if "fixme" in markers:
                    marker_conditions.append("has_fixme = true")
                if "todo" in markers:
                    marker_conditions.append("has_todo = true")
                if "security" in markers:
                    marker_conditions.append("has_security = true")
                
                if marker_conditions:
                    sql_query += " AND " + " AND ".join(marker_conditions)
            
            sql_query += f" ORDER BY similarity DESC LIMIT {top_k}"
            
            result = await session.execute(text(sql_query), params)
            rows = result.fetchall()
        
        results = []
        for row in rows:
            metadata = row.metadata
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            results.append(VectorSearchResult(
                id=str(row.id),
                file_path=row.file_path,
                content=row.content,
                similarity=float(row.similarity),
                has_deprecated=row.has_deprecated,
                has_fixme=row.has_fixme,
                has_todo=row.has_todo,
                has_security=row.has_security,
                metadata=metadata if metadata else {}
            ))
        
        search_time = (time.time() - start_time) * 1000
        self._query_count += 1
        self._total_query_time += search_time
        
        logger.debug(f"Keyword search completed in {search_time:.2f}ms, found {len(results)} results")
        
        return results
    
    async def hybrid_search(
        self,
        workspace_id: str,
        query_embedding: List[float],
        query_text: str = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        filters: Dict[str, bool] = None,
        weights: Dict[str, float] = None
    ) -> List[VectorSearchResult]:
        """
        混合搜索（向量 + 关键词）
        
        Args:
            workspace_id: 工作空间 ID
            query_embedding: 查询嵌入向量
            query_text: 查询文本（用于关键词搜索）
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            filters: 标量过滤器
            weights: 权重配置
            
        Returns:
            搜索结果列表
        """
        import time
        start_time = time.time()
        
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        pgvector_embedding = self._embedding_to_pgvector(query_embedding)
        
        if weights is None:
            weights = {"vector": 0.7, "keyword": 0.3}
        
        async with self.async_session() as session:
            sql_query = text(f"""
                WITH vector_results AS (
                    SELECT 
                        id, file_path, content,
                        has_deprecated, has_fixme, has_todo, has_security,
                        metadata, created_at,
                        1 - (embedding <=> :query_vector::vector) as vector_similarity
                    FROM screenplay.document_embeddings
                    WHERE workspace_id = :workspace_id
                    AND (embedding <=> :query_vector::vector) < :threshold
                )
                SELECT 
                    id, file_path, content,
                    has_deprecated, has_fixme, has_todo, has_security,
                    metadata, created_at,
                    vector_similarity,
                    (
                        SELECT MAX(similarity(content, :query_text))
                        FROM screenplay.document_embeddings d2
                        WHERE d2.id = vector_results.id
                    ) as keyword_similarity
                FROM vector_results
            """)
            
            params = {
                "query_vector": pgvector_embedding,
                "workspace_id": workspace_id,
                "threshold": 1 - similarity_threshold,
                "query_text": query_text or ""
            }
            
            result = await session.execute(sql_query, params)
            rows = result.fetchall()
        
        scored_results = []
        for row in rows:
            vector_score = float(row.vector_similarity) if row.vector_similarity else 0
            keyword_score = float(row.keyword_similarity) if row.keyword_similarity else 0
            
            combined_score = (
                weights.get("vector", 0.7) * vector_score +
                weights.get("keyword", 0.3) * keyword_score
            )
            
            if combined_score >= similarity_threshold:
                metadata = row.metadata
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                scored_results.append((
                    VectorSearchResult(
                        id=str(row.id),
                        file_path=row.file_path,
                        content=row.content,
                        similarity=combined_score,
                        has_deprecated=row.has_deprecated,
                        has_fixme=row.has_fixme,
                        has_todo=row.has_todo,
                        has_security=row.has_security,
                        metadata=metadata if metadata else {}
                    ),
                    combined_score
                ))
        
        scored_results.sort(key=lambda x: x[1], reverse=True)
        results = [r[0] for r in scored_results[:top_k]]
        
        search_time = (time.time() - start_time) * 1000
        self._query_count += 1
        self._total_query_time += search_time
        
        logger.debug(f"Hybrid search completed in {search_time:.2f}ms, found {len(results)} results")
        
        return results
    
    async def delete_documents_by_workspace(self, workspace_id: str) -> int:
        """
        删除工作空间的所有文档
        
        Args:
            workspace_id: 工作空间 ID
            
        Returns:
            删除的文档数量
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        async with self.async_session() as session:
            result = await session.execute(
                delete(DocumentEmbeddingModel).where(
                    DocumentEmbeddingModel.workspace_id == workspace_id
                )
            )
            await session.commit()
            
            return result.rowcount
    
    async def get_stats(self, workspace_id: str = None) -> Dict[str, Any]:
        """
        获取统计信息
        
        Args:
            workspace_id: 工作空间 ID（可选）
            
        Returns:
            统计信息字典
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        async with self.async_session() as session:
            query = select(
                func.count(DocumentEmbeddingModel.id).label('total_docs'),
                func.sum(DocumentEmbeddingModel.file_size).label('total_size'),
                func.avg(DocumentEmbeddingModel.line_count).label('avg_lines'),
                func.count(func.distinct(DocumentEmbeddingModel.workspace_id)).label('workspace_count')
            )
            
            if workspace_id:
                query = query.where(
                    DocumentEmbeddingModel.workspace_id == workspace_id
                )
            
            result = await session.execute(query)
            row = result.one()
            
            return {
                "total_documents": row.total_docs or 0,
                "total_size_bytes": row.total_size or 0,
                "average_lines": row.avg_lines or 0,
                "workspace_count": row.workspace_count or 0,
                "query_count": self._query_count,
                "total_query_time_ms": self._total_query_time,
                "average_query_time_ms": (
                    self._total_query_time / self._query_count 
                    if self._query_count > 0 else 0
                )
            }
