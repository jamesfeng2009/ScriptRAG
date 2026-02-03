"""向量数据库服务 - PostgreSQL + pgvector 集成"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class VectorSearchResult(BaseModel):
    """向量搜索结果"""
    id: str
    file_path: str
    content: str
    similarity: float
    has_deprecated: bool = False
    has_fixme: bool = False
    has_todo: bool = False
    has_security: bool = False
    metadata: Dict[str, Any] = {}


class IVectorDBService(ABC):
    """向量数据库服务接口（抽象）"""
    
    @abstractmethod
    async def index_document(
        self,
        workspace_id: str,
        file_path: str,
        content: str,
        embedding: List[float],
        language: Optional[str] = None,
        has_deprecated: bool = False,
        has_fixme: bool = False,
        has_todo: bool = False,
        has_security: bool = False,
        metadata: Optional[Dict[str, Any]] = None
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
            
        Returns:
            文档 ID
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        workspace_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        搜索相似文档
        
        Args:
            workspace_id: 工作空间 ID
            query_embedding: 查询嵌入向量
            top_k: 返回结果数量
            filters: 可选过滤器
            
        Returns:
            相似文档列表
        """
        pass
    
    @abstractmethod
    async def delete_documents(
        self,
        document_ids: List[str]
    ) -> int:
        """
        删除文档
        
        Args:
            document_ids: 要删除的文档 ID 列表
            
        Returns:
            删除的文档数量
        """
        pass
    
    @abstractmethod
    async def delete_workspace(
        self,
        workspace_id: str
    ) -> int:
        """
        删除工作空间的所有文档
        
        Args:
            workspace_id: 工作空间 ID
            
        Returns:
            删除的文档数量
        """
        pass
    
    @abstractmethod
    async def get_document_count(
        self,
        workspace_id: str
    ) -> int:
        """
        获取工作空间的文档数量
        
        Args:
            workspace_id: 工作空间 ID
            
        Returns:
            文档数量
        """
        pass


class PGVectorService(IVectorDBService):
    """PostgreSQL + pgvector 向量数据库服务"""
    
    def __init__(self, db_pool):
        """
        初始化 pgvector 服务
        
        Args:
            db_pool: AsyncPG 连接池
        """
        self.db_pool = db_pool
        self._table_initialized = False
    
    async def initialize(self):
        """初始化连接池"""
        try:
            import asyncpg
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size
            )
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {str(e)}")
            raise
    
    async def index_document(
        self,
        workspace_id: str,
        file_path: str,
        content: str,
        embedding: List[float],
        language: Optional[str] = None,
        has_deprecated: bool = False,
        has_fixme: bool = False,
        has_todo: bool = False,
        has_security: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """索引文档"""
        if not self.pool:
            raise Exception("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                # 使用 UPSERT 语法（ON CONFLICT）
                result = await conn.fetchrow(
                    """
                    INSERT INTO code_documents (
                        workspace_id, file_path, content, embedding, language,
                        has_deprecated, has_fixme, has_todo, has_security, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (workspace_id, file_path) 
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        language = EXCLUDED.language,
                        has_deprecated = EXCLUDED.has_deprecated,
                        has_fixme = EXCLUDED.has_fixme,
                        has_todo = EXCLUDED.has_todo,
                        has_security = EXCLUDED.has_security,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    RETURNING id
                    """,
                    workspace_id, file_path, content, embedding, language,
                    has_deprecated, has_fixme, has_todo, has_security,
                    metadata or {}
                )
                doc_id = str(result['id'])
                logger.info(f"Indexed document: {file_path} (ID: {doc_id})")
                return doc_id
        except Exception as e:
            logger.error(f"Failed to index document {file_path}: {str(e)}")
            raise
    
    async def search(
        self,
        workspace_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        搜索相似文档
        
        Args:
            workspace_id: 工作空间 ID
            query_embedding: 查询嵌入向量
            top_k: 返回结果数量
            filters: 可选过滤器
            
        Returns:
            相似文档列表
        """
        if not self.pool:
            raise Exception("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM search_similar_documents($1, $2, $3)
                    """,
                    workspace_id, query_embedding, top_k
                )
                
                results = [
                    VectorSearchResult(
                        id=str(row['id']),
                        file_path=row['file_path'],
                        content=row['content'],
                        similarity=row['similarity'],
                        has_deprecated=row['has_deprecated'],
                        has_fixme=row['has_fixme'],
                        has_todo=row['has_todo'],
                        has_security=row['has_security']
                    )
                    for row in rows
                ]
                
                logger.info(f"Vector search returned {len(results)} results")
                return results
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise
    
    async def delete_documents(
        self,
        document_ids: List[str]
    ) -> int:
        """
        删除文档
        
        Args:
            document_ids: 要删除的文档 ID 列表
            
        Returns:
            删除的文档数量
        """
        if not self.pool:
            raise Exception("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM code_documents
                    WHERE id = ANY($1)
                    """,
                    document_ids
                )
                deleted = int(result.split()[-1])
                logger.info(f"Deleted {deleted} documents")
                return deleted
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise
    
    async def delete_workspace(
        self,
        workspace_id: str
    ) -> int:
        """
        删除工作空间的所有文档
        
        Args:
            workspace_id: 工作空间 ID
            
        Returns:
            删除的文档数量
        """
        if not self.pool:
            raise Exception("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM code_documents
                    WHERE workspace_id = $1
                    """,
                    workspace_id
                )
                deleted = int(result.split()[-1])
                logger.info(f"Deleted {deleted} documents for workspace {workspace_id}")
                return deleted
        except Exception as e:
            logger.error(f"Failed to delete workspace: {str(e)}")
            raise
    
    async def get_document_count(
        self,
        workspace_id: str
    ) -> int:
        """
        获取工作空间的文档数量
        
        Args:
            workspace_id: 工作空间 ID
            
        Returns:
            文档数量
        """
        if not self.pool:
            raise Exception("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM code_documents
                    WHERE workspace_id = $1
                    """,
                    workspace_id
                )
                return count or 0
        except Exception as e:
            logger.error(f"Failed to get document count: {str(e)}")
            raise
    
    async def close(self):
        """关闭数据库连接"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
