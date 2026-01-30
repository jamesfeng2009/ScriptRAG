"""Vector Database Service - PostgreSQL + pgvector integration"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
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
    async def vector_search(
        self,
        workspace_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """
        向量相似度搜索
        
        Args:
            workspace_id: 工作空间 ID
            query_embedding: 查询嵌入向量
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            
        Returns:
            搜索结果列表
        """
        pass
    
    @abstractmethod
    async def hybrid_search(
        self,
        workspace_id: str,
        query_embedding: List[float],
        keyword_filters: Optional[Dict[str, bool]] = None,
        top_k: int = 5
    ) -> List[VectorSearchResult]:
        """
        混合搜索（向量 + 关键词）
        
        Args:
            workspace_id: 工作空间 ID
            query_embedding: 查询嵌入向量
            keyword_filters: 关键词过滤器（has_deprecated, has_fixme, has_todo, has_security）
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        pass
    
    @abstractmethod
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
            是否删除成功
        """
        pass
    
    @abstractmethod
    async def close(self):
        """关闭数据库连接"""
        pass


class PostgresVectorDBService(IVectorDBService):
    """
    PostgreSQL + pgvector 向量数据库服务实现
    
    功能：
    - 文档索引与嵌入存储
    - 向量相似度搜索（HNSW 索引）
    - 混合搜索（向量 + 标量过滤）
    - 连接池和错误重试
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        min_pool_size: int = 10,
        max_pool_size: int = 20
    ):
        """
        初始化 PostgreSQL 向量数据库服务
        
        Args:
            host: 数据库主机
            port: 数据库端口
            database: 数据库名称
            user: 用户名
            password: 密码
            min_pool_size: 最小连接池大小
            max_pool_size: 最大连接池大小
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.pool = None
    
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
    
    async def vector_search(
        self,
        workspace_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """向量相似度搜索"""
        if not self.pool:
            raise Exception("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM search_similar_documents($1, $2, $3, $4)
                    """,
                    workspace_id, query_embedding, top_k, similarity_threshold
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
    
    async def hybrid_search(
        self,
        workspace_id: str,
        query_embedding: List[float],
        keyword_filters: Optional[Dict[str, bool]] = None,
        top_k: int = 5
    ) -> List[VectorSearchResult]:
        """混合搜索（向量 + 关键词）"""
        if not self.pool:
            raise Exception("Connection pool not initialized")
        
        try:
            import json
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM hybrid_search_documents($1, $2, $3::jsonb, $4)
                    """,
                    workspace_id, query_embedding, 
                    json.dumps(keyword_filters or {}), top_k
                )
                
                results = []
                for row in rows:
                    # 从数据库获取完整的文档信息
                    doc_row = await conn.fetchrow(
                        """
                        SELECT has_deprecated, has_fixme, has_todo, has_security, metadata
                        FROM code_documents
                        WHERE id = $1
                        """,
                        row['id']
                    )
                    
                    results.append(VectorSearchResult(
                        id=str(row['id']),
                        file_path=row['file_path'],
                        content=row['content'],
                        similarity=row['similarity'],
                        has_deprecated=doc_row['has_deprecated'] if doc_row else False,
                        has_fixme=doc_row['has_fixme'] if doc_row else False,
                        has_todo=doc_row['has_todo'] if doc_row else False,
                        has_security=doc_row['has_security'] if doc_row else False,
                        metadata=doc_row['metadata'] if doc_row else {}
                    ))
                
                logger.info(f"Hybrid search returned {len(results)} results")
                return results
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            raise
    
    async def delete_document(
        self,
        workspace_id: str,
        file_path: str
    ) -> bool:
        """删除文档"""
        if not self.pool:
            raise Exception("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM code_documents
                    WHERE workspace_id = $1 AND file_path = $2
                    """,
                    workspace_id, file_path
                )
                deleted = result.split()[-1] == '1'
                if deleted:
                    logger.info(f"Deleted document: {file_path}")
                return deleted
        except Exception as e:
            logger.error(f"Failed to delete document {file_path}: {str(e)}")
            raise
    
    async def close(self):
        """关闭数据库连接"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
