"""
双写管理器
在迁移期间同时写入 PostgreSQL 和 Milvus，确保数据一致性
"""

import asyncio
import asyncpg
from pymilvus import Collection, connections
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid


logger = logging.getLogger(__name__)


class DualWriteManager:
    """双写管理器 - 同时写入 PostgreSQL 和 Milvus"""
    
    def __init__(
        self,
        pg_pool: asyncpg.Pool,
        milvus_collection: Collection,
        enable_milvus: bool = False
    ):
        """
        初始化双写管理器
        
        Args:
            pg_pool: PostgreSQL 连接池
            milvus_collection: Milvus 集合
            enable_milvus: 是否启用 Milvus 写入（灰度开关）
        """
        self.pg_pool = pg_pool
        self.milvus_collection = milvus_collection
        self.enable_milvus = enable_milvus
        
    async def insert_document(
        self,
        workspace_id: str,
        file_path: str,
        content: str,
        embedding: List[float],
        language: str,
        has_deprecated: bool = False,
        has_fixme: bool = False,
        has_todo: bool = False,
        has_security: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        插入文档（双写）
        
        Returns:
            文档 ID
        """
        doc_id = str(uuid.uuid4())
        now = datetime.now()
        
        # 写入 PostgreSQL
        try:
            await self._insert_to_postgres(
                doc_id=doc_id,
                workspace_id=workspace_id,
                file_path=file_path,
                content=content,
                embedding=embedding,
                language=language,
                has_deprecated=has_deprecated,
                has_fixme=has_fixme,
                has_todo=has_todo,
                has_security=has_security,
                metadata=metadata or {},
                created_at=now,
                updated_at=now
            )
            logger.debug(f"Inserted document {doc_id} to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to insert to PostgreSQL: {e}")
            raise
        
        # 写入 Milvus（如果启用）
        if self.enable_milvus:
            try:
                await self._insert_to_milvus(
                    doc_id=doc_id,
                    workspace_id=workspace_id,
                    file_path=file_path,
                    content=content,
                    embedding=embedding,
                    language=language,
                    has_deprecated=has_deprecated,
                    has_fixme=has_fixme,
                    has_todo=has_todo,
                    has_security=has_security,
                    created_at=now,
                    updated_at=now
                )
                logger.debug(f"Inserted document {doc_id} to Milvus")
            except Exception as e:
                logger.error(f"Failed to insert to Milvus: {e}")
                # Milvus 写入失败不影响主流程
                # 可以通过增量同步补齐
        
        return doc_id
    
    async def _insert_to_postgres(
        self,
        doc_id: str,
        workspace_id: str,
        file_path: str,
        content: str,
        embedding: List[float],
        language: str,
        has_deprecated: bool,
        has_fixme: bool,
        has_todo: bool,
        has_security: bool,
        metadata: Dict[str, Any],
        created_at: datetime,
        updated_at: datetime
    ):
        """插入到 PostgreSQL"""
        async with self.pg_pool.acquire() as conn:
            query = """
                INSERT INTO code_documents (
                    id, workspace_id, file_path, content, embedding,
                    language, has_deprecated, has_fixme, has_todo, has_security,
                    metadata, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
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
                    updated_at = EXCLUDED.updated_at
            """
            await conn.execute(
                query,
                doc_id,
                workspace_id,
                file_path,
                content,
                embedding,
                language,
                has_deprecated,
                has_fixme,
                has_todo,
                has_security,
                metadata,
                created_at,
                updated_at
            )
    
    async def _insert_to_milvus(
        self,
        doc_id: str,
        workspace_id: str,
        file_path: str,
        content: str,
        embedding: List[float],
        language: str,
        has_deprecated: bool,
        has_fixme: bool,
        has_todo: bool,
        has_security: bool,
        created_at: datetime,
        updated_at: datetime
    ):
        """插入到 Milvus"""
        # 先删除已存在的记录（如果有）
        expr = f'id == "{doc_id}"'
        self.milvus_collection.delete(expr)
        
        # 插入新记录
        entities = [
            [doc_id],
            [workspace_id],
            [file_path],
            [content[:65535]],  # 截断到最大长度
            [embedding],
            [language or ""],
            [has_deprecated],
            [has_fixme],
            [has_todo],
            [has_security],
            [int(created_at.timestamp() * 1000)],
            [int(updated_at.timestamp() * 1000)]
        ]
        
        self.milvus_collection.insert(entities)
    
    async def update_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        has_deprecated: Optional[bool] = None,
        has_fixme: Optional[bool] = None,
        has_todo: Optional[bool] = None,
        has_security: Optional[bool] = None
    ):
        """
        更新文档（双写）
        """
        now = datetime.now()
        
        # 更新 PostgreSQL
        try:
            await self._update_postgres(
                doc_id=doc_id,
                content=content,
                embedding=embedding,
                has_deprecated=has_deprecated,
                has_fixme=has_fixme,
                has_todo=has_todo,
                has_security=has_security,
                updated_at=now
            )
            logger.debug(f"Updated document {doc_id} in PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to update PostgreSQL: {e}")
            raise
        
        # 更新 Milvus（如果启用）
        if self.enable_milvus:
            try:
                # Milvus 不支持原地更新，需要先查询再删除再插入
                await self._update_milvus(doc_id, content, embedding, has_deprecated, has_fixme, has_todo, has_security, now)
                logger.debug(f"Updated document {doc_id} in Milvus")
            except Exception as e:
                logger.error(f"Failed to update Milvus: {e}")
    
    async def _update_postgres(
        self,
        doc_id: str,
        content: Optional[str],
        embedding: Optional[List[float]],
        has_deprecated: Optional[bool],
        has_fixme: Optional[bool],
        has_todo: Optional[bool],
        has_security: Optional[bool],
        updated_at: datetime
    ):
        """更新 PostgreSQL"""
        updates = []
        params = []
        param_idx = 1
        
        if content is not None:
            updates.append(f"content = ${param_idx}")
            params.append(content)
            param_idx += 1
        
        if embedding is not None:
            updates.append(f"embedding = ${param_idx}")
            params.append(embedding)
            param_idx += 1
        
        if has_deprecated is not None:
            updates.append(f"has_deprecated = ${param_idx}")
            params.append(has_deprecated)
            param_idx += 1
        
        if has_fixme is not None:
            updates.append(f"has_fixme = ${param_idx}")
            params.append(has_fixme)
            param_idx += 1
        
        if has_todo is not None:
            updates.append(f"has_todo = ${param_idx}")
            params.append(has_todo)
            param_idx += 1
        
        if has_security is not None:
            updates.append(f"has_security = ${param_idx}")
            params.append(has_security)
            param_idx += 1
        
        updates.append(f"updated_at = ${param_idx}")
        params.append(updated_at)
        param_idx += 1
        
        params.append(doc_id)
        
        if updates:
            query = f"""
                UPDATE code_documents
                SET {', '.join(updates)}
                WHERE id = ${param_idx}
            """
            async with self.pg_pool.acquire() as conn:
                await conn.execute(query, *params)
    
    async def _update_milvus(
        self,
        doc_id: str,
        content: Optional[str],
        embedding: Optional[List[float]],
        has_deprecated: Optional[bool],
        has_fixme: Optional[bool],
        has_todo: Optional[bool],
        has_security: Optional[bool],
        updated_at: datetime
    ):
        """更新 Milvus（删除后重新插入）"""
        # 查询现有记录
        expr = f'id == "{doc_id}"'
        results = self.milvus_collection.query(
            expr=expr,
            output_fields=["*"]
        )
        
        if not results:
            logger.warning(f"Document {doc_id} not found in Milvus")
            return
        
        existing = results[0]
        
        # 删除旧记录
        self.milvus_collection.delete(expr)
        
        # 准备新数据
        new_content = content if content is not None else existing["content"]
        new_embedding = embedding if embedding is not None else existing["embedding"]
        new_has_deprecated = has_deprecated if has_deprecated is not None else existing["has_deprecated"]
        new_has_fixme = has_fixme if has_fixme is not None else existing["has_fixme"]
        new_has_todo = has_todo if has_todo is not None else existing["has_todo"]
        new_has_security = has_security if has_security is not None else existing["has_security"]
        
        # 插入新记录
        entities = [
            [doc_id],
            [existing["workspace_id"]],
            [existing["file_path"]],
            [new_content[:65535]],
            [new_embedding],
            [existing["language"]],
            [new_has_deprecated],
            [new_has_fixme],
            [new_has_todo],
            [new_has_security],
            [existing["created_at"]],
            [int(updated_at.timestamp() * 1000)]
        ]
        
        self.milvus_collection.insert(entities)
    
    async def delete_document(self, doc_id: str):
        """
        删除文档（双写）
        """
        # 删除 PostgreSQL
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("DELETE FROM code_documents WHERE id = $1", doc_id)
            logger.debug(f"Deleted document {doc_id} from PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to delete from PostgreSQL: {e}")
            raise
        
        # 删除 Milvus（如果启用）
        if self.enable_milvus:
            try:
                expr = f'id == "{doc_id}"'
                self.milvus_collection.delete(expr)
                logger.debug(f"Deleted document {doc_id} from Milvus")
            except Exception as e:
                logger.error(f"Failed to delete from Milvus: {e}")
    
    def enable_milvus_write(self):
        """启用 Milvus 写入（灰度切流）"""
        self.enable_milvus = True
        logger.info("Milvus write enabled")
    
    def disable_milvus_write(self):
        """禁用 Milvus 写入"""
        self.enable_milvus = False
        logger.info("Milvus write disabled")
