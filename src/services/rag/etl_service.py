"""ETL Service - 数据写入流水线

功能：
- 文件 Hash 查重（秒传）
- 文本读取与分块
- 向量化与存储
- Delete-then-Insert 原子更新
"""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.config import get_database_config, get_llm_config
from src.services.document_chunker import SmartChunker, create_smart_chunker
from src.services.database.postgres import PostgresService
from src.services.rag.document_repository import (
    DocumentRepository,
    DocumentFile,
    FileStatus
)
from src.services.llm.service import LLMService


logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """分块数据"""
    content: str
    index: int
    metadata: Dict[str, Any]


@dataclass
class IngestResult:
    """摄入结果"""
    status: str  # "success", "fast_upload", "failed"
    source_id: str
    chunk_count: int
    error_msg: Optional[str] = None


class ETLService:
    """ETL 服务 - 数据写入流水线

    流程：
    1. 计算文件 Hash
    2. 查重（秒传）
    3. 读取 + 分块
    4. 向量化
    5. 写入向量库
    """

    def __init__(
        self,
        document_repo: DocumentRepository = None,
        chunker: SmartChunker = None,
        llm_service: LLMService = None,
        vector_store: PostgresService = None,
        workspace_id: str = "default"
    ):
        """
        初始化 ETL 服务

        Args:
            document_repo: 文件 Repository
            chunker: 文档分块器
            llm_service: LLM 服务（用于 embedding）
            vector_store: 向量数据库服务
            workspace_id: 工作空间 ID
        """
        self.document_repo = document_repo
        self.chunker = chunker
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.workspace_id = workspace_id

        if self.document_repo is None:
            self.document_repo = DocumentRepository()

        if self.chunker is None:
            self.chunker = create_smart_chunker({
                'chunk_size': 500,
                'min_chunk_size': 50
            })

        if self.llm_service is None:
            llm_config = get_llm_config()
            self.llm_service = LLMService()

        if self.vector_store is None:
            db_config = get_database_config()
            self.vector_store = PostgresService({
                'host': db_config.host,
                'port': db_config.port,
                'database': db_config.database,
                'user': db_config.user,
                'password': db_config.password
            })

        logger.info("ETLService initialized")

    async def initialize(self) -> None:
        """初始化服务"""
        await self.document_repo.initialize()
        await self.vector_store.connect()
        logger.info("ETLService ready")

    async def close(self) -> None:
        """关闭服务"""
        await self.document_repo.close()
        await self.vector_store.disconnect()

    async def ingest(
        self,
        file_path: str,
        source_id: str = None,
        file_name: str = None
    ) -> IngestResult:
        """
        文档摄入

        Args:
            file_path: 文件路径
            source_id: 文档唯一标识（可选，自动生成）
            file_name: 原始文件名（可选，从路径提取）

        Returns:
            摄入结果
        """
        try:
            file_path_obj = Path(file_path)

            if file_name is None:
                file_name = file_path_obj.name

            if source_id is None:
                source_id = str(uuid.uuid4())

            logger.info(f"开始摄入文件: {file_name}, source_id={source_id}")

            file_hash = self._compute_hash(file_path)
            logger.info(f"文件 Hash: {file_hash[:16]}...")

            existing = self.document_repo.get_by_hash(file_hash)
            if existing and existing.status == FileStatus.INDEXED:
                logger.info(f"秒传成功: file_hash={file_hash[:16]}")
                return IngestResult(
                    status="fast_upload",
                    source_id=existing.id,
                    chunk_count=existing.chunk_count
                )

            if existing:
                self.document_repo.update_status(source_id, FileStatus.INDEXING)
            else:
                doc_file = DocumentFile(
                    id=source_id,
                    file_name=file_name,
                    file_hash=file_hash,
                    status=FileStatus.INDEXING
                )
                self.document_repo.create(doc_file)

            content = file_path_obj.read_text(encoding="utf-8")

            chunks = self._chunk(content)
            logger.info(f"分块完成: {len(chunks)} 个 chunk")

            embeddings = await self._embed_batch([c.content for c in chunks])

            await self._atomic_upsert(source_id, chunks, embeddings)

            self.document_repo.update_status(
                source_id,
                FileStatus.INDEXED,
                chunk_count=len(chunks)
            )

            logger.info(f"文档摄入成功: source_id={source_id}")
            return IngestResult(
                status="success",
                source_id=source_id,
                chunk_count=len(chunks)
            )

        except Exception as e:
            logger.error(f"文档摄入失败: {e}")
            self.document_repo.update_status(
                source_id or "",
                FileStatus.FAILED,
                error_msg=str(e)
            )
            return IngestResult(
                status="failed",
                source_id=source_id or "",
                chunk_count=0,
                error_msg=str(e)
            )

    def _compute_hash(self, file_path: str) -> str:
        """计算文件 SHA-256"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _chunk(self, content: str) -> List[Chunk]:
        """分块 - 二元法则"""
        chunks = self.chunker.chunk_text(content, "memory")
        return [
            Chunk(
                content=c.content,
                index=i,
                metadata={
                    'start_line': c.metadata.start_line,
                    'end_line': c.metadata.end_line
                }
            )
            for i, c in enumerate(chunks)
        ]

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量向量化"""
        embeddings = []
        batch_size = 50

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = await self.llm_service.embed(batch)
                embeddings.extend(response)
            except Exception as e:
                logger.error(f"Embedding 批次 {i//batch_size} 失败: {e}")
                for text in batch:
                    embedding = await self.llm_service.embed([text])
                    embeddings.append(embedding[0])

        return embeddings

    async def _atomic_upsert(
        self,
        source_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]]
    ) -> None:
        """原子写入 - 先删后插"""
        await self._delete_by_source_id(source_id)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            await self._upsert_chunk(source_id, chunk, embedding)

    async def _delete_by_source_id(self, source_id: str) -> None:
        """根据 source_id 删除数据"""
        await self.vector_store.execute(
            """
            DELETE FROM code_documents
            WHERE workspace_id = $1 AND file_path LIKE $2
            """,
            self.workspace_id,
            f"{source_id}_%"
        )

    async def _upsert_chunk(
        self,
        source_id: str,
        chunk: Chunk,
        embedding: List[float]
    ) -> None:
        """写入单个 chunk"""
        file_path = f"{source_id}_{chunk.index}"

        await self.vector_store.execute(
            """
            INSERT INTO code_documents (
                workspace_id, file_path, content, embedding,
                has_deprecated, has_fixme, has_todo, has_security
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (workspace_id, file_path)
            DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
            """,
            self.workspace_id,
            file_path,
            chunk.content,
            embedding,
            False, False, False, False
        )


async def create_etl_service(workspace_id: str = "default") -> ETLService:
    """工厂方法：创建 ETL 服务"""
    db_config = get_database_config()

    document_repo = DocumentRepository()
    await document_repo.initialize()

    pg_service = PostgresService({
        'host': db_config.host,
        'port': db_config.port,
        'database': db_config.database,
        'user': db_config.user,
        'password': db_config.password
    })
    await pg_service.connect()

    llm_service = LLMService()

    return ETLService(
        document_repo=document_repo,
        vector_store=pg_service,
        llm_service=llm_service,
        workspace_id=workspace_id
    )
