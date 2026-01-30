"""
PostgreSQL 到 Milvus 数据迁移脚本
支持批量迁移、增量同步和数据验证
"""

import asyncio
import asyncpg
from pymilvus import connections, Collection, utility
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json
from tqdm import tqdm

from milvus_schema import (
    MilvusSchemaDefinition,
    COLLECTION_NAME,
    get_partition_name
)


logger = logging.getLogger(__name__)


class PostgresToMilvusMigration:
    """PostgreSQL 到 Milvus 数据迁移器"""
    
    def __init__(
        self,
        pg_config: Dict[str, Any],
        milvus_config: Dict[str, Any],
        batch_size: int = 1000
    ):
        """
        初始化迁移器
        
        Args:
            pg_config: PostgreSQL 连接配置
            milvus_config: Milvus 连接配置
            batch_size: 批量处理大小
        """
        self.pg_config = pg_config
        self.milvus_config = milvus_config
        self.batch_size = batch_size
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.milvus_collection: Optional[Collection] = None
        
    async def connect(self):
        """建立数据库连接"""
        # 连接 PostgreSQL
        self.pg_pool = await asyncpg.create_pool(**self.pg_config)
        logger.info("Connected to PostgreSQL")
        
        # 连接 Milvus
        connections.connect(
            alias="default",
            host=self.milvus_config["host"],
            port=self.milvus_config["port"]
        )
        logger.info("Connected to Milvus")
        
        # 初始化或获取集合
        await self._init_collection()
    
    async def _init_collection(self):
        """初始化 Milvus 集合"""
        if utility.has_collection(COLLECTION_NAME):
            logger.info(f"Collection {COLLECTION_NAME} already exists")
            self.milvus_collection = Collection(COLLECTION_NAME)
        else:
            logger.info(f"Creating collection {COLLECTION_NAME}")
            schema = MilvusSchemaDefinition.get_code_documents_schema()
            self.milvus_collection = Collection(
                name=COLLECTION_NAME,
                schema=schema
            )
            
            # 创建索引
            index_params = MilvusSchemaDefinition.get_index_params()
            self.milvus_collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            logger.info("Index created successfully")
        
        # 加载集合到内存
        self.milvus_collection.load()
        logger.info("Collection loaded into memory")
    
    async def migrate_all(self, workspace_ids: Optional[List[str]] = None):
        """
        迁移所有数据
        
        Args:
            workspace_ids: 要迁移的工作空间 ID 列表，None 表示迁移所有
        """
        logger.info("Starting full migration")
        
        # 获取总记录数
        total_count = await self._get_total_count(workspace_ids)
        logger.info(f"Total records to migrate: {total_count}")
        
        # 分批迁移
        offset = 0
        with tqdm(total=total_count, desc="Migrating") as pbar:
            while offset < total_count:
                batch = await self._fetch_batch(offset, workspace_ids)
                if not batch:
                    break
                
                await self._insert_batch(batch)
                offset += len(batch)
                pbar.update(len(batch))
        
        logger.info("Full migration completed")
    
    async def _get_total_count(self, workspace_ids: Optional[List[str]] = None) -> int:
        """获取总记录数"""
        async with self.pg_pool.acquire() as conn:
            if workspace_ids:
                query = """
                    SELECT COUNT(*) FROM code_documents
                    WHERE workspace_id = ANY($1)
                """
                count = await conn.fetchval(query, workspace_ids)
            else:
                query = "SELECT COUNT(*) FROM code_documents"
                count = await conn.fetchval(query)
            return count
    
    async def _fetch_batch(
        self,
        offset: int,
        workspace_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """从 PostgreSQL 获取一批数据"""
        async with self.pg_pool.acquire() as conn:
            if workspace_ids:
                query = """
                    SELECT 
                        id::text,
                        workspace_id::text,
                        file_path,
                        content,
                        embedding,
                        language,
                        has_deprecated,
                        has_fixme,
                        has_todo,
                        has_security,
                        EXTRACT(EPOCH FROM created_at)::bigint * 1000 as created_at,
                        EXTRACT(EPOCH FROM updated_at)::bigint * 1000 as updated_at
                    FROM code_documents
                    WHERE workspace_id = ANY($1)
                    ORDER BY id
                    LIMIT $2 OFFSET $3
                """
                rows = await conn.fetch(query, workspace_ids, self.batch_size, offset)
            else:
                query = """
                    SELECT 
                        id::text,
                        workspace_id::text,
                        file_path,
                        content,
                        embedding,
                        language,
                        has_deprecated,
                        has_fixme,
                        has_todo,
                        has_security,
                        EXTRACT(EPOCH FROM created_at)::bigint * 1000 as created_at,
                        EXTRACT(EPOCH FROM updated_at)::bigint * 1000 as updated_at
                    FROM code_documents
                    ORDER BY id
                    LIMIT $1 OFFSET $2
                """
                rows = await conn.fetch(query, self.batch_size, offset)
            
            return [dict(row) for row in rows]
    
    async def _insert_batch(self, batch: List[Dict[str, Any]]):
        """插入一批数据到 Milvus"""
        if not batch:
            return
        
        # 转换数据格式
        entities = self._convert_to_milvus_format(batch)
        
        # 插入数据
        try:
            self.milvus_collection.insert(entities)
            logger.debug(f"Inserted {len(batch)} records")
        except Exception as e:
            logger.error(f"Failed to insert batch: {e}")
            raise
    
    def _convert_to_milvus_format(self, batch: List[Dict[str, Any]]) -> List[List[Any]]:
        """将 PostgreSQL 数据转换为 Milvus 格式"""
        entities = [
            [row["id"] for row in batch],
            [row["workspace_id"] for row in batch],
            [row["file_path"] for row in batch],
            [row["content"][:65535] for row in batch],  # 截断到最大长度
            [row["embedding"] for row in batch],
            [row["language"] or "" for row in batch],
            [row["has_deprecated"] for row in batch],
            [row["has_fixme"] for row in batch],
            [row["has_todo"] for row in batch],
            [row["has_security"] for row in batch],
            [row["created_at"] for row in batch],
            [row["updated_at"] for row in batch],
        ]
        return entities
    
    async def migrate_incremental(self, since_timestamp: datetime):
        """
        增量迁移（迁移指定时间之后的数据）
        
        Args:
            since_timestamp: 起始时间戳
        """
        logger.info(f"Starting incremental migration since {since_timestamp}")
        
        async with self.pg_pool.acquire() as conn:
            query = """
                SELECT 
                    id::text,
                    workspace_id::text,
                    file_path,
                    content,
                    embedding,
                    language,
                    has_deprecated,
                    has_fixme,
                    has_todo,
                    has_security,
                    EXTRACT(EPOCH FROM created_at)::bigint * 1000 as created_at,
                    EXTRACT(EPOCH FROM updated_at)::bigint * 1000 as updated_at
                FROM code_documents
                WHERE updated_at > $1
                ORDER BY updated_at
            """
            rows = await conn.fetch(query, since_timestamp)
            
            batch = [dict(row) for row in rows]
            if batch:
                await self._insert_batch(batch)
                logger.info(f"Incremental migration completed: {len(batch)} records")
            else:
                logger.info("No new records to migrate")
    
    async def verify_migration(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        验证迁移结果
        
        Args:
            sample_size: 抽样验证的记录数
            
        Returns:
            验证结果统计
        """
        logger.info("Starting migration verification")
        
        # 获取 PostgreSQL 总数
        pg_count = await self._get_total_count()
        
        # 获取 Milvus 总数
        milvus_count = self.milvus_collection.num_entities
        
        # 抽样验证
        async with self.pg_pool.acquire() as conn:
            query = """
                SELECT id::text, embedding
                FROM code_documents
                ORDER BY RANDOM()
                LIMIT $1
            """
            samples = await conn.fetch(query, sample_size)
        
        matched = 0
        for sample in samples:
            # 在 Milvus 中查询
            expr = f'id == "{sample["id"]}"'
            results = self.milvus_collection.query(
                expr=expr,
                output_fields=["id", "embedding"]
            )
            
            if results and len(results) > 0:
                matched += 1
        
        verification_result = {
            "pg_count": pg_count,
            "milvus_count": milvus_count,
            "count_match": pg_count == milvus_count,
            "sample_size": sample_size,
            "sample_matched": matched,
            "sample_match_rate": matched / sample_size if sample_size > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Verification result: {json.dumps(verification_result, indent=2)}")
        return verification_result
    
    async def close(self):
        """关闭连接"""
        if self.pg_pool:
            await self.pg_pool.close()
            logger.info("PostgreSQL connection closed")
        
        if self.milvus_collection:
            self.milvus_collection.release()
            connections.disconnect("default")
            logger.info("Milvus connection closed")


async def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # PostgreSQL 配置
    pg_config = {
        "host": "localhost",
        "port": 5432,
        "database": "screenplay_db",
        "user": "postgres",
        "password": "password"
    }
    
    # Milvus 配置
    milvus_config = {
        "host": "localhost",
        "port": 19530
    }
    
    # 创建迁移器
    migrator = PostgresToMilvusMigration(
        pg_config=pg_config,
        milvus_config=milvus_config,
        batch_size=1000
    )
    
    try:
        # 连接数据库
        await migrator.connect()
        
        # 执行全量迁移
        await migrator.migrate_all()
        
        # 验证迁移结果
        verification = await migrator.verify_migration(sample_size=100)
        
        if verification["count_match"] and verification["sample_match_rate"] > 0.95:
            logger.info("Migration verification passed!")
        else:
            logger.warning("Migration verification failed!")
            
    finally:
        await migrator.close()


if __name__ == "__main__":
    asyncio.run(main())
