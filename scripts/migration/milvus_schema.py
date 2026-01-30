"""
Milvus Collection Schema Definition
与 PostgreSQL 兼容的 Milvus 集合定义
"""

from pymilvus import CollectionSchema, FieldSchema, DataType
from typing import Dict, Any


class MilvusSchemaDefinition:
    """Milvus 集合 Schema 定义"""
    
    @staticmethod
    def get_code_documents_schema() -> CollectionSchema:
        """
        定义 code_documents 集合 schema
        与 PostgreSQL code_documents 表兼容
        """
        fields = [
            # 主键
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=100,
                is_primary=True,
                description="文档唯一标识符"
            ),
            
            # 工作空间 ID
            FieldSchema(
                name="workspace_id",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="工作空间标识符"
            ),
            
            # 文件路径
            FieldSchema(
                name="file_path",
                dtype=DataType.VARCHAR,
                max_length=500,
                description="文件路径"
            ),
            
            # 文件内容
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,  # Milvus VARCHAR 最大长度
                description="文件内容"
            ),
            
            # 向量嵌入（核心字段）
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=1536,  # OpenAI text-embedding-3-large 维度
                description="向量嵌入"
            ),
            
            # 编程语言
            FieldSchema(
                name="language",
                dtype=DataType.VARCHAR,
                max_length=50,
                description="编程语言"
            ),
            
            # 标记字段（用于混合搜索）
            FieldSchema(
                name="has_deprecated",
                dtype=DataType.BOOL,
                description="是否包含废弃标记"
            ),
            
            FieldSchema(
                name="has_fixme",
                dtype=DataType.BOOL,
                description="是否包含 FIXME 标记"
            ),
            
            FieldSchema(
                name="has_todo",
                dtype=DataType.BOOL,
                description="是否包含 TODO 标记"
            ),
            
            FieldSchema(
                name="has_security",
                dtype=DataType.BOOL,
                description="是否包含安全标记"
            ),
            
            # 时间戳（Unix 时间戳，毫秒）
            FieldSchema(
                name="created_at",
                dtype=DataType.INT64,
                description="创建时间戳（毫秒）"
            ),
            
            FieldSchema(
                name="updated_at",
                dtype=DataType.INT64,
                description="更新时间戳（毫秒）"
            ),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="代码文档集合，包含向量嵌入和元数据",
            enable_dynamic_field=True  # 允许动态字段以支持未来扩展
        )
        
        return schema
    
    @staticmethod
    def get_index_params() -> Dict[str, Any]:
        """
        获取索引参数配置
        使用 HNSW 索引以获得最佳性能
        """
        return {
            "metric_type": "COSINE",  # 余弦相似度
            "index_type": "HNSW",     # HNSW 索引
            "params": {
                "M": 16,              # HNSW 参数：每个节点的最大连接数
                "efConstruction": 256  # 构建时的搜索深度
            }
        }
    
    @staticmethod
    def get_search_params() -> Dict[str, Any]:
        """
        获取搜索参数配置
        """
        return {
            "metric_type": "COSINE",
            "params": {
                "ef": 128  # 搜索时的深度
            }
        }
    
    @staticmethod
    def get_collection_properties() -> Dict[str, Any]:
        """
        获取集合属性配置
        """
        return {
            "collection.ttl.seconds": 0,  # 不自动删除数据
            "collection.autocompaction.enabled": "true",  # 启用自动压缩
        }


# 集合名称常量
COLLECTION_NAME = "code_documents"

# 分区策略（按工作空间分区）
PARTITION_PREFIX = "workspace_"


def get_partition_name(workspace_id: str) -> str:
    """根据工作空间 ID 生成分区名称"""
    return f"{PARTITION_PREFIX}{workspace_id}"
