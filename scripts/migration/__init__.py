"""
数据库迁移工具包
PostgreSQL + pgvector -> Milvus 迁移工具集
"""

from .milvus_schema import MilvusSchemaDefinition, COLLECTION_NAME, get_partition_name
from .postgres_to_milvus import PostgresToMilvusMigration
from .dual_write_manager import DualWriteManager
from .gradual_cutover import GradualCutoverManager, CutoverScheduler
from .migration_monitor import (
    MigrationMonitor,
    MigrationThresholds,
    MetricsCollector,
    MigrationRecommendation
)
from .migration_orchestrator import MigrationOrchestrator, MigrationPhase

__all__ = [
    # Schema
    "MilvusSchemaDefinition",
    "COLLECTION_NAME",
    "get_partition_name",
    
    # Migration
    "PostgresToMilvusMigration",
    
    # Dual Write
    "DualWriteManager",
    
    # Cutover
    "GradualCutoverManager",
    "CutoverScheduler",
    
    # Monitoring
    "MigrationMonitor",
    "MigrationThresholds",
    "MetricsCollector",
    "MigrationRecommendation",
    
    # Orchestration
    "MigrationOrchestrator",
    "MigrationPhase",
]

__version__ = "1.0.0"
