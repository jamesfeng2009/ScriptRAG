"""
迁移编排器
协调整个迁移流程：监控 -> 迁移 -> 双写 -> 灰度切流 -> 完全切换
"""

import asyncio
import asyncpg
from pymilvus import connections, Collection
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import json

from migration_monitor import MigrationMonitor, MigrationRecommendation, MigrationThresholds
from postgres_to_milvus import PostgresToMilvusMigration
from dual_write_manager import DualWriteManager
from gradual_cutover import GradualCutoverManager, CutoverScheduler


logger = logging.getLogger(__name__)


class MigrationPhase:
    """迁移阶段枚举"""
    MONITORING = "monitoring"           # 监控阶段
    PREPARATION = "preparation"         # 准备阶段
    FULL_MIGRATION = "full_migration"   # 全量迁移
    DUAL_WRITE = "dual_write"           # 双写阶段
    GRADUAL_CUTOVER = "gradual_cutover" # 灰度切流
    COMPLETED = "completed"             # 完成


class MigrationOrchestrator:
    """迁移编排器"""
    
    def __init__(
        self,
        pg_config: Dict[str, Any],
        milvus_config: Dict[str, Any]
    ):
        """
        初始化迁移编排器
        
        Args:
            pg_config: PostgreSQL 连接配置
            milvus_config: Milvus 连接配置
        """
        self.pg_config = pg_config
        self.milvus_config = milvus_config
        self.current_phase = MigrationPhase.MONITORING
        
        # 组件
        self.monitor: Optional[MigrationMonitor] = None
        self.migrator: Optional[PostgresToMilvusMigration] = None
        self.dual_write_manager: Optional[DualWriteManager] = None
        self.cutover_manager: Optional[GradualCutoverManager] = None
        
        # 状态
        self.migration_started = False
        self.migration_completed = False
        
    async def initialize(self):
        """初始化所有组件"""
        logger.info("Initializing migration orchestrator")
        
        # 初始化监控器
        self.monitor = MigrationMonitor(
            pg_config=self.pg_config,
            alert_callbacks=[self._handle_migration_alert]
        )
        await self.monitor.connect()
        
        logger.info("Migration orchestrator initialized")
    
    async def _handle_migration_alert(self, alert_data: Dict[str, Any]):
        """
        处理迁移告警
        
        Args:
            alert_data: 告警数据
        """
        logger.warning(f"Migration alert received: {alert_data['type']}")
        
        # 如果还在监控阶段，且收到告警，建议开始迁移
        if self.current_phase == MigrationPhase.MONITORING and not self.migration_started:
            metrics = alert_data["metrics"]
            recommendation = MigrationRecommendation.analyze_metrics(metrics)
            
            logger.info(f"Migration recommendation: {json.dumps(recommendation, indent=2)}")
            
            if recommendation["priority"] in ["critical", "high"]:
                logger.warning("Migration is recommended! Please run start_migration() to begin.")
    
    async def start_monitoring(self):
        """启动监控"""
        logger.info("Starting migration monitoring")
        self.current_phase = MigrationPhase.MONITORING
        
        # 启动后台监控任务
        asyncio.create_task(self.monitor.start_monitoring(interval_seconds=60))
    
    async def start_migration(self):
        """开始迁移流程"""
        if self.migration_started:
            logger.warning("Migration already started")
            return
        
        logger.info("=" * 80)
        logger.info("STARTING MIGRATION PROCESS")
        logger.info("=" * 80)
        
        self.migration_started = True
        
        try:
            # 阶段 1: 准备
            await self._phase_preparation()
            
            # 阶段 2: 全量迁移
            await self._phase_full_migration()
            
            # 阶段 3: 启用双写
            await self._phase_enable_dual_write()
            
            # 阶段 4: 灰度切流
            await self._phase_gradual_cutover()
            
            # 阶段 5: 完成
            await self._phase_completion()
            
            self.migration_completed = True
            logger.info("=" * 80)
            logger.info("MIGRATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    async def _phase_preparation(self):
        """阶段 1: 准备"""
        logger.info("Phase 1: Preparation")
        self.current_phase = MigrationPhase.PREPARATION
        
        # 收集当前指标
        metrics = await self.monitor.collect_metrics()
        logger.info(f"Current metrics: {json.dumps(metrics, indent=2)}")
        
        # 生成迁移建议
        recommendation = MigrationRecommendation.analyze_metrics(metrics)
        logger.info(f"Migration recommendation: {json.dumps(recommendation, indent=2)}")
        
        # 初始化迁移器
        self.migrator = PostgresToMilvusMigration(
            pg_config=self.pg_config,
            milvus_config=self.milvus_config,
            batch_size=1000
        )
        await self.migrator.connect()
        
        logger.info("Preparation phase completed")
    
    async def _phase_full_migration(self):
        """阶段 2: 全量迁移"""
        logger.info("Phase 2: Full Migration")
        self.current_phase = MigrationPhase.FULL_MIGRATION
        
        # 执行全量迁移
        await self.migrator.migrate_all()
        
        # 验证迁移结果
        verification = await self.migrator.verify_migration(sample_size=100)
        logger.info(f"Migration verification: {json.dumps(verification, indent=2)}")
        
        if not verification["count_match"]:
            raise Exception("Migration verification failed: count mismatch")
        
        if verification["sample_match_rate"] < 0.95:
            raise Exception(f"Migration verification failed: sample match rate too low ({verification['sample_match_rate']:.2%})")
        
        logger.info("Full migration phase completed")
    
    async def _phase_enable_dual_write(self):
        """阶段 3: 启用双写"""
        logger.info("Phase 3: Enable Dual Write")
        self.current_phase = MigrationPhase.DUAL_WRITE
        
        # 初始化双写管理器
        pg_pool = await asyncpg.create_pool(**self.pg_config)
        milvus_collection = Collection("code_documents")
        
        self.dual_write_manager = DualWriteManager(
            pg_pool=pg_pool,
            milvus_collection=milvus_collection,
            enable_milvus=True  # 启用 Milvus 写入
        )
        
        logger.info("Dual write enabled")
        logger.info("All new writes will go to both PostgreSQL and Milvus")
        
        # 等待一段时间以确保双写稳定
        logger.info("Waiting 5 minutes to ensure dual write stability...")
        await asyncio.sleep(300)
        
        logger.info("Dual write phase completed")
    
    async def _phase_gradual_cutover(self):
        """阶段 4: 灰度切流"""
        logger.info("Phase 4: Gradual Cutover")
        self.current_phase = MigrationPhase.GRADUAL_CUTOVER
        
        # 初始化切流管理器
        pg_pool = await asyncpg.create_pool(**self.pg_config)
        milvus_collection = Collection("code_documents")
        
        self.cutover_manager = GradualCutoverManager(
            pg_pool=pg_pool,
            milvus_collection=milvus_collection
        )
        
        # 设置切流策略
        self.cutover_manager.set_cutover_strategy("workspace")
        
        # 创建切流调度器
        scheduler = CutoverScheduler(self.cutover_manager)
        
        # 执行渐进式切流
        logger.info("Starting gradual cutover process")
        logger.info("This will take several days to complete")
        
        # 注意：在生产环境中，这个过程会持续数天
        # 这里为了演示，可以缩短时间或手动控制
        await scheduler.execute_gradual_cutover()
        
        logger.info("Gradual cutover phase completed")
    
    async def _phase_completion(self):
        """阶段 5: 完成"""
        logger.info("Phase 5: Completion")
        self.current_phase = MigrationPhase.COMPLETED
        
        # 最终验证
        logger.info("Performing final verification")
        
        # 对比 PostgreSQL 和 Milvus 的搜索结果
        test_embedding = [0.1] * 1536  # 测试向量
        comparison = await self.cutover_manager.compare_results(
            workspace_id="test-workspace",
            query_embedding=test_embedding,
            top_k=10
        )
        
        logger.info(f"Final comparison: {json.dumps(comparison, indent=2)}")
        
        # 禁用 PostgreSQL 写入（可选）
        logger.info("Migration completed. PostgreSQL can now be used as read-only backup.")
        
        logger.info("Completion phase finished")
    
    async def get_status(self) -> Dict[str, Any]:
        """
        获取迁移状态
        
        Returns:
            状态信息
        """
        status = {
            "current_phase": self.current_phase,
            "migration_started": self.migration_started,
            "migration_completed": self.migration_completed,
            "timestamp": datetime.now().isoformat()
        }
        
        # 添加当前指标
        if self.monitor:
            try:
                metrics = await self.monitor.collect_metrics()
                status["metrics"] = metrics
                
                # 添加迁移建议
                recommendation = MigrationRecommendation.analyze_metrics(metrics)
                status["recommendation"] = recommendation
            except Exception as e:
                logger.error(f"Failed to collect metrics: {e}")
        
        return status
    
    async def close(self):
        """关闭所有连接"""
        logger.info("Closing migration orchestrator")
        
        if self.monitor:
            self.monitor.stop_monitoring()
            await self.monitor.close()
        
        if self.migrator:
            await self.migrator.close()
        
        logger.info("Migration orchestrator closed")


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
    
    # 创建编排器
    orchestrator = MigrationOrchestrator(
        pg_config=pg_config,
        milvus_config=milvus_config
    )
    
    try:
        # 初始化
        await orchestrator.initialize()
        
        # 启动监控
        await orchestrator.start_monitoring()
        
        # 等待用户决定开始迁移
        logger.info("Monitoring started. Waiting for migration trigger...")
        logger.info("To start migration manually, call orchestrator.start_migration()")
        
        # 在实际使用中，可以通过 API 或命令行触发迁移
        # 这里为了演示，等待 10 秒后自动开始
        await asyncio.sleep(10)
        
        # 获取状态
        status = await orchestrator.get_status()
        logger.info(f"Current status: {json.dumps(status, indent=2)}")
        
        # 如果建议迁移，则开始迁移
        if status.get("recommendation", {}).get("priority") in ["critical", "high"]:
            logger.info("Starting migration based on recommendation")
            await orchestrator.start_migration()
        else:
            logger.info("No immediate migration needed. Continue monitoring.")
            # 保持监控运行
            await asyncio.sleep(3600)  # 运行 1 小时
        
    except KeyboardInterrupt:
        logger.info("Orchestrator interrupted by user")
    finally:
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(main())
