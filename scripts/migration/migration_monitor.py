"""
迁移触发条件监控器
监控向量数量、搜索 QPS、P99 延迟、存储大小等指标
当达到阈值时触发迁移告警
"""

import asyncio
import asyncpg
from pymilvus import connections, utility
from typing import Dict, Any, Optional, List, Callable
import logging
from datetime import datetime, timedelta
import time
from collections import deque
import psutil
import json


logger = logging.getLogger(__name__)


class MigrationThresholds:
    """迁移触发阈值配置"""
    
    # 向量数量阈值（100 万）
    VECTOR_COUNT = 1_000_000
    
    # 搜索 QPS 阈值（100）
    SEARCH_QPS = 100
    
    # P99 延迟阈值（500ms）
    P99_LATENCY_MS = 500
    
    # 存储大小阈值（100GB）
    STORAGE_SIZE_GB = 100


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, window_size: int = 60):
        """
        初始化指标收集器
        
        Args:
            window_size: 滑动窗口大小（秒）
        """
        self.window_size = window_size
        self.search_timestamps = deque(maxlen=10000)  # 搜索时间戳队列
        self.latencies = deque(maxlen=10000)  # 延迟队列
        
    def record_search(self, latency_ms: float):
        """
        记录一次搜索
        
        Args:
            latency_ms: 搜索延迟（毫秒）
        """
        now = time.time()
        self.search_timestamps.append(now)
        self.latencies.append(latency_ms)
    
    def get_qps(self) -> float:
        """
        计算当前 QPS
        
        Returns:
            每秒查询数
        """
        now = time.time()
        cutoff = now - self.window_size
        
        # 统计窗口内的查询数
        recent_searches = sum(1 for ts in self.search_timestamps if ts >= cutoff)
        
        return recent_searches / self.window_size
    
    def get_p99_latency(self) -> float:
        """
        计算 P99 延迟
        
        Returns:
            P99 延迟（毫秒）
        """
        if not self.latencies:
            return 0.0
        
        sorted_latencies = sorted(self.latencies)
        p99_index = int(len(sorted_latencies) * 0.99)
        
        return sorted_latencies[p99_index] if p99_index < len(sorted_latencies) else sorted_latencies[-1]
    
    def get_avg_latency(self) -> float:
        """
        计算平均延迟
        
        Returns:
            平均延迟（毫秒）
        """
        if not self.latencies:
            return 0.0
        
        return sum(self.latencies) / len(self.latencies)


class MigrationMonitor:
    """迁移监控器"""
    
    def __init__(
        self,
        pg_config: Dict[str, Any],
        alert_callbacks: Optional[List[Callable]] = None
    ):
        """
        初始化迁移监控器
        
        Args:
            pg_config: PostgreSQL 连接配置
            alert_callbacks: 告警回调函数列表
        """
        self.pg_config = pg_config
        self.alert_callbacks = alert_callbacks or []
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.metrics_collector = MetricsCollector()
        self.monitoring = False
        
        # 告警状态（避免重复告警）
        self.alert_states = {
            "vector_count": False,
            "search_qps": False,
            "p99_latency": False,
            "storage_size": False
        }
    
    async def connect(self):
        """建立数据库连接"""
        self.pg_pool = await asyncpg.create_pool(**self.pg_config)
        logger.info("Connected to PostgreSQL for monitoring")
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """
        启动监控
        
        Args:
            interval_seconds: 监控间隔（秒）
        """
        self.monitoring = True
        logger.info(f"Starting migration monitoring (interval: {interval_seconds}s)")
        
        while self.monitoring:
            try:
                # 收集指标
                metrics = await self.collect_metrics()
                
                # 检查阈值
                await self.check_thresholds(metrics)
                
                # 记录指标
                logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
                
                # 等待下一次检查
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        logger.info("Stopping migration monitoring")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """
        收集所有监控指标
        
        Returns:
            指标字典
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "vector_count": await self._get_vector_count(),
            "search_qps": self.metrics_collector.get_qps(),
            "p99_latency_ms": self.metrics_collector.get_p99_latency(),
            "avg_latency_ms": self.metrics_collector.get_avg_latency(),
            "storage_size_gb": await self._get_storage_size_gb(),
        }
        
        return metrics
    
    async def _get_vector_count(self) -> int:
        """获取向量数量"""
        async with self.pg_pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM code_documents")
            return count
    
    async def _get_storage_size_gb(self) -> float:
        """获取存储大小（GB）"""
        async with self.pg_pool.acquire() as conn:
            query = """
                SELECT pg_total_relation_size('code_documents') / (1024.0 * 1024.0 * 1024.0) AS size_gb
            """
            size_gb = await conn.fetchval(query)
            return float(size_gb) if size_gb else 0.0
    
    async def check_thresholds(self, metrics: Dict[str, Any]):
        """
        检查是否达到迁移阈值
        
        Args:
            metrics: 当前指标
        """
        # 检查向量数量
        if metrics["vector_count"] >= MigrationThresholds.VECTOR_COUNT:
            if not self.alert_states["vector_count"]:
                await self._trigger_alert(
                    "vector_count",
                    f"Vector count ({metrics['vector_count']}) exceeded threshold ({MigrationThresholds.VECTOR_COUNT})",
                    metrics
                )
                self.alert_states["vector_count"] = True
        else:
            self.alert_states["vector_count"] = False
        
        # 检查搜索 QPS
        if metrics["search_qps"] >= MigrationThresholds.SEARCH_QPS:
            if not self.alert_states["search_qps"]:
                await self._trigger_alert(
                    "search_qps",
                    f"Search QPS ({metrics['search_qps']:.2f}) exceeded threshold ({MigrationThresholds.SEARCH_QPS})",
                    metrics
                )
                self.alert_states["search_qps"] = True
        else:
            self.alert_states["search_qps"] = False
        
        # 检查 P99 延迟
        if metrics["p99_latency_ms"] >= MigrationThresholds.P99_LATENCY_MS:
            if not self.alert_states["p99_latency"]:
                await self._trigger_alert(
                    "p99_latency",
                    f"P99 latency ({metrics['p99_latency_ms']:.2f}ms) exceeded threshold ({MigrationThresholds.P99_LATENCY_MS}ms)",
                    metrics
                )
                self.alert_states["p99_latency"] = True
        else:
            self.alert_states["p99_latency"] = False
        
        # 检查存储大小
        if metrics["storage_size_gb"] >= MigrationThresholds.STORAGE_SIZE_GB:
            if not self.alert_states["storage_size"]:
                await self._trigger_alert(
                    "storage_size",
                    f"Storage size ({metrics['storage_size_gb']:.2f}GB) exceeded threshold ({MigrationThresholds.STORAGE_SIZE_GB}GB)",
                    metrics
                )
                self.alert_states["storage_size"] = True
        else:
            self.alert_states["storage_size"] = False
    
    async def _trigger_alert(
        self,
        alert_type: str,
        message: str,
        metrics: Dict[str, Any]
    ):
        """
        触发告警
        
        Args:
            alert_type: 告警类型
            message: 告警消息
            metrics: 当前指标
        """
        logger.warning(f"MIGRATION ALERT [{alert_type}]: {message}")
        
        alert_data = {
            "type": alert_type,
            "message": message,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # 调用所有告警回调
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    async def close(self):
        """关闭连接"""
        if self.pg_pool:
            await self.pg_pool.close()
            logger.info("PostgreSQL monitoring connection closed")


class MigrationRecommendation:
    """迁移建议生成器"""
    
    @staticmethod
    def analyze_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析指标并生成迁移建议
        
        Args:
            metrics: 当前指标
            
        Returns:
            分析结果和建议
        """
        recommendations = []
        urgency_score = 0
        
        # 分析向量数量
        vector_ratio = metrics["vector_count"] / MigrationThresholds.VECTOR_COUNT
        if vector_ratio >= 1.0:
            recommendations.append({
                "metric": "vector_count",
                "status": "critical",
                "message": f"向量数量已达到阈值的 {vector_ratio:.1%}，建议立即迁移"
            })
            urgency_score += 40
        elif vector_ratio >= 0.8:
            recommendations.append({
                "metric": "vector_count",
                "status": "warning",
                "message": f"向量数量已达到阈值的 {vector_ratio:.1%}，建议准备迁移"
            })
            urgency_score += 20
        
        # 分析搜索 QPS
        qps_ratio = metrics["search_qps"] / MigrationThresholds.SEARCH_QPS
        if qps_ratio >= 1.0:
            recommendations.append({
                "metric": "search_qps",
                "status": "critical",
                "message": f"搜索 QPS 已达到阈值的 {qps_ratio:.1%}，建议立即迁移"
            })
            urgency_score += 30
        elif qps_ratio >= 0.8:
            recommendations.append({
                "metric": "search_qps",
                "status": "warning",
                "message": f"搜索 QPS 已达到阈值的 {qps_ratio:.1%}，建议准备迁移"
            })
            urgency_score += 15
        
        # 分析 P99 延迟
        latency_ratio = metrics["p99_latency_ms"] / MigrationThresholds.P99_LATENCY_MS
        if latency_ratio >= 1.0:
            recommendations.append({
                "metric": "p99_latency",
                "status": "critical",
                "message": f"P99 延迟已达到阈值的 {latency_ratio:.1%}，建议立即迁移"
            })
            urgency_score += 20
        elif latency_ratio >= 0.8:
            recommendations.append({
                "metric": "p99_latency",
                "status": "warning",
                "message": f"P99 延迟已达到阈值的 {latency_ratio:.1%}，建议准备迁移"
            })
            urgency_score += 10
        
        # 分析存储大小
        storage_ratio = metrics["storage_size_gb"] / MigrationThresholds.STORAGE_SIZE_GB
        if storage_ratio >= 1.0:
            recommendations.append({
                "metric": "storage_size",
                "status": "critical",
                "message": f"存储大小已达到阈值的 {storage_ratio:.1%}，建议立即迁移"
            })
            urgency_score += 10
        elif storage_ratio >= 0.8:
            recommendations.append({
                "metric": "storage_size",
                "status": "warning",
                "message": f"存储大小已达到阈值的 {storage_ratio:.1%}，建议准备迁移"
            })
            urgency_score += 5
        
        # 生成总体建议
        if urgency_score >= 50:
            overall_recommendation = "立即开始迁移到 Milvus"
            priority = "critical"
        elif urgency_score >= 30:
            overall_recommendation = "建议在 1-2 周内开始迁移准备"
            priority = "high"
        elif urgency_score >= 15:
            overall_recommendation = "建议在 1 个月内开始迁移准备"
            priority = "medium"
        else:
            overall_recommendation = "暂时无需迁移，继续监控"
            priority = "low"
        
        return {
            "urgency_score": urgency_score,
            "priority": priority,
            "overall_recommendation": overall_recommendation,
            "detailed_recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }


# 告警回调示例
async def send_email_alert(alert_data: Dict[str, Any]):
    """发送邮件告警（示例）"""
    logger.info(f"Sending email alert: {alert_data['message']}")
    # 实际实现需要集成邮件服务


async def send_slack_alert(alert_data: Dict[str, Any]):
    """发送 Slack 告警（示例）"""
    logger.info(f"Sending Slack alert: {alert_data['message']}")
    # 实际实现需要集成 Slack API


async def log_to_database(alert_data: Dict[str, Any]):
    """记录告警到数据库（示例）"""
    logger.info(f"Logging alert to database: {alert_data['message']}")
    # 实际实现需要写入数据库


async def main():
    """主函数 - 监控示例"""
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
    
    # 创建监控器
    monitor = MigrationMonitor(
        pg_config=pg_config,
        alert_callbacks=[
            send_email_alert,
            send_slack_alert,
            log_to_database
        ]
    )
    
    try:
        # 连接数据库
        await monitor.connect()
        
        # 启动监控（每 60 秒检查一次）
        await monitor.start_monitoring(interval_seconds=60)
        
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    finally:
        monitor.stop_monitoring()
        await monitor.close()


if __name__ == "__main__":
    asyncio.run(main())
