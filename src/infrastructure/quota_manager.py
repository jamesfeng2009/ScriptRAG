"""配额管理 - 资源配额控制与执行

本模块提供配额管理功能：
1. API 调用速率限制
2. 资源配额跟踪
3. 配额执行
4. 配额警告和通知
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
import asyncpg


logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """配额跟踪的资源类型"""
    API_CALL = "api_call"
    LLM_CALL = "llm_call"
    EMBEDDING_GENERATION = "embedding_generation"
    VECTOR_SEARCH = "vector_search"
    SCREENPLAY_GENERATION = "screenplay_generation"
    STORAGE_BYTES = "storage_bytes"


class QuotaPeriod(str, Enum):
    """配额周期类型"""
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"


class QuotaLimit:
    """配额限制配置"""
    
    def __init__(
        self,
        resource_type: ResourceType,
        limit: int,
        period: QuotaPeriod = QuotaPeriod.DAILY,
        alert_threshold: float = 0.8
    ):
        self.resource_type = resource_type
        self.limit = limit
        self.period = period
        self.alert_threshold = alert_threshold  # Alert when usage reaches this percentage


# Default quota limits by plan
DEFAULT_QUOTA_LIMITS = {
    "free": {
        ResourceType.API_CALL: QuotaLimit(ResourceType.API_CALL, 100, QuotaPeriod.DAILY),
        ResourceType.LLM_CALL: QuotaLimit(ResourceType.LLM_CALL, 500, QuotaPeriod.DAILY),
        ResourceType.EMBEDDING_GENERATION: QuotaLimit(ResourceType.EMBEDDING_GENERATION, 1000, QuotaPeriod.DAILY),
        ResourceType.VECTOR_SEARCH: QuotaLimit(ResourceType.VECTOR_SEARCH, 1000, QuotaPeriod.DAILY),
        ResourceType.SCREENPLAY_GENERATION: QuotaLimit(ResourceType.SCREENPLAY_GENERATION, 10, QuotaPeriod.DAILY),
        ResourceType.STORAGE_BYTES: QuotaLimit(ResourceType.STORAGE_BYTES, 1024 * 1024 * 100, QuotaPeriod.MONTHLY),  # 100MB
    },
    "basic": {
        ResourceType.API_CALL: QuotaLimit(ResourceType.API_CALL, 1000, QuotaPeriod.DAILY),
        ResourceType.LLM_CALL: QuotaLimit(ResourceType.LLM_CALL, 5000, QuotaPeriod.DAILY),
        ResourceType.EMBEDDING_GENERATION: QuotaLimit(ResourceType.EMBEDDING_GENERATION, 10000, QuotaPeriod.DAILY),
        ResourceType.VECTOR_SEARCH: QuotaLimit(ResourceType.VECTOR_SEARCH, 10000, QuotaPeriod.DAILY),
        ResourceType.SCREENPLAY_GENERATION: QuotaLimit(ResourceType.SCREENPLAY_GENERATION, 100, QuotaPeriod.DAILY),
        ResourceType.STORAGE_BYTES: QuotaLimit(ResourceType.STORAGE_BYTES, 1024 * 1024 * 1024, QuotaPeriod.MONTHLY),  # 1GB
    },
    "pro": {
        ResourceType.API_CALL: QuotaLimit(ResourceType.API_CALL, 10000, QuotaPeriod.DAILY),
        ResourceType.LLM_CALL: QuotaLimit(ResourceType.LLM_CALL, 50000, QuotaPeriod.DAILY),
        ResourceType.EMBEDDING_GENERATION: QuotaLimit(ResourceType.EMBEDDING_GENERATION, 100000, QuotaPeriod.DAILY),
        ResourceType.VECTOR_SEARCH: QuotaLimit(ResourceType.VECTOR_SEARCH, 100000, QuotaPeriod.DAILY),
        ResourceType.SCREENPLAY_GENERATION: QuotaLimit(ResourceType.SCREENPLAY_GENERATION, 1000, QuotaPeriod.DAILY),
        ResourceType.STORAGE_BYTES: QuotaLimit(ResourceType.STORAGE_BYTES, 1024 * 1024 * 1024 * 10, QuotaPeriod.MONTHLY),  # 10GB
    },
    "enterprise": {
        ResourceType.API_CALL: QuotaLimit(ResourceType.API_CALL, 100000, QuotaPeriod.DAILY),
        ResourceType.LLM_CALL: QuotaLimit(ResourceType.LLM_CALL, 500000, QuotaPeriod.DAILY),
        ResourceType.EMBEDDING_GENERATION: QuotaLimit(ResourceType.EMBEDDING_GENERATION, 1000000, QuotaPeriod.DAILY),
        ResourceType.VECTOR_SEARCH: QuotaLimit(ResourceType.VECTOR_SEARCH, 1000000, QuotaPeriod.DAILY),
        ResourceType.SCREENPLAY_GENERATION: QuotaLimit(ResourceType.SCREENPLAY_GENERATION, 10000, QuotaPeriod.DAILY),
        ResourceType.STORAGE_BYTES: QuotaLimit(ResourceType.STORAGE_BYTES, 1024 * 1024 * 1024 * 100, QuotaPeriod.MONTHLY),  # 100GB
    }
}


class QuotaExceededException(Exception):
    """超出配额异常"""
    
    def __init__(self, resource_type: ResourceType, current: int, limit: int):
        self.resource_type = resource_type
        self.current = current
        self.limit = limit
        super().__init__(
            f"配额超出 {resource_type}: {current}/{limit}"
        )


class QuotaManager:
    """配额管理和执行服务"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        """
        初始化配额管理器
        
        Args:
            db_pool: AsyncPG 连接池
        """
        self.db_pool = db_pool
    
    def _get_period_start(self, period: QuotaPeriod) -> datetime:
        """
        获取当前周期的开始时间
        
        Args:
            period: 配额周期
            
        Returns:
            周期的开始时间
        """
        now = datetime.utcnow()
        
        if period == QuotaPeriod.HOURLY:
            return now.replace(minute=0, second=0, microsecond=0)
        elif period == QuotaPeriod.DAILY:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == QuotaPeriod.MONTHLY:
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        return now
    
    async def get_quota_limits(self, tenant_id: str, plan: str) -> Dict[ResourceType, QuotaLimit]:
        """
        获取租户的配额限制
        
        Args:
            tenant_id: 租户 ID
            plan: 订阅计划
            
        Returns:
            资源类型到配额限制的字典
        """
        # Get default limits for plan
        limits = DEFAULT_QUOTA_LIMITS.get(plan, DEFAULT_QUOTA_LIMITS["free"]).copy()
        
        # Check for custom tenant limits in database
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT settings FROM tenants WHERE id = $1
                    """,
                    tenant_id
                )
                
                if row and row['settings']:
                    custom_limits = row['settings'].get('quota_limits', {})
                    
                    # Override with custom limits
                    for resource_type_str, custom_limit in custom_limits.items():
                        try:
                            resource_type = ResourceType(resource_type_str)
                            if resource_type in limits:
                                limits[resource_type].limit = custom_limit.get('limit', limits[resource_type].limit)
                                limits[resource_type].alert_threshold = custom_limit.get(
                                    'alert_threshold',
                                    limits[resource_type].alert_threshold
                                )
                        except ValueError:
                            logger.warning(f"Unknown resource type in custom limits: {resource_type_str}")
        except Exception as e:
            logger.error(f"Failed to get custom quota limits: {str(e)}")
        
        return limits
    
    async def get_current_usage(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        period: QuotaPeriod
    ) -> int:
        """
        获取资源类型的当前使用量
        
        Args:
            tenant_id: 租户 ID
            resource_type: 资源类型
            period: 配额周期
            
        Returns:
            当前使用量
        """
        try:
            period_start = self._get_period_start(period)
            
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT COALESCE(SUM(usage_count), 0) as total_usage
                    FROM quota_usage
                    WHERE tenant_id = $1
                        AND resource_type = $2
                        AND time >= $3
                    """,
                    tenant_id,
                    resource_type.value,
                    period_start
                )
                
                return int(row['total_usage']) if row else 0
        except Exception as e:
            logger.error(f"获取当前使用量失败: {str(e)}")
            return 0
    
    async def increment_usage(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        count: int = 1
    ) -> None:
        """
        增加资源类型的使用量
        
        Args:
            tenant_id: 租户 ID
            resource_type: 资源类型
            count: 要增加的使用量
        """
        try:
            now = datetime.utcnow()
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO quota_usage (time, tenant_id, resource_type, usage_count)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (time, tenant_id, resource_type)
                    DO UPDATE SET usage_count = quota_usage.usage_count + $4
                    """,
                    now,
                    tenant_id,
                    resource_type.value,
                    count
                )
        except Exception as e:
            logger.error(f"Failed to increment usage: {str(e)}")
    
    async def check_quota(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        plan: str,
        count: int = 1
    ) -> bool:
        """
        检查配额是否允许请求的使用量
        
        Args:
            tenant_id: 租户 ID
            resource_type: 资源类型
            plan: 订阅计划
            count: 请求的使用量
            
        Returns:
            配额允许返回 True，否则返回 False
            
        Raises:
            QuotaExceededException: 配额超出时抛出
        """
        try:
            # 获取配额限制
            limits = await self.get_quota_limits(tenant_id, plan)
            
            if resource_type not in limits:
                # 未定义限制，允许
                return True
            
            quota_limit = limits[resource_type]
            
            # 获取当前使用量
            current_usage = await self.get_current_usage(
                tenant_id,
                resource_type,
                quota_limit.period
            )
            
            # 检查增加使用量是否会超出限制
            if current_usage + count > quota_limit.limit:
                raise QuotaExceededException(
                    resource_type,
                    current_usage + count,
                    quota_limit.limit
                )
            
            # 检查是否接近警告阈值
            usage_percentage = (current_usage + count) / quota_limit.limit
            if usage_percentage >= quota_limit.alert_threshold:
                await self._send_quota_alert(
                    tenant_id,
                    resource_type,
                    current_usage + count,
                    quota_limit.limit,
                    usage_percentage
                )
            
            return True
        except QuotaExceededException:
            raise
        except Exception as e:
            logger.error(f"检查配额失败: {str(e)}")
            # 出错时允许操作（故障开放）
            return True
    
    async def enforce_quota(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        plan: str,
        count: int = 1
    ) -> None:
        """
        执行配额检查并在允许时增加使用量
        
        Args:
            tenant_id: 租户 ID
            resource_type: 资源类型
            plan: 订阅计划
            count: 使用量
            
        Raises:
            QuotaExceededException: 配额超出时抛出
        """
        # 检查配额
        await self.check_quota(tenant_id, resource_type, plan, count)
        
        # 增加使用量
        await self.increment_usage(tenant_id, resource_type, count)
    
    async def _send_quota_alert(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        current_usage: int,
        limit: int,
        usage_percentage: float
    ) -> None:
        """
        在达到阈值时发送配额警告
        
        Args:
            tenant_id: 租户 ID
            resource_type: 资源类型
            current_usage: 当前使用量
            limit: 配额限制
            usage_percentage: 使用百分比
        """
        try:
            logger.warning(
                f"租户 {tenant_id} 配额警告: "
                f"{resource_type} 使用量达到 {usage_percentage:.1%} "
                f"({current_usage}/{limit})"
            )
            
            # TODO: 实现实际的警告机制（邮件、webhook 等）
            # 目前仅记录警告
            
        except Exception as e:
            logger.error(f"发送配额警告失败: {str(e)}")
    
    async def get_usage_stats(
        self,
        tenant_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        获取租户的使用统计
        
        Args:
            tenant_id: 租户 ID
            start_time: 开始时间过滤
            end_time: 结束时间过滤
            
        Returns:
            包含使用统计的字典
        """
        try:
            query = """
                SELECT 
                    resource_type,
                    SUM(usage_count) as total_usage,
                    MIN(time) as first_usage,
                    MAX(time) as last_usage
                FROM quota_usage
                WHERE tenant_id = $1
            """
            params = [tenant_id]
            param_idx = 2
            
            if start_time:
                query += f" AND time >= ${param_idx}"
                params.append(start_time)
                param_idx += 1
            
            if end_time:
                query += f" AND time <= ${param_idx}"
                params.append(end_time)
                param_idx += 1
            
            query += " GROUP BY resource_type"
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return {
                    "tenant_id": tenant_id,
                    "usage": [dict(row) for row in rows]
                }
        except Exception as e:
            logger.error(f"Failed to get usage stats: {str(e)}")
            return {"tenant_id": tenant_id, "usage": []}
    
    async def reset_quota(
        self,
        tenant_id: str,
        resource_type: Optional[ResourceType] = None
    ) -> bool:
        """
        Reset quota usage for a tenant (admin operation)
        
        Args:
            tenant_id: Tenant ID
            resource_type: Specific resource type to reset (None for all)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.db_pool.acquire() as conn:
                if resource_type:
                    await conn.execute(
                        """
                        DELETE FROM quota_usage
                        WHERE tenant_id = $1 AND resource_type = $2
                        """,
                        tenant_id,
                        resource_type.value
                    )
                else:
                    await conn.execute(
                        """
                        DELETE FROM quota_usage
                        WHERE tenant_id = $1
                        """,
                        tenant_id
                    )
            
            logger.info(f"Reset quota for tenant {tenant_id}, resource: {resource_type or 'all'}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset quota: {str(e)}")
            return False
