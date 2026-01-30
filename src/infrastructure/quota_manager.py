"""Quota Management - Resource quota control and enforcement

This module provides quota management functionality:
1. API call rate limiting
2. Resource quota tracking
3. Quota enforcement
4. Quota alerts and notifications
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
import asyncpg


logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Resource types for quota tracking"""
    API_CALL = "api_call"
    LLM_CALL = "llm_call"
    EMBEDDING_GENERATION = "embedding_generation"
    VECTOR_SEARCH = "vector_search"
    SCREENPLAY_GENERATION = "screenplay_generation"
    STORAGE_BYTES = "storage_bytes"


class QuotaPeriod(str, Enum):
    """Quota period types"""
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"


class QuotaLimit:
    """Quota limit configuration"""
    
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
    """Exception raised when quota is exceeded"""
    
    def __init__(self, resource_type: ResourceType, current: int, limit: int):
        self.resource_type = resource_type
        self.current = current
        self.limit = limit
        super().__init__(
            f"Quota exceeded for {resource_type}: {current}/{limit}"
        )


class QuotaManager:
    """Service for quota management and enforcement"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        """
        Initialize quota manager
        
        Args:
            db_pool: AsyncPG connection pool
        """
        self.db_pool = db_pool
    
    def _get_period_start(self, period: QuotaPeriod) -> datetime:
        """
        Get the start time for the current period
        
        Args:
            period: Quota period
            
        Returns:
            Start datetime for the period
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
        Get quota limits for a tenant
        
        Args:
            tenant_id: Tenant ID
            plan: Subscription plan
            
        Returns:
            Dictionary of resource type to quota limit
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
        Get current usage for a resource type
        
        Args:
            tenant_id: Tenant ID
            resource_type: Resource type
            period: Quota period
            
        Returns:
            Current usage count
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
            logger.error(f"Failed to get current usage: {str(e)}")
            return 0
    
    async def increment_usage(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        count: int = 1
    ) -> None:
        """
        Increment usage for a resource type
        
        Args:
            tenant_id: Tenant ID
            resource_type: Resource type
            count: Usage count to increment
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
        Check if quota allows the requested usage
        
        Args:
            tenant_id: Tenant ID
            resource_type: Resource type
            plan: Subscription plan
            count: Requested usage count
            
        Returns:
            True if quota allows, False otherwise
            
        Raises:
            QuotaExceededException: If quota is exceeded
        """
        try:
            # Get quota limits
            limits = await self.get_quota_limits(tenant_id, plan)
            
            if resource_type not in limits:
                # No limit defined, allow
                return True
            
            quota_limit = limits[resource_type]
            
            # Get current usage
            current_usage = await self.get_current_usage(
                tenant_id,
                resource_type,
                quota_limit.period
            )
            
            # Check if adding count would exceed limit
            if current_usage + count > quota_limit.limit:
                raise QuotaExceededException(
                    resource_type,
                    current_usage + count,
                    quota_limit.limit
                )
            
            # Check if approaching alert threshold
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
            logger.error(f"Failed to check quota: {str(e)}")
            # On error, allow the operation (fail open)
            return True
    
    async def enforce_quota(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        plan: str,
        count: int = 1
    ) -> None:
        """
        Enforce quota and increment usage if allowed
        
        Args:
            tenant_id: Tenant ID
            resource_type: Resource type
            plan: Subscription plan
            count: Usage count
            
        Raises:
            QuotaExceededException: If quota is exceeded
        """
        # Check quota
        await self.check_quota(tenant_id, resource_type, plan, count)
        
        # Increment usage
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
        Send quota alert when threshold is reached
        
        Args:
            tenant_id: Tenant ID
            resource_type: Resource type
            current_usage: Current usage
            limit: Quota limit
            usage_percentage: Usage percentage
        """
        try:
            logger.warning(
                f"Quota alert for tenant {tenant_id}: "
                f"{resource_type} usage at {usage_percentage:.1%} "
                f"({current_usage}/{limit})"
            )
            
            # TODO: Implement actual alert mechanism (email, webhook, etc.)
            # For now, just log the alert
            
        except Exception as e:
            logger.error(f"Failed to send quota alert: {str(e)}")
    
    async def get_usage_stats(
        self,
        tenant_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get usage statistics for a tenant
        
        Args:
            tenant_id: Tenant ID
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            Dictionary with usage statistics
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
