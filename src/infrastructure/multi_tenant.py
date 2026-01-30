"""Multi-Tenant Support - Tenant isolation and management

This module provides multi-tenant functionality:
1. Tenant context management
2. Tenant-level data isolation
3. Tenant-level configuration
4. Tenant validation and access control
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager
import asyncpg
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class TenantConfig(BaseModel):
    """Tenant configuration model"""
    id: str
    name: str
    plan: str = "free"  # free, basic, pro, enterprise
    quota_limit: int = 1000
    is_active: bool = True
    settings: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class TenantContext:
    """Thread-local tenant context for request isolation"""
    
    def __init__(self):
        self._tenant_id: Optional[str] = None
        self._user_id: Optional[str] = None
        self._workspace_id: Optional[str] = None
    
    @property
    def tenant_id(self) -> Optional[str]:
        return self._tenant_id
    
    @tenant_id.setter
    def tenant_id(self, value: str):
        self._tenant_id = value
    
    @property
    def user_id(self) -> Optional[str]:
        return self._user_id
    
    @user_id.setter
    def user_id(self, value: str):
        self._user_id = value
    
    @property
    def workspace_id(self) -> Optional[str]:
        return self._workspace_id
    
    @workspace_id.setter
    def workspace_id(self, value: str):
        self._workspace_id = value
    
    def clear(self):
        """Clear tenant context"""
        self._tenant_id = None
        self._user_id = None
        self._workspace_id = None
    
    def is_set(self) -> bool:
        """Check if tenant context is set"""
        return self._tenant_id is not None


# Global tenant context instance
tenant_context = TenantContext()


class MultiTenantService:
    """Service for multi-tenant management"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        """
        Initialize multi-tenant service
        
        Args:
            db_pool: AsyncPG connection pool
        """
        self.db_pool = db_pool
    
    async def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """
        Get tenant configuration
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            TenantConfig if found, None otherwise
        """
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, name, plan, quota_limit, is_active, settings, created_at
                    FROM tenants
                    WHERE id = $1
                    """,
                    tenant_id
                )
                
                if row:
                    return TenantConfig(
                        id=str(row['id']),
                        name=row['name'],
                        plan=row['plan'],
                        quota_limit=row['quota_limit'],
                        is_active=row['is_active'],
                        settings=row['settings'] or {},
                        created_at=row['created_at']
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to get tenant {tenant_id}: {str(e)}")
            return None
    
    async def create_tenant(
        self,
        name: str,
        plan: str = "free",
        quota_limit: int = 1000,
        settings: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a new tenant
        
        Args:
            name: Tenant name
            plan: Subscription plan
            quota_limit: API call quota limit
            settings: Tenant-specific settings
            
        Returns:
            Tenant ID if successful, None otherwise
        """
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO tenants (name, plan, quota_limit, settings, is_active, created_at)
                    VALUES ($1, $2, $3, $4, TRUE, $5)
                    RETURNING id
                    """,
                    name,
                    plan,
                    quota_limit,
                    settings or {},
                    datetime.utcnow()
                )
                
                tenant_id = str(row['id'])
                logger.info(f"Created tenant: {tenant_id} ({name})")
                return tenant_id
        except Exception as e:
            logger.error(f"Failed to create tenant: {str(e)}")
            return None
    
    async def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        plan: Optional[str] = None,
        quota_limit: Optional[int] = None,
        settings: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None
    ) -> bool:
        """
        Update tenant configuration
        
        Args:
            tenant_id: Tenant ID
            name: New tenant name
            plan: New subscription plan
            quota_limit: New quota limit
            settings: New settings
            is_active: Active status
            
        Returns:
            True if successful, False otherwise
        """
        try:
            updates = []
            params = []
            param_idx = 1
            
            if name is not None:
                updates.append(f"name = ${param_idx}")
                params.append(name)
                param_idx += 1
            
            if plan is not None:
                updates.append(f"plan = ${param_idx}")
                params.append(plan)
                param_idx += 1
            
            if quota_limit is not None:
                updates.append(f"quota_limit = ${param_idx}")
                params.append(quota_limit)
                param_idx += 1
            
            if settings is not None:
                updates.append(f"settings = ${param_idx}")
                params.append(settings)
                param_idx += 1
            
            if is_active is not None:
                updates.append(f"is_active = ${param_idx}")
                params.append(is_active)
                param_idx += 1
            
            if not updates:
                return True
            
            updates.append(f"updated_at = ${param_idx}")
            params.append(datetime.utcnow())
            param_idx += 1
            
            params.append(tenant_id)
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    f"""
                    UPDATE tenants
                    SET {', '.join(updates)}
                    WHERE id = ${param_idx}
                    """,
                    *params
                )
            
            logger.info(f"Updated tenant: {tenant_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update tenant {tenant_id}: {str(e)}")
            return False
    
    async def delete_tenant(self, tenant_id: str) -> bool:
        """
        Delete a tenant (soft delete by setting is_active=False)
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE tenants
                    SET is_active = FALSE, updated_at = $1
                    WHERE id = $2
                    """,
                    datetime.utcnow(),
                    tenant_id
                )
            
            logger.info(f"Deleted tenant: {tenant_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete tenant {tenant_id}: {str(e)}")
            return False
    
    async def list_tenants(
        self,
        is_active: Optional[bool] = None,
        plan: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[TenantConfig]:
        """
        List tenants with optional filters
        
        Args:
            is_active: Filter by active status
            plan: Filter by plan
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of TenantConfig objects
        """
        try:
            query = """
                SELECT id, name, plan, quota_limit, is_active, settings, created_at
                FROM tenants
                WHERE 1=1
            """
            params = []
            param_idx = 1
            
            if is_active is not None:
                query += f" AND is_active = ${param_idx}"
                params.append(is_active)
                param_idx += 1
            
            if plan is not None:
                query += f" AND plan = ${param_idx}"
                params.append(plan)
                param_idx += 1
            
            query += f" ORDER BY created_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
            params.extend([limit, offset])
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return [
                    TenantConfig(
                        id=str(row['id']),
                        name=row['name'],
                        plan=row['plan'],
                        quota_limit=row['quota_limit'],
                        is_active=row['is_active'],
                        settings=row['settings'] or {},
                        created_at=row['created_at']
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to list tenants: {str(e)}")
            return []
    
    async def validate_tenant_access(
        self,
        tenant_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Validate that a tenant is active and user has access
        
        Args:
            tenant_id: Tenant ID
            user_id: User ID (optional)
            
        Returns:
            True if access is valid, False otherwise
        """
        try:
            # Check tenant is active
            tenant = await self.get_tenant(tenant_id)
            if not tenant or not tenant.is_active:
                logger.warning(f"Tenant {tenant_id} is not active or not found")
                return False
            
            # If user_id provided, check user belongs to tenant
            if user_id:
                async with self.db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT id FROM users
                        WHERE id = $1 AND tenant_id = $2 AND is_active = TRUE
                        """,
                        user_id,
                        tenant_id
                    )
                    
                    if not row:
                        logger.warning(f"User {user_id} does not belong to tenant {tenant_id}")
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to validate tenant access: {str(e)}")
            return False
    
    async def get_tenant_workspaces(
        self,
        tenant_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all workspaces for a tenant
        
        Args:
            tenant_id: Tenant ID
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of workspace dictionaries
        """
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, name, description, settings, created_at, updated_at
                    FROM workspaces
                    WHERE tenant_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                    """,
                    tenant_id,
                    limit,
                    offset
                )
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get tenant workspaces: {str(e)}")
            return []
    
    @asynccontextmanager
    async def tenant_scope(
        self,
        tenant_id: str,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None
    ):
        """
        Context manager for tenant-scoped operations
        
        Args:
            tenant_id: Tenant ID
            user_id: User ID (optional)
            workspace_id: Workspace ID (optional)
            
        Yields:
            TenantContext
            
        Raises:
            ValueError: If tenant access validation fails
        """
        # Validate tenant access
        if not await self.validate_tenant_access(tenant_id, user_id):
            raise ValueError(f"Invalid tenant access: {tenant_id}")
        
        # Set tenant context
        tenant_context.tenant_id = tenant_id
        tenant_context.user_id = user_id
        tenant_context.workspace_id = workspace_id
        
        try:
            yield tenant_context
        finally:
            # Clear tenant context
            tenant_context.clear()


def require_tenant_context():
    """
    Decorator to require tenant context for a function
    
    Raises:
        ValueError: If tenant context is not set
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if not tenant_context.is_set():
                raise ValueError("Tenant context is not set")
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def get_current_tenant_id() -> Optional[str]:
    """Get current tenant ID from context"""
    return tenant_context.tenant_id


def get_current_user_id() -> Optional[str]:
    """Get current user ID from context"""
    return tenant_context.user_id


def get_current_workspace_id() -> Optional[str]:
    """Get current workspace ID from context"""
    return tenant_context.workspace_id
