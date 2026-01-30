"""Audit Logger - Comprehensive audit logging for compliance

This module provides audit logging functionality:
1. Record all user operations
2. Persist audit logs to database
3. Query and search audit logs
4. Compliance and security tracking
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import asyncpg
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Audit action types"""
    # Authentication actions
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGE = "password_change"
    
    # Tenant actions
    TENANT_CREATE = "tenant_create"
    TENANT_UPDATE = "tenant_update"
    TENANT_DELETE = "tenant_delete"
    
    # User actions
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    USER_ROLE_CHANGE = "user_role_change"
    
    # Workspace actions
    WORKSPACE_CREATE = "workspace_create"
    WORKSPACE_UPDATE = "workspace_update"
    WORKSPACE_DELETE = "workspace_delete"
    
    # Screenplay actions
    SCREENPLAY_GENERATE = "screenplay_generate"
    SCREENPLAY_VIEW = "screenplay_view"
    SCREENPLAY_DELETE = "screenplay_delete"
    SCREENPLAY_EXPORT = "screenplay_export"
    
    # Document actions
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_DELETE = "document_delete"
    DOCUMENT_INDEX = "document_index"
    
    # Configuration actions
    CONFIG_UPDATE = "config_update"
    QUOTA_UPDATE = "quota_update"
    
    # API actions
    API_CALL = "api_call"
    API_KEY_CREATE = "api_key_create"
    API_KEY_REVOKE = "api_key_revoke"
    
    # System actions
    SYSTEM_BACKUP = "system_backup"
    SYSTEM_RESTORE = "system_restore"
    SYSTEM_MAINTENANCE = "system_maintenance"


class ResourceType(str, Enum):
    """Resource types for audit logging"""
    TENANT = "tenant"
    USER = "user"
    WORKSPACE = "workspace"
    SCREENPLAY = "screenplay"
    DOCUMENT = "document"
    CONFIG = "config"
    API_KEY = "api_key"
    SYSTEM = "system"


class AuditLogEntry(BaseModel):
    """Audit log entry model"""
    id: Optional[str] = None
    tenant_id: str
    user_id: Optional[str] = None
    action: AuditAction
    resource_type: ResourceType
    resource_id: Optional[str] = None
    details: Dict[str, Any] = {}
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class AuditLogger:
    """Service for audit logging"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        """
        Initialize audit logger
        
        Args:
            db_pool: AsyncPG connection pool
        """
        self.db_pool = db_pool
    
    async def log(
        self,
        tenant_id: str,
        action: AuditAction,
        resource_type: ResourceType,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[str]:
        """
        Log an audit event
        
        Args:
            tenant_id: Tenant ID
            action: Action performed
            resource_type: Type of resource
            user_id: User ID (optional)
            resource_id: Resource ID (optional)
            details: Additional details (optional)
            ip_address: IP address (optional)
            user_agent: User agent string (optional)
            
        Returns:
            Audit log ID if successful, None otherwise
        """
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO audit_logs (
                        tenant_id, user_id, action, resource_type, resource_id,
                        details, ip_address, user_agent, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                    """,
                    tenant_id,
                    user_id,
                    action.value,
                    resource_type.value,
                    resource_id,
                    details or {},
                    ip_address,
                    user_agent,
                    datetime.utcnow()
                )
                
                audit_id = str(row['id'])
                logger.debug(f"Audit log created: {audit_id} - {action.value}")
                return audit_id
        except Exception as e:
            logger.error(f"Failed to create audit log: {str(e)}")
            # Don't raise - audit logging failures shouldn't break the application
            return None
    
    async def log_authentication(
        self,
        tenant_id: str,
        user_id: str,
        action: AuditAction,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Log authentication event
        
        Args:
            tenant_id: Tenant ID
            user_id: User ID
            action: Authentication action
            success: Whether authentication succeeded
            ip_address: IP address
            user_agent: User agent string
            details: Additional details
            
        Returns:
            Audit log ID if successful, None otherwise
        """
        audit_details = details or {}
        audit_details['success'] = success
        
        return await self.log(
            tenant_id=tenant_id,
            action=action,
            resource_type=ResourceType.USER,
            user_id=user_id,
            resource_id=user_id,
            details=audit_details,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    async def log_api_call(
        self,
        tenant_id: str,
        user_id: Optional[str],
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: int,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[str]:
        """
        Log API call
        
        Args:
            tenant_id: Tenant ID
            user_id: User ID (optional)
            endpoint: API endpoint
            method: HTTP method
            status_code: Response status code
            response_time_ms: Response time in milliseconds
            ip_address: IP address
            user_agent: User agent string
            
        Returns:
            Audit log ID if successful, None otherwise
        """
        return await self.log(
            tenant_id=tenant_id,
            action=AuditAction.API_CALL,
            resource_type=ResourceType.SYSTEM,
            user_id=user_id,
            details={
                'endpoint': endpoint,
                'method': method,
                'status_code': status_code,
                'response_time_ms': response_time_ms
            },
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    async def query_logs(
        self,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[ResourceType] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditLogEntry]:
        """
        Query audit logs with filters
        
        Args:
            tenant_id: Filter by tenant ID
            user_id: Filter by user ID
            action: Filter by action
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of AuditLogEntry objects
        """
        try:
            query = """
                SELECT 
                    id, tenant_id, user_id, action, resource_type, resource_id,
                    details, ip_address, user_agent, created_at
                FROM audit_logs
                WHERE 1=1
            """
            params = []
            param_idx = 1
            
            if tenant_id:
                query += f" AND tenant_id = ${param_idx}"
                params.append(tenant_id)
                param_idx += 1
            
            if user_id:
                query += f" AND user_id = ${param_idx}"
                params.append(user_id)
                param_idx += 1
            
            if action:
                query += f" AND action = ${param_idx}"
                params.append(action.value)
                param_idx += 1
            
            if resource_type:
                query += f" AND resource_type = ${param_idx}"
                params.append(resource_type.value)
                param_idx += 1
            
            if resource_id:
                query += f" AND resource_id = ${param_idx}"
                params.append(resource_id)
                param_idx += 1
            
            if start_time:
                query += f" AND created_at >= ${param_idx}"
                params.append(start_time)
                param_idx += 1
            
            if end_time:
                query += f" AND created_at <= ${param_idx}"
                params.append(end_time)
                param_idx += 1
            
            query += f" ORDER BY created_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
            params.extend([limit, offset])
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return [
                    AuditLogEntry(
                        id=str(row['id']),
                        tenant_id=str(row['tenant_id']),
                        user_id=str(row['user_id']) if row['user_id'] else None,
                        action=AuditAction(row['action']),
                        resource_type=ResourceType(row['resource_type']),
                        resource_id=str(row['resource_id']) if row['resource_id'] else None,
                        details=row['details'] or {},
                        ip_address=row['ip_address'],
                        user_agent=row['user_agent'],
                        created_at=row['created_at']
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to query audit logs: {str(e)}")
            return []
    
    async def get_user_activity(
        self,
        tenant_id: str,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """
        Get activity logs for a specific user
        
        Args:
            tenant_id: Tenant ID
            user_id: User ID
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of results
            
        Returns:
            List of AuditLogEntry objects
        """
        return await self.query_logs(
            tenant_id=tenant_id,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    async def get_resource_history(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        resource_id: str,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """
        Get audit history for a specific resource
        
        Args:
            tenant_id: Tenant ID
            resource_type: Resource type
            resource_id: Resource ID
            limit: Maximum number of results
            
        Returns:
            List of AuditLogEntry objects
        """
        return await self.query_logs(
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            limit=limit
        )
    
    async def get_security_events(
        self,
        tenant_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """
        Get security-related audit events
        
        Args:
            tenant_id: Tenant ID
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of results
            
        Returns:
            List of AuditLogEntry objects
        """
        security_actions = [
            AuditAction.LOGIN,
            AuditAction.LOGIN_FAILED,
            AuditAction.PASSWORD_CHANGE,
            AuditAction.USER_ROLE_CHANGE,
            AuditAction.API_KEY_CREATE,
            AuditAction.API_KEY_REVOKE
        ]
        
        try:
            query = """
                SELECT 
                    id, tenant_id, user_id, action, resource_type, resource_id,
                    details, ip_address, user_agent, created_at
                FROM audit_logs
                WHERE tenant_id = $1
                    AND action = ANY($2)
            """
            params = [tenant_id, [a.value for a in security_actions]]
            param_idx = 3
            
            if start_time:
                query += f" AND created_at >= ${param_idx}"
                params.append(start_time)
                param_idx += 1
            
            if end_time:
                query += f" AND created_at <= ${param_idx}"
                params.append(end_time)
                param_idx += 1
            
            query += f" ORDER BY created_at DESC LIMIT ${param_idx}"
            params.append(limit)
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return [
                    AuditLogEntry(
                        id=str(row['id']),
                        tenant_id=str(row['tenant_id']),
                        user_id=str(row['user_id']) if row['user_id'] else None,
                        action=AuditAction(row['action']),
                        resource_type=ResourceType(row['resource_type']),
                        resource_id=str(row['resource_id']) if row['resource_id'] else None,
                        details=row['details'] or {},
                        ip_address=row['ip_address'],
                        user_agent=row['user_agent'],
                        created_at=row['created_at']
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get security events: {str(e)}")
            return []
    
    async def get_statistics(
        self,
        tenant_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get audit log statistics
        
        Args:
            tenant_id: Tenant ID
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            Dictionary with statistics
        """
        try:
            query = """
                SELECT 
                    action,
                    resource_type,
                    COUNT(*) as count,
                    COUNT(DISTINCT user_id) as unique_users
                FROM audit_logs
                WHERE tenant_id = $1
            """
            params = [tenant_id]
            param_idx = 2
            
            if start_time:
                query += f" AND created_at >= ${param_idx}"
                params.append(start_time)
                param_idx += 1
            
            if end_time:
                query += f" AND created_at <= ${param_idx}"
                params.append(end_time)
                param_idx += 1
            
            query += " GROUP BY action, resource_type ORDER BY count DESC"
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return {
                    "tenant_id": tenant_id,
                    "statistics": [dict(row) for row in rows]
                }
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {str(e)}")
            return {"tenant_id": tenant_id, "statistics": []}
    
    async def export_logs(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime,
        format: str = "json"
    ) -> Optional[str]:
        """
        Export audit logs for compliance
        
        Args:
            tenant_id: Tenant ID
            start_time: Start time
            end_time: End time
            format: Export format (json, csv)
            
        Returns:
            Export data as string, None if failed
        """
        try:
            logs = await self.query_logs(
                tenant_id=tenant_id,
                start_time=start_time,
                end_time=end_time,
                limit=10000  # Large limit for export
            )
            
            if format == "json":
                import json
                return json.dumps([log.dict() for log in logs], indent=2, default=str)
            elif format == "csv":
                import csv
                import io
                
                output = io.StringIO()
                if logs:
                    fieldnames = logs[0].dict().keys()
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    for log in logs:
                        writer.writerow(log.dict())
                
                return output.getvalue()
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
        except Exception as e:
            logger.error(f"Failed to export audit logs: {str(e)}")
            return None
