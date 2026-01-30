"""Unit tests for commercial features

Tests for:
1. Multi-tenant support
2. Quota management
3. Audit logging
4. Redis caching
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import asyncpg

from src.infrastructure.multi_tenant import (
    MultiTenantService,
    TenantConfig,
    tenant_context
)
from src.infrastructure.quota_manager import (
    QuotaManager,
    ResourceType,
    QuotaPeriod,
    QuotaExceededException
)
from src.infrastructure.audit_logger import (
    AuditLogger,
    AuditAction,
    ResourceType as AuditResourceType
)
from src.services.database.redis_cache import RedisCacheService


@pytest.fixture
def mock_db_pool():
    """Mock database pool"""
    pool = AsyncMock(spec=asyncpg.Pool)
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    return pool


@pytest.fixture
def multi_tenant_service(mock_db_pool):
    """Create multi-tenant service"""
    return MultiTenantService(mock_db_pool)


@pytest.fixture
def quota_manager(mock_db_pool):
    """Create quota manager"""
    return QuotaManager(mock_db_pool)


@pytest.fixture
def audit_logger(mock_db_pool):
    """Create audit logger"""
    return AuditLogger(mock_db_pool)


class TestMultiTenantService:
    """Tests for multi-tenant service"""
    
    @pytest.mark.asyncio
    async def test_get_tenant(self, multi_tenant_service, mock_db_pool):
        """Test getting tenant configuration"""
        # Mock database response
        mock_row = {
            'id': 'tenant-123',
            'name': 'Test Tenant',
            'plan': 'pro',
            'quota_limit': 10000,
            'is_active': True,
            'settings': {'custom': 'value'},
            'created_at': datetime.utcnow()
        }
        
        conn = await mock_db_pool.acquire().__aenter__()
        conn.fetchrow.return_value = mock_row
        
        # Get tenant
        tenant = await multi_tenant_service.get_tenant('tenant-123')
        
        # Verify
        assert tenant is not None
        assert tenant.id == 'tenant-123'
        assert tenant.name == 'Test Tenant'
        assert tenant.plan == 'pro'
        assert tenant.is_active is True
    
    @pytest.mark.asyncio
    async def test_create_tenant(self, multi_tenant_service, mock_db_pool):
        """Test creating a new tenant"""
        # Mock database response
        mock_row = {'id': 'new-tenant-123'}
        
        conn = await mock_db_pool.acquire().__aenter__()
        conn.fetchrow.return_value = mock_row
        
        # Create tenant
        tenant_id = await multi_tenant_service.create_tenant(
            name='New Tenant',
            plan='basic',
            quota_limit=1000
        )
        
        # Verify
        assert tenant_id == 'new-tenant-123'
        conn.fetchrow.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_tenant_access(self, multi_tenant_service, mock_db_pool):
        """Test tenant access validation"""
        # Mock tenant exists and is active
        mock_tenant_row = {
            'id': 'tenant-123',
            'name': 'Test Tenant',
            'plan': 'pro',
            'quota_limit': 10000,
            'is_active': True,
            'settings': {},
            'created_at': datetime.utcnow()
        }
        
        mock_user_row = {'id': 'user-123'}
        
        conn = await mock_db_pool.acquire().__aenter__()
        conn.fetchrow.side_effect = [mock_tenant_row, mock_user_row]
        
        # Validate access
        is_valid = await multi_tenant_service.validate_tenant_access(
            'tenant-123',
            'user-123'
        )
        
        # Verify
        assert is_valid is True


class TestQuotaManager:
    """Tests for quota manager"""
    
    @pytest.mark.asyncio
    async def test_get_current_usage(self, quota_manager, mock_db_pool):
        """Test getting current usage"""
        # Mock database response
        mock_row = {'total_usage': 500}
        
        conn = await mock_db_pool.acquire().__aenter__()
        conn.fetchrow.return_value = mock_row
        
        # Get usage
        usage = await quota_manager.get_current_usage(
            'tenant-123',
            ResourceType.API_CALL,
            QuotaPeriod.DAILY
        )
        
        # Verify
        assert usage == 500
    
    @pytest.mark.asyncio
    async def test_increment_usage(self, quota_manager, mock_db_pool):
        """Test incrementing usage"""
        conn = await mock_db_pool.acquire().__aenter__()
        
        # Increment usage
        await quota_manager.increment_usage(
            'tenant-123',
            ResourceType.API_CALL,
            count=5
        )
        
        # Verify database call
        conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_quota_exceeded(self, quota_manager, mock_db_pool):
        """Test quota exceeded exception"""
        # Mock current usage at limit
        mock_row = {'total_usage': 1000}
        mock_tenant_row = {
            'settings': {}
        }
        
        conn = await mock_db_pool.acquire().__aenter__()
        conn.fetchrow.side_effect = [mock_tenant_row, mock_row]
        
        # Check quota should raise exception
        with pytest.raises(QuotaExceededException) as exc_info:
            await quota_manager.check_quota(
                'tenant-123',
                ResourceType.API_CALL,
                'free',
                count=1
            )
        
        # Verify exception details
        assert exc_info.value.resource_type == ResourceType.API_CALL
        assert exc_info.value.current == 1001
        assert exc_info.value.limit == 100  # Free plan limit


class TestAuditLogger:
    """Tests for audit logger"""
    
    @pytest.mark.asyncio
    async def test_log_audit_event(self, audit_logger, mock_db_pool):
        """Test logging audit event"""
        # Mock database response
        mock_row = {'id': 'audit-123'}
        
        conn = await mock_db_pool.acquire().__aenter__()
        conn.fetchrow.return_value = mock_row
        
        # Log event
        audit_id = await audit_logger.log(
            tenant_id='tenant-123',
            action=AuditAction.USER_CREATE,
            resource_type=AuditResourceType.USER,
            user_id='user-123',
            resource_id='new-user-456',
            details={'email': 'test@example.com'}
        )
        
        # Verify
        assert audit_id == 'audit-123'
        conn.fetchrow.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_logs(self, audit_logger, mock_db_pool):
        """Test querying audit logs"""
        # Mock database response
        mock_rows = [
            {
                'id': 'audit-1',
                'tenant_id': 'tenant-123',
                'user_id': 'user-123',
                'action': 'user_create',
                'resource_type': 'user',
                'resource_id': 'new-user-1',
                'details': {},
                'ip_address': '127.0.0.1',
                'user_agent': 'test-agent',
                'created_at': datetime.utcnow()
            }
        ]
        
        conn = await mock_db_pool.acquire().__aenter__()
        conn.fetch.return_value = mock_rows
        
        # Query logs
        logs = await audit_logger.query_logs(
            tenant_id='tenant-123',
            action=AuditAction.USER_CREATE
        )
        
        # Verify
        assert len(logs) == 1
        assert logs[0].id == 'audit-1'
        assert logs[0].action == AuditAction.USER_CREATE


class TestRedisCacheService:
    """Tests for Redis cache service"""
    
    @pytest.fixture
    def redis_cache(self):
        """Create Redis cache service"""
        return RedisCacheService(host='localhost', port=6379)
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, redis_cache):
        """Test cache key generation"""
        key = redis_cache._generate_key('test', 'arg1', 'arg2')
        assert key == 'test:arg1:arg2'
    
    @pytest.mark.asyncio
    async def test_content_hashing(self, redis_cache):
        """Test content hashing"""
        hash1 = redis_cache._hash_content('test content')
        hash2 = redis_cache._hash_content('test content')
        hash3 = redis_cache._hash_content('different content')
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16  # Truncated to 16 chars
    
    @pytest.mark.asyncio
    async def test_cache_operations_without_connection(self, redis_cache):
        """Test cache operations without connection (should not fail)"""
        # Without connection, operations should return None/False gracefully
        result = await redis_cache.get('test-key')
        assert result is None
        
        success = await redis_cache.set('test-key', 'value')
        assert success is False
        
        exists = await redis_cache.exists('test-key')
        assert exists is False


def test_tenant_context():
    """Test tenant context management"""
    # Initially not set
    assert not tenant_context.is_set()
    
    # Set context
    tenant_context.tenant_id = 'tenant-123'
    tenant_context.user_id = 'user-456'
    tenant_context.workspace_id = 'workspace-789'
    
    # Verify
    assert tenant_context.is_set()
    assert tenant_context.tenant_id == 'tenant-123'
    assert tenant_context.user_id == 'user-456'
    assert tenant_context.workspace_id == 'workspace-789'
    
    # Clear context
    tenant_context.clear()
    assert not tenant_context.is_set()
    assert tenant_context.tenant_id is None
