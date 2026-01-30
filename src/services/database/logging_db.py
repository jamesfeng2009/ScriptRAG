"""Database Logging Service - Persists logs to PostgreSQL

This module provides functionality to persist logs to the database:
1. LLM call logs (llm_call_logs table)
2. Execution logs (execution_logs table)
3. Audit logs (audit_logs table)
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import asyncpg


logger = logging.getLogger(__name__)


class LoggingDBService:
    """Service for persisting logs to PostgreSQL"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        """
        Initialize logging database service
        
        Args:
            db_pool: AsyncPG connection pool
        """
        self.db_pool = db_pool
    
    async def log_llm_call(
        self,
        session_id: Optional[str],
        provider: str,
        model: str,
        task_type: str,
        status: str,
        response_time_ms: Optional[int] = None,
        token_count: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Persist LLM call log to database
        
        Args:
            session_id: Session ID (optional)
            provider: LLM provider name
            model: Model name
            task_type: Task type (high_performance/lightweight/embedding)
            status: Call status (success/failed)
            response_time_ms: Response time in milliseconds
            token_count: Number of tokens used
            error_message: Error message if failed
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO llm_call_logs (
                        session_id, provider, model, task_type, status,
                        response_time_ms, token_count, error_message, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    session_id,
                    provider,
                    model,
                    task_type,
                    status,
                    response_time_ms,
                    token_count,
                    error_message,
                    datetime.utcnow()
                )
        except Exception as e:
            logger.error(f"Failed to persist LLM call log: {str(e)}")
            # Don't raise - logging failures shouldn't break the application
    
    async def log_execution(
        self,
        session_id: str,
        agent_name: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Persist execution log to database
        
        Args:
            session_id: Session ID
            agent_name: Name of the agent
            action: Action performed
            details: Additional details (JSON)
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO execution_logs (
                        session_id, agent_name, action, details, created_at
                    ) VALUES ($1, $2, $3, $4, $5)
                    """,
                    session_id,
                    agent_name,
                    action,
                    details or {},
                    datetime.utcnow()
                )
        except Exception as e:
            logger.error(f"Failed to persist execution log: {str(e)}")
    
    async def log_audit(
        self,
        tenant_id: str,
        user_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """
        Persist audit log to database
        
        Args:
            tenant_id: Tenant ID
            user_id: User ID (optional)
            action: Action performed
            resource_type: Type of resource
            resource_id: Resource ID (optional)
            details: Additional details (JSON)
            ip_address: IP address
            user_agent: User agent string
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO audit_logs (
                        tenant_id, user_id, action, resource_type, resource_id,
                        details, ip_address, user_agent, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    tenant_id,
                    user_id,
                    action,
                    resource_type,
                    resource_id,
                    details or {},
                    ip_address,
                    user_agent,
                    datetime.utcnow()
                )
        except Exception as e:
            logger.error(f"Failed to persist audit log: {str(e)}")
    
    async def get_llm_call_stats(
        self,
        session_id: Optional[str] = None,
        provider: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get LLM call statistics
        
        Args:
            session_id: Filter by session ID
            provider: Filter by provider
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            Dictionary with statistics
        """
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT 
                        provider,
                        model,
                        COUNT(*) as total_calls,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_calls,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_calls,
                        AVG(response_time_ms) as avg_response_time_ms,
                        SUM(token_count) as total_tokens
                    FROM llm_call_logs
                    WHERE 1=1
                """
                params = []
                param_idx = 1
                
                if session_id:
                    query += f" AND session_id = ${param_idx}"
                    params.append(session_id)
                    param_idx += 1
                
                if provider:
                    query += f" AND provider = ${param_idx}"
                    params.append(provider)
                    param_idx += 1
                
                if start_time:
                    query += f" AND created_at >= ${param_idx}"
                    params.append(start_time)
                    param_idx += 1
                
                if end_time:
                    query += f" AND created_at <= ${param_idx}"
                    params.append(end_time)
                    param_idx += 1
                
                query += " GROUP BY provider, model"
                
                rows = await conn.fetch(query, *params)
                
                return {
                    "stats": [dict(row) for row in rows]
                }
        except Exception as e:
            logger.error(f"Failed to get LLM call stats: {str(e)}")
            return {"stats": []}
