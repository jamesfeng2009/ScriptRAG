"""数据库日志服务 - 将日志持久化到 PostgreSQL

本模块提供将日志持久化到数据库的功能：
1. LLM 调用日志（llm_call_logs 表）
2. 执行日志（execution_logs 表）
3. 审计日志（audit_logs 表）
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import asyncpg


logger = logging.getLogger(__name__)


class LoggingDBService:
    """用于将日志持久化到 PostgreSQL 的服务"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        """
        初始化日志数据库服务
        
        Args:
            db_pool: AsyncPG 连接池
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
        将 LLM 调用日志持久化到数据库
        
        Args:
            session_id: 会话 ID（可选）
            provider: LLM 提供商名称
            model: 模型名称
            task_type: 任务类型（high_performance/lightweight/embedding）
            status: 调用状态（success/failed）
            response_time_ms: 响应时间（毫秒）
            token_count: 使用的 token 数量
            error_message: 错误消息（如果失败）
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
            # 不要抛出异常 - 日志失败不应中断应用程序
    
    async def log_execution(
        self,
        session_id: str,
        agent_name: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        将执行日志持久化到数据库
        
        Args:
            session_id: 会话 ID
            agent_name: 智能体名称
            action: 执行的操作
            details: 附加详情（JSON）
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
        将审计日志持久化到数据库
        
        Args:
            tenant_id: 租户 ID
            user_id: 用户 ID（可选）
            action: 执行的操作
            resource_type: 资源类型
            resource_id: 资源 ID（可选）
            details: 附加详情（JSON）
            ip_address: IP 地址
            user_agent: 用户代理字符串
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
        获取 LLM 调用统计
        
        Args:
            session_id: 按会话 ID 过滤
            provider: 按提供商过滤
            start_time: 开始时间过滤
            end_time: 结束时间过滤
            
        Returns:
            包含统计信息的字典
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
