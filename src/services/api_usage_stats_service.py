"""API Usage Stats Service - API 使用统计服务"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.utils.database_utils import build_db_url

logger = logging.getLogger(__name__)


class APIUsageStatsService:
    """API 使用统计服务"""
    
    def __init__(self, config=None):
        """
        初始化 API 使用统计服务
        
        Args:
            config: 数据库配置对象（可选，默认从 config.py 加载）
        """
        self._config = config
        self._engine = None
        self._session_factory = None
        self._initialized = False
    
    def _get_db_url(self) -> str:
        """获取数据库连接 URL"""
        if self._config is not None:
            db_config = self._config
        else:
            db_config = None
        return build_db_url(db_config)
    
    async def initialize(self):
        """初始化连接池"""
        try:
            self._engine = create_async_engine(
                self._get_db_url(),
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            
            self._session_factory = sessionmaker(
                self._engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            self._initialized = True
            logger.info("APIUsageStatsService initialized")
        except Exception as e:
            logger.error(f"Failed to initialize APIUsageStatsService: {e}")
            raise
    
    async def close(self):
        """关闭连接池"""
        if self._engine:
            await self._engine.dispose()
            self._initialized = False
            logger.info("APIUsageStatsService closed")
    
    async def health_check(self) -> bool:
        """健康检查"""
        if not self._initialized:
            return False
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    
    async def record_usage(
        self,
        provider: str,
        model: str,
        operation: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        workspace_id: Optional[str] = None
    ) -> str:
        """记录 API 使用"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        total_tokens = prompt_tokens + completion_tokens
        
        async with self._session_factory() as session:
            query = text("""
                INSERT INTO screenplay.api_usage_stats (
                    workspace_id, provider, model, operation,
                    prompt_tokens, completion_tokens, total_tokens,
                    cost_usd, date, hour
                ) VALUES (
                    :workspace_id, :provider, :model, :operation,
                    :prompt_tokens, :completion_tokens, :total_tokens,
                    :cost_usd, CURRENT_DATE, EXTRACT(HOUR FROM NOW())
                )
                RETURNING id
            """)
            
            result = await session.execute(query, {
                'workspace_id': workspace_id,
                'provider': provider,
                'model': model,
                'operation': operation,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'cost_usd': cost_usd
            })
            await session.commit()
            
            return str(result.scalar())
    
    async def get_daily_summary(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """获取每日汇总"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        conditions = ["date >= CURRENT_DATE - (INTERVAL '1 day' * :days)"]
        params = {'days': days}
        
        if provider:
            conditions.append("provider = :provider")
            params['provider'] = provider
        if model:
            conditions.append("model = :model")
            params['model'] = model
        
        where_clause = " AND ".join(conditions)
        
        async with self._session_factory() as session:
            query = text(f"""
                SELECT date, provider, model,
                       SUM(prompt_tokens) as total_prompt,
                       SUM(completion_tokens) as total_completion,
                       SUM(total_tokens) as total_tokens,
                       SUM(cost_usd) as total_cost,
                       SUM(request_count) as total_requests
                FROM screenplay.api_usage_stats
                WHERE {where_clause}
                GROUP BY date, provider, model
                ORDER BY date DESC, total_cost DESC
            """)
            
            result = await session.execute(query, params)
            rows = result.fetchall()
            
            summaries = []
            for row in rows:
                summaries.append({
                    'date': row[0].isoformat() if row[0] else None,
                    'provider': row[1],
                    'model': row[2],
                    'total_prompt_tokens': row[3] or 0,
                    'total_completion_tokens': row[4] or 0,
                    'total_tokens': row[5] or 0,
                    'total_cost_usd': round(row[6], 6) if row[6] else 0,
                    'total_requests': row[7] or 0
                })
            
            return summaries
    
    async def get_provider_summary(
        self,
        workspace_id: Optional[str] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """获取提供商汇总"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            if workspace_id:
                query = text("""
                    SELECT provider, model,
                           SUM(prompt_tokens) as total_prompt,
                           SUM(completion_tokens) as total_completion,
                           SUM(total_tokens) as total_tokens,
                           SUM(cost_usd) as total_cost,
                           SUM(request_count) as total_requests,
                           COUNT(DISTINCT date) as active_days
                    FROM screenplay.api_usage_stats
                    WHERE workspace_id = :workspace_id
                          AND date >= CURRENT_DATE - INTERVAL '1 day' * :days
                    GROUP BY provider, model
                    ORDER BY total_cost DESC
                """)
                params = {'workspace_id': workspace_id, 'days': days}
            else:
                query = text("""
                    SELECT provider, model,
                           SUM(prompt_tokens) as total_prompt,
                           SUM(completion_tokens) as total_completion,
                           SUM(total_tokens) as total_tokens,
                           SUM(cost_usd) as total_cost,
                           SUM(request_count) as total_requests,
                           COUNT(DISTINCT date) as active_days
                    FROM screenplay.api_usage_stats
                    WHERE date >= CURRENT_DATE - INTERVAL '1 day' * :days
                    GROUP BY provider, model
                    ORDER BY total_cost DESC
                """)
                params = {'days': days}
            
            result = await session.execute(query, params)
            rows = result.fetchall()
            
            summaries = []
            for row in rows:
                summaries.append({
                    'provider': row[0],
                    'model': row[1],
                    'total_prompt_tokens': row[2] or 0,
                    'total_completion_tokens': row[3] or 0,
                    'total_tokens': row[4] or 0,
                    'total_cost_usd': round(row[5], 6) if row[5] else 0,
                    'total_requests': row[6] or 0,
                    'active_days': row[7] or 0
                })
            
            return summaries
    
    async def get_total_stats(self, workspace_id: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """获取总统计"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            if workspace_id:
                query = text("""
                    SELECT 
                        SUM(prompt_tokens) as total_prompt,
                        SUM(completion_tokens) as total_completion,
                        SUM(total_tokens) as total_tokens,
                        SUM(cost_usd) as total_cost,
                        SUM(request_count) as total_requests,
                        COUNT(DISTINCT date) as active_days,
                        COUNT(DISTINCT provider) as provider_count,
                        COUNT(DISTINCT model) as model_count
                    FROM screenplay.api_usage_stats
                    WHERE workspace_id = :workspace_id
                          AND date >= CURRENT_DATE - INTERVAL '1 day' * :days
                """)
                params = {'workspace_id': workspace_id, 'days': days}
            else:
                query = text("""
                    SELECT 
                        SUM(prompt_tokens) as total_prompt,
                        SUM(completion_tokens) as total_completion,
                        SUM(total_tokens) as total_tokens,
                        SUM(cost_usd) as total_cost,
                        SUM(request_count) as total_requests,
                        COUNT(DISTINCT date) as active_days,
                        COUNT(DISTINCT provider) as provider_count,
                        COUNT(DISTINCT model) as model_count
                    FROM screenplay.api_usage_stats
                    WHERE date >= CURRENT_DATE - INTERVAL '1 day' * :days
                """)
                params = {'days': days}
            
            result = await session.execute(query, params)
            row = result.fetchone()
            
            return {
                'period_days': days,
                'total_prompt_tokens': row[0] or 0,
                'total_completion_tokens': row[1] or 0,
                'total_tokens': row[2] or 0,
                'total_cost_usd': round(row[3], 6) if row[3] else 0,
                'total_requests': row[4] or 0,
                'active_days': row[5] or 0,
                'provider_count': row[6] or 0,
                'model_count': row[7] or 0
            }
    
    async def get_cost_by_day(self, workspace_id: Optional[str] = None, days: int = 7) -> List[Dict[str, Any]]:
        """获取每日成本趋势"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            if workspace_id:
                query = text("""
                    SELECT date,
                           SUM(cost_usd) as daily_cost,
                           SUM(total_tokens) as daily_tokens,
                           SUM(request_count) as daily_requests
                    FROM screenplay.api_usage_stats
                    WHERE workspace_id = :workspace_id
                          AND date >= CURRENT_DATE - INTERVAL '1 day' * :days
                    GROUP BY date
                    ORDER BY date DESC
                """)
                params = {'workspace_id': workspace_id, 'days': days}
            else:
                query = text("""
                    SELECT date,
                           SUM(cost_usd) as daily_cost,
                           SUM(total_tokens) as daily_tokens,
                           SUM(request_count) as daily_requests
                    FROM screenplay.api_usage_stats
                    WHERE date >= CURRENT_DATE - INTERVAL '1 day' * :days
                    GROUP BY date
                    ORDER BY date DESC
                """)
                params = {'days': days}
            
            result = await session.execute(query, params)
            rows = result.fetchall()
            
            trends = []
            for row in rows:
                trends.append({
                    'date': row[0].isoformat() if row[0] else None,
                    'daily_cost_usd': round(row[1], 6) if row[1] else 0,
                    'daily_tokens': row[2] or 0,
                    'daily_requests': row[3] or 0
                })
            
            return trends
    
    async def cleanup_old_stats(self, retention_days: int = 90) -> int:
        """清理旧统计数据"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            query = text("""
                SELECT screenplay.cleanup_old_api_stats(:retention_days)
            """)
            
            result = await session.execute(query, {'retention_days': retention_days})
            deleted_count = result.scalar()
            await session.commit()
            
            return deleted_count or 0
