"""Retrieval Logs Service - 检索日志管理服务"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.utils.database_utils import build_db_url

logger = logging.getLogger(__name__)


class RetrievalLogsService:
    """检索日志服务"""
    
    def __init__(self, config=None):
        """
        初始化检索日志服务
        
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
            logger.info("RetrievalLogsService initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RetrievalLogsService: {e}")
            raise
    
    async def close(self):
        """关闭连接池"""
        if self._engine:
            await self._engine.dispose()
            self._initialized = False
            logger.info("RetrievalLogsService closed")
    
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
    
    async def log_retrieval(
        self,
        workspace_id: str,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        result_count: int = 0,
        total_score: float = 0.0,
        search_strategy: str = "hybrid",
        duration_ms: int = 0,
        token_usage: Optional[Dict[str, Any]] = None,
        cost_usd: float = 0.0,
        user_id: Optional[str] = None
    ) -> str:
        """记录检索日志"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        log_id = str(uuid4())
        embedding_str = f"[{','.join(map(str, query_embedding))}]" if query_embedding else None
        token_usage_json = None
        if token_usage:
            import json
            token_usage_json = json.dumps(token_usage)
        
        async with self._session_factory() as session:
            insert_query = text("""
                INSERT INTO screenplay.retrieval_logs (
                    id, workspace_id, user_id, query, query_embedding,
                    top_k, result_count, total_score, search_strategy,
                    duration_ms, token_usage, cost_usd
                ) VALUES (
                    :id, :workspace_id, :user_id, :q, :embedding,
                    :top_k, :result_count, :total_score, :search_strategy,
                    :duration_ms, :token_usage, :cost_usd
                )
                RETURNING id
            """)
            
            result = await session.execute(insert_query, {
                'id': log_id,
                'workspace_id': workspace_id,
                'user_id': user_id,
                'q': query,
                'embedding': embedding_str,
                'top_k': top_k,
                'result_count': result_count,
                'total_score': total_score,
                'search_strategy': search_strategy,
                'duration_ms': duration_ms,
                'token_usage': token_usage_json,
                'cost_usd': cost_usd
            })
            await session.commit()
            
            return str(result.scalar())
    
    async def get_log(self, log_id: str) -> Optional[Dict[str, Any]]:
        """获取检索日志"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            query = text("""
                SELECT id, workspace_id, user_id, query, query_embedding,
                       top_k, result_count, total_score, search_strategy,
                       duration_ms, token_usage, cost_usd, created_at
                FROM screenplay.retrieval_logs
                WHERE id = :id
            """)
            
            result = await session.execute(query, {'id': log_id})
            row = result.fetchone()
            
            if row:
                return {
                    'id': str(row[0]),
                    'workspace_id': row[1],
                    'user_id': row[2],
                    'query': row[3],
                    'query_embedding': row[4],
                    'top_k': row[5],
                    'result_count': row[6],
                    'total_score': row[7],
                    'search_strategy': row[8],
                    'duration_ms': row[9],
                    'token_usage': row[10],
                    'cost_usd': row[11],
                    'created_at': row[12].isoformat() if row[12] else None
                }
            return None
    
    async def list_logs(
        self,
        workspace_id: str,
        days: int = 7,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """列出检索日志"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            query = text("""
                SELECT id, workspace_id, user_id, query, query_embedding,
                       top_k, result_count, total_score, search_strategy,
                       duration_ms, token_usage, cost_usd, created_at
                FROM screenplay.retrieval_logs
                WHERE workspace_id = :workspace_id
                      AND created_at >= NOW() - INTERVAL '1 day' * :days
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """)
            
            result = await session.execute(query, {
                'workspace_id': workspace_id,
                'days': days,
                'limit': limit,
                'offset': offset
            })
            rows = result.fetchall()
            
            logs = []
            for row in rows:
                logs.append({
                    'id': str(row[0]),
                    'workspace_id': row[1],
                    'user_id': row[2],
                    'query': row[3],
                    'query_embedding': row[4],
                    'top_k': row[5],
                    'result_count': row[6],
                    'total_score': row[7],
                    'search_strategy': row[8],
                    'duration_ms': row[9],
                    'token_usage': row[10],
                    'cost_usd': row[11],
                    'created_at': row[12].isoformat() if row[12] else None
                })
            
            return logs
    
    async def get_popular_queries(
        self,
        workspace_id: str,
        days: int = 7,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """获取热门查询"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            query = text("""
                SELECT LEFT(query, 100) as query_preview,
                       COUNT(*) as query_count,
                       AVG(result_count) as avg_results,
                       SUM(cost_usd) as total_cost,
                       AVG(duration_ms) as avg_duration_ms
                FROM screenplay.retrieval_logs
                WHERE workspace_id = :workspace_id
                      AND created_at >= NOW() - INTERVAL '1 day' * :days
                GROUP BY LEFT(query, 100)
                ORDER BY query_count DESC
                LIMIT :limit
            """)
            
            result = await session.execute(query, {
                'workspace_id': workspace_id,
                'days': days,
                'limit': limit
            })
            rows = result.fetchall()
            
            queries = []
            for row in rows:
                queries.append({
                    'query_preview': row[0],
                    'query_count': row[1],
                    'avg_results': round(row[2], 2) if row[2] else 0,
                    'total_cost': round(row[3], 6) if row[3] else 0,
                    'avg_duration_ms': round(row[4], 2) if row[4] else 0
                })
            
            return queries
    
    async def get_stats(self, workspace_id: str, days: int = 7) -> Dict[str, Any]:
        """获取检索统计"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            total_query = text("""
                SELECT COUNT(*) as total,
                       AVG(result_count) as avg_results,
                       AVG(duration_ms) as avg_duration_ms,
                       SUM(cost_usd) as total_cost
                FROM screenplay.retrieval_logs
                WHERE workspace_id = :workspace_id
                      AND created_at >= NOW() - INTERVAL '1 day' * :days
            """)
            
            result = await session.execute(total_query, {
                'workspace_id': workspace_id,
                'days': days
            })
            row = result.fetchone()
            
            strategy_query = text("""
                SELECT search_strategy,
                       COUNT(*) as count,
                       AVG(result_count) as avg_results
                FROM screenplay.retrieval_logs
                WHERE workspace_id = :workspace_id
                      AND created_at >= NOW() - INTERVAL '1 day' * :days
                GROUP BY search_strategy
                ORDER BY count DESC
            """)
            
            strategy_result = await session.execute(strategy_query, {
                'workspace_id': workspace_id,
                'days': days
            })
            strategy_rows = strategy_result.fetchall()
            
            return {
                'workspace_id': workspace_id,
                'period_days': days,
                'total_retrievals': row[0] or 0,
                'avg_results_per_query': round(row[1], 2) if row[1] else 0,
                'avg_duration_ms': round(row[2], 2) if row[2] else 0,
                'total_cost_usd': round(row[3], 6) if row[3] else 0,
                'by_strategy': [
                    {
                        'strategy': sr[0],
                        'count': sr[1],
                        'avg_results': round(sr[2], 2) if sr[2] else 0
                    }
                    for sr in strategy_rows
                ]
            }
    
    async def cleanup_old_logs(self, retention_days: int = 30) -> int:
        """清理旧日志"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        async with self._session_factory() as session:
            query = text("""
                SELECT screenplay.cleanup_old_logs(:retention_days)
            """)
            
            result = await session.execute(query, {'retention_days': retention_days})
            deleted_count = result.scalar()
            await session.commit()
            
            return deleted_count or 0
