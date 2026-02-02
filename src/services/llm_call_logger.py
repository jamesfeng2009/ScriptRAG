"""
LLM 调用记录所有 LLM API服务

用于记录 调用，便于追踪模型使用情况和问题排查。
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ..config import get_database_config

logger = logging.getLogger(__name__)


class LLMCallLogger:
    """LLM 调用记录器"""

    _instance = None
    _engine = None
    _session_factory = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_initialized(self):
        """确保数据库连接已初始化"""
        if self._initialized and self._engine is not None:
            return
        
        logger.debug(f"LLMCallLogger: Initializing... (initialized={self._initialized})")
        
        try:
            db_config = get_database_config()
            logger.debug(f"LLMCallLogger: Using config - {db_config.host}:{db_config.port}/{db_config.database}")
            
            database_url = db_config.url
            
            self._engine = create_engine(
                database_url,
                poolclass=None,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=False,
                connect_args={
                    "options": "-c search_path=screenplay"
                }
            )
            self._session_factory = sessionmaker(bind=self._engine)
            self._initialized = True
            logger.info(f"LLMCallLogger: Database connection initialized successfully")
        except Exception as e:
            logger.error(f"LLMCallLogger: Failed to initialize: {e}")
            self._engine = None
            self._session_factory = None
            self._initialized = False

    def log_call(
        self,
        task_id: Optional[str],
        provider: str,
        model: str,
        request_type: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        response_time_ms: Optional[float] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
        cost_estimate: Optional[float] = None,
        request_preview: Optional[str] = None,
        response_preview: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """记录一次 LLM 调用"""
        self._ensure_initialized()
        
        if self._engine is None:
            logger.warning("LLMCallLogger: Database not available, skipping log")
            return False

        total_tokens = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        query = text("""
            INSERT INTO llm_call_logs (
                task_id, provider, model, request_type,
                input_tokens, output_tokens, total_tokens,
                response_time_ms, status, error_message, error_code,
                cost_estimate, request_preview, response_preview,
                extra_data, created_at
            ) VALUES (
                :task_id, :provider, :model, :request_type,
                :input_tokens, :output_tokens, :total_tokens,
                :response_time_ms, :status, :error_message, :error_code,
                :cost_estimate, :request_preview, :response_preview,
                :extra_data, NOW()
            )
        """)

        try:
            with self._engine.connect() as conn:
                conn.execute(query, {
                    "task_id": task_id,
                    "provider": provider,
                    "model": model,
                    "request_type": request_type,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "response_time_ms": response_time_ms,
                    "status": status,
                    "error_message": error_message,
                    "error_code": error_code,
                    "cost_estimate": cost_estimate,
                    "request_preview": request_preview[:1000] if request_preview else None,
                    "response_preview": response_preview[:1000] if response_preview else None,
                    "extra_data": extra_data
                })
                conn.commit()
            
            logger.debug(f"LLM call logged: {provider}/{model} ({status})")
            return True
        except Exception as e:
            logger.error(f"Failed to log LLM call: {e}")
            return False

    def get_provider_stats(self, provider: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        """获取调用统计"""
        self._ensure_initialized()
        
        if self._engine is None:
            return {"error": "Database not available"}

        try:
            where_clause = "WHERE provider = :provider" if provider else ""
            params = {"provider": provider, "days": days} if provider else {"days": days}

            query = text(f"""
                SELECT 
                    provider,
                    model,
                    COUNT(*) as total_calls,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_calls,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_calls,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    AVG(response_time_ms) as avg_response_time_ms
                FROM llm_call_logs
                {where_clause}
                AND created_at > NOW() - INTERVAL '{days} days'
                GROUP BY provider, model
                ORDER BY total_calls DESC
            """)

            with self._engine.connect() as conn:
                result = conn.execute(query, params)
                rows = result.fetchall()

            stats = []
            for row in rows:
                stats.append({
                    "provider": row[0],
                    "model": row[1],
                    "total_calls": row[2],
                    "success_calls": row[3],
                    "failed_calls": row[4],
                    "total_input_tokens": row[5] or 0,
                    "total_output_tokens": row[6] or 0,
                    "avg_response_time_ms": round(row[7] or 0, 2)
                })

            return {"stats": stats, "period_days": days}
        except Exception as e:
            logger.error(f"Failed to get provider stats: {e}")
            return {"error": str(e)}

    def get_task_calls(self, task_id: str) -> list:
        """获取某个任务的所有 LLM 调用"""
        self._ensure_initialized()
        
        if self._engine is None:
            return []

        try:
            query = text("""
                SELECT 
                    id, provider, model, request_type,
                    input_tokens, output_tokens, response_time_ms,
                    status, created_at
                FROM llm_call_logs
                WHERE task_id = :task_id
                ORDER BY created_at ASC
            """)

            with self._engine.connect() as conn:
                result = conn.execute(query, {"task_id": task_id})
                rows = result.fetchall()

            calls = []
            for row in rows:
                calls.append({
                    "id": row[0],
                    "provider": row[1],
                    "model": row[2],
                    "request_type": row[3],
                    "input_tokens": row[4],
                    "output_tokens": row[5],
                    "response_time_ms": row[6],
                    "status": row[7],
                    "created_at": row[8].isoformat() if row[8] else None
                })

            return calls
        except Exception as e:
            logger.error(f"Failed to get task calls: {e}")
            return []


def get_llm_call_logger() -> LLMCallLogger:
    """获取 LLM 调用记录器单例"""
    return LLMCallLogger()
