"""Logging Configuration - Structured logging setup"""

import logging
import sys
import json
import traceback
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """用于结构化日志的 JSON 格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data, ensure_ascii=False)


class AgentLogger:
    """
    支持结构化日志的增强型智能体操作日志记录器。
    
    提供以下日志记录方法：
    - 智能体转换
    - 转向触发
    - RAG 检索结果
    - 技能切换
    - 事实检查结果
    - 重试尝试
    - 带堆栈跟踪的错误
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_agent_transition(
        self,
        from_agent: str,
        to_agent: str,
        step_id: Optional[int] = None,
        reason: Optional[str] = None
    ):
        """Log agent transition with timestamp"""
        extra_data = {
            "event_type": "agent_transition",
            "from_agent": from_agent,
            "to_agent": to_agent,
            "step_id": step_id,
            "reason": reason
        }
        self.logger.info(
            f"Agent transition: {from_agent} -> {to_agent}",
            extra={"extra_data": extra_data}
        )
    
    def log_pivot_trigger(
        self,
        step_id: int,
        pivot_reason: str,
        conflict_details: Optional[Dict[str, Any]] = None
    ):
        """Log pivot trigger with reason"""
        extra_data = {
            "event_type": "pivot_trigger",
            "step_id": step_id,
            "pivot_reason": pivot_reason,
            "conflict_details": conflict_details
        }
        self.logger.warning(
            f"Pivot triggered for step {step_id}: {pivot_reason}",
            extra={"extra_data": extra_data}
        )
    
    def log_retrieval_result(
        self,
        step_id: int,
        doc_count: int,
        sources: list,
        retrieval_method: str,
        confidence_scores: Optional[list] = None
    ):
        """Log RAG retrieval results with sources"""
        extra_data = {
            "event_type": "retrieval_result",
            "step_id": step_id,
            "doc_count": doc_count,
            "sources": sources,
            "retrieval_method": retrieval_method,
            "confidence_scores": confidence_scores
        }
        self.logger.info(
            f"Retrieved {doc_count} documents for step {step_id}",
            extra={"extra_data": extra_data}
        )
    
    def log_skill_switch(
        self,
        step_id: int,
        from_skill: str,
        to_skill: str,
        trigger_reason: str,
        complexity_score: Optional[float] = None
    ):
        """Log skill switch with trigger reason"""
        extra_data = {
            "event_type": "skill_switch",
            "step_id": step_id,
            "from_skill": from_skill,
            "to_skill": to_skill,
            "trigger_reason": trigger_reason,
            "complexity_score": complexity_score
        }
        self.logger.info(
            f"Skill switch for step {step_id}: {from_skill} -> {to_skill} ({trigger_reason})",
            extra={"extra_data": extra_data}
        )
    
    def log_fact_check_result(
        self,
        step_id: int,
        is_valid: bool,
        hallucinations: list,
        verification_method: str
    ):
        """Log fact checker verification results"""
        extra_data = {
            "event_type": "fact_check_result",
            "step_id": step_id,
            "is_valid": is_valid,
            "hallucination_count": len(hallucinations),
            "hallucinations": hallucinations,
            "verification_method": verification_method
        }
        if is_valid:
            self.logger.info(
                f"Fact check passed for step {step_id}",
                extra={"extra_data": extra_data}
            )
        else:
            self.logger.warning(
                f"Fact check failed for step {step_id}: {len(hallucinations)} hallucinations detected",
                extra={"extra_data": extra_data}
            )
    
    def log_retry_attempt(
        self,
        step_id: int,
        retry_count: int,
        max_retries: int,
        reason: str
    ):
        """Log retry attempts"""
        extra_data = {
            "event_type": "retry_attempt",
            "step_id": step_id,
            "retry_count": retry_count,
            "max_retries": max_retries,
            "reason": reason
        }
        self.logger.warning(
            f"Retry attempt {retry_count}/{max_retries} for step {step_id}: {reason}",
            extra={"extra_data": extra_data}
        )
    
    def log_degradation(
        self,
        step_id: int,
        degradation_type: str,
        reason: str,
        action_taken: str
    ):
        """Log graceful degradation actions"""
        extra_data = {
            "event_type": "degradation",
            "step_id": step_id,
            "degradation_type": degradation_type,
            "reason": reason,
            "action_taken": action_taken
        }
        self.logger.warning(
            f"Degradation for step {step_id}: {degradation_type} - {action_taken}",
            extra={"extra_data": extra_data}
        )
    
    def log_error_with_context(
        self,
        error: Exception,
        context: Dict[str, Any],
        agent_name: Optional[str] = None
    ):
        """Log error with full context and stack trace"""
        extra_data = {
            "event_type": "error",
            "agent_name": agent_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        self.logger.error(
            f"Error in {agent_name or 'unknown'}: {str(error)}",
            exc_info=True,
            extra={"extra_data": extra_data}
        )


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    use_json: bool = False
):
    """
    配置应用程序的结构化日志。
    
    Args:
        level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        log_file: 日志输出的可选文件路径
        format_string: 可选的自定义格式字符串
        use_json: 使用 JSON 结构化日志格式
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    # Create formatter
    if use_json:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(format_string)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set third-party library log levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)


def get_agent_logger(name: str) -> AgentLogger:
    """
    获取增强型智能体日志记录器实例
    
    Args:
        name: 日志记录器名称（通常使用 __name__）
    
    Returns:
        AgentLogger 实例
    """
    return AgentLogger(logging.getLogger(name))
