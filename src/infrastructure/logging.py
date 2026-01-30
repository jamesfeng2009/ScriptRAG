"""Logging Configuration - Structured logging setup"""

import logging
import sys
import json
import traceback
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
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
    Enhanced logger for agent operations with structured logging support.
    
    Provides methods for logging:
    - Agent transitions
    - Pivot triggers
    - RAG retrieval results
    - Skill switches
    - Fact checker results
    - Retry attempts
    - Errors with stack traces
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


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    use_json: bool = False
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        format_string: Optional custom format string
        use_json: Use JSON structured logging format
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


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def get_agent_logger(name: str) -> AgentLogger:
    """
    Get an enhanced agent logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        AgentLogger instance
    """
    return AgentLogger(logging.getLogger(name))
