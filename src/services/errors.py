"""Service Errors - Unified error handling for service layer"""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import time


class ErrorSeverity(Enum):
    """错误严重级别"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ErrorCategory(Enum):
    """错误分类"""
    RETRIEVAL = "retrieval"
    LLM = "llm"
    RAG = "rag"
    CACHE = "cache"
    SESSION = "session"
    STORAGE = "storage"
    MONITORING = "monitoring"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class ServiceErrorContext:
    """错误上下文信息"""
    service_name: str
    operation: str
    timestamp: float = field(default_factory=time.time)
    request_id: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


class ServiceError(Exception):
    """服务层错误基类"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[ServiceErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.context = context
        self.cause = cause
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp,
            "context": {
                "service_name": self.context.service_name if self.context else None,
                "operation": self.context.operation if self.context else None,
            } if self.context else None,
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class RetrievalServiceError(ServiceError):
    """检索服务错误"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "RETRIEVAL_ERROR",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ServiceErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            category=ErrorCategory.RETRIEVAL,
            context=context,
            cause=cause
        )


class LLMServiceError(ServiceError):
    """LLM 服务错误"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "LLM_ERROR",
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[ServiceErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            category=ErrorCategory.LLM,
            context=context,
            cause=cause
        )


class RAGServiceError(ServiceError):
    """RAG 服务错误"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "RAG_ERROR",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ServiceErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            category=ErrorCategory.RAG,
            context=context,
            cause=cause
        )


class CacheServiceError(ServiceError):
    """缓存服务错误"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "CACHE_ERROR",
        severity: ErrorSeverity = ErrorSeverity.LOW,
        context: Optional[ServiceErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            category=ErrorCategory.CACHE,
            context=context,
            cause=cause
        )


class SessionServiceError(ServiceError):
    """会话服务错误"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "SESSION_ERROR",
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[ServiceErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            category=ErrorCategory.SESSION,
            context=context,
            cause=cause
        )


class StorageServiceError(ServiceError):
    """存储服务错误"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "STORAGE_ERROR",
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[ServiceErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            category=ErrorCategory.STORAGE,
            context=context,
            cause=cause
        )


class ValidationError(ServiceError):
    """验证错误"""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        error_code: str = "VALIDATION_ERROR",
        severity: ErrorSeverity = ErrorSeverity.LOW,
        context: Optional[ServiceErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        self.field_name = field_name
        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            category=ErrorCategory.VALIDATION,
            context=context,
            cause=cause
        )
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["field_name"] = self.field_name
        return result


def create_error_context(
    service_name: str,
    operation: str,
    request_id: Optional[str] = None,
    **additional_info
) -> ServiceErrorContext:
    """创建错误上下文"""
    return ServiceErrorContext(
        service_name=service_name,
        operation=operation,
        request_id=request_id,
        additional_info=additional_info
    )
