"""Infrastructure Layer - Data access, logging, monitoring

本模块包含基础设施相关组件：
- error_handler.py: 原始错误处理（v1）
- error_handler_v2.py: v2.1 错误处理（推荐）
- logging.py: 日志配置
- metrics.py: 性能指标
- audit_logger.py: 审计日志
- error_handler.py: 统一错误处理
"""

from . import error_handler
from . import logging as infra_logging
from . import metrics
from . import audit_logger

__all__ = [
    "error_handler",
    "logging",
    "metrics",
    "audit_logger",
]

try:
    from . import error_handler_v2
    __all__.append("error_handler_v2")
except ImportError:
    pass
