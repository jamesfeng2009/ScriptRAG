"""数据库工具模块 - 统一的数据库连接配置管理

提供数据库 URL 构建、连接池管理等统一功能。
消除各服务中的重复代码。
"""

from typing import Optional
from ..config import get_database_config, DatabaseConfig


def build_db_url(db_config: Optional[DatabaseConfig] = None) -> str:
    """构建数据库连接 URL

    Args:
        db_config: 数据库配置对象。如果为 None，则从全局配置获取。

    Returns:
        PostgreSQL 异步连接 URL 字符串
    """
    if db_config is None:
        db_config = get_database_config()

    return (
        f"postgresql+asyncpg://{db_config.user}:{db_config.password}"
        f"@{db_config.host}:{db_config.port}/{db_config.database}"
    )


def build_sync_db_url(db_config: Optional[DatabaseConfig] = None) -> str:
    """构建同步数据库连接 URL

    Args:
        db_config: 数据库配置对象。如果为 None，则从全局配置获取。

    Returns:
        PostgreSQL 同步连接 URL 字符串
    """
    if db_config is None:
        db_config = get_database_config()

    return (
        f"postgresql://{db_config.user}:{db_config.password}"
        f"@{db_config.host}:{db_config.port}/{db_config.database}"
    )
