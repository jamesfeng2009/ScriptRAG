import asyncio
import os
import sys
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy import inspect

from alembic import context

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入实体类
from src.domain.entities import Base

# 这是 Alembic 的配置对象
config = context.config

# 从环境变量读取数据库配置
DATABASE_URL = (
    f"postgresql+asyncpg://{os.getenv('POSTGRES_USER', 'postgres')}:"
    f"{os.getenv('POSTGRES_PASSWORD', '123456')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', 5432)}/"
    f"{os.getenv('POSTGRES_DB', 'term_sheet_db')}"
)
config.set_main_option("sqlalchemy.url", DATABASE_URL)

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 设置目标元数据
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线运行迁移（不需要活动数据库连接）"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """在给定连接上运行迁移"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # 支持 SQLite 和其他数据库的批量操作
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """异步运行迁移"""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations() -> None:
    """确定运行迁移的方式并执行"""
    if context.is_offline_mode():
        run_migrations_offline()
    else:
        asyncio.run(run_async_migrations())


if __name__ == "__main__":
    run_migrations()
