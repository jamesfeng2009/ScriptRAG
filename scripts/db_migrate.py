#!/usr/bin/env python3
"""
数据库迁移管理脚本

使用 Alembic 进行数据库版本管理和迁移。
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def run_command(cmd: list, cwd: str = None) -> int:
    """运行命令并返回退出码"""
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")
    
    result = subprocess.run(cmd, cwd=cwd or project_root)
    return result.returncode


def check_database_connection():
    """检查数据库连接"""
    try:
        import asyncpg
        import asyncio
        
        async def test_connection():
            conn = await asyncpg.connect(
                host="localhost",
                port=5433,
                user="postgres",
                password="123456",
                database="Screenplay"
            )
            await conn.close()
            return True
        
        asyncio.run(test_connection())
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def init_alembic():
    """初始化 Alembic（如果尚未初始化）"""
    alembic_dir = project_root / "alembic"
    if not alembic_dir.exists():
        print("Initializing Alembic...")
        return run_command(["alembic", "init", "alembic"])
    else:
        print("Alembic already initialized")
        return 0


def create_migration(message: str, autogenerate: bool = True):
    """创建新的迁移文件"""
    cmd = ["alembic", "revision"]
    if autogenerate:
        cmd.append("--autogenerate")
    cmd.extend(["-m", message])
    
    return run_command(cmd)


def upgrade_database(revision: str = "head"):
    """升级数据库到指定版本"""
    return run_command(["alembic", "upgrade", revision])


def downgrade_database(revision: str):
    """降级数据库到指定版本"""
    return run_command(["alembic", "downgrade", revision])


def show_current_revision():
    """显示当前数据库版本"""
    return run_command(["alembic", "current"])


def show_history():
    """显示迁移历史"""
    return run_command(["alembic", "history", "--verbose"])


def show_heads():
    """显示当前头部版本"""
    return run_command(["alembic", "heads"])


def stamp_database(revision: str):
    """标记数据库版本（不执行迁移）"""
    return run_command(["alembic", "stamp", revision])


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("""
数据库迁移管理脚本

用法:
    python scripts/db_migrate.py <command> [args]

命令:
    init                    - 初始化 Alembic
    check                   - 检查数据库连接
    create <message>        - 创建新的迁移文件（自动生成）
    create-empty <message>  - 创建空的迁移文件
    upgrade [revision]      - 升级数据库（默认到最新版本）
    downgrade <revision>    - 降级数据库到指定版本
    current                 - 显示当前数据库版本
    history                 - 显示迁移历史
    heads                   - 显示头部版本
    stamp <revision>        - 标记数据库版本（不执行迁移）

示例:
    python scripts/db_migrate.py check
    python scripts/db_migrate.py create "Add user table"
    python scripts/db_migrate.py upgrade
    python scripts/db_migrate.py downgrade -1
    python scripts/db_migrate.py current
        """)
        return 1
    
    command = sys.argv[1]
    
    if command == "init":
        return init_alembic()
    
    elif command == "check":
        return 0 if check_database_connection() else 1
    
    elif command == "create":
        if len(sys.argv) < 3:
            print("Error: Migration message required")
            return 1
        message = sys.argv[2]
        return create_migration(message, autogenerate=True)
    
    elif command == "create-empty":
        if len(sys.argv) < 3:
            print("Error: Migration message required")
            return 1
        message = sys.argv[2]
        return create_migration(message, autogenerate=False)
    
    elif command == "upgrade":
        revision = sys.argv[2] if len(sys.argv) > 2 else "head"
        return upgrade_database(revision)
    
    elif command == "downgrade":
        if len(sys.argv) < 3:
            print("Error: Revision required for downgrade")
            return 1
        revision = sys.argv[2]
        return downgrade_database(revision)
    
    elif command == "current":
        return show_current_revision()
    
    elif command == "history":
        return show_history()
    
    elif command == "heads":
        return show_heads()
    
    elif command == "stamp":
        if len(sys.argv) < 3:
            print("Error: Revision required for stamp")
            return 1
        revision = sys.argv[2]
        return stamp_database(revision)
    
    else:
        print(f"Unknown command: {command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())