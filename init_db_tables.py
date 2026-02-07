#!/usr/bin/env python
"""初始化数据库表"""

import asyncio
from src.services.persistence.task_persistence_service import TaskDatabaseService
from src.services.persistence.agent_execution_persistence_service import AgentExecutionDatabaseService


async def init_tables():
    """创建所有缺失的表"""
    print("🔧 初始化数据库表...")

    task_service = TaskDatabaseService.create_from_env()
    agent_service = AgentExecutionDatabaseService.create_from_env()

    try:
        await task_service.create_tables()
        print("✅ tasks 表创建成功")

        await agent_service.create_tables()
        print("✅ agent_executions 表创建成功")

    except Exception as e:
        print(f"❌ 创建表失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await task_service.close()
        await agent_service.close()


if __name__ == "__main__":
    asyncio.run(init_tables())
