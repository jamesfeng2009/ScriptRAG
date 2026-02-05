#!/usr/bin/env python3
"""直接连接数据库创建阴阳师技能"""

import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime


async def test_connection():
    """测试数据库连接"""
    # 尝试多个端口
    ports = [5432, 5433]
    host = "localhost"
    user = "postgres"
    password = "123456"
    database = "Screenplay"

    for port in ports:
        try:
            print(f"🔗 尝试连接 {host}:{port}/{database}...")
            conn = await asyncpg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database
            )
            print(f"✅ 连接成功！端口: {port}")
            
            # 检查表是否存在
            result = await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            tables = [r['table_name'] for r in result]
            print(f"📊 数据库中的表: {tables}")
            
            await conn.close()
            return port
        except Exception as e:
            print(f"❌ 端口 {port} 失败: {e}")
    
    return None


async def create_skills(port: int):
    """创建技能到数据库"""
    conn = await asyncpg.connect(
        host="localhost",
        port=port,
        user="postgres",
        password="123456",
        database="Screenplay"
    )

    now = datetime.now()

    # 技能1: mysterious_fantasy
    try:
        await conn.execute("""
            INSERT INTO skills (skill_name, description, tone, compatible_with, prompt_config, is_enabled, is_default, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (skill_name) DO UPDATE SET
                description = EXCLUDED.description,
                tone = EXCLUDED.tone,
                compatible_with = EXCLUDED.compatible_with,
                prompt_config = EXCLUDED.prompt_config,
                updated_at = EXCLUDED.updated_at
        """, 
            "mysterious_fantasy",
            "神秘玄幻风格，适合阴阳师题材",
            "神秘、玄幻、古典",
            '["standard_tutorial"]',
            '{"system_prompt": "你是一个神秘的阴阳师，用玄幻古典的语言风格写作，注重意境和氛围营造。", "temperature": 0.8, "max_tokens": 2000}',
            True,
            False,
            now,
            now
        )
        print("✅ 技能 mysterious_fantasy 创建成功")
    except Exception as e:
        print(f"❌ 技能1失败: {e}")

    # 技能2: hot_battle
    try:
        await conn.execute("""
            INSERT INTO skills (skill_name, description, tone, compatible_with, prompt_config, is_enabled, is_default, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (skill_name) DO UPDATE SET
                description = EXCLUDED.description,
                tone = EXCLUDED.tone,
                compatible_with = EXCLUDED.compatible_with,
                prompt_config = EXCLUDED.prompt_config,
                updated_at = EXCLUDED.updated_at
        """,
            "hot_battle",
            "热血战斗风格，适合阴阳师题材",
            "热血、激昂、战斗",
            '["standard_tutorial"]',
            '{"system_prompt": "你是一个热血的战斗导演，用激昂的语言风格写作，注重动作和战斗场面。", "temperature": 0.9, "max_tokens": 2000}',
            True,
            False,
            now,
            now
        )
        print("✅ 技能 hot_battle 创建成功")
    except Exception as e:
        print(f"❌ 技能2失败: {e}")

    # 验证
    result = await conn.fetch("SELECT skill_name, description, tone FROM skills ORDER BY id")
    print(f"\n📊 数据库中现有 {len(result)} 个技能:")
    for skill in result:
        print(f"  - {skill['skill_name']}: {skill['description']}")

    await conn.close()


async def main():
    print("="*60)
    print("🎎 直接连接数据库创建阴阳师技能")
    print("="*60 + "\n")

    # 测试连接
    port = await test_connection()
    if port is None:
        print("❌ 无法连接到数据库")
        return

    # 创建技能
    print("\n📝 创建技能...")
    await create_skills(port)
    
    print("\n" + "="*60)
    print("🎉 技能创建完成！")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
