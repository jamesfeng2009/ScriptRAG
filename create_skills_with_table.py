#!/usr/bin/env python3
"""创建 skills 表并添加阴阳师技能"""

import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime


async def create_table_and_skills():
    """创建 skills 表并添加技能"""
    conn = await asyncpg.connect(
        host="localhost",
        port=5433,
        user="postgres",
        password="123456",
        database="Screenplay"
    )

    now = datetime.now()

    # 1. 创建 skills 表
    print("📝 创建 skills 表...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id SERIAL PRIMARY KEY,
            skill_name VARCHAR(100) UNIQUE NOT NULL,
            description TEXT,
            tone VARCHAR(255),
            compatible_with JSONB DEFAULT '[]',
            prompt_config JSONB DEFAULT '{}',
            is_enabled BOOLEAN DEFAULT true,
            is_default BOOLEAN DEFAULT false,
            extra_data JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✅ skills 表创建成功")

    # 2. 创建索引
    print("📝 创建索引...")
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_skills_skill_name ON skills(skill_name)
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_skills_enabled ON skills(is_enabled)
    """)
    print("✅ 索引创建成功")

    # 3. 插入技能1: mysterious_fantasy
    print("\n📝 插入技能 mysterious_fantasy...")
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
    print("✅ 技能 mysterious_fantasy 插入成功")

    # 4. 插入技能2: hot_battle
    print("\n📝 插入技能 hot_battle...")
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
    print("✅ 技能 hot_battle 插入成功")

    # 5. 验证
    print("\n📊 验证技能...")
    result = await conn.fetch("SELECT id, skill_name, description, tone, is_enabled FROM skills ORDER BY id")
    print(f"\n✅ 数据库中现有 {len(result)} 个技能:\n")
    for skill in result:
        status = "✅ 启用" if skill['is_enabled'] else "❌ 禁用"
        print(f"  [{skill['id']}] {skill['skill_name']}")
        print(f"      描述: {skill['description']}")
        print(f"      语调: {skill['tone']}")
        print(f"      状态: {status}\n")

    await conn.close()


async def main():
    print("="*60)
    print("🎎 创建 skills 表并添加阴阳师技能")
    print("="*60 + "\n")

    await create_table_and_skills()
    
    print("="*60)
    print("🎉 技能创建完成！")
    print("="*60)
    print("\n📝 使用方法：")
    print("1. 启动 API 服务器: uvicorn src.presentation.api:app --reload")
    print("2. 访问 http://localhost:8000/docs")
    print("3. 使用 /generate 接口生成剧本，指定 skill:")
    print('   {"topic": "阴阳师安培晴明退治恶鬼", "skill": {"initial_skill": "mysterious_fantasy"}}')
    print("4. 运行时切换技能: POST /adjust/{task_id} {action: 'switch_skill', 'skill': 'hot_battle'}")


if __name__ == "__main__":
    asyncio.run(main())
