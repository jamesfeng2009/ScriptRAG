#!/usr/bin/env python3
"""创建阴阳师主题的技能（持久化到数据库）"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DatabaseConfig
from src.services.skill_persistence_service import SkillRecord


async def create_skills_with_db():
    """使用数据库创建技能"""
    try:
        from src.services.skill_persistence_service import SkillDatabaseService, SkillService
        
        print("🔗 尝试连接数据库...")
        db_config = DatabaseConfig.from_env()
        print(f"   主机: {db_config.host}, 端口: {db_config.port}, 数据库: {db_config.database}")
        
        skill_db_service = SkillDatabaseService(
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            user=db_config.user,
            password=db_config.password,
            echo=db_config.echo
        )
        
        await skill_db_service.connect()
        print("✅ 数据库连接成功\n")
        
        skill_service = SkillService(skill_db_service, enable_cache=True)
        
        now = datetime.now()
        
        skill1 = SkillRecord(
            skill_name="mysterious_fantasy",
            description="神秘玄幻风格，适合阴阳师题材",
            tone="神秘、玄幻、古典",
            compatible_with=["standard_tutorial"],
            prompt_config={
                "system_prompt": """你是一个神秘的阴阳师，用玄幻古典的语言风格写作，注重意境和氛围营造。

写作风格要求：
1. 语言古朴典雅，带有平安时代的韵味
2. 注重意境的描写，如月色、雾气、符咒等
3. 善用五感描写，营造神秘氛围
4. 战斗场景要飘逸灵动，讲究招式美感
5. 人物对话要符合古代阴阳师的身份

示例：
- "月华如练，一道银白色的光幕笼罩着夜空。"
- "晴明轻摇折扇，扇面上绘制的神秘符文泛起淡淡金光。"
- "恶鬼咆哮之声震落檐下风铃，却见晴明神色自若，唇角含笑。" """,
                "temperature": 0.8,
                "max_tokens": 2000
            },
            is_enabled=True,
            is_default=False,
            created_at=now,
            updated_at=now
        )
        
        await skill_service.create(skill1)
        print(f"✅ 技能1创建成功: {skill1.skill_name}\n")
        print(f"📜 技能1: {skill1.skill_name}")
        print(f"   描述: {skill1.description}")
        print(f"   语调: {skill1.tone}\n")
        
        skill2 = SkillRecord(
            skill_name="hot_battle",
            description="热血战斗风格，适合阴阳师题材",
            tone="热血、激昂、战斗",
            compatible_with=["standard_tutorial"],
            prompt_config={
                "system_prompt": """你是一个热血的战斗导演，用激昂的语言风格写作，注重动作和战斗场面。

写作风格要求：
1. 节奏紧凑，场面宏大
2. 动作描写要淋漓尽致，拳拳到肉
3. 战斗口号要震撼人心，激昂澎湃
4. 人物要有英雄气概，临危不惧
5. 关键时刻要有爆发力，让读者热血沸腾

示例：
- "刀光如虹，恶鬼的头颅应声而落！"
- "晴明眼中燃起金色的火焰，符咒化作漫天火雨！"
- "这一刻，天地为之变色！恶鬼发出撕心裂肺的惨叫！" """,
                "temperature": 0.9,
                "max_tokens": 2000
            },
            is_enabled=True,
            is_default=False,
            created_at=now,
            updated_at=now
        )
        
        await skill_service.create(skill2)
        print(f"✅ 技能2创建成功: {skill2.skill_name}\n")
        print(f"📜 技能2: {skill2.skill_name}")
        print(f"   描述: {skill2.description}")
        print(f"   语调: {skill2.tone}\n")
        
        all_skills = await skill_service.get_all()
        print(f"📋 当前共有 {len(all_skills)} 个技能:")
        for s in all_skills:
            print(f"   - {s.skill_name} ({s.tone})")
        
        await skill_db_service.close()
        print("\n🎉 技能创建完成！")
        
    except Exception as e:
        print(f"❌ 数据库错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*60)
    print("🎎 创建阴阳师主题 Skills（持久化到数据库）")
    print("="*60 + "\n")
    
    asyncio.run(create_skills_with_db())
    
    print("\n" + "="*60)
    print("📝 使用方法：")
    print("="*60)
    print("1. 启动 API 服务器: uvicorn src.presentation.api:app --reload")
    print("2. 访问 http://localhost:8000/docs")
    print("3. 使用 /generate 接口生成剧本，指定 skill:")
    print('   {"topic": "阴阳师安培晴明退治恶鬼", "skill": {"initial_skill": "mysterious_fantasy"}}')
    print("4. 运行时切换技能: POST /adjust/{task_id} {action: 'switch_skill', skill: 'hot_battle'}")
