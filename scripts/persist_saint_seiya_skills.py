#!/usr/bin/env python
"""
将圣斗士星矢主题的技能配置持久化到 Screenplay 数据库的 skills 表

使用方法:
    python scripts/persist_saint_seiya_skills.py --persist
"""

import argparse
import asyncio
import yaml
from pathlib import Path

from src.services.persistence.skill_persistence_service import SkillDatabaseService, SkillRecord


async def persist_saint_seiya_skills():
    """将圣斗士星矢的技能配置持久化到 Screenplay 数据库"""
    print("=" * 60)
    print("圣斗士星矢技能落库工具")
    print("=" * 60)

    skill_service = SkillDatabaseService.create_from_env()
    await skill_service.connect()
    theme_path = Path(__file__).parent.parent / "config" / "themes" / "saint_seiya.yaml"

    if not theme_path.exists():
        print(f"❌ 主题配置文件不存在: {theme_path}")
        return

    with open(theme_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)

    skills_config = raw_config.get('skills', {})
    print(f"\n📋 找到 {len(skills_config)} 个圣斗士星矢技能配置:")
    print()

    existing_skills = []
    new_skills = []

    for skill_id, skill_config in skills_config.items():
        prompt_config = skill_config.get('prompt_config', {})
        existing_record = await skill_service.get(skill_id)

        if existing_record:
            print(f"   ✅ {skill_id} - 已存在")
            existing_skills.append(skill_id)
        else:
            print(f"   🆕 {skill_id} - 新增")
            new_skills.append(skill_id)

        print(f"      描述: {skill_config.get('description', '')}")
        print(f"      语气: {skill_config.get('tone', 'neutral')}")
        print(f"      Temperature: {prompt_config.get('temperature', 0.7)}")
        print()

    print("-" * 60)
    print(f"📊 统计:")
    print(f"   - 已存在: {len(existing_skills)} 个")
    print(f"   - 新增: {len(new_skills)} 个")
    print("-" * 60)

    if not new_skills:
        print("\n✅ 所有技能已存在，无需新增")
        return

    print(f"\n🚀 开始持久化 {len(new_skills)} 个圣斗士技能到 Screenplay 数据库...")
    print()

    for skill_id in new_skills:
        skill_config = skills_config[skill_id]
        prompt_config = skill_config.get('prompt_config', {})

        try:
            record = SkillRecord(
                skill_name=skill_id,
                description=skill_config.get('description', ''),
                tone=skill_config.get('tone', 'neutral'),
                prompt_config=prompt_config
            )

            await skill_service.create(record)
            print(f"   ✅ 成功保存: {skill_id}")

        except Exception as e:
            print(f"   ❌ 保存失败: {skill_id} - {str(e)}")

    print()
    print("=" * 60)
    print("✅ 技能持久化完成")
    print("=" * 60)

    await show_all_skills()


async def show_all_skills():
    """显示 Screenplay 数据库中所有的技能"""
    print("\n📋 Screenplay 数据库 skills 表中的所有技能:")
    print("-" * 60)

    skill_service = SkillDatabaseService.create_from_env()
    await skill_service.connect()
    all_skills = await skill_service.get_all()

    for skill in all_skills:
        print(f"   • {skill.skill_name}")
        desc = skill.description or ''
        print(f"     描述: {desc[:50]}..." if len(desc) > 50 else f"     描述: {desc}")
        print(f"     语气: {skill.tone}")
        print(f"     启用: {'是' if skill.is_enabled else '否'}")
        print()

    print(f"📊 总计: {len(all_skills)} 个技能")
    print("-" * 60)


async def main():
    parser = argparse.ArgumentParser(description="圣斗士星矢技能落库工具")
    parser.add_argument(
        '--show', '-s',
        action='store_true',
        help='显示 Screenplay 数据库中所有技能'
    )
    parser.add_argument(
        '--persist', '-p',
        action='store_true',
        help='将圣斗士星矢技能持久化到数据库'
    )

    args = parser.parse_args()

    if args.show:
        await show_all_skills()
    elif args.persist:
        await persist_saint_seiya_skills()
    else:
        await persist_saint_seiya_skills()


if __name__ == "__main__":
    asyncio.run(main())
