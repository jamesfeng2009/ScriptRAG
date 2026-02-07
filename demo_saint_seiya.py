#!/usr/bin/env python
"""
圣斗士星矢剧本生成示例

演示如何使用主题配置文件生成剧本：
1. 检测用户意图的主题
2. 获取可用的技能选项
3. 根据用户选择的技能生成剧本
4. 将技能和任务记录持久化到数据库

支持两种模式：
- 直接LLM模式（默认）：快速测试
- 完整工作流模式：经过所有 agent，生成更丰富的落库记录

使用方法:
    # 直接LLM模式
    python demo_saint_seiya.py --skill heated_battle
    python demo_saint_seiya.py --skill strategic_approach

    # 完整工作流模式（经过 agent）
    python demo_saint_seiya.py --skill heated_battle --workflow

    # 交互模式
    python demo_saint_seiya.py --interactive
"""

import argparse
import logging
import yaml
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.config import get_llm_config
from src.services.llm.service import LLMService
from src.domain.skill_loader import SkillConfigLoader
from src.services.persistence.skill_persistence_service import SkillDatabaseService, SkillRecord
from src.services.persistence.task_persistence_service import TaskDatabaseService, TaskRecord
from src.application.orchestrator import WorkflowOrchestrator
from src.services.retrieval_service import RetrievalService
from src.services.parser.tree_sitter_parser import TreeSitterParser
from src.services.core.summarization_service import SummarizationService
from src.services.knowledge.universal_knowledge_service import UniversalKnowledgeRetrievalService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_workflow_services():
    """初始化工作流所需的服务"""
    print("🔧 初始化工作流服务...")

    config_path = Path(__file__).parent / "config" / "skills.yaml"
    themes_dir = Path(__file__).parent / "config" / "themes"

    llm_config = get_llm_config()

    with open('config.yaml', 'r') as f:
        config_data = yaml.safe_load(f)

    llm_providers = config_data.get("llm", {}).setdefault("providers", {})
    if llm_config.glm_api_key:
        llm_providers.setdefault("glm", {})["api_key"] = llm_config.glm_api_key
        llm_providers.setdefault("glm", {})["base_url"] = "https://open.bigmodel.cn/api/paas/v4"

    llm_service = LLMService(config_data.get('llm', {}))
    retrieval_service = UniversalKnowledgeRetrievalService(
        base_knowledge_dir=str(Path(__file__).parent / "data" / "knowledge"),
        default_theme="saint_seiya",
        enable_theme_detection=True
    )
    parser_service = TreeSitterParser()
    summarization_service = SummarizationService(llm_service)

    theme_loader = SkillConfigLoader(
        config_path=str(config_path),
        themes_dir=str(themes_dir)
    )

    skill_service = SkillDatabaseService.create_from_env()
    task_service = TaskDatabaseService.create_from_env()

    orchestrator = WorkflowOrchestrator(
        llm_service=llm_service,
        retrieval_service=retrieval_service,
        parser_service=parser_service,
        summarization_service=summarization_service,
        enable_agentic_rag=True,
        enable_dynamic_adjustment=False,
        enable_task_stack=False,
        enable_tools=False
    )

    print("✅ 工作流服务初始化完成 (简化模式)")
    return llm_service, theme_loader, skill_service, task_service, orchestrator


def detect_and_show_theme(theme_loader, user_query: str):
    """检测并显示主题信息"""
    print("\n" + "=" * 60)
    print(f"用户输入: {user_query}")
    print("=" * 60)

    theme_id = theme_loader.detect_theme(user_query)

    if not theme_id:
        print("\n❌ 未检测到主题，请尝试包含圣斗士相关关键词")
        return None

    print(f"\n✅ 检测到主题: {theme_id}")

    theme = theme_loader.load_theme(theme_id)
    if theme:
        print(f"   名称: {theme['name']}")
        print(f"   描述: {theme['description']}")

    return theme_id


def show_skill_options(theme_loader, theme_id: str):
    """显示可用的技能选项"""
    options = theme_loader.get_theme_skill_options(theme_id)

    print("\n📋 可用的技能选项:")
    for i, opt in enumerate(options, 1):
        icon = opt.get('icon', '•')
        print(f"   {i}. {icon} {opt['name']}")
        print(f"      {opt['description']}")
        print(f"      触发词: {opt.get('trigger_keywords', [])}")
        print()

    return options


def get_user_choice(options: list) -> Optional[str]:
    """获取用户选择的技能"""
    print("请选择技能编号 (直接回车使用默认技能): ", end='')
    choice = input().strip()

    if not choice:
        return options[0]['id'] if options else None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            return options[idx]['id']
        print("无效选择，使用默认技能")
        return options[0]['id'] if options else None
    except ValueError:
        print("请输入数字")
        return None


def generate_script_with_skill(llm_service, theme_loader, skill_service, task_service, theme_id: str, skill_id: str, user_query: str, step_description: str = "星矢来到狮子宫门前"):
    """使用指定技能生成剧本片段，并持久化到数据库（直接LLM模式）"""
    import asyncio

    skills = theme_loader.get_theme_skills(theme_id)

    if skill_id not in skills:
        print(f"❌ 技能 {skill_id} 不存在")
        return None

    skill_config = skills[skill_id]
    prompt_config = skill_config.get('prompt_config', {})
    system_prompt = prompt_config.get('system_prompt', '')
    user_template = prompt_config.get('user_template', '')

    user_prompt = user_template.format(
        step_description=step_description,
        retrieved_content="黄金圣斗士艾欧里亚正在狮子宫内等待挑战者..."
    )

    print(f"\n🎬 生成剧本片段 - 技能: {skill_id}")
    print("-" * 60)

    async def do_chat():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = await llm_service.chat_completion(
            messages=messages,
            temperature=prompt_config.get('temperature', 0.7),
            max_tokens=prompt_config.get('max_tokens', 2000)
        )
        return response

    try:
        content = asyncio.run(do_chat())

        print(content)

        if content:
            print("\n💾 持久化数据到数据库...")

            async def persist_data():
                skill_record = await skill_service.get(skill_id)
                if not skill_record:
                    print(f"   保存技能: {skill_id}")
                    await skill_service.create(SkillRecord(
                        skill_name=skill_id,
                        description=skill_config.get('description', ''),
                        tone=skill_config.get('tone', 'neutral'),
                        prompt_config=prompt_config
                    ))

                task_id = str(uuid.uuid4())
                print(f"   保存任务: {task_id}")
                await task_service.create(TaskRecord(
                    task_id=task_id,
                    status="completed",
                    topic=user_query[:200] if len(user_query) > 200 else user_query,
                    context="",
                    current_skill=skill_id,
                    screenplay=content[:10000] if len(content) > 10000 else content,
                    request_data={
                        "theme": theme_id,
                        "skill": skill_id,
                        "step_description": step_description
                    }
                ))

            asyncio.run(persist_data())
            print("   ✅ 数据持久化完成")

        return content

    except Exception as e:
        print(f"❌ 生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def generate_script_with_workflow(orchestrator, theme_loader, skill_service, task_service, skill_id: str, user_query: str):
    """使用完整工作流生成剧本（经过所有agent）"""
    print(f"\n🚀 启动完整工作流 - 技能: {skill_id}")
    print("=" * 60)

    initial_state = {
        "user_topic": user_query,
        "chat_history": [],
        "messages": [],
        "enable_dynamic_adjustment": True,
        "current_skill": skill_id
    }

    try:
        result = await orchestrator.execute(
            initial_state=initial_state,
            recursion_limit=100
        )

        if result["success"]:
            state = result["state"]
            screenplay = state.get("final_screenplay", "") or state.get("screenplay", "")
            outline = state.get("outline", [])
            skill_history = state.get("skill_history", [])

            print("\n✅ 工作流执行完成")
            print(f"   生成的剧本长度: {len(screenplay)} 字符")
            print(f"   大纲步骤数: {len(outline)}")
            print(f"   技能切换次数: {len(skill_history)}")

            if screenplay:
                print("\n📜 生成的剧本片段:")
                print("-" * 60)
                print(screenplay[:2000] + "..." if len(screenplay) > 2000 else screenplay)

            task_id = state.get("task_id", str(uuid.uuid4()))
            print(f"\n💾 任务ID: {task_id}")

            return {
                "success": True,
                "screenplay": screenplay,
                "outline": outline,
                "task_id": task_id
            }
        else:
            print(f"❌ 工作流执行失败: {result.get('error', '未知错误')}")
            return {
                "success": False,
                "error": result.get('error', '未知错误')
            }

    except Exception as e:
        print(f"❌ 工作流执行异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def interactive_mode(llm_service, theme_loader, skill_service, task_service):
    """交互模式"""
    print("\n🌟 圣斗士星矢剧本生成器 - 交互模式")
    print("输入 'quit' 退出")
    print("-" * 40)

    default_queries = [
        "我想看圣斗士星矢攻打十二宫",
        "星矢要用热血战斗击败黄金圣斗士",
        "星矢想要用智慧击败对手",
        "我想看星矢和伙伴们的感情故事"
    ]

    while True:
        print("\n预设问题 (直接回车使用第一个):")
        for i, q in enumerate(default_queries, 1):
            print(f"   {i}. {q}")
        print("   0. 自定义输入")

        choice = input("\n请选择 (或直接回车): ").strip()

        if choice == '0':
            query = input("请输入你的需求: ").strip()
        elif choice == '':
            query = default_queries[0]
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(default_queries):
                    query = default_queries[idx]
                else:
                    query = default_queries[0]
            except ValueError:
                query = default_queries[0]

        if query.lower() == 'quit':
            break

        theme_id = detect_and_show_theme(theme_loader, query)
        if not theme_id:
            continue

        options = show_skill_options(theme_loader, theme_id)
        skill_id = get_user_choice(options)

        if skill_id:
            generate_script_with_skill(
                llm_service,
                theme_loader,
                skill_service,
                task_service,
                theme_id,
                skill_id,
                query
            )


def main():
    parser = argparse.ArgumentParser(description="圣斗士星矢剧本生成示例")
    parser.add_argument(
        '--skill',
        choices=['heated_battle', 'strategic_approach', 'emotional_bond'],
        help='指定技能'
    )
    parser.add_argument(
        '--query',
        type=str,
        default="圣斗士星矢攻打十二宫",
        help='用户查询'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='交互模式'
    )
    parser.add_argument(
        '--workflow', '-w',
        action='store_true',
        help='使用完整工作流模式（经过所有 agent）'
    )

    args = parser.parse_args()

    if args.workflow:
        print("🌀 使用完整工作流模式（经过所有 agent）")
        llm_service, theme_loader, skill_service, task_service, orchestrator = init_workflow_services()

        if args.interactive:
            print("⚠️  完整工作流模式暂不支持交互模式，使用直接LLM模式")
            args.workflow = False

        if args.skill:
            theme_id = detect_and_show_theme(theme_loader, args.query)
            if theme_id:
                asyncio.run(generate_script_with_workflow(
                    orchestrator,
                    theme_loader,
                    skill_service,
                    task_service,
                    args.skill,
                    args.query
                ))
        else:
            print("请使用 --skill 指定技能")
            print("示例: python demo_saint_seiya.py --workflow --skill heated_battle")
        return

    print("🔧 加载服务...")
    llm_service, theme_loader, skill_service, task_service = load_services()
    print("✅ 服务加载完成")

    if args.interactive:
        interactive_mode(llm_service, theme_loader, skill_service, task_service)
        return

    if args.skill:
        theme_id = detect_and_show_theme(theme_loader, args.query)
        if theme_id:
            generate_script_with_skill(
                llm_service,
                theme_loader,
                skill_service,
                task_service,
                theme_id,
                args.skill,
                args.query,
                step_description="星矢站在狮子宫门前，面对黄金圣斗士艾欧里亚"
            )
    else:
        theme_id = detect_and_show_theme(theme_loader, args.query)
        if theme_id:
            options = show_skill_options(theme_loader, theme_id)
            print("\n💡 使用 --skill 参数指定技能，例如:")
            print(f"   python demo_saint_seiya.py --skill {options[0]['id']}")
            print(f"   python demo_saint_seiya.py --workflow --skill {options[0]['id']}  # 完整工作流模式")
            print(f"   python demo_saint_seiya.py --interactive  # 交互模式")


if __name__ == "__main__":
    main()
