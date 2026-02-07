#!/usr/bin/env python
"""快速测试 _record_agent_execution 修复"""

import asyncio
import yaml
from pathlib import Path
from src.config import get_llm_config
from src.services.llm.service import LLMService
from src.domain.skill_loader import SkillConfigLoader
from src.application.orchestrator import WorkflowOrchestrator
from src.services.parser.tree_sitter_parser import TreeSitterParser
from src.services.core.summarization_service import SummarizationService
from src.services.mocks import MockRetrievalService


async def quick_test():
    print("🔧 初始化服务...")

    with open('config.yaml', 'r') as f:
        config_data = yaml.safe_load(f)

    llm_config = get_llm_config()
    llm_providers = config_data.get("llm", {}).setdefault("providers", {})
    if llm_config.glm_api_key:
        llm_providers.setdefault("glm", {})["api_key"] = llm_config.glm_api_key
        llm_providers.setdefault("glm", {})["base_url"] = "https://open.bigmodel.cn/api/paas/v4"

    llm_service = LLMService(config_data.get('llm', {}))
    retrieval_service = MockRetrievalService()
    parser_service = TreeSitterParser()
    summarization_service = SummarizationService(llm_service)

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

    print("🚀 启动工作流测试...")

    initial_state = {
        "user_topic": "星矢要攻打狮子宫",
        "chat_history": [],
        "messages": [],
        "enable_dynamic_adjustment": False,
        "current_skill": "heated_battle"
    }

    result = await orchestrator.execute(
        initial_state=initial_state,
        recursion_limit=20
    )

    if result["success"]:
        print("\n✅ 工作流执行成功!")
        print(f"   剧本长度: {len(result['state'].get('screenplay', ''))} 字符")
    else:
        print("\n❌ 工作流执行失败")

    return result


if __name__ == "__main__":
    asyncio.run(quick_test())
