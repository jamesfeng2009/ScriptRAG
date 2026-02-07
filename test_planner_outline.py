#!/usr/bin/env python
"""测试 planner 生成 outline"""

import asyncio
import yaml
from src.config import get_llm_config
from src.services.llm.service import LLMService
from src.domain.agents.node_factory import NodeFactory


async def test_planner():
    config_path = "config.yaml"

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    llm_config = get_llm_config()
    llm_providers = config_data.get("llm", {}).setdefault("providers", {})
    if llm_config.glm_api_key:
        llm_providers.setdefault("glm", {})["api_key"] = llm_config.glm_api_key
        llm_providers.setdefault("glm", {})["base_url"] = "https://open.bigmodel.cn/api/paas/v4"

    llm_service = LLMService(config_data.get('llm', {}))

    node_factory = NodeFactory(
        llm_service=llm_service,
        retrieval_service=None,
        parser_service=None,
        summarization_service=None
    )

    state = {
        "user_topic": "星矢要攻打狮子宫",
        "project_context": ""
    }

    print("🧪 测试 planner 生成 outline...")
    print("=" * 60)

    result = await node_factory.planner_node(state)

    outline = result.get("outline", [])
    print(f"\n✅ 生成了 {len(outline)} 个步骤")
    print("\n步骤详情:")
    for i, step in enumerate(outline):
        print(f"  [{i}] step_id: {step.get('step_id')}")
        print(f"      title: {step.get('title')}")
        print(f"      description: {step.get('description', '')[:50]}...")
        print()


if __name__ == "__main__":
    asyncio.run(test_planner())
