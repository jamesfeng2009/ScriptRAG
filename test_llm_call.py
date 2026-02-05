#!/usr/bin/env python
"""Test LLM service initialization and chat completion"""

import yaml
import os
from dotenv import load_dotenv

load_dotenv()

from src.config import get_llm_config
from src.services.llm.service import LLMService

# 加载 YAML 配置
with open('config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

llm_config = get_llm_config()

# 模拟 init_services 中的代码
llm_providers = config_data.get("llm", {}).setdefault("providers", {})

if llm_config.glm_api_key:
    llm_providers.setdefault("glm", {})["api_key"] = llm_config.glm_api_key
    llm_providers.setdefault("glm", {})["base_url"] = "https://open.bigmodel.cn/api/paas/v4"

print('=== 初始化 LLM 服务 ===')
try:
    llm_service = LLMService(config_data.get('llm', {}))
    print('✅ LLM service initialized successfully')
    # 检查可用的 adapter
    print(f'   Available providers: {list(llm_service.adapters.keys())}')
except Exception as e:
    print(f'❌ LLM service initialization failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)

print('\n=== 测试 LLM chat_completion ===')
try:
    import asyncio
    
    async def test_chat():
        response = await llm_service.chat_completion(
            messages=[
                {"role": "system", "content": "你是一个专业的剧本写作助手。"},
                {"role": "user", "content": "你好，请简单介绍一下你自己。"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response
    
    result = asyncio.run(test_chat())
    print('✅ LLM chat_completion succeeded')
    print(f'   Response: {result[:200]}...' if len(result) > 200 else f'   Response: {result}')
except Exception as e:
    print(f'❌ LLM chat_completion failed: {e}')
    import traceback
    traceback.print_exc()
