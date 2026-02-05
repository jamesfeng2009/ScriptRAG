#!/usr/bin/env python
"""Test the chat message endpoint logic"""

import yaml
import os
from dotenv import load_dotenv
import asyncio

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

llm_service = LLMService(config_data.get('llm', {}))
print(f'✅ LLM service initialized, providers: {list(llm_service.adapters.keys())}')

# 模拟端点中的调用
async def test_chat():
    config = {
        "skill": "standard_tutorial",
        "enable_rag": False,
        "temperature": 0.7
    }
    
    request_message = "我要创作一个以阴阳师安培晴明为主角的剧本。基本设定：- 时代：平安时代- 地点：京都- 主角：安培晴明- 请帮我完善世界观设定。"
    
    # 模拟 history
    history = []
    
    effective_skill = config.get("skill")
    effective_rag = config.get("enable_rag", False)
    
    context = ""
    
    # 构建 full_prompt
    history_text = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in [{"role": "user", "content": request_message}]
    ])
    
    full_prompt = f"""[对话历史]
{history_text}

[用户新请求]
{request_message}
"""
    
    if context:
        full_prompt = f"{context}\n\n{full_prompt}"
    
    print(f'=== 测试 chat_completion ===')
    print(f'messages: {full_prompt[:200]}...')
    print(f'temperature: {config.get("temperature", 0.7)}')
    print(f'max_tokens: 3000')
    
    try:
        response_text = await llm_service.chat_completion(
            messages=[
                {"role": "system", "content": "你是一个专业的剧本写作助手。"},
                {"role": "user", "content": full_prompt}
            ],
            temperature=config.get("temperature", 0.7),
            max_tokens=3000
        )
        print('✅ chat_completion succeeded!')
        print(f'Response: {response_text[:300]}...' if len(response_text) > 300 else f'Response: {response_text}')
    except Exception as e:
        print(f'❌ chat_completion failed: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(test_chat())
