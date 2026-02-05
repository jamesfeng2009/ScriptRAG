#!/usr/bin/env python
"""Test the full chat message endpoint logic"""

import yaml
import os
from dotenv import load_dotenv
import asyncio
from datetime import datetime

load_dotenv()

from src.config import get_llm_config
from src.services.llm.service import LLMService

# 模拟 ChatHistoryManager
class MockChatHistoryManager:
    _sessions = {}
    _messages = {}
    
    @classmethod
    def create_session(cls, session_id, mode, config):
        session = {
            "session_id": session_id,
            "mode": mode,
            "config": config,
            "created_at": datetime.now(),
            "message_count": 0
        }
        cls._sessions[session_id] = session
        cls._messages[session_id] = []
        return session
    
    @classmethod
    def get_session(cls, session_id):
        return cls._sessions.get(session_id)
    
    @classmethod
    def add_message(cls, session_id, role, content):
        if session_id not in cls._messages:
            cls._messages[session_id] = []
        cls._messages[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
        if session_id in cls._sessions:
            cls._sessions[session_id]["message_count"] += 1
    
    @classmethod
    def get_history(cls, session_id):
        return cls._messages.get(session_id, [])

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
rag_service = None  # RAG service not available
skill_service = None  # Skill service not available

print(f'✅ LLM service initialized, providers: {list(llm_service.adapters.keys())}')
print(f'✅ rag_service: {rag_service}')
print(f'✅ skill_service: {skill_service}')

# 模拟端点中的调用
async def test_send_message():
    session_id = "chat_test123"
    
    # 创建 session
    config = {
        "skill": "standard_tutorial",
        "enable_rag": False,
        "temperature": 0.7
    }
    MockChatHistoryManager.create_session(session_id, "agent", config)
    
    request_message = "我要创作一个以阴阳师安培晴明为主角的剧本。基本设定：- 时代：平安时代- 地点：京都- 主角：安培晴明- 请帮我完善世界观设定。"
    
    # 1. 添加用户消息到内存
    print('\n=== 步骤 1: 添加用户消息 ===')
    MockChatHistoryManager.add_message(session_id, "user", request_message)
    print(f'✅ 用户消息已添加')
    
    # 2. 获取 history
    history = MockChatHistoryManager.get_history(session_id)
    print(f'✅ History length: {len(history)}')
    
    # 3. 处理 skill 和 rag
    effective_skill = config.get("skill")
    effective_rag = config.get("enable_rag", False)
    
    print(f'\n=== 步骤 2: 处理配置 ===')
    print(f'effective_skill: {effective_skill}')
    print(f'effective_rag: {effective_rag}')
    
    context = ""
    
    # 4. 构建 full_prompt
    print(f'\n=== 步骤 3: 构建 prompt ===')
    history_text = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in history[-10:]
    ])
    
    full_prompt = f"""[对话历史]
{history_text}

[用户新请求]
{request_message}
"""
    
    if context:
        full_prompt = f"{context}\n\n{full_prompt}"
    
    # 5. 调用 LLM
    print(f'\n=== 步骤 4: 调用 LLM ===')
    if not llm_service:
        print('❌ LLM service not available')
        return
    
    try:
        response_text = await llm_service.chat_completion(
            messages=[
                {"role": "system", "content": "你是一个专业的剧本写作助手。"},
                {"role": "user", "content": full_prompt}
            ],
            temperature=config.get("temperature", 0.7),
            max_tokens=3000
        )
        print('✅ LLM chat_completion succeeded!')
        print(f'Response preview: {response_text[:200]}...' if len(response_text) > 200 else f'Response: {response_text}')
        
        # 6. 添加 assistant 消息
        MockChatHistoryManager.add_message(session_id, "assistant", response_text)
        print(f'\n✅ Assistant 消息已添加')
        print(f'Total messages: {len(MockChatHistoryManager.get_history(session_id))}')
        
    except Exception as e:
        print(f'❌ LLM chat_completion failed: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(test_send_message())
