#!/usr/bin/env python
"""Test LLM configuration and connectivity"""

import yaml
import os
from dotenv import load_dotenv

load_dotenv()

from src.config import get_llm_config

# 加载 YAML 配置
with open('config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

llm_config = get_llm_config()

print('=== LLMConfig from env ===')
print(f'glm_api_key: {"***" if llm_config.glm_api_key else "MISSING"}')
print(f'openai_api_key: {"***" if llm_config.openai_api_key else "MISSING"}')
print(f'qwen_api_key: {"***" if llm_config.qwen_api_key else "MISSING"}')

print('\n=== YAML providers before ===')
providers = config_data.get("llm", {}).get("providers", {})
for provider, config in providers.items():
    has_key = bool(config.get("api_key"))
    print(f'{provider}: api_key={"***" if has_key else "MISSING"}')

# 模拟 init_services 中的代码
llm_providers = config_data.get("llm", {}).setdefault("providers", {})

if llm_config.glm_api_key:
    llm_providers.setdefault("glm", {})["api_key"] = llm_config.glm_api_key
    llm_providers.setdefault("glm", {})["base_url"] = "https://open.bigmodel.cn/api/paas/v4"
    print('\n✅ GLM API key loaded from env to YAML config')
else:
    print('\n❌ GLM API key NOT loaded (glm_api_key is None)')

print('\n=== YAML providers after ===')
providers = config_data.get("llm", {}).get("providers", {})
for provider, config in providers.items():
    has_key = bool(config.get("api_key"))
    print(f'{provider}: api_key={"***" if has_key else "MISSING"}')
    if provider == 'glm':
        print(f'  base_url: {config.get("base_url")}')
