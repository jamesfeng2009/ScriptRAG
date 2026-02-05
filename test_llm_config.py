#!/usr/bin/env python
"""Test LLM configuration and connectivity"""

import yaml
import os
from dotenv import load_dotenv

load_dotenv()

# 加载配置
with open('config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

# 检查 LLM 配置
llm_config = config_data.get('llm', {})
providers = llm_config.get('providers', {})

print('=== LLM 配置 ===')
for provider, config in providers.items():
    has_key = bool(config.get('api_key'))
    mask_key = config.get('api_key', '')[:10] + '...' if has_key else 'MISSING'
    print(f'{provider}: api_key={mask_key}')
    if provider == 'glm':
        print(f'  base_url: {config.get("base_url")}')

print('\n=== 环境变量 ===')
print(f'ACTIVE_LLM_PROVIDER: {os.getenv("ACTIVE_LLM_PROVIDER")}')
print(f'GLM_API_KEY exists: {bool(os.getenv("GLM_API_KEY"))}')
