"""Unit tests for LLM Service"""

import pytest
from src.services.llm.adapter import LLMProviderConfig, ModelMapping
from src.services.llm.openai_adapter import OpenAICompatibleAdapter


def test_llm_provider_config_creation():
    """测试 LLM 提供商配置创建"""
    config = LLMProviderConfig(
        provider="openai",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        timeout=60,
        max_retries=3
    )
    
    assert config.provider == "openai"
    assert config.api_key == "test-key"
    assert config.timeout == 60
    assert config.max_retries == 3


def test_model_mapping_creation():
    """测试模型映射配置创建"""
    mapping = ModelMapping(
        high_performance="gpt-4o",
        lightweight="gpt-4o-mini",
        embedding="text-embedding-3-large"
    )
    
    assert mapping.high_performance == "gpt-4o"
    assert mapping.lightweight == "gpt-4o-mini"
    assert mapping.embedding == "text-embedding-3-large"


def test_openai_adapter_initialization():
    """测试 OpenAI 适配器初始化"""
    config = LLMProviderConfig(
        provider="openai",
        api_key="test-key",
        base_url="https://api.openai.com/v1"
    )
    
    mapping = ModelMapping(
        high_performance="gpt-4o",
        lightweight="gpt-4o-mini",
        embedding="text-embedding-3-large"
    )
    
    adapter = OpenAICompatibleAdapter(config, mapping)
    
    assert adapter.config.provider == "openai"
    assert adapter.get_model_name("high_performance") == "gpt-4o"
    assert adapter.get_model_name("lightweight") == "gpt-4o-mini"
    assert adapter.get_model_name("embedding") == "text-embedding-3-large"


def test_llm_service_config_structure():
    """测试 LLM 服务配置结构"""
    config = {
        "providers": {
            "openai": {
                "provider": "openai",
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1"
            }
        },
        "model_mappings": {
            "openai": {
                "high_performance": "gpt-4o",
                "lightweight": "gpt-4o-mini",
                "embedding": "text-embedding-3-large"
            }
        },
        "active_provider": "openai",
        "fallback_providers": []
    }
    
    assert "providers" in config
    assert "model_mappings" in config
    assert "active_provider" in config
    assert config["active_provider"] == "openai"
