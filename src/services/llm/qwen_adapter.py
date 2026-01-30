"""Qwen Adapter - Tongyi Qianwen LLM provider implementation"""

from .openai_adapter import OpenAICompatibleAdapter


class QwenAdapter(OpenAICompatibleAdapter):
    """
    通义千问适配器
    
    使用 OpenAI 兼容接口
    """
    pass
