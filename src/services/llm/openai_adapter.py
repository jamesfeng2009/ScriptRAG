"""OpenAI Adapter - OpenAI LLM provider implementation"""

from typing import List, Dict, Any, Optional, Literal
from openai import AsyncOpenAI
from .adapter import LLMAdapter, LLMProviderConfig, ModelMapping


class OpenAICompatibleAdapter(LLMAdapter):
    """
    统一的 OpenAI 兼容适配器
    
    支持所有 OpenAI 兼容的提供商：
    - OpenAI
    - Qwen (通义千问)
    - MiniMax
    - GLM (智谱)
    """
    
    def __init__(self, config: LLMProviderConfig, model_mapping: ModelMapping):
        """
        初始化 OpenAI 兼容适配器
        
        Args:
            config: 提供商配置
            model_mapping: 模型映射配置
        """
        super().__init__(config, model_mapping)
        
        # 初始化 OpenAI 客户端
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        聊天补全接口
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            **kwargs: 其他参数
            
        Returns:
            生成的文本内容
        """
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Chat completion failed for {self.config.provider}: {str(e)}")
    
    async def embedding(
        self,
        texts: List[str],
        model: str
    ) -> List[List[float]]:
        """
        嵌入向量生成接口
        
        Args:
            texts: 文本列表
            model: 嵌入模型名称
            
        Returns:
            嵌入向量列表
        """
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise Exception(f"Embedding generation failed for {self.config.provider}: {str(e)}")
    
    def get_model_name(
        self, 
        task_type: Literal["high_performance", "lightweight", "embedding"]
    ) -> str:
        """
        获取任务类型对应的模型名称
        
        Args:
            task_type: 任务类型
            
        Returns:
            模型名称
        """
        return getattr(self.model_mapping, task_type)
