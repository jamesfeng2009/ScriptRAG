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
    
    async def chat_completion_with_tools(
        self,
        messages: List[Dict[str, str]],
        model: str,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        支持工具调用的聊天补全接口
        
        Args:
            messages: 消息列表
            model: 模型名称
            tools: 工具定义列表，OpenAI tools 格式
            tool_choice: 工具选择配置，可选 {"type": "function", "function": {"name": "xxx"}}
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            **kwargs: 其他参数
            
        Returns:
            {
                "content": str,  # 文本响应（当没有工具调用时）
                "tool_calls": List[Dict],  # 工具调用列表
                "finish_reason": str  # 结束原因：stop/tool_calls/function_call
            }
        """
        try:
            request_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            if tools:
                request_params["tools"] = tools
            
            if tool_choice:
                request_params["tool_choice"] = tool_choice
            
            response = await self.client.chat.completions.create(**request_params)
            
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            
            tool_calls = []
            if message.tool_calls:
                for call in message.tool_calls:
                    tool_calls.append({
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments
                        }
                    })
            
            return {
                "content": message.content if message.content else "",
                "tool_calls": tool_calls,
                "finish_reason": finish_reason if finish_reason else "stop"
            }
            
        except Exception as e:
            raise Exception(f"Chat completion with tools failed for {self.config.provider}: {str(e)}")
    
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
