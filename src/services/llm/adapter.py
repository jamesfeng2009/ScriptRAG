"""LLM Adapter - Abstract base class for LLM providers"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class LLMProviderConfig(BaseModel):
    """LLM 提供商配置"""
    provider: Literal["openai", "qwen", "minimax", "glm"]
    api_key: str
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3


class ModelMapping(BaseModel):
    """模型映射配置"""
    high_performance: str  # 用于规划器、导演、转向管理器、事实检查器
    lightweight: str       # 用于编剧、编译器
    embedding: str         # 用于向量嵌入


class ToolCall(BaseModel):
    """工具调用信息"""
    id: str = Field(..., description="工具调用 ID")
    type: str = Field(default="function", description="调用类型")
    function: Dict[str, str] = Field(..., description="函数信息，包含 name 和 arguments")


class ToolChoice(BaseModel):
    """工具选择配置"""
    type: Literal["function", "none"] = Field(default="function", description="工具调用类型")
    function: Optional[Dict[str, str]] = Field(None, description="指定调用的函数")


class LLMAdapter(ABC):
    """LLM 适配器抽象基类"""
    
    def __init__(self, config: LLMProviderConfig, model_mapping: ModelMapping):
        """
        初始化 LLM 适配器
        
        Args:
            config: 提供商配置
            model_mapping: 模型映射配置
        """
        self.config = config
        self.model_mapping = model_mapping
    
    @abstractmethod
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
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            model: 模型名称
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成 token 数
            **kwargs: 其他提供商特定参数
            
        Returns:
            生成的文本内容
        """
        pass
    
    @abstractmethod
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
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            model: 模型名称
            tools: 工具定义列表，OpenAI tools 格式
            tool_choice: 工具选择配置，可选 {"type": "function", "function": {"name": "xxx"}}
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成 token 数
            **kwargs: 其他提供商特定参数
            
        Returns:
            {
                "content": str,  # 文本响应（当没有工具调用时）
                "tool_calls": List[ToolCall],  # 工具调用列表
                "finish_reason": str  # 结束原因：stop/tool_calls/function_call
            }
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
