"""LLM Service - Unified LLM service with automatic fallback"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Literal
import yaml
from pathlib import Path

from .adapter import LLMAdapter, LLMProviderConfig, ModelMapping
from .openai_adapter import OpenAICompatibleAdapter
from .qwen_adapter import QwenAdapter
from .minimax_adapter import MiniMaxAdapter
from .glm_adapter import GLMAdapter
from ...infrastructure.error_handler import (
    RateLimitError,
    MalformedResponseError,
    ProviderUnavailableError,
    ErrorHandler
)
from ...services.llm_call_logger import get_llm_call_logger


logger = logging.getLogger(__name__)


class LLMService:
    """
    统一的 LLM 服务
    
    功能：
    - 管理多个 LLM 提供商适配器
    - 自动回退机制
    - 指数退避重试策略
    - LLM 调用日志记录（内存和数据库）
    """
    
    # 适配器类映射
    ADAPTER_CLASSES = {
        "openai": OpenAICompatibleAdapter,
        "qwen": QwenAdapter,
        "minimax": MiniMaxAdapter,
        "glm": GLMAdapter
    }
    
    def __init__(
        self,
        config: Dict[str, Any],
        logging_db_service: Optional[Any] = None,
        session_id: Optional[str] = None
    ):
        """
        初始化 LLM 服务
        
        Args:
            config: 配置字典，包含 providers、model_mappings、active_provider、fallback_providers
            logging_db_service: 数据库日志服务（可选）
            session_id: 会话 ID（可选，用于日志关联）
        """
        self.config = config
        self.adapters: Dict[str, LLMAdapter] = {}
        self.logging_db_service = logging_db_service
        self.session_id = session_id
        self._initialize_adapters()
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "LLMService":
        """
        从 YAML 配置文件加载 LLM 服务
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            LLMService 实例
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return cls(config)
    
    def _initialize_adapters(self):
        """初始化所有配置的适配器"""
        for provider_name, provider_config_dict in self.config["providers"].items():
            adapter_class = self.ADAPTER_CLASSES.get(provider_name)
            if adapter_class:
                try:
                    # 创建配置对象
                    provider_config = LLMProviderConfig(**provider_config_dict)
                    model_mapping_dict = self.config["model_mappings"][provider_name]
                    model_mapping = ModelMapping(**model_mapping_dict)
                    
                    # 初始化适配器
                    self.adapters[provider_name] = adapter_class(provider_config, model_mapping)
                    logger.info(f"Initialized adapter for provider: {provider_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize adapter for {provider_name}: {str(e)}")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        task_type: Literal["high_performance", "lightweight"],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        统一的聊天补全接口，支持自动回退和指数退避重试
        
        Args:
            messages: 消息列表
            task_type: 任务类型（high_performance 或 lightweight）
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            **kwargs: 其他参数
            
        Returns:
            生成的文本内容
        """
        active_provider = self.config["active_provider"]
        fallback_providers = self.config.get("fallback_providers", [])
        
        providers_to_try = [active_provider] + fallback_providers
        
        for provider_name in providers_to_try:
            if provider_name not in self.adapters:
                logger.warning(f"Provider {provider_name} not available, skipping")
                continue
            
            try:
                adapter = self.adapters[provider_name]
                model = adapter.get_model_name(task_type)
                
                # 使用指数退避重试
                async def _call_with_retry():
                    start_time = time.time()
                    result = await adapter.chat_completion(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                    response_time = int((time.time() - start_time) * 1000)
                    return result, response_time
                
                result, response_time = await ErrorHandler.retry_with_exponential_backoff(
                    _call_with_retry,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=60.0,
                    exponential_base=2.0
                )
                
                # 记录成功的调用
                self._log_llm_call(
                    provider=provider_name,
                    model=model,
                    task_type=task_type,
                    status="success",
                    response_time_ms=response_time
                )
                
                return result
                
            except (RateLimitError, MalformedResponseError) as e:
                logger.error(f"LLM call failed for {provider_name} after retries: {str(e)}")
                
                # 记录失败的调用
                self._log_llm_call(
                    provider=provider_name,
                    model=adapter.get_model_name(task_type) if provider_name in self.adapters else "unknown",
                    task_type=task_type,
                    status="failed",
                    error_message=str(e)
                )
                
                # 记录提供商切换事件
                if provider_name != providers_to_try[-1]:
                    next_provider = providers_to_try[providers_to_try.index(provider_name) + 1]
                    logger.warning(f"Switching from provider {provider_name} to {next_provider}")
                    self._log_provider_switch(
                        from_provider=provider_name,
                        to_provider=next_provider,
                        reason=str(e)
                    )
                
                # 如果是最后一个提供商，抛出异常
                if provider_name == providers_to_try[-1]:
                    raise ProviderUnavailableError(f"All LLM providers failed. Last error: {str(e)}")
                
                # 否则继续尝试下一个提供商
                continue
                
            except Exception as e:
                logger.error(f"Unexpected LLM error for {provider_name}: {str(e)}")
                
                # 记录失败的调用
                self._log_llm_call(
                    provider=provider_name,
                    model=adapter.get_model_name(task_type) if provider_name in self.adapters else "unknown",
                    task_type=task_type,
                    status="failed",
                    error_message=str(e)
                )
                
                # 记录提供商切换事件
                if provider_name != providers_to_try[-1]:
                    next_provider = providers_to_try[providers_to_try.index(provider_name) + 1]
                    logger.warning(f"Switching from provider {provider_name} to {next_provider} due to error")
                    self._log_provider_switch(
                        from_provider=provider_name,
                        to_provider=next_provider,
                        reason=str(e)
                    )
                
                # 如果是最后一个提供商，抛出异常
                if provider_name == providers_to_try[-1]:
                    raise ProviderUnavailableError(f"All LLM providers failed. Last error: {str(e)}")
                
                # 否则继续尝试下一个提供商
                continue
    
    async def embedding(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        统一的嵌入向量生成接口，支持自动回退和指数退避重试
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        active_provider = self.config["active_provider"]
        fallback_providers = self.config.get("fallback_providers", [])
        
        providers_to_try = [active_provider] + fallback_providers
        
        for provider_name in providers_to_try:
            if provider_name not in self.adapters:
                logger.warning(f"Provider {provider_name} not available, skipping")
                continue
            
            try:
                adapter = self.adapters[provider_name]
                model = adapter.get_model_name("embedding")
                
                # 使用指数退避重试
                async def _call_with_retry():
                    start_time = time.time()
                    result = await adapter.embedding(texts=texts, model=model)
                    response_time = int((time.time() - start_time) * 1000)
                    return result, response_time
                
                result, response_time = await ErrorHandler.retry_with_exponential_backoff(
                    _call_with_retry,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=60.0,
                    exponential_base=2.0
                )
                
                # 记录成功的调用
                self._log_llm_call(
                    provider=provider_name,
                    model=model,
                    task_type="embedding",
                    status="success",
                    response_time_ms=response_time
                )
                
                return result
                
            except (RateLimitError, MalformedResponseError) as e:
                logger.error(f"Embedding generation failed for {provider_name} after retries: {str(e)}")
                
                # 记录失败的调用
                self._log_llm_call(
                    provider=provider_name,
                    model=adapter.get_model_name("embedding") if provider_name in self.adapters else "unknown",
                    task_type="embedding",
                    status="failed",
                    error_message=str(e)
                )
                
                # 记录提供商切换事件
                if provider_name != providers_to_try[-1]:
                    next_provider = providers_to_try[providers_to_try.index(provider_name) + 1]
                    logger.warning(f"Switching from provider {provider_name} to {next_provider}")
                    self._log_provider_switch(
                        from_provider=provider_name,
                        to_provider=next_provider,
                        reason=str(e)
                    )
                
                # 如果是最后一个提供商，抛出异常
                if provider_name == providers_to_try[-1]:
                    raise ProviderUnavailableError(f"All embedding providers failed. Last error: {str(e)}")
                
                # 否则继续尝试下一个提供商
                continue
                
            except Exception as e:
                logger.error(f"Unexpected embedding error for {provider_name}: {str(e)}")
                
                # 记录失败的调用
                self._log_llm_call(
                    provider=provider_name,
                    model=adapter.get_model_name("embedding") if provider_name in self.adapters else "unknown",
                    task_type="embedding",
                    status="failed",
                    error_message=str(e)
                )
                
                # 记录提供商切换事件
                if provider_name != providers_to_try[-1]:
                    next_provider = providers_to_try[providers_to_try.index(provider_name) + 1]
                    logger.warning(f"Switching from provider {provider_name} to {next_provider} due to error")
                    self._log_provider_switch(
                        from_provider=provider_name,
                        to_provider=next_provider,
                        reason=str(e)
                    )
                
                # 如果是最后一个提供商，抛出异常
                if provider_name == providers_to_try[-1]:
                    raise ProviderUnavailableError(f"All embedding providers failed. Last error: {str(e)}")
                
                # 否则继续尝试下一个提供商
                continue
    
    def _log_llm_call(
        self,
        provider: str,
        model: str,
        task_type: str,
        status: str,
        response_time_ms: Optional[int] = None,
        token_count: Optional[int] = None,
        error_message: Optional[str] = None
    ):
        """
        记录 LLM 调用（内存日志和数据库持久化）
        
        Args:
            provider: 提供商名称
            model: 模型名称
            task_type: 任务类型
            status: 状态（success/failed）
            response_time_ms: 响应时间（毫秒）
            token_count: token 数量
            error_message: 错误信息
        """
        log_data = {
            "provider": provider,
            "model": model,
            "task_type": task_type,
            "status": status,
            "response_time_ms": response_time_ms,
            "token_count": token_count,
            "error_message": error_message
        }
        
        if status == "success":
            logger.info(f"LLM call succeeded: {log_data}")
        else:
            logger.error(f"LLM call failed: {log_data}")
        
        try:
            llm_logger = get_llm_call_logger()
            llm_logger.log_call(
                task_id=self.session_id,
                provider=provider,
                model=model,
                request_type=task_type,
                response_time_ms=response_time_ms,
                status=status,
                error_message=error_message
            )
        except Exception as e:
            logger.warning(f"Failed to log LLM call: {e}")
    
    def _log_provider_switch(
        self,
        from_provider: str,
        to_provider: str,
        reason: str
    ):
        """
        记录提供商切换事件
        
        Args:
            from_provider: 原提供商
            to_provider: 新提供商
            reason: 切换原因
        """
        log_data = {
            "from_provider": from_provider,
            "to_provider": to_provider,
            "reason": reason
        }
        
        logger.warning(f"Provider switch: {log_data}")
        
        # 数据库持久化（异步，不阻塞主流程）
        if self.logging_db_service:
            try:
                # 创建任务但不等待完成
                asyncio.create_task(
                    self.logging_db_service.log_execution(
                        session_id=self.session_id,
                        agent_name="LLMService",
                        action="provider_switch",
                        details=log_data
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to schedule provider switch logging: {str(e)}")
