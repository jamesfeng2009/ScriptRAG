"""LLM Service - Unified LLM service with automatic fallback and cost control"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Literal, Tuple
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


class BudgetExceededError(Exception):
    """预算超出异常"""
    pass


class TokenEstimator:
    """Token 估算器
    
    支持 tiktoken 和回退字符估算两种方式。
    """
    
    def __init__(self):
        self._tiktoken_model = None
        self._try_init_tiktoken()
    
    def _try_init_tiktoken(self):
        """尝试初始化 tiktoken"""
        try:
            import tiktoken
            self._tiktoken_model = tiktoken.encoding_for_model("gpt-4")
        except ImportError:
            logger.warning("tiktoken not available, using character-based estimation")
        except Exception:
            logger.warning("Failed to initialize tiktoken, using character-based estimation")
    
    def estimate(self, text: str) -> int:
        """估算文本的 token 数量
        
        Args:
            text: 输入文本
            
        Returns:
            估算的 token 数量
        """
        if self._tiktoken_model:
            try:
                return len(self._tiktoken_model.encode(text))
            except Exception:
                pass
        
        return len(text) // 4
    
    def estimate_messages(self, messages: List[Dict[str, str]]) -> int:
        """估算消息列表的 token 数量
        
        Args:
            messages: 消息列表
            
        Returns:
            估算的 token 数量
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            total += self.estimate(f"[{role}] {content}")
        return total


class AdaptiveCompressionController:
    """自适应压缩控制器
    
    根据提供商配置和优先级动态调整压缩策略。
    """
    
    def __init__(
        self,
        provider: str,
        model: str,
        enable_compression: bool = True
    ):
        self.provider_name = provider
        self.model = model
        self.enable_compression = enable_compression
        
        self._config = None
        self._load_provider_config()
    
    def _load_provider_config(self):
        """延迟加载提供商配置"""
        try:
            from ...services.rag.cost_control import LLMProvider, PROVIDER_CONFIGS
            self._llm_provider = LLMProvider
            self._provider_configs = PROVIDER_CONFIGS
            self._config = self._get_config(self.provider_name)
        except ImportError:
            self._llm_provider = None
            self._provider_configs = None
            self._config = None
    
    def _get_config(self, provider_name: str):
        """获取提供商配置"""
        if not self._provider_configs:
            return None
        
        mapping = {
            "openai": self._llm_provider.OPENAI,
            "anthropic": self._llm_provider.ANTHROPIC,
            "google": self._llm_provider.GOOGLE,
            "qwen": self._llm_provider.QWEN,
            "lmstudio": self._llm_provider.LMSTUDIO,
            "ollama": self._llm_provider.OLLAMA
        }
        
        provider_enum = mapping.get(provider_name.lower(), self._llm_provider.OTHER)
        return self._provider_configs.get(provider_enum)
    
    def get_compression_params(self, priority: int) -> Dict[str, Any]:
        """获取压缩参数
        
        Args:
            priority: 优先级 (1-10)
            
        Returns:
            压缩参数字典
        """
        if not self.enable_compression or not self._config:
            return {"enabled": False}
        
        strategy = self._config.default_strategy
        preserve_start = self._config.recommended_preserve_start
        preserve_end = self._config.recommended_preserve_end
        
        if priority <= 3:
            preserve_start = max(1, preserve_start - 1)
            preserve_end = max(1, preserve_end - 1)
        elif priority >= 8:
            preserve_start = min(5, preserve_start + 1)
            preserve_end = min(5, preserve_end + 1)
        
        return {
            "enabled": True,
            "strategy": strategy,
            "preserve_start": preserve_start,
            "preserve_end": preserve_end,
            "max_context_tokens": self._config.max_context_tokens
        }
    
    def get_max_input_tokens(self) -> int:
        """获取最大输入 token 数"""
        if not self._config:
            return 32000
        return int(self._config.max_context_tokens * 0.8)


class LLMService:
    """
    统一的 LLM 服务
    
    功能：
    - 管理多个 LLM 提供商适配器
    - 自动回退机制
    - 指数退避重试策略
    - LLM 调用日志记录（内存和数据库）
    - 成本控制和上下文压缩
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
        session_id: Optional[str] = None,
        cost_controller: Optional["CostController"] = None,
        enable_compression: bool = True
    ):
        """
        初始化 LLM 服务
        
        Args:
            config: 配置字典，包含 providers、model_mappings、active_provider、fallback_providers
            logging_db_service: 数据库日志服务（可选）
            session_id: 会话 ID（可选，用于日志关联）
            cost_controller: 成本控制器（可选）
            enable_compression: 是否启用上下文压缩（默认启用）
        """
        self.config = config
        self.adapters: Dict[str, LLMAdapter] = {}
        self.logging_db_service = logging_db_service
        self.session_id = session_id
        self.cost_controller = cost_controller
        self.enable_compression = enable_compression
        
        self._initialize_adapters()
        
        self.token_estimator = TokenEstimator()
        
        active_provider = config.get("active_provider", "qwen")
        self.compression_controller = AdaptiveCompressionController(
            provider=active_provider,
            model="",
            enable_compression=enable_compression
        )
        
        self.context_compressor = None
        self._middle_strategy = None
        
        if self.enable_compression:
            try:
                from ...services.rag.cost_control import ContextCompressor, MiddleRemovalStrategy
                self.context_compressor = ContextCompressor(
                    max_tokens=4000,
                    compression_ratio=0.5,
                    preserve_key_info=True
                )
                self._middle_strategy = MiddleRemovalStrategy()
            except ImportError:
                logger.warning("Failed to import compression components")
    
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
                    provider_config = LLMProviderConfig(**provider_config_dict)
                    model_mapping_dict = self.config["model_mappings"][provider_name]
                    model_mapping = ModelMapping(**model_mapping_dict)
                    
                    self.adapters[provider_name] = adapter_class(provider_config, model_mapping)
                    logger.info(f"Initialized adapter for provider: {provider_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize adapter for {provider_name}: {str(e)}")
    
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """估算消息的 token 数量
        
        Args:
            messages: 消息列表
            
        Returns:
            估算的 token 数量
        """
        return self.token_estimator.estimate_messages(messages)
    
    async def _check_budget(
        self,
        estimated_tokens: int,
        task_type: str
    ) -> Tuple[bool, str]:
        """检查预算是否充足
        
        Args:
            estimated_tokens: 预估 token 数量
            task_type: 任务类型
            
        Returns:
            (是否允许, 消息)
        """
        if not self.cost_controller:
            return True, "Cost controller not configured"
        
        return await self.cost_controller.check_budget(
            estimated_tokens=estimated_tokens,
            operation=f"llm_{task_type}"
        )
    
    async def _compress_messages(
        self,
        messages: List[Dict[str, str]],
        priority: int = 5
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """压缩消息内容
        
        Args:
            messages: 原始消息列表
            priority: 优先级 (1-10)
            
        Returns:
            (压缩后的消息列表, 压缩元数据)
        """
        if not self.context_compressor or not self._middle_strategy:
            return messages, {"compressed": False, "reason": "compression_disabled"}
        
        try:
            from ...services.rag.cost_control import CompressionStrategyType
            
            compression_params = self.compression_controller.get_compression_params(priority)
            if not compression_params["enabled"]:
                return messages, {"compressed": False, "reason": "compression_disabled"}
            
            strategy_type = compression_params["strategy"]
            
            combined_content = "\n".join([
                msg.get("content", "")
                for msg in messages
                if msg.get("role") == "user"
            ])
            
            if not combined_content:
                return messages, {"compressed": False, "reason": "no_content"}
            
            estimated_tokens = self._estimate_tokens(messages)
            target_tokens = int(estimated_tokens * 0.6)
            
            if strategy_type == CompressionStrategyType.MIDDLE_REMOVAL:
                if self._middle_strategy:
                    compressed_messages = self._middle_strategy.compress(messages, target_tokens)
                    new_tokens = self._estimate_tokens(compressed_messages)
                    
                    return compressed_messages, {
                        "compressed": True,
                        "strategy": "middle_removal",
                        "original_tokens": estimated_tokens,
                        "compressed_tokens": new_tokens,
                        "ratio": new_tokens / estimated_tokens if estimated_tokens > 0 else 1.0
                    }
            elif strategy_type == CompressionStrategyType.OLDEST_REMOVAL:
                if hasattr(self.context_compressor, 'strategies'):
                    old_strategy = self.context_compressor.strategies.get(
                        CompressionStrategyType.OLDEST_REMOVAL
                    )
                    if old_strategy:
                        compressed_messages = old_strategy.compress(messages, target_tokens)
                        new_tokens = self._estimate_tokens(compressed_messages)
                        
                        return compressed_messages, {
                            "compressed": True,
                            "strategy": "oldest_removal",
                            "original_tokens": estimated_tokens,
                            "compressed_tokens": new_tokens,
                            "ratio": new_tokens / estimated_tokens if estimated_tokens > 0 else 1.0
                        }
            
            return messages, {"compressed": False, "reason": "strategy_not_available"}
            
        except Exception as e:
            logger.warning(f"Failed to compress messages: {e}")
            return messages, {"compressed": False, "reason": str(e)}
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        task_type: Literal["high_performance", "lightweight"] = "high_performance",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        priority: int = 5,
        **kwargs
    ) -> str:
        """
        统一的聊天补全接口，支持自动回退、指数退避重试、成本控制和上下文压缩
        
        降级链路：
        1. 估算 token → 检查预算
        2. 预算不足 → 尝试压缩
        3. 压缩后 → 重新检查预算
        4. 仍不足 → 切换提供商
        5. 所有提供商失败 → 抛出 BudgetExceededError
        
        Args:
            messages: 消息列表
            task_type: 任务类型（high_performance 或 lightweight）
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            priority: 优先级 (1-10)，用于压缩策略
            **kwargs: 其他参数
            
        Returns:
            生成的文本内容
        """
        active_provider = self.config.get("active_provider", "qwen")
        
        return await self._execute_with_fallback(
            messages=messages,
            task_type=task_type,
            temperature=temperature,
            max_tokens=max_tokens,
            priority=priority,
            providers_to_try=[active_provider] + self.config.get("fallback_providers", []),
            **kwargs
        )
    
    async def _execute_with_fallback(
        self,
        messages: List[Dict[str, str]],
        task_type: Literal["high_performance", "lightweight"] = "high_performance",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        priority: int = 5,
        providers_to_try: List[str] = None,
        **kwargs
    ) -> str:
        """执行 LLM 调用，支持多级降级
        
        降级链路：预算检查 → 压缩 → 重新检查 → 提供商回退 → 拒绝
        
        Args:
            messages: 消息列表
            task_type: 任务类型
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            priority: 优先级
            providers_to_try: 按顺序尝试的提供商列表
            **kwargs: 其他参数
            
        Returns:
            生成的文本内容
        """
        if providers_to_try is None:
            providers_to_try = [self.config.get("active_provider", "qwen")]
        
        compression_metadata = {}
        last_error = None
        
        for provider_idx, provider_name in enumerate(providers_to_try):
            if provider_name not in self.adapters:
                logger.warning(f"Provider {provider_name} not available, skipping")
                continue
            
            try:
                adapter = self.adapters[provider_name]
                model = adapter.get_model_name(task_type)
                
                max_input_tokens = self.compression_controller.get_max_input_tokens()
                
                estimated_tokens = self._estimate_tokens(messages)
                
                if estimated_tokens > max_input_tokens:
                    logger.info(
                        f"Token count ({estimated_tokens}) exceeds provider limit ({max_input_tokens}), "
                        f"attempting compression for {provider_name}"
                    )
                    
                    messages, compression_metadata = await self._compress_messages(
                        messages, priority
                    )
                    
                    estimated_tokens = self._estimate_tokens(messages)
                    
                    if estimated_tokens > max_input_tokens:
                        logger.warning(
                            f"Compression insufficient: {estimated_tokens} tokens, "
                            f"provider limit: {max_input_tokens}"
                        )
                
                if self.cost_controller:
                    can_proceed, budget_message = await self._check_budget(
                        estimated_tokens, task_type
                    )
                    
                    if not can_proceed:
                        if self.enable_compression and compression_metadata.get("compressed"):
                            logger.info(
                                f"Budget still exceeded after compression, "
                                f"trying fallback provider"
                            )
                        else:
                            budget_error = BudgetExceededError(
                                f"Budget exceeded: {budget_message}"
                            )
                            
                            self._log_llm_call(
                                provider=provider_name,
                                model=model,
                                task_type=task_type,
                                status="budget_exceeded",
                                token_count=estimated_tokens,
                                error_message=budget_message
                            )
                            
                            if provider_idx == len(providers_to_try) - 1:
                                raise budget_error
                            
                            continue
                
                return await self._call_with_retry(
                    adapter=adapter,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    task_type=task_type,
                    provider_name=provider_name,
                    **kwargs
                )
                
            except (RateLimitError, MalformedResponseError, ProviderUnavailableError) as e:
                last_error = e
                logger.error(f"LLM call failed for {provider_name}: {str(e)}")
                
                self._log_llm_call(
                    provider=provider_name,
                    model=adapter.get_model_name(task_type) if provider_name in self.adapters else "unknown",
                    task_type=task_type,
                    status="failed",
                    error_message=str(e)
                )
                
                if provider_idx < len(providers_to_try) - 1:
                    next_provider = providers_to_try[provider_idx + 1]
                    logger.warning(f"Switching from {provider_name} to {next_provider}")
                    self._log_provider_switch(
                        from_provider=provider_name,
                        to_provider=next_provider,
                        reason=str(e)
                    )
                else:
                    raise ProviderUnavailableError(
                        f"All providers failed. Last error: {str(e)}"
                    )
                    
            except BudgetExceededError as e:
                last_error = e
                
                if provider_idx < len(providers_to_try) - 1:
                    next_provider = providers_to_try[provider_idx + 1]
                    logger.warning(
                        f"Budget exceeded for {provider_name}, "
                        f"switching to {next_provider}"
                    )
                    self._log_provider_switch(
                        from_provider=provider_name,
                        to_provider=next_provider,
                        reason="budget_exceeded"
                    )
                else:
                    raise
                    
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error for {provider_name}: {str(e)}")
                
                if provider_idx < len(providers_to_try) - 1:
                    next_provider = providers_to_try[provider_idx + 1]
                    logger.warning(f"Switching from {provider_name} to {next_provider} due to error")
                    self._log_provider_switch(
                        from_provider=provider_name,
                        to_provider=next_provider,
                        reason=str(e)
                    )
                else:
                    raise ProviderUnavailableError(
                        f"All providers failed. Last error: {str(e)}"
                    )
        
        raise ProviderUnavailableError("No providers available")
    
    async def _call_with_retry(
        self,
        adapter: LLMAdapter,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        task_type: str,
        provider_name: str,
        **kwargs
    ) -> str:
        """使用指数退避重试调用 LLM
        
        Args:
            adapter: LLM 适配器
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            task_type: 任务类型
            provider_name: 提供商名称
            **kwargs: 其他参数
            
        Returns:
            生成的文本内容
        """
        async def _call_with_retry_inner():
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
            _call_with_retry_inner,
            max_retries=3,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0
        )
        
        self._log_llm_call(
            provider=provider_name,
            model=model,
            task_type=task_type,
            status="success",
            response_time_ms=response_time,
            token_count=self._estimate_tokens(messages)
        )
        
        return result
    
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
