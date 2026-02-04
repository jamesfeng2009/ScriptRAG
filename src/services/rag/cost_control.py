"""
RAG Cost Control and Context Compression System

功能：
1. Token 预算控制 - 防止 API 超出预算
2. 上下文压缩 - 减少 token 消耗
3. 成本监控 - 跟踪 API 调用成本
4. 智能摘要 - 压缩检索结果

使用示例：
```python
from src.services.rag.cost_control import CostController, TokenBudget
from src.services.rag.context_compressor import ContextCompressor

# 初始化成本控制器
cost_controller = CostController(
    max_tokens_per_request=8000,
    max_cost_per_day=10.0,
    budget_alert_threshold=0.8
)

# 初始化上下文压缩器
compressor = ContextCompressor(
    max_tokens=4000,
    compression_ratio=0.5,
    preserve_key_info=True
)

# 压缩检索结果
compressed_context = await compressor.compress(
    query="如何实现异步操作",
    retrieved_documents=results,
    llm_service=llm_service
)
```
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel
import hashlib

logger = logging.getLogger(__name__)


class CostLevel(Enum):
    """成本级别"""
    LOW = "low"      # < $0.01
    MEDIUM = "medium"  # $0.01 - $0.1
    HIGH = "high"    # $0.1 - $0.5
    CRITICAL = "critical"  # > $0.5


class MessagePriority(Enum):
    """消息优先级枚举
    
    用于控制消息在压缩过程中的保留顺序
    """
    CRITICAL = 0  # 关键消息（系统提示、工具调用请求）
    HIGH = 1      # 高优先级（用户请求、工具响应、长消息）
    NORMAL = 2    # 普通消息（一般对话）
    LOW = 3       # 低优先级（简短确认、闲聊）


class CompressionStrategyType(Enum):
    """压缩策略类型枚举"""
    MIDDLE_REMOVAL = "middle_removal"    # 中间移除策略
    OLDEST_REMOVAL = "oldest_removal"    # 最旧移除策略


class LLMProvider(Enum):
    """LLM 供应商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    QWEN = "qwen"
    LMSTUDIO = "lmstudio"
    OLLAMA = "ollama"
    OTHER = "other"


@dataclass
class ProviderConfig:
    """供应商配置"""
    provider: LLMProvider
    model: str
    default_strategy: CompressionStrategyType
    max_context_tokens: int
    recommended_preserve_start: int
    recommended_preserve_end: int


PROVIDER_CONFIGS: Dict[LLMProvider, ProviderConfig] = {
    LLMProvider.OPENAI: ProviderConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        default_strategy=CompressionStrategyType.MIDDLE_REMOVAL,
        max_context_tokens=128000,
        recommended_preserve_start=3,
        recommended_preserve_end=3
    ),
    LLMProvider.ANTHROPIC: ProviderConfig(
        provider=LLMProvider.ANTHROPIC,
        model="claude-3",
        default_strategy=CompressionStrategyType.OLDEST_REMOVAL,
        max_context_tokens=200000,
        recommended_preserve_start=2,
        recommended_preserve_end=4
    ),
    LLMProvider.GOOGLE: ProviderConfig(
        provider=LLMProvider.GOOGLE,
        model="gemini-pro",
        default_strategy=CompressionStrategyType.MIDDLE_REMOVAL,
        max_context_tokens=1000000,
        recommended_preserve_start=3,
        recommended_preserve_end=3
    ),
    LLMProvider.QWEN: ProviderConfig(
        provider=LLMProvider.QWEN,
        model="qwen-turbo",
        default_strategy=CompressionStrategyType.MIDDLE_REMOVAL,
        max_context_tokens=32000,
        recommended_preserve_start=3,
        recommended_preserve_end=3
    ),
    LLMProvider.LMSTUDIO: ProviderConfig(
        provider=LLMProvider.LMSTUDIO,
        model="local",
        default_strategy=CompressionStrategyType.MIDDLE_REMOVAL,
        max_context_tokens=32768,
        recommended_preserve_start=2,
        recommended_preserve_end=2
    ),
    LLMProvider.OLLAMA: ProviderConfig(
        provider=LLMProvider.OLLAMA,
        model="local",
        default_strategy=CompressionStrategyType.MIDDLE_REMOVAL,
        max_context_tokens=16384,
        recommended_preserve_start=2,
        recommended_preserve_end=2
    ),
}


@dataclass
class TokenUsage:
    """Token 使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    
    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost=self.cost + other.cost
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost
        }


@dataclass
class CostRecord:
    """成本记录"""
    timestamp: datetime
    operation: str
    tokens: TokenUsage
    model: str
    details: Dict[str, Any] = field(default_factory=dict)


class CostController:
    """
    LLM API 成本控制器
    
    功能：
    - Token 预算管理
    - 成本监控和警报
    - 速率限制
    - 使用统计
    """
    
    # OpenAI GPT-4o 定价 (每 1M tokens)
    PRICING = {
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        "qwen-max": {"input": 0.02, "output": 0.06},
        "qwen-turbo": {"input": 0.008, "output": 0.024},
        "glm-4": {"input": 0.05, "output": 0.15},
        "glm-3-turbo": {"input": 0.005, "output": 0.015},
    }
    
    def __init__(
        self,
        max_tokens_per_request: int = 8000,
        max_tokens_per_day: int = 500000,
        max_cost_per_day: float = 10.0,
        budget_alert_threshold: float = 0.8,
        provider: str = "openai",
        model: str = "gpt-4o-mini"
    ):
        """
        初始化成本控制器
        
        Args:
            max_tokens_per_request: 每次请求最大 token 数
            max_tokens_per_day: 每日最大 token 数
            max_cost_per_day: 每日最大成本（美元）
            budget_alert_threshold: 预算警报阈值（0-1）
            provider: LLM 提供商
            model: 默认模型
        """
        self.max_tokens_per_request = max_tokens_per_request
        self.max_tokens_per_day = max_tokens_per_day
        self.max_cost_per_day = max_cost_per_day
        self.budget_alert_threshold = budget_alert_threshold
        self.provider = provider
        self.model = model
        
        self._daily_usage: Dict[str, TokenUsage] = {}
        self._daily_cost: float = 0.0
        self._last_reset: datetime = datetime.now()
        self._cost_history: List[CostRecord] = []
        self._lock = asyncio.Lock()
    
    def _get_pricing(self, model: str) -> Dict[str, float]:
        """获取模型定价"""
        return self.PRICING.get(model, {"input": 0.01, "output": 0.03})
    
    def _should_reset_daily(self) -> bool:
        """检查是否应该重置每日统计"""
        now = datetime.now()
        return (now - self._last_reset).days >= 1
    
    async def _reset_daily_stats(self):
        """重置每日统计"""
        async with self._lock:
            if self._should_reset_daily():
                self._daily_usage = {}
                self._daily_cost = 0.0
                self._last_reset = datetime.now()
                logger.info("Daily cost statistics reset")
    
    def _calculate_cost(self, model: str, usage: TokenUsage) -> float:
        """计算 API 调用成本"""
        pricing = self._get_pricing(model)
        cost = (usage.prompt_tokens * pricing["input"] / 1_000_000 +
                usage.completion_tokens * pricing["output"] / 1_000_000)
        return cost
    
    async def check_budget(
        self,
        estimated_tokens: int,
        operation: str = "general"
    ) -> Tuple[bool, str]:
        """
        检查是否超出预算
        
        Args:
            estimated_tokens: 预估 token 数量
            operation: 操作名称
            
        Returns:
            (是否允许, 消息)
        """
        await self._reset_daily_stats()
        
        # 检查每日 token 限制
        current_tokens = sum(
            usage.total_tokens for usage in self._daily_usage.values()
        )
        
        if current_tokens + estimated_tokens > self.max_tokens_per_day:
            return False, f"Daily token limit exceeded: {current_tokens}/{self.max_tokens_per_day}"
        
        # 检查每日成本限制
        estimated_cost = (estimated_tokens / 1_000_000) * 0.01  # 估算
        if self._daily_cost + estimated_cost > self.max_cost_per_day:
            return False, f"Daily cost limit exceeded: ${self._daily_cost:.2f}/${self.max_cost_per_day}"
        
        # 检查单次请求限制
        if estimated_tokens > self.max_tokens_per_request:
            return False, f"Request token limit exceeded: {estimated_tokens}/{self.max_tokens_per_request}"
        
        return True, "Budget check passed"
    
    async def record_usage(
        self,
        operation: str,
        model: str,
        usage: TokenUsage,
        details: Dict[str, Any] = None
    ):
        """记录 API 使用情况"""
        async with self._lock:
            # 计算成本
            cost = self._calculate_cost(model, usage)
            total_usage = self._daily_usage.get(operation, TokenUsage())
            total_usage = total_usage + usage
            self._daily_usage[operation] = total_usage
            self._daily_cost += cost
            
            # 记录历史
            record = CostRecord(
                timestamp=datetime.now(),
                operation=operation,
                tokens=usage,
                model=model,
                details=details or {}
            )
            self._cost_history.append(record)
            
            # 只保留最近 1000 条记录
            if len(self._cost_history) > 1000:
                self._cost_history = self._cost_history[-1000:]
            
            # 检查是否需要发出警报
            cost_ratio = self._daily_cost / self.max_cost_per_day
            if cost_ratio >= self.budget_alert_threshold:
                logger.warning(
                    f"Cost alert: ${self._daily_cost:.2f} "
                    f"({cost_ratio * 100:.1f}% of daily budget)"
                )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        return {
            "daily_cost": self._daily_cost,
            "daily_token_limit": self.max_tokens_per_day,
            "daily_cost_limit": self.max_cost_per_day,
            "usage_by_operation": {
                op: usage.to_dict() 
                for op, usage in self._daily_usage.items()
            },
            "cost_level": self.get_cost_level().value
        }
    
    def get_cost_level(self) -> CostLevel:
        """获取当前成本级别"""
        ratio = self._daily_cost / self.max_cost_per_day
        
        if ratio < 0.2:
            return CostLevel.LOW
        elif ratio < 0.5:
            return CostLevel.MEDIUM
        elif ratio < 0.8:
            return CostLevel.HIGH
        else:
            return CostLevel.CRITICAL
    
    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的使用历史"""
        records = self._cost_history[-limit:]
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "operation": r.operation,
                "model": r.model,
                "tokens": r.tokens.to_dict(),
                "cost": self._calculate_cost(r.model, r.tokens)
            }
            for r in records
        ]


class TokenBudget:
    """
    Token 预算管理器
    
    用于管理对话或任务的 token 预算
    """
    
    def __init__(
        self,
        max_tokens: int = 12000,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95
    ):
        """
        初始化 token 预算
        
        Args:
            max_tokens: 最大 token 数
            warning_threshold: 警告阈值
            critical_threshold: 临界阈值
        """
        self.max_tokens = max_tokens
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.used_tokens = 0
        self.turns = 0
    
    def check(self, tokens: int = 0) -> Tuple[bool, str]:
        """
        检查是否超出预算
        
        Args:
            tokens: 预估要使用的 token 数
            
        Returns:
            (是否允许, 消息)
        """
        new_total = self.used_tokens + tokens
        
        if new_total >= self.max_tokens * self.critical_threshold:
            return False, f"CRITICAL: Token budget nearly exhausted ({new_total}/{self.max_tokens})"
        elif new_total >= self.max_tokens * self.warning_threshold:
            return False, f"WARNING: Token budget running low ({new_total}/{self.max_tokens})"
        
        return True, f"OK: {new_total}/{self.max_tokens} tokens"
    
    def use(self, tokens: int):
        """使用 token"""
        self.used_tokens += tokens
        self.turns += 1
    
    def get_remaining(self) -> int:
        """获取剩余 token"""
        return max(0, self.max_tokens - self.used_tokens)
    
    def get_usage_ratio(self) -> float:
        """获取使用比例"""
        return self.used_tokens / self.max_tokens


@dataclass
class CompressedMessage:
    """压缩后的消息"""
    content: str
    role: str
    priority: MessagePriority
    token_count: int
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    preserved: bool = False  # 是否被保留（不可移除）
    original_index: int = -1  # 原始消息索引


class CompressionStrategy(ABC):
    """压缩策略基类"""
    
    def __init__(
        self,
        preserve_start: int = 3,
        preserve_end: int = 3,
        min_messages: int = 2
    ):
        """
        初始化压缩策略
        
        Args:
            preserve_start: 保留开头消息数量
            preserve_end: 保留结尾消息数量
            min_messages: 最少保留消息数量
        """
        self.preserve_start = preserve_start
        self.preserve_end = preserve_end
        self.min_messages = min_messages
    
    @abstractmethod
    def compress(
        self,
        messages: List[Dict[str, Any]],
        target_token_count: int,
        token_estimator: callable
    ) -> List[Dict[str, Any]]:
        """执行压缩
        
        Args:
            messages: 消息列表
            target_token_count: 目标token数量
            token_estimator: token估算函数
            
        Returns:
            压缩后的消息列表
        """
        pass
    
    def _assign_priorities(
        self,
        messages: List[Dict[str, Any]],
        preserve_indices: set
    ) -> List[CompressedMessage]:
        """为消息分配优先级
        
        Args:
            messages: 消息列表
            preserve_indices: 需要保留的消息索引集合
            
        Returns:
            带优先级排序的消息列表
        """
        prioritized = []
        
        for idx, msg in enumerate(messages):
            content = msg.get("content", "") or ""
            role = msg.get("role", "user")
            
            # 判断是否保留
            preserved = idx in preserve_indices
            
            # 分配优先级
            priority = self._calculate_priority(msg, idx, len(messages))
            
            # 估算token
            token_count = self._estimate_tokens(content)
            
            prioritized.append(CompressedMessage(
                content=content,
                role=role,
                priority=priority,
                token_count=token_count,
                timestamp=datetime.now(),
                metadata=msg.get("metadata", {}),
                preserved=preserved,
                original_index=idx
            ))
        
        # 按优先级排序（低优先级在前，优先移除）
        prioritized.sort(key=lambda x: (0 if x.preserved else 1, x.priority.value, x.original_index))
        
        return prioritized
    
    def _calculate_priority(
        self,
        msg: Dict[str, Any],
        index: int,
        total: int
    ) -> MessagePriority:
        """计算单条消息的优先级
        
        Args:
            msg: 消息字典
            index: 消息索引
            total: 总消息数
            
        Returns:
            消息优先级
        """
        content = msg.get("content", "") or ""
        role = msg.get("role", "assistant")
        
        # 系统消息和工具调用为最高优先级
        if role == "system":
            return MessagePriority.CRITICAL
        
        if msg.get("tool_calls") or content.startswith("Function call:"):
            return MessagePriority.CRITICAL
        
        if role == "tool":
            return MessagePriority.HIGH
        
        # 对话开头和结尾保留高优先级
        if index < self.preserve_start or index >= total - self.preserve_end:
            return MessagePriority.HIGH
        
        # 长消息提高优先级
        if len(content) > 800:
            return MessagePriority.HIGH
        
        # 短消息且无问号降低优先级
        if len(content) < 20 and "?" not in content:
            return MessagePriority.LOW
        
        # 函数调用结果
        if "function_call" in msg or "tool_results" in msg:
            return MessagePriority.HIGH
        
        return MessagePriority.NORMAL
    
    def _estimate_tokens(self, text: str) -> int:
        """估算token数量
        
        Args:
            text: 文本内容
            
        Returns:
            估算的token数量
        """
        return len(text) // 4


class MiddleRemovalStrategy(CompressionStrategy):
    """中间移除策略
    
    原理：保留对话的开始和结束部分，移除中间的消息
    适用场景：保持对话流程和最近上下文最重要的情况
    """
    
    def __init__(
        self,
        preserve_start: int = 3,
        preserve_end: int = 3,
        min_messages: int = 2
    ):
        """
        初始化中间移除策略
        
        Args:
            preserve_start: 保留开头消息数量
            preserve_end: 保留结尾消息数量
            min_messages: 最少保留消息数量
        """
        super().__init__(preserve_start, preserve_end, min_messages)
        logger.info(f"初始化 MiddleRemovalStrategy: preserve_start={preserve_start}, preserve_end={preserve_end}")
    
    def compress(
        self,
        messages: List[Dict[str, Any]],
        target_token_count: int,
        token_estimator: callable = None
    ) -> List[Dict[str, Any]]:
        """执行中间移除压缩
        
        Args:
            messages: 消息列表
            target_token_count: 目标token数量
            token_estimator: token估算函数（可选）
            
        Returns:
            压缩后的消息列表
        """
        if len(messages) <= self.min_messages:
            return messages.copy()
        
        current_tokens = sum(
            self._estimate_tokens(msg.get("content", ""))
            for msg in messages
        )
        
        if current_tokens <= target_token_count:
            return messages.copy()
        
        total = len(messages)
        
        # 确定保留区域
        preserve_indices = set()
        
        # 保留开头消息
        for i in range(min(self.preserve_start, total // 2)):
            preserve_indices.add(i)
        
        # 保留结尾消息
        for i in range(max(total - self.preserve_end, total // 2), total):
            preserve_indices.add(i)
        
        # 至少保留最小消息数
        while len(preserve_indices) < self.min_messages and len(preserve_indices) < total:
            preserve_indices.add(total - len(preserve_indices) - 1)
        
        # 分配优先级并排序
        prioritized = self._assign_priorities(messages, preserve_indices)
        
        # 移除低优先级消息直到达到目标
        result = []
        removed_indices = set()
        
        # 先收集要保留的消息
        for pm in prioritized:
            if pm.preserved or pm.priority == MessagePriority.CRITICAL:
                result.append(messages[pm.original_index])
            elif current_tokens > target_token_count:
                # 移除低优先级消息
                removed_indices.add(pm.original_index)
                current_tokens -= pm.token_count
            else:
                result.append(messages[pm.original_index])
        
        # 按原始顺序排序结果
        result.sort(key=lambda x: messages.index(x))
        
        logger.info(
            f"MiddleRemovalStrategy: 原始={len(messages)}, 压缩后={len(result)}, "
            f"移除={len(messages) - len(result)}"
        )
        
        return result


class OldestRemovalStrategy(CompressionStrategy):
    """最旧移除策略
    
    原理：优先移除最早的消息，保留较新的消息
    适用场景：长对话中最近上下文最重要的情况
    """
    
    def __init__(
        self,
        min_messages: int = 2,
        priority_weight: float = 0.7
    ):
        """
        初始化最旧移除策略
        
        Args:
            min_messages: 最少保留消息数量
            priority_weight: 优先级权重（时间vs优先级）
        """
        super().__init__(preserve_start=0, preserve_end=0, min_messages=min_messages)
        self.priority_weight = priority_weight
        logger.info(f"初始化 OldestRemovalStrategy: min_messages={min_messages}")
    
    def compress(
        self,
        messages: List[Dict[str, Any]],
        target_token_count: int,
        token_estimator: callable = None
    ) -> List[Dict[str, Any]]:
        """执行最旧移除压缩
        
        Args:
            messages: 消息列表
            target_token_count: 目标token数量
            token_estimator: token估算函数（可选）
            
        Returns:
            压缩后的消息列表
        """
        if len(messages) <= self.min_messages:
            return messages.copy()
        
        current_tokens = sum(
            self._estimate_tokens(msg.get("content", ""))
            for msg in messages
        )
        
        if current_tokens <= target_token_count:
            return messages.copy()
        
        total = len(messages)
        
        # 所有消息默认不保留（除非是关键消息）
        preserve_indices = set()
        
        # 分配优先级
        prioritized = self._assign_priorities(messages, preserve_indices)
        
        # 按优先级和时间排序：低优先级在前，同优先级按原始索引（旧消息在前）
        prioritized.sort(key=lambda x: (
            x.priority.value,  # 优先级排序
            x.original_index   # 索引排序（小的在前=旧消息）
        ))
        
        # 移除低优先级消息
        result = []
        removed_count = 0
        
        for pm in prioritized:
            if current_tokens <= target_token_count:
                result.append(messages[pm.original_index])
                continue
            
            # 关键消息必须保留
            if pm.priority == MessagePriority.CRITICAL:
                result.append(messages[pm.original_index])
                continue
            
            # 检查是否已达到最小消息数
            remaining_count = total - removed_count - 1
            if remaining_count <= self.min_messages:
                result.append(messages[pm.original_index])
                continue
            
            # 移除消息
            removed_count += 1
            current_tokens -= pm.token_count
        
        # 按原始顺序排序
        result.sort(key=lambda x: messages.index(x))
        
        logger.info(
            f"OldestRemovalStrategy: 原始={len(messages)}, 压缩后={len(result)}, "
            f"移除={len(messages) - len(result)}"
        )
        
        return result


@dataclass
class CompressionHistory:
    """压缩历史记录"""
    timestamp: datetime
    original_count: int
    compressed_count: int
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    strategy: str
    reason: str


class ContextCompressor:
    """
    上下文压缩器
    
    功能：
    - 智能摘要检索结果
    - 消息历史压缩
    - 移除冗余信息
    - 保留关键信息
    - 压缩长文本
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        compression_ratio: float = 0.5,
        preserve_key_info: bool = True,
        llm_service: Any = None,
        compression_threshold: float = 0.9,
        check_interval: int = 5,
        enable_compression: bool = True,
        default_strategy: CompressionStrategyType = CompressionStrategyType.MIDDLE_REMOVAL
    ):
        """
        初始化上下文压缩器
        
        Args:
            max_tokens: 压缩后最大 token 数
            compression_ratio: 压缩比例 (0-1)
            preserve_key_info: 是否保留关键信息
            llm_service: LLM 服务（用于智能摘要）
            compression_threshold: 压缩触发阈值（0-1），达到此比例时触发压缩
            check_interval: 压缩检查时间间隔（秒）
            enable_compression: 是否启用压缩
            default_strategy: 默认压缩策略
        """
        self.max_tokens = max_tokens
        self.compression_ratio = compression_ratio
        self.preserve_key_info = preserve_key_info
        self.llm_service = llm_service
        self.compression_threshold = compression_threshold
        self.check_interval = check_interval
        self.enable_compression = enable_compression
        self.default_strategy = default_strategy
        
        # 初始化压缩策略
        self.strategies = {
            CompressionStrategyType.MIDDLE_REMOVAL: MiddleRemovalStrategy(
                preserve_start=3,
                preserve_end=3,
                min_messages=2
            ),
            CompressionStrategyType.OLDEST_REMOVAL: OldestRemovalStrategy(
                min_messages=2
            )
        }
        
        # 压缩状态管理
        self.last_compression_check: Optional[datetime] = None
        self.compression_history: List[CompressionHistory] = []
        self.current_strategy = default_strategy
        
        logger.info(
            f"ContextCompressor 初始化完成: max_tokens={max_tokens}, "
            f"threshold={compression_threshold}, interval={check_interval}s, "
            f"strategy={default_strategy.value}"
        )
    
    async def compress(
        self,
        query: str,
        retrieved_documents: List[Any],
        llm_service: Any = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        压缩检索结果
        
        Args:
            query: 查询文本
            retrieved_documents: 检索到的文档
            llm_service: LLM 服务
            
        Returns:
            压缩后的上下文
        """
        # 估算原始 token 数
        original_tokens = self._estimate_tokens(retrieved_documents)
        
        # 如果不需要压缩
        if original_tokens <= self.max_tokens:
            return {
                "compressed": False,
                "original_tokens": original_tokens,
                "compressed_tokens": original_tokens,
                "documents": retrieved_documents,
                "summary": None
            }
        
        # 使用 LLM 进行智能摘要
        if llm_service or self.llm_service:
            service = llm_service or self.llm_service
            summary = await self._llm_summarize(query, retrieved_documents, service)
            compressed_docs = await self._smart_compress(
                query, retrieved_documents, summary
            )
        else:
            # 使用规则压缩
            compressed_docs = self._rule_based_compress(retrieved_documents)
        
        compressed_tokens = self._estimate_tokens(compressed_docs)
        
        return {
            "compressed": True,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": compressed_tokens / original_tokens,
            "documents": compressed_docs,
            "summary": summary if llm_service or self.llm_service else None
        }
    
    def _estimate_tokens(self, documents: List[Any]) -> int:
        """估算 token 数量"""
        total_chars = 0
        for doc in documents:
            if hasattr(doc, 'content'):
                total_chars += len(doc.content)
            elif isinstance(doc, dict) and 'content' in doc:
                total_chars += len(doc['content'])
            elif isinstance(doc, str):
                total_chars += len(doc)
        
        # 粗略估算：1 token ≈ 4 characters
        return total_chars // 4
    
    async def _llm_summarize(
        self,
        query: str,
        documents: List[Any],
        llm_service: Any
    ) -> str:
        """使用 LLM 生成摘要"""
        
        # 合并文档内容
        content = "\n\n".join([
            doc.content if hasattr(doc, 'content') else str(doc)
            for doc in documents
        ])
        
        # 限制输入长度
        max_input = 8000  # tokens
        if len(content) > max_input * 4:
            content = content[:max_input * 4] + "..."
        
        prompt = f"""
请根据以下查询和相关文档，生成一个简洁的摘要：

查询：{query}

文档内容：
{content}

要求：
1. 摘要应突出与查询相关的信息
2. 保留关键的技术细节和代码示例
3. 移除冗余和重复的信息
4. 摘要长度适中，便于后续处理

摘要：
"""
        
        try:
            response = await llm_service.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                task_type="general"
            )
            return response
        except Exception as e:
            logger.error(f"Failed to generate LLM summary: {e}")
            return ""
    
    async def _smart_compress(
        self,
        query: str,
        documents: List[Any],
        summary: str
    ) -> List[Any]:
        """智能压缩文档"""
        # 如果有摘要，只返回摘要
        if summary:
            from src.services.retrieval.strategies import RetrievalResult
            
            return [RetrievalResult(
                id="compressed-summary",
                file_path="COMPRESSED_SUMMARY",
                content=summary,
                similarity=1.0,
                confidence=1.0,
                has_deprecated=False,
                has_fixme=False,
                has_todo=False,
                has_security=False,
                metadata={"compressed": True, "original_count": len(documents)}
            )]
        
        # 否则使用规则压缩
        return self._rule_based_compress(documents)
    
    def _rule_based_compress(self, documents: List[Any]) -> List[Any]:
        """基于规则的压缩"""
        compressed = []
        seen_content = set()
        
        for doc in documents:
            content = doc.content if hasattr(doc, 'content') else str(doc)
            
            # 跳过重复内容
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # 移除过长注释和空白
            lines = content.split('\n')
            filtered_lines = []
            for line in lines:
                stripped = line.strip()
                # 跳过纯注释行
                if stripped.startswith('#') and len(stripped) < 50:
                    continue
                # 跳过连续空行
                if stripped == '' and filtered_lines and filtered_lines[-1].strip() == '':
                    continue
                filtered_lines.append(line)
            
            compressed_content = '\n'.join(filtered_lines)
            
            # 截断过长内容
            max_chars = 2000
            if len(compressed_content) > max_chars:
                compressed_content = compressed_content[:max_chars] + "\n... [truncated]"
            
            # 创建压缩后的文档
            if hasattr(doc, 'with_content'):
                compressed.append(doc.with_content(compressed_content))
            else:
                compressed.append(doc)
        
        return compressed
    
    def should_compress(self, current_token_count: int, max_tokens: int = None) -> Tuple[bool, str]:
        """检查是否应该触发压缩
        
        Args:
            current_token_count: 当前 token 数量
            max_tokens: 最大 token 数（可选，默认使用实例配置）
            
        Returns:
            (是否应该压缩, 原因)
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        # 检查压缩是否启用
        if not self.enable_compression:
            return False, "compression_disabled"
        
        # 检查是否有分词器
        if current_token_count <= 0:
            return False, "no_tokens"
        
        # 检查时间间隔
        if self.last_compression_check:
            time_since_last = (datetime.now() - self.last_compression_check).total_seconds()
            if time_since_last < self.check_interval:
                return False, f"interval_not_reached:{int(time_since_last)}s"
        
        # 检查阈值
        utilization = current_token_count / max_tokens
        if utilization < self.compression_threshold:
            return False, f"below_threshold:{utilization:.2%}"
        
        return True, f"threshold_reached:{utilization:.2%}"
    
    async def compress_messages(
        self,
        messages: List[Dict[str, Any]],
        target_token_count: int = None,
        strategy_type: CompressionStrategyType = None,
        reason: str = "manual"
    ) -> Dict[str, Any]:
        """压缩消息历史
        
        Args:
            messages: 原始消息列表
            target_token_count: 目标 token 数量（可选，默认使用 max_tokens）
            strategy_type: 压缩策略类型（可选，默认使用默认策略）
            reason: 压缩触发原因
            
        Returns:
            压缩结果字典
        """
        if target_token_count is None:
            target_token_count = int(self.max_tokens * self.compression_ratio)
        
        if strategy_type is None:
            strategy_type = self.default_strategy
        
        # 估算原始 token 数
        original_tokens = sum(
            self._estimate_message_tokens(msg)
            for msg in messages
        )
        
        # 如果不需要压缩
        if original_tokens <= target_token_count:
            return {
                "compressed": False,
                "original_count": len(messages),
                "compressed_count": len(messages),
                "original_tokens": original_tokens,
                "compressed_tokens": original_tokens,
                "compression_ratio": 1.0,
                "strategy": strategy_type.value,
                "reason": reason,
                "messages": messages
            }
        
        # 获取策略
        strategy = self.strategies.get(strategy_type)
        if not strategy:
            logger.warning(f"未知策略类型: {strategy_type}，使用默认中间移除策略")
            strategy = self.strategies[CompressionStrategyType.MIDDLE_REMOVAL]
        
        # 执行压缩
        compressed_messages = strategy.compress(messages, target_token_count)
        
        # 估算压缩后的 token 数
        compressed_tokens = sum(
            self._estimate_message_tokens(msg)
            for msg in compressed_messages
        )
        
        # 记录压缩历史
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        history = CompressionHistory(
            timestamp=datetime.now(),
            original_count=len(messages),
            compressed_count=len(compressed_messages),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            strategy=strategy_type.value,
            reason=reason
        )
        self.compression_history.append(history)
        
        # 限制历史记录数量
        if len(self.compression_history) > 100:
            self.compression_history = self.compression_history[-100:]
        
        logger.info(
            f"消息压缩完成: 原始={len(messages)}, 压缩后={len(compressed_messages)}, "
            f"策略={strategy_type.value}, 原因={reason}"
        )
        
        return {
            "compressed": True,
            "original_count": len(messages),
            "compressed_count": len(compressed_messages),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            "strategy": strategy_type.value,
            "reason": reason,
            "messages": compressed_messages
        }
    
    async def check_and_compress(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = None,
        strategy_type: CompressionStrategyType = None
    ) -> Dict[str, Any]:
        """检查并执行压缩
        
        Args:
            messages: 消息列表
            max_tokens: 最大 token 数
            strategy_type: 压缩策略类型
            
        Returns:
            压缩结果
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        # 计算当前 token 数
        current_tokens = sum(
            self._estimate_message_tokens(msg)
            for msg in messages
        )
        
        # 检查是否应该压缩
        should_compress, reason = self.should_compress(current_tokens, max_tokens)
        
        if not should_compress:
            return {
                "compressed": False,
                "reason": reason,
                "messages": messages,
                "current_tokens": current_tokens,
                "max_tokens": max_tokens
            }
        
        # 执行压缩
        target_tokens = int(max_tokens * self.compression_ratio)
        result = await self.compress_messages(
            messages=messages,
            target_token_count=target_tokens,
            strategy_type=strategy_type,
            reason=reason
        )
        
        # 更新最后检查时间
        self.last_compression_check = datetime.now()
        
        return result
    
    def _estimate_message_tokens(self, message: Dict[str, Any]) -> int:
        """估算消息的 token 数量
        
        Args:
            message: 消息字典
            
        Returns:
            估算的 token 数量
        """
        content = ""
        
        # 获取内容
        if isinstance(message, dict):
            content = message.get("content", "") or ""
            
            # 特殊处理工具调用
            if message.get("tool_calls"):
                content += f" tool_calls: {len(message['tool_calls'])} calls"
            
            if message.get("role") == "tool":
                content = f"[Tool Response] {content}"
        else:
            content = str(message)
        
        # 粗略估算：1 token ≈ 4 characters
        return len(content) // 4
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计信息
        
        Returns:
            统计信息字典
        """
        if not self.compression_history:
            return {
                "total_compressions": 0,
                "average_compression_ratio": 0,
                "strategy_usage": {},
                "recent_compressions": [],
                "current_strategy": self.current_strategy.value,
                "compression_enabled": self.enable_compression,
                "threshold": self.compression_threshold,
                "check_interval": self.check_interval
            }
        
        total_compressions = len(self.compression_history)
        
        # 计算平均压缩比
        ratios = [h.compression_ratio for h in self.compression_history]
        avg_ratio = sum(ratios) / len(ratios)
        
        # 统计策略使用情况
        strategy_usage = {}
        for h in self.compression_history:
            strategy_usage[h.strategy] = strategy_usage.get(h.strategy, 0) + 1
        
        return {
            "total_compressions": total_compressions,
            "average_compression_ratio": avg_ratio,
            "strategy_usage": strategy_usage,
            "recent_compressions": [
                {
                    "timestamp": h.timestamp.isoformat(),
                    "original_count": h.original_count,
                    "compressed_count": h.compressed_count,
                    "strategy": h.strategy,
                    "reason": h.reason
                }
                for h in self.compression_history[-10:]
            ],
            "current_strategy": self.current_strategy.value,
            "compression_enabled": self.enable_compression,
            "threshold": self.compression_threshold,
            "check_interval": self.check_interval
        }
    
    def set_strategy(self, strategy_type: CompressionStrategyType):
        """设置压缩策略
        
        Args:
            strategy_type: 策略类型
        """
        self.current_strategy = strategy_type
        logger.info(f"压缩策略已切换为: {strategy_type.value}")
    
    def reset_compression_history(self):
        """重置压缩历史"""
        self.compression_history = []
        self.last_compression_check = None
        logger.info("压缩历史已重置")


class SmartRetriever:
    """
    智能检索器
    
    结合成本控制和上下文压缩的检索器
    """
    
    def __init__(
        self,
        vector_db: Any,
        llm_service: Any,
        cost_controller: CostController = None,
        context_compressor: ContextCompressor = None,
        config: Dict[str, Any] = None
    ):
        """
        初始化智能检索器
        
        Args:
            vector_db: 向量数据库
            llm_service: LLM 服务
            cost_controller: 成本控制器
            context_compressor: 上下文压缩器
            config: 配置
        """
        self.vector_db = vector_db
        self.llm_service = llm_service
        self.cost_controller = cost_controller or CostController()
        self.context_compressor = context_compressor or ContextCompressor()
        self.config = config or {}
    
    async def smart_search(
        self,
        query: str,
        workspace_id: str,
        top_k: int = 5,
        use_compression: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        智能检索
        
        Args:
            query: 查询文本
            workspace_id: 工作空间 ID
            top_k: 返回结果数量
            use_compression: 是否使用压缩
            
        Returns:
            检索结果和元数据
        """
        start_time = datetime.now()
        
        # 1. 生成查询嵌入
        embedding_start = datetime.now()
        estimated_embedding_tokens = 100  # 估算
        
        can_proceed, message = await self.cost_controller.check_budget(
            estimated_embedding_tokens, "embedding"
        )
        if not can_proceed:
            logger.warning(f"Budget check failed: {message}")
        
        try:
            query_embedding = await self.llm_service.embedding(query)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            query_embedding = None
        
        embedding_time = (datetime.now() - embedding_start).total_seconds()
        
        # 2. 向量搜索
        search_start = datetime.now()
        try:
            results = await self.vector_db.hybrid_search(
                workspace_id=workspace_id,
                query_embedding=query_embedding,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            results = []
        
        search_time = (datetime.now() - search_start).total_seconds()
        
        # 3. 上下文压缩
        compression_metadata = {}
        if use_compression and results:
            compression_start = datetime.now()
            
            compressed = await self.context_compressor.compress(
                query=query,
                retrieved_documents=results,
                llm_service=self.llm_service
            )
            
            compression_metadata = {
                "was_compressed": compressed["compressed"],
                "original_tokens": compressed.get("original_tokens", 0),
                "compressed_tokens": compressed.get("compressed_tokens", 0),
                "compression_ratio": compressed.get("compression_ratio", 1.0)
            }
            
            final_results = compressed.get("documents", results)
        else:
            final_results = results
        
        compression_time = (datetime.now() - compression_start).total_seconds() if use_compression else 0
        
        # 4. 记录成本
        estimated_tokens = self.context_compressor._estimate_tokens(results)
        await self.cost_controller.record_usage(
            operation="smart_search",
            model=self.config.get("model", "gpt-4o-mini"),
            usage=TokenUsage(
                prompt_tokens=estimated_tokens,
                completion_tokens=len(query) // 4,
                total_tokens=estimated_tokens + len(query) // 4
            ),
            details={
                "query_length": len(query),
                "result_count": len(results),
                "compression_used": use_compression
            }
        )
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "query": query,
            "results": final_results,
            "metadata": {
                "total_time": total_time,
                "embedding_time": embedding_time,
                "search_time": search_time,
                "compression_time": compression_time,
                "result_count": len(final_results),
                "cost_level": self.cost_controller.get_cost_level().value,
                "usage_stats": self.cost_controller.get_usage_stats(),
                **compression_metadata
            }
        }


@dataclass
class ToolCallPair:
    """工具调用对
    
    记录一次完整的工具调用过程，包括请求和响应。
    """
    request_index: int
    response_index: int
    tool_name: str
    request_message: Dict[str, Any]
    response_message: Dict[str, Any]
    is_valid: bool = True


class ToolCallPairValidator:
    """工具调用对验证器
    
    功能：
    1. 检测消息流中的工具调用对（请求+响应）
    2. 确保工具调用对在压缩过程中不被分离
    3. 验证工具调用对的完整性
    
    工具调用对模式：
    - assistant 消息包含 tool_calls 或 function_call
    - 后续必须有对应的 user 消息包含 tool_results 或 function_response
    """
    
    def __init__(self):
        self.tool_call_patterns = [
            ("tool_calls", "tool_results"),
            ("function_call", "function_response"),
            ("tool_calls", "content"),  # OpenAI 格式
        ]
    
    def detect_tool_calls(self, messages: List[Dict[str, Any]]) -> List[ToolCallPair]:
        """检测消息流中的工具调用对
        
        Args:
            messages: 消息列表
            
        Returns:
            检测到的工具调用对列表
        """
        tool_calls = []
        i = 0
        
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")
            
            if role == "assistant":
                tool_call = self._extract_tool_call(msg)
                if tool_call:
                    # 查找对应的响应
                    response_index = self._find_tool_response(
                        messages, i, tool_call.get("function", {}).get("name") or tool_call.get("name")
                    )
                    
                    if response_index is not None:
                        pair = ToolCallPair(
                            request_index=i,
                            response_index=response_index,
                            tool_name=tool_call.get("function", {}).get("name") or tool_call.get("name", "unknown"),
                            request_message=msg,
                            response_message=messages[response_index]
                        )
                        tool_calls.append(pair)
                        i = response_index + 1
                        continue
            
            i += 1
        
        return tool_calls
    
    def _extract_tool_call(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从消息中提取工具调用信息"""
        # OpenAI 格式
        if "tool_calls" in message:
            tc = message["tool_calls"]
            if isinstance(tc, list) and len(tc) > 0:
                return tc[0]
        
        # 传统格式
        if "function_call" in message:
            return {"function": message["function_call"]}
        
        return None
    
    def _find_tool_response(
        self,
        messages: List[Dict[str, Any]],
        start_index: int,
        tool_name: str
    ) -> Optional[int]:
        """查找工具调用的响应消息"""
        for i in range(start_index + 1, min(start_index + 10, len(messages))):
            msg = messages[i]
            role = msg.get("role", "")
            
            if role == "user":
                # 检查是否是工具响应
                content = msg.get("content", "")
                
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "tool_result":
                            result_tool = item.get("tool_use_id") or item.get("name")
                            if result_tool == tool_name or result_tool in str(item):
                                return i
                
                # 检查传统格式
                if "tool_results" in msg:
                    for result in msg["tool_results"]:
                        if result.get("function") == tool_name:
                            return i
                
                if "function_response" in msg:
                    if msg["function_response"].get("name") == tool_name:
                        return i
        
        return None
    
    def validate_pairs_for_compression(
        self,
        messages: List[Dict[str, Any]],
        indices_to_remove: List[int]
    ) -> Tuple[bool, List[str]]:
        """验证压缩计划不会破坏工具调用对
        
        Args:
            messages: 原始消息列表
            indices_to_remove: 计划删除的消息索引列表
            
        Returns:
            (是否有效, 错误消息列表)
        """
        errors = []
        tool_pairs = self.detect_tool_calls(messages)
        
        for pair in tool_pairs:
            request_removed = pair.request_index in indices_to_remove
            response_removed = pair.response_index in indices_to_remove
            
            if request_removed and not response_removed:
                errors.append(
                    f"工具调用 '{pair.tool_name}' 的请求被删除，但响应保留，会导致上下文不完整"
                )
            elif response_removed and not request_removed:
                errors.append(
                    f"工具调用 '{pair.tool_name}' 的响应被删除，但请求保留，会导致上下文不完整"
                )
            elif request_removed and response_removed:
                errors.append(
                    f"工具调用对 '{pair.tool_name}' 被完全删除，可能丢失重要信息"
                )
        
        return len(errors) == 0, errors
    
    def get_protected_indices(self, messages: List[Dict[str, Any]]) -> List[int]:
        """获取应该被保护的索引列表（工具调用对所在的索引）"""
        protected = []
        tool_pairs = self.detect_tool_calls(messages)
        
        for pair in tool_pairs:
            protected.append(pair.request_index)
            protected.append(pair.response_index)
        
        return sorted(set(protected))


class ReferencePreserver:
    """对话引用保护器
    
    功能：
    1. 检测消息中的引用词（代词、指示词等）
    2. 保留引用指向的上下文信息
    3. 确保压缩后引用仍然可解析
    
    引用词类型：
    - 代词：这、那、它、他们、这个、那个
    - 指示词：上述、前面、以下、前文
    - 时间引用：刚才、之前、之后、现在
    - 上下文引用：如前所述、综上所述、如上所示
    """
    
    def __init__(self):
        self.reference_patterns = {
            "pronouns": [
                r"这", r"那", r"它", r"它们", r"这个", r"那个",
                r"其", r"其余", r"其他", r"此人", r"此事"
            ],
            "demonstratives": [
                r"上述", r"前面", r"下文", r"以下", r"前文",
                r"上方", r"下方", r"左侧", r"右侧"
            ],
            "temporal": [
                r"刚才", r"之前", r"之后", r"现在", r"刚才",
                r"刚才", r"稍早", r"稍后", r"此前", r"此后"
            ],
            "contextual": [
                r"如前所述", r"综上所述", r"如上所示", r"前文提到",
                r"如前所述", r"如上所述", r"据此", r"因此"
            ]
        }
    
    def detect_references(self, messages: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """检测每条消息中的引用词
        
        Returns:
            {消息索引: [检测到的引用词列表]}
        """
        references = {}
        
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            
            detected = []
            for ref_type, patterns in self.reference_patterns.items():
                for pattern in patterns:
                    import re
                    if re.search(pattern, content):
                        detected.append(f"[{ref_type}] {pattern}")
            
            if detected:
                references[i] = detected
        
        return references
    
    def find_reference_targets(
        self,
        messages: List[Dict[str, Any]],
        reference_msg_index: int
    ) -> List[int]:
        """查找引用词可能指向的目标消息索引
        
        Args:
            messages: 消息列表
            reference_msg_index: 包含引用词的消息索引
            
        Returns:
            可能的目标消息索引列表（按相关性排序）
        """
        if reference_msg_index == 0:
            return []
        
        targets = []
        ref_msg = messages[reference_msg_index]
        ref_content = ref_msg.get("content", "")
        
        if isinstance(ref_content, list):
            ref_content = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in ref_content
            )
        
        for i in range(reference_msg_index):
            target_msg = messages[i]
            target_content = target_msg.get("content", "")
            
            if isinstance(target_content, list):
                target_content = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in target_content
                )
            
            score = self._calculate_reference_score(ref_content, target_content)
            if score > 0:
                targets.append((i, score))
        
        targets.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in targets[:5]]
    
    def _calculate_reference_score(self, ref_content: str, target_content: str) -> float:
        """计算引用关联度分数"""
        import re
        score = 0.0
        
        # 提取关键实体
        ref_entities = self._extract_entities(ref_content)
        target_entities = self._extract_entities(target_content)
        
        # 计算重叠
        overlap = len(ref_entities & target_entities)
        score += overlap * 0.3
        
        # 检查关键词重叠
        ref_keywords = set(re.findall(r"\b\w+\b", ref_content.lower()))
        target_keywords = set(re.findall(r"\b\w+\b", target_content.lower()))
        
        keywords_overlap = len(ref_keywords & target_keywords)
        if len(ref_keywords) > 0:
            score += (keywords_overlap / len(ref_keywords)) * 0.5
        
        # 检查内容长度相似度（引用通常指向相似长度的内容）
        len_ratio = min(len(target_content), len(ref_content)) / max(len(target_content), len(ref_content))
        score += len_ratio * 0.2
        
        return score
    
    def _extract_entities(self, text: str) -> set:
        """从文本中提取关键实体"""
        import re
        entities = set()
        
        # 提取被引用的词（如「...」中的内容）
        quoted = re.findall(r'[「"\'"]([^"\'"」]+)["\'"」]', text)
        entities.update(quoted)
        
        # 提取技术术语（连续的字母数字下划线）
        terms = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text)
        entities.update([t for t in terms if len(t) > 2])
        
        return entities
    
    def get_protected_indices(
        self,
        messages: List[Dict[str, Any]],
        compression_indices: List[int]
    ) -> List[int]:
        """获取应该被保护的索引以保留引用关系
        
        Args:
            messages: 消息列表
            compression_indices: 计划压缩的消息索引
            
        Returns:
            应该额外保护的索引列表
        """
        protected = set(compression_indices)
        references = self.detect_references(messages)
        
        for ref_msg_index in references.keys():
            if ref_msg_index in protected:
                continue
            
            targets = self.find_reference_targets(messages, ref_msg_index)
            
            for target in targets:
                if target in compression_indices:
                    protected.add(ref_msg_index)
                    protected.add(target)
                    break
        
        return sorted(protected)


class AdvancedCompressionProtector:
    """高级压缩保护器
    
    整合工具调用对保护和对话引用保护，
    为压缩过程提供全面的上下文保护。
    
    功能：
    1. 检测和保护工具调用对
    2. 检测和保护引用关系
    3. 提供安全的压缩建议
    4. 验证压缩后的完整性
    """
    
    def __init__(self):
        self.tool_validator = ToolCallPairValidator()
        self.reference_preserver = ReferencePreserver()
    
    def analyze_compression_safety(
        self,
        messages: List[Dict[str, Any]],
        indices_to_remove: List[int]
    ) -> Dict[str, Any]:
        """分析压缩计划的安全性
        
        Args:
            messages: 原始消息列表
            indices_to_remove: 计划删除的消息索引
            
        Returns:
            安全分析结果
        """
        analysis = {
            "is_safe": True,
            "tool_pairs": {
                "detected": [],
                "broken_pairs": [],
                "fully_removed": []
            },
            "references": {
                "detected": {},
                "at_risk": []
            },
            "suggested_protected": [],
            "warnings": []
        }
        
        tool_pairs = self.tool_validator.detect_tool_calls(messages)
        analysis["tool_pairs"]["detected"] = [
            {"request": p.request_index, "response": p.response_index, "tool": p.tool_name}
            for p in tool_pairs
        ]
        
        for pair in tool_pairs:
            request_removed = pair.request_index in indices_to_remove
            response_removed = pair.response_index in indices_to_remove
            
            if request_removed and not response_removed:
                analysis["tool_pairs"]["broken_pairs"].append({
                    "tool": pair.tool_name,
                    "issue": "request_removed",
                    "request_index": pair.request_index
                })
                analysis["is_safe"] = False
                analysis["warnings"].append(
                    f"警告: 工具 '{pair.tool_name}' 的请求被删除，但响应保留"
                )
            elif response_removed and not request_removed:
                analysis["tool_pairs"]["broken_pairs"].append({
                    "tool": pair.tool_name,
                    "issue": "response_removed",
                    "response_index": pair.response_index
                })
                analysis["is_safe"] = False
                analysis["warnings"].append(
                    f"警告: 工具 '{pair.tool_name}' 的响应被删除，但请求保留"
                )
            elif request_removed and response_removed:
                analysis["tool_pairs"]["fully_removed"].append({
                    "tool": pair.tool_name,
                    "request_index": pair.request_index,
                    "response_index": pair.response_index
                })
                analysis["warnings"].append(
                    f"注意: 工具调用对 '{pair.tool_name}' 被完全删除"
                )
        
        references = self.reference_preserver.detect_references(messages)
        analysis["references"]["detected"] = {
            str(k): v for k, v in references.items()
        }
        
        for ref_idx in references:
            targets = self.reference_preserver.find_reference_targets(messages, ref_idx)
            at_risk = [t for t in targets if t in indices_to_remove]
            if at_risk:
                analysis["references"]["at_risk"].append({
                    "reference_message_index": ref_idx,
                    "target_indices_at_risk": at_risk
                })
                analysis["warnings"].append(
                    f"警告: 消息 {ref_idx} 中的引用指向可能被删除的消息 {at_risk}"
                )
        
        analysis["suggested_protected"] = list(set(
            indices_to_remove + 
            self.tool_validator.get_protected_indices(messages) +
            self.reference_preserver.get_protected_indices(messages, indices_to_remove)
        ))
        
        return analysis
    
    def get_safe_compression_indices(
        self,
        messages: List[Dict[str, Any]],
        target_indices: List[int]
    ) -> List[int]:
        """获取安全的压缩索引（排除受保护的内容）"""
        protected = set(self.tool_validator.get_protected_indices(messages))
        protected.update(
            self.reference_preserver.get_protected_indices(messages, target_indices)
        )
        
        return [idx for idx in target_indices if idx not in protected]
    
    def validate_compression_result(
        self,
        original_messages: List[Dict[str, Any]],
        compressed_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """验证压缩结果的完整性
        
        Returns:
            验证结果
        """
        result = {
            "is_valid": True,
            "tool_pairs_preserved": 0,
            "tool_pairs_lost": 0,
            "references_valid": True,
            "issues": []
        }
        
        original_pairs = self.tool_validator.detect_tool_calls(original_messages)
        compressed_indices = list(range(len(compressed_messages)))
        
        for pair in original_pairs:
            req_in_compressed = pair.request_index < len(compressed_messages)
            resp_in_compressed = pair.response_index < len(compressed_messages)
            
            if req_in_compressed and resp_in_compressed:
                result["tool_pairs_preserved"] += 1
            else:
                result["tool_pairs_lost"] += 1
                result["issues"].append(
                    f"工具调用对 '{pair.tool_name}' 未完整保留"
                )
                result["is_valid"] = False
        
        references = self.reference_preserver.detect_references(compressed_messages)
        if len(references) > 0:
            for ref_idx in references:
                targets = self.reference_preserver.find_reference_targets(
                    compressed_messages, ref_idx
                )
                if not targets:
                    result["references_valid"] = False
                    result["issues"].append(
                        f"消息 {ref_idx} 中的引用可能无法解析"
                    )
        
        return result


# 使用示例
async def example_usage():
    """使用示例"""
    
    # 1. 初始化组件
    cost_controller = CostController(
        max_tokens_per_request=8000,
        max_tokens_per_day=500000,
        max_cost_per_day=10.0,
        provider="qwen",
        model="qwen-turbo"
    )
    
    context_compressor = ContextCompressor(
        max_tokens=4000,
        compression_ratio=0.5,
        preserve_key_info=True
    )
    
    # 2. 创建智能检索器
    smart_retriever = SmartRetriever(
        vector_db=None,  # 实际使用时传入真实的 vector_db
        llm_service=None,  # 实际使用时传入真实的 llm_service
        cost_controller=cost_controller,
        context_compressor=context_compressor,
        config={"model": "qwen-turbo"}
    )
    
    # 3. 检查预算
    can_proceed, message = await cost_controller.check_budget(5000, "test_query")
    print(f"Budget check: {message}")
    
    # 4. 获取使用统计
    stats = cost_controller.get_usage_stats()
    print(f"Current cost level: {stats['cost_level']}")
    print(f"Daily cost: ${stats['daily_cost']:.4f}")
    
    # 5. 记录使用
    await cost_controller.record_usage(
        operation="test_query",
        model="qwen-turbo",
        usage=TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500
        )
    )
    
    print("Usage stats:", cost_controller.get_usage_stats())


if __name__ == "__main__":
    asyncio.run(example_usage())
