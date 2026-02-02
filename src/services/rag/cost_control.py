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


class ContextCompressor:
    """
    上下文压缩器
    
    功能：
    - 智能摘要检索结果
    - 移除冗余信息
    - 保留关键信息
    - 压缩长文本
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        compression_ratio: float = 0.5,
        preserve_key_info: bool = True,
        llm_service: Any = None
    ):
        """
        初始化上下文压缩器
        
        Args:
            max_tokens: 压缩后最大 token 数
            compression_ratio: 压缩比例 (0-1)
            preserve_key_info: 是否保留关键信息
            llm_service: LLM 服务（用于智能摘要）
        """
        self.max_tokens = max_tokens
        self.compression_ratio = compression_ratio
        self.preserve_key_info = preserve_key_info
        self.llm_service = llm_service
    
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
