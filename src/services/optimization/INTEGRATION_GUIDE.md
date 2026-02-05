# 优化模块集成指南

## 概述

本文档指导如何将 `src/services/optimization/` 目录下的优化模块集成到现有系统中。

## 目录结构

```
src/services/optimization/
├── __init__.py              # 模块导出
├── smart_skip.py           # 智能跳过优化
├── async_io.py            # 异步I/O优化
├── llm_optimizer.py       # LLM优化
├── integration.py         # 集成示例（本文档）
└── usage_example.py       # 使用示例
```

## 快速集成（5分钟）

### 步骤1：更新配置文件

```yaml
# config.yaml

# 性能配置
performance:
  max_concurrent_requests: 10
  max_concurrent_embeddings: 10
  min_pool_size: 5
  max_pool_size: 20
  
# 压缩配置
compression:
  enabled: true
  strategy: hybrid
  max_tokens: 4000
  
# 智能跳过配置
smart_skip:
  enabled: true
  quality_threshold: 0.85
  complexity_threshold: 0.8
  
# 并行检索配置
parallel_retrieval:
  enabled: true
  vector_weight: 0.6
  keyword_weight: 0.4
```

### 步骤2：修改 LLM 服务

**文件**: `src/services/llm/service.py`

```python
"""LLM Service - 集成优化版本"""

import logging
from typing import List, Dict, Any, Optional
from .adapter import LLMAdapter

logger = logging.getLogger(__name__)


class OptimizedLLMService:
    """
    优化后的LLM服务
    
    集成:
    - 上下文压缩
    - 智能模型选择
    - 请求缓存
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logging_db_service: Optional[Any] = None
    ):
        self.config = config
        self.adapters: Dict[str, LLMAdapter] = {}
        
        # 初始化优化组件
        from ..optimization import (
            ContextCompressor,
            SmartModelSelector,
            LLMOptimizer
        )
        
        self.compressor = ContextCompressor(
            max_tokens=4000,
            compression_ratio=0.5,
            preserve_key_info=True
        )
        
        self.model_selector = SmartModelSelector(
            config=config,
            enable_adaptive_selection=True
        )
        
        self.optimizer = LLMOptimizer(
            llm_config=config,
            enable_compression=True,
            enable_smart_model=True
        )
        
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        """初始化适配器"""
        from .glm_adapter import GLMAdapter
        from .qwen_adapter import QwenAdapter
        from .openai_adapter import OpenAICompatibleAdapter
        
        adapter_classes = {
            "glm": GLMAdapter,
            "qwen": QwenAdapter,
            "openai": OpenAICompatibleAdapter
        }
        
        for provider_name, provider_config in self.config.get("providers", {}).items():
            adapter_class = adapter_classes.get(provider_name)
            if adapter_class:
                self.adapters[provider_name] = adapter_class(
                    provider_config,
                    self.config["model_mappings"].get(provider_name, {})
                )
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        task_type: str = "lightweight",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        优化的聊天补全接口
        
        集成智能模型选择和上下文压缩。
        """
        # 智能模型选择
        estimated_tokens = self._estimate_tokens(messages)
        selection = self.model_selector.select_model(
            task_type=task_type,
            messages=messages,
            estimated_tokens=estimated_tokens
        )
        
        logger.info(
            f"Selected model: {selection.selected_model} "
            f"(complexity: {selection.confidence:.2f})"
        )
        
        # 上下文压缩
        if estimated_tokens > 3000:
            last_msg = messages[-1] if messages else {}
            content = last_msg.get('content', '')
            
            compression_result = self.compressor.compress(content, strategy="hybrid")
            
            if compression_result.compression_ratio < 0.8:
                messages[-1]['content'] = compression_result.preserved_content
                logger.info(
                    f"Compressed context: {compression_result.original_tokens} → "
                    f"{compression_result.compressed_tokens} tokens"
                )
        
        # 执行LLM调用
        provider_name = selection.provider
        adapter = self.adapters.get(provider_name)
        
        if not adapter:
            raise ValueError(f"No adapter for provider: {provider_name}")
        
        model = selection.selected_model
        
        return await self._call_with_retry(
            adapter=adapter,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def _call_with_retry(
        self,
        adapter: LLMAdapter,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """带重试的LLM调用"""
        max_retries = kwargs.pop('max_retries', 3)
        
        for attempt in range(max_retries):
            try:
                return await adapter.chat_completion(
                    messages=messages,
                    model=model,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
    
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """估算token数量"""
        return sum(len(msg.get('content', '')) // 4 for msg in messages)
```

### 步骤3：修改 Navigator 节点

**文件**: `src/domain/agents/navigator.py`

```python
"""导航器智能体 - 集成优化版本"""

import asyncio
import logging
from typing import List, Optional

from ..models import SharedState, RetrievedDocument
from ...services.retrieval_service import RetrievalService
from ...services.parser.tree_sitter_parser import IParserService
from ...services.summarization_service import SummarizationService
from ...services.optimization import (
    SmartSkipOptimizer,
    CacheBasedSkipper,
    ParallelExecutor
)
from ...infrastructure.logging import get_agent_logger

logger = logging.getLogger(__name__)
agent_logger = get_agent_logger(__name__)


class OptimizedNavigator:
    """
    优化后的导航器
    
    集成:
    - 并行检索（向量 + 关键词）
    - 智能跳过
    - 缓存优化
    """
    
    def __init__(self):
        self.skip_optimizer = SmartSkipOptimizer()
        self.cache_skipper = CacheBasedSkipper()
        self.parallel_executor = ParallelExecutor(max_concurrency=10)
    
    async def retrieve_content(
        self,
        state: SharedState,
        retrieval_service: RetrievalService,
        parser_service: IParserService,
        summarization_service: SummarizationService,
        workspace_id: str
    ) -> SharedState:
        """
        优化的检索内容
        
        特性:
        1. 检查缓存是否命中
        2. 并行执行向量和关键词搜索
        3. 并行处理检索结果
        """
        try:
            # 1. 智能跳过检查
            if await self._should_skip_retrieval(state):
                logger.info("Skipping redundant retrieval due to cache hit")
                return state
            
            current_step = state.outline[state.current_step_index]
            query = current_step.description
            
            # 2. 并行检索
            retrieval_results = await self._parallel_retrieve(
                state=state,
                retrieval_service=retrieval_service,
                query=query,
                workspace_id=workspace_id
            )
            
            if not retrieval_results:
                state.retrieved_docs = []
                return state
            
            # 3. 并行处理结果
            state.retrieved_docs = await self._parallel_process_results(
                retrieval_results=retrieval_results,
                parser_service=parser_service,
                summarization_service=summarization_service
            )
            
            logger.info(
                f"Navigator: Retrieved {len(state.retrieved_docs)} documents"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Navigator failed: {str(e)}")
            state.retrieved_docs = []
            return state
    
    async def _should_skip_retrieval(self, state: SharedState) -> bool:
        """检查是否应该跳过检索"""
        if not state.retrieved_docs:
            return False
        
        cache_key = f"navigator:{state.current_step.step_id}"
        decision = self.cache_skipper.should_skip_processing(
            cache_key,
            min_hits_for_skip=2
        )
        return decision.should_skip
    
    async def _parallel_retrieve(
        self,
        state: SharedState,
        retrieval_service: RetrievalService,
        query: str,
        workspace_id: str
    ) -> List:
        """并行执行向量和关键词搜索"""
        async def vector_search():
            return await retrieval_service.retrieve_with_strategy(
                workspace_id=workspace_id,
                query=query,
                strategy_name="vector_search",
                top_k=5
            )
        
        async def keyword_search():
            return await retrieval_service.retrieve_with_strategy(
                workspace_id=workspace_id,
                query=query,
                strategy_name="keyword_search",
                top_k=5
            )
        
        # 并行执行
        results = await self.parallel_executor.execute_parallel({
            'vector': vector_search,
            'keyword': keyword_search
        })
        
        # 合并结果
        vector_results = results.get('vector', [])
        keyword_results = results.get('keyword', [])
        
        return self._merge_results(vector_results, keyword_results)
    
    def _merge_results(self, vector_results: List, keyword_results: List) -> List:
        """合并检索结果"""
        if not vector_results:
            return keyword_results
        if not keyword_results:
            return vector_results
        
        # 加权合并
        scored = []
        for r in vector_results:
            r_copy = r
            r_copy._score = getattr(r, 'confidence', 0.5) * 0.6
            r_copy._source = 'vector'
            scored.append(r_copy)
        
        for r in keyword_results:
            r_copy = r
            r_copy._score = getattr(r, 'confidence', 0.5) * 0.4
            r_copy._source = 'keyword'
            scored.append(r_copy)
        
        # 排序去重
        scored.sort(key=lambda x: getattr(x, '_score', 0), reverse=True)
        
        seen = set()
        unique = []
        for r in scored:
            source = getattr(r, 'source', str(r.file_path))
            if source not in seen:
                unique.append(r)
                seen.add(source)
        
        return unique[:10]
    
    async def _parallel_process_results(
        self,
        retrieval_results: List,
        parser_service: IParserService,
        summarization_service: SummarizationService
    ) -> List[RetrievedDocument]:
        """并行处理检索结果"""
        async def process_single(result):
            try:
                parsed = parser_service.parse(
                    file_path=result.file_path,
                    content=result.content
                )
                
                content = result.content
                summary = None
                
                if summarization_service.check_size(result.content):
                    summarized = await summarization_service.summarize_document(
                        content=result.content,
                        file_path=result.file_path,
                        parsed_code=parsed
                    )
                    content = summarized.summary
                    summary = summarized.summary
                
                return RetrievedDocument(
                    content=content,
                    source=result.file_path,
                    confidence=getattr(result, 'confidence', 0.5),
                    metadata={
                        'similarity': getattr(result, 'similarity', 0.0),
                        'search_source': getattr(result, 'strategy_name', 'unknown'),
                        'language': parsed.language,
                        'was_summarized': summary is not None,
                        'retrieval_source': getattr(result, '_source', 'hybrid')
                    },
                    summary=summary
                )
            except Exception as e:
                logger.error(f"Failed to process result: {e}")
                return RetrievedDocument(
                    content=result.content,
                    source=result.file_path,
                    confidence=0.0,
                    metadata={'error': str(e)}
                )
        
        tasks = [process_single(r) for r in retrieval_results]
        results = await asyncio.gather(*tasks)
        
        return [r for r in results if isinstance(r, RetrievedDocument)]
```

### 步骤4：修改 Director 节点

**文件**: `src/domain/agents/director.py`

```python
"""导演智能体 - 集成优化版本"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from ..models import SharedState, RetrievedDocument, OutlineStep
from ...services.llm.service import LLMService
from ...services.optimization import SmartSkipOptimizer
from ...infrastructure.logging import get_agent_logger

logger = logging.getLogger(__name__)
agent_logger = get_agent_logger(__name__)


class OptimizedDirector:
    """
    优化后的导演智能体
    
    集成:
    - 智能跳过低复杂度评估
    - 质量评估优化
    """
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.skip_optimizer = SmartSkipOptimizer()
    
    async def evaluate_and_decide(
        self,
        state: SharedState
    ) -> SharedState:
        """
        优化的导演决策
        
        特性:
        1. 高质量内容跳过详细评估
        2. 智能复杂度处理
        """
        try:
            # 1. 智能跳过检查
            if self._should_skip_evaluation(state):
                logger.info("Skipping detailed evaluation due to high quality")
                state.add_log_entry(
                    agent_name="director",
                    action="skipped_quality",
                    details={"reason": "high_quality_content"}
                )
                return state
            
            current_step = state.get_current_step()
            
            # 2. 复杂度评估
            complexity = await self._assess_complexity(
                current_step=current_step,
                retrieved_docs=state.retrieved_docs
            )
            
            # 3. 基于复杂度的处理
            processing_mode = self.skip_optimizer.complexity_skipper.get_processing_mode(
                complexity
            )
            
            logger.info(
                f"Director: complexity={complexity:.2f}, "
                f"mode={processing_mode}"
            )
            
            if processing_mode == 'minimal':
                return await self._quick_evaluation(state, current_step)
            elif processing_mode == 'reduced':
                return await self._standard_evaluation(state, current_step)
            else:
                return await self._detailed_evaluation(state, current_step)
            
        except Exception as e:
            logger.error(f"Director evaluation failed: {str(e)}")
            return state
    
    def _should_skip_evaluation(self, state: SharedState) -> bool:
        """检查是否应该跳过评估"""
        if not state.retrieved_docs:
            return False
        
        content = str(state.retrieved_docs[0].content)[:500]
        quality = self.skip_optimizer.quality_assessor.assess_quality(content)
        
        return quality >= 0.9
    
    async def _assess_complexity(
        self,
        current_step: OutlineStep,
        retrieved_docs: List[RetrievedDocument]
    ) -> float:
        """评估复杂度"""
        if not retrieved_docs:
            return 0.5
        
        docs_summary = "\n".join([
            doc.content[:200] for doc in retrieved_docs[:3]
        ])
        
        complexity_indicators = [
            len(current_step.description) / 500,
            len(docs_summary) / 2000,
            sum(1 for d in retrieved_docs if d.metadata.get('has_deprecated', False)) / 3
        ]
        
        return min(1.0, sum(complexity_indicators) / len(complexity_indicators))
    
    async def _quick_evaluation(
        self,
        state: SharedState,
        current_step: OutlineStep
    ) -> SharedState:
        """快速评估（高质量内容）"""
        state.pivot_triggered = False
        state.add_log_entry(
            agent_name="director",
            action="quick_approved",
            details={"step_id": current_step.step_id}
        )
        return state
    
    async def _standard_evaluation(
        self,
        state: SharedState,
        current_step: OutlineStep
    ) -> SharedState:
        """标准评估"""
        has_conflict, conflict_type, _ = self._detect_conflicts(
            current_step,
            state.retrieved_docs
        )
        
        if has_conflict:
            state.pivot_triggered = True
            state.pivot_reason = conflict_type
        else:
            state.pivot_triggered = False
        
        return state
    
    async def _detailed_evaluation(
        self,
        state: SharedState,
        current_step: OutlineStep
    ) -> SharedState:
        """详细评估（低质量内容）"""
        has_conflict, conflict_type, details = self._detect_conflicts(
            current_step,
            state.retrieved_docs
        )
        
        complexity = await self._assess_complexity(
            current_step,
            state.retrieved_docs
        )
        
        if has_conflict:
            state.pivot_triggered = True
            state.pivot_reason = conflict_type
        elif complexity > 0.7:
            state.pivot_triggered = True
            state.pivot_reason = "high_complexity"
        else:
            state.pivot_triggered = False
        
        return state
    
    def _detect_conflicts(
        self,
        current_step: OutlineStep,
        retrieved_docs: List[RetrievedDocument]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """检测冲突"""
        deprecated = [d for d in retrieved_docs if d.metadata.get('has_deprecated', False)]
        
        if deprecated:
            return True, "deprecation_conflict", f"Found {len(deprecated)} deprecated docs"
        
        return False, None, None
```

### 步骤5：更新应用入口

**文件**: `src/application/orchestrator.py`

```python
"""Orchestrator - 集成优化版本"""

import logging
from typing import Dict, Any

from ..llm.service import LLMService
from ..retrieval_service import RetrievalService
from ..services.optimization import Optimization集成管理器

logger = logging.getLogger(__name__)


class OptimizedOrchestrator:
    """
    优化后的编排器
    
    集成所有优化组件，提供统一的服务访问接口。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化优化管理器
        self.optimizer_manager = Optimization集成管理器(
            config=config,
            redis_config=config.get('database', {}).get('redis')
        )
    
    async def initialize(self):
        """初始化所有服务"""
        await self.optimizer_manager.initialize()
        
        # 初始化LLM服务
        self.llm_service = self.optimizer_manager.get_optimized_llm_service()
        self.llm_service.set_original_service(
            LLMService(config=self.config)
        )
        
        # 初始化检索服务
        self.retrieval_service = self.optimizer_manager.get_optimized_retrieval_service()
        
        logger.info("Optimized orchestrator initialized")
    
    async def shutdown(self):
        """关闭所有服务"""
        await self.optimizer_manager.shutdown()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.optimizer_manager.get_integration_stats()
```

## 独立组件使用

### 1. 仅使用智能跳过

```python
from src.services.optimization import SmartSkipOptimizer

optimizer = SmartSkipOptimizer()

# 评估是否应该跳过
decisions = optimizer.evaluate_skip_decision(
    content="你的内容",
    complexity_score=0.5,
    cache_key="unique_key"
)

overall = optimizer.get_overall_skip_decision(decisions)
print(f"Should skip: {overall.should_skip}")
```

### 2. 仅使用并行检索

```python
from src.services.optimization import AsyncIOOptimizer

optimizer = AsyncIOOptimizer(max_concurrency=10)

# 定义检索函数
async def vector_search(query):
    return [{"type": "vector", "query": query}]

async def keyword_search(query):
    return [{"type": "keyword", "query": query}]

# 并行执行
results = await optimizer.parallel_retrieve(
    vector_search_func=vector_search,
    keyword_search_func=keyword_search,
    query="测试查询",
    workspace_id="default",
    top_k=10
)
```

### 3. 仅使用LLM优化

```python
from src.services.optimization import LLMOptimizer

optimizer = LLMOptimizer(
    llm_config=config,
    enable_compression=True,
    enable_smart_model=True
)

# 优化的LLM调用
response, metadata = await optimizer.optimized_chat_completion(
    messages=[{"role": "user", "content": "你好"}],
    task_type="lightweight",
    llm_call_func=your_original_llm_function
)
```

## 监控和调优

### 获取性能统计

```python
stats = optimizer_manager.get_integration_stats()

print(f"初始化状态: {stats['initialized']}")
print(f"Redis启用: {stats['redis_enabled']}")
print(f"并行统计: {stats['async_optimizer_stats']}")
print(f"LLM优化统计: {stats['llm_optimizer_stats']}")
```

### 调优建议

1. **并行度调优**
   ```python
   # 高并发场景
   optimizer = AsyncIOOptimizer(max_concurrency=20)
   
   # 低并发场景
   optimizer = AsyncIOOptimizer(max_concurrency=5)
   ```

2. **压缩率调优**
   ```python
   # 激进压缩（适合长上下文）
   compressor = ContextCompressor(
       max_tokens=2000,
       compression_ratio=0.3
   )
   
   # 保守压缩（适合短上下文）
   compressor = ContextCompressor(
       max_tokens=6000,
       compression_ratio=0.7
   )
   ```

3. **跳过阈值调优**
   ```python
   # 严格模式（更多跳过）
   optimizer = SmartSkipOptimizer(
       quality_threshold=0.9,   # 提高质量阈值
       complexity_threshold=0.9   # 提高复杂度阈值
   )
   
   # 宽松模式（更少跳过）
   optimizer = SmartSkipOptimizer(
       quality_threshold=0.7,   # 降低质量阈值
       complexity_threshold=0.6   # 降低复杂度阈值
   )
   ```

## 常见问题

### Q1: 优化不生效怎么办？

1. 检查配置是否正确加载
2. 查看日志中的优化决策
3. 验证组件是否正确初始化

```python
# 调试：打印所有优化决策
optimizer = SmartSkipOptimizer()
decisions = optimizer.evaluate_skip_decision(...)
for key, decision in decisions.items():
    print(f"{key}: {decision.__dict__}")
```

### Q2: 如何禁用特定优化？

```python
# 禁用智能跳过
optimizer = SmartSkipOptimizer(
    enable_quality_skip=False,
    enable_complexity_skip=False,
    enable_cache_skip=False
)

# 禁用LLM压缩
llm_optimizer = LLMOptimizer(
    enable_compression=False,
    enable_smart_model=True
)

# 禁用并行检索（使用串行）
navigator = OptimizedNavigator()
# 不调用_parallel_retrieve，改用原有逻辑
```

### Q3: 性能下降怎么办？

1. **检查是否过度压缩**
   ```python
   compressor = ContextCompressor(
       max_tokens=6000,  # 提高限制
       compression_ratio=0.7  # 降低压缩率
   )
   ```

2. **调整并行度**
   ```python
   # 降低并行度
   executor = ParallelExecutor(max_concurrency=5)
   ```

3. **增加缓存TTL**
   ```python
   quality_assessor = QualityAssessor(
       cache_ttl=600  # 延长缓存时间
   )
   ```

## 总结

集成优化模块的步骤：

1. ✅ 更新配置文件
2. ✅ 修改 LLM 服务（集成压缩和模型选择）
3. ✅ 修改 Navigator 节点（集成并行检索）
4. ✅ 修改 Director 节点（集成智能跳过）
5. ✅ 更新 Orchestrator（统一管理）
6. ✅ 监控和调优

预计性能提升：**3-5倍**
