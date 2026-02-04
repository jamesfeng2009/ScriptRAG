# RAG剧本生成系统 - 全面技术审查报告

**审查日期**: 2026-02-04  
**审查范围**: Agent隔离、Function Call、上下文压缩、Token限制处理

---

## 执行摘要

### ✅ 已实现的功能

1. **上下文压缩和Token限制** - 完整实现
2. **LLM服务和适配器架构** - 完整实现
3. **错误处理和重试机制** - 完整实现
4. **日志和监控系统** - 完整实现

### ⚠️ 部分实现的功能

1. **Agent上下文隔离** - 理论设计完成，实际未强制执行
2. **Function Call支持** - 仅在幻觉检测中有简单实现

### ❌ 缺失的功能

1. **强制的Agent上下文隔离机制**
2. **完整的Function Calling支持**
3. **动态上下文压缩触发机制**

---

## 1. Agent上下文隔离分析

### 1.1 当前状态

**架构模式**: SharedState全局共享模式

```python
# 所有Agent共享同一个状态对象
class SharedState(BaseModel):
    user_topic: str
    outline: List[OutlineStep]
    retrieved_docs: List[RetrievedDocument]
    fragments: List[ScreenplayFragment]
    current_skill: str
    execution_log: List[Dict[str, Any]]
```

**数据流**:
```
Planner → Navigator → Director → Writer → FactChecker → Compiler
    ↓         ↓          ↓          ↓          ↓            ↓
  SharedState (全局共享，所有Agent可读写)
```

### 1.2 隔离设计文档

✅ **已完成**: `docs/CONTEXT_ISOLATION_REVIEW.md`
- 详细的架构分析
- 混合隔离方案设计
- 数据所有权定义
- 实施路线图

### 1.3 实际实现情况

❌ **未实现强制隔离**:
- 没有访问控制装饰器
- 没有只读视图机制
- 没有Agent工作空间隔离
- 所有Agent可以自由修改SharedState的任何字段

⚠️ **存在的问题**:
```python
# 任何Agent都可以这样做，没有限制
state.current_skill = "new_skill"  # 没有权限检查
state.outline.append(new_step)     # 没有所有权验证
state.fragments = []               # 可以清空其他Agent的数据
```

### 1.4 推荐的改进方案

**方案A: 数据访问装饰器** (推荐立即实施)

```python
from functools import wraps
from typing import Set

class DataAccessControl:
    """数据访问控制装饰器"""
    
    @staticmethod
    def agent_access(
        agent_name: str,
        reads: Set[str],
        writes: Set[str]
    ):
        def decorator(func):
            @wraps(func)
            async def wrapper(state: SharedState, *args, **kwargs):
                # 记录访问
                state.add_log_entry(
                    agent_name=agent_name,
                    action="data_access",
                    details={
                        "reads": list(reads),
                        "writes": list(writes)
                    }
                )
                
                # 执行前快照
                before_snapshot = state.model_dump()
                
                # 执行Agent逻辑
                result = await func(state, *args, **kwargs)
                
                # 验证只修改了声明的字段
                after_snapshot = state.model_dump()
                modified_fields = {
                    k for k in after_snapshot 
                    if after_snapshot[k] != before_snapshot[k]
                }
                
                unauthorized = modified_fields - writes - {"execution_log", "updated_at"}
                if unauthorized:
                    logger.warning(
                        f"Agent {agent_name} modified unauthorized fields: {unauthorized}"
                    )
                
                return result
            return wrapper
        return decorator

# 使用示例
@DataAccessControl.agent_access(
    agent_name="planner",
    reads={"user_topic", "project_context"},
    writes={"outline"}
)
async def plan_outline(state: SharedState, llm_service: LLMService):
    # 只能读取 user_topic, project_context
    # 只能写入 outline
    pass
```

**方案B: 只读视图** (可选)

```python
class ReadOnlyStateView:
    """只读状态视图"""
    
    def __init__(self, state: SharedState, allowed_fields: frozenset):
        self._state = state
        self._allowed_fields = allowed_fields
    
    def __getattr__(self, name: str):
        if name not in self._allowed_fields:
            raise AttributeError(f"Field {name} not accessible")
        return getattr(self._state, name)
    
    def __setattr__(self, name: str, value: Any):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            raise AttributeError("Cannot modify read-only view")
```

---

## 2. Function Call支持分析

### 2.1 当前状态

❌ **基本未实现**

仅在幻觉检测中有简单的函数调用检测:

```python
# src/domain/agents/fact_checker.py
function_calls = re.findall(r'`(\w+)\(\)`', fragment.content)
for func_call in function_calls:
    if not re.search(r'\b' + re.escape(func_call) + r'\b', all_source_content):
        # 检测到幻觉的函数调用
```

### 2.2 缺失的功能

1. **LLM Function Calling API支持**
   - OpenAI的tools/functions参数
   - 函数定义和参数schema
   - 函数调用结果处理

2. **Agent工具系统**
   - 工具注册机制
   - 工具执行框架
   - 工具结果验证

3. **多轮对话支持**
   - Function call → Execute → Return result → Continue

### 2.3 推荐实现方案

**阶段1: LLM适配器增强**

```python
# src/services/llm/adapter.py
class LLMAdapter(ABC):
    
    @abstractmethod
    async def chat_completion_with_tools(
        self,
        messages: List[Dict[str, str]],
        model: str,
        tools: List[Dict[str, Any]],  # OpenAI tools format
        tool_choice: str = "auto",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        支持工具调用的聊天补全
        
        Returns:
            {
                "content": str,  # 文本响应
                "tool_calls": [  # 工具调用列表
                    {
                        "id": "call_xxx",
                        "type": "function",
                        "function": {
                            "name": "search_code",
                            "arguments": "{\"query\": \"async function\"}"
                        }
                    }
                ],
                "finish_reason": "tool_calls" | "stop"
            }
        """
        pass
```

**阶段2: 工具系统**

```python
# src/domain/tools/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel

class ToolDefinition(BaseModel):
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema

class Tool(ABC):
    """工具基类"""
    
    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """获取工具定义"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """执行工具"""
        pass

# src/domain/tools/retrieval_tool.py
class RetrievalTool(Tool):
    """检索工具"""
    
    def __init__(self, retrieval_service: RetrievalService, workspace_id: str):
        self.retrieval_service = retrieval_service
        self.workspace_id = workspace_id
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_code",
            description="Search for relevant code snippets in the codebase",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    
    async def execute(self, query: str, top_k: int = 5) -> List[Dict]:
        """执行检索"""
        results = await self.retrieval_service.hybrid_search(
            workspace_id=self.workspace_id,
            query=query,
            top_k=top_k
        )
        return [
            {
                "file_path": r.file_path,
                "content": r.content[:500],  # 截断
                "similarity": r.similarity
            }
            for r in results
        ]
```

**阶段3: Agent工具集成**

```python
# src/domain/agents/writer.py
async def generate_fragment_with_tools(
    state: SharedState,
    llm_service: LLMService,
    tools: List[Tool]
) -> SharedState:
    """支持工具调用的片段生成"""
    
    # 准备工具定义
    tool_definitions = [tool.get_definition().dict() for tool in tools]
    tool_map = {tool.get_definition().name: tool for tool in tools}
    
    messages = [
        {"role": "system", "content": "You are a screenplay writer..."},
        {"role": "user", "content": f"Write about: {current_step.title}"}
    ]
    
    max_iterations = 5
    for i in range(max_iterations):
        # 调用LLM
        response = await llm_service.chat_completion_with_tools(
            messages=messages,
            tools=tool_definitions,
            task_type="lightweight"
        )
        
        # 如果没有工具调用，返回结果
        if response["finish_reason"] == "stop":
            content = response["content"]
            break
        
        # 执行工具调用
        if response["finish_reason"] == "tool_calls":
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                
                # 执行工具
                tool = tool_map[tool_name]
                result = await tool.execute(**tool_args)
                
                # 添加工具结果到消息
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result)
                })
    
    # 创建片段
    fragment = ScreenplayFragment(
        step_id=current_step.step_id,
        content=content,
        skill_used=state.current_skill
    )
    
    state.fragments.append(fragment)
    return state
```

---

## 3. 上下文压缩和Token限制

### 3.1 当前实现状态

✅ **完整实现**: `src/services/rag/cost_control.py`

**核心组件**:

1. **CostController** - 成本控制器
   ```python
   cost_controller = CostController(
       max_tokens_per_request=8000,
       max_tokens_per_day=500000,
       max_cost_per_day=10.0
   )
   ```

2. **TokenBudget** - Token预算管理
   ```python
   budget = TokenBudget(
       max_tokens=12000,
       warning_threshold=0.8,
       critical_threshold=0.95
   )
   ```

3. **ContextCompressor** - 上下文压缩器
   ```python
   compressor = ContextCompressor(
       max_tokens=4000,
       compression_ratio=0.5,
       preserve_key_info=True
   )
   ```

4. **SmartRetriever** - 智能检索器
   - 集成成本控制
   - 自动上下文压缩
   - 使用统计

### 3.2 压缩策略

**策略1: 规则压缩**
```python
def _rule_based_compress(self, documents: List[Any]) -> List[Any]:
    # 1. 去重
    # 2. 移除长注释
    # 3. 移除连续空行
    # 4. 截断过长内容
```

**策略2: LLM智能摘要**
```python
async def _llm_summarize(self, query: str, documents: List[Any]) -> str:
    # 使用LLM生成针对查询的摘要
    # 保留关键技术细节和代码示例
```

### 3.3 Token限制处理

**多层次限制**:

1. **请求级别**
   ```python
   max_tokens_per_request = 8000  # 单次请求限制
   ```

2. **每日级别**
   ```python
   max_tokens_per_day = 500000  # 每日总量限制
   max_cost_per_day = 10.0      # 每日成本限制
   ```

3. **会话级别**
   ```python
   budget = TokenBudget(max_tokens=12000)  # 单个会话限制
   ```

**超限处理**:
```python
can_proceed, message = await cost_controller.check_budget(
    estimated_tokens=5000,
    operation="query"
)

if not can_proceed:
    # 触发压缩或拒绝请求
    logger.warning(f"Budget exceeded: {message}")
```

### 3.4 存在的问题

⚠️ **未集成到主流程**:
- `cost_control.py`是独立模块
- 主要的Agent节点没有使用成本控制
- 没有自动触发压缩的机制

⚠️ **缺少动态调整**:
- 压缩比例是固定的
- 没有根据剩余预算动态调整
- 没有优先级队列

### 3.5 推荐改进

**改进1: 集成到LLM Service**

```python
# src/services/llm/service.py
class LLMService:
    def __init__(
        self,
        config: Dict[str, Any],
        cost_controller: Optional[CostController] = None
    ):
        self.cost_controller = cost_controller or CostController()
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        task_type: str,
        **kwargs
    ) -> str:
        # 估算token数
        estimated_tokens = self._estimate_tokens(messages)
        
        # 检查预算
        can_proceed, message = await self.cost_controller.check_budget(
            estimated_tokens,
            operation=task_type
        )
        
        if not can_proceed:
            # 尝试压缩消息
            messages = await self._compress_messages(messages)
            estimated_tokens = self._estimate_tokens(messages)
            
            # 再次检查
            can_proceed, message = await self.cost_controller.check_budget(
                estimated_tokens,
                operation=task_type
            )
            
            if not can_proceed:
                raise BudgetExceededError(message)
        
        # 调用LLM
        result = await self._call_llm(messages, **kwargs)
        
        # 记录使用
        await self.cost_controller.record_usage(
            operation=task_type,
            model=model,
            usage=TokenUsage(...)
        )
        
        return result
```

**改进2: 动态压缩策略**

```python
class AdaptiveCompressor:
    """自适应压缩器"""
    
    def __init__(self, budget: TokenBudget):
        self.budget = budget
    
    async def compress(
        self,
        content: str,
        priority: int = 5  # 1-10
    ) -> str:
        """根据剩余预算动态压缩"""
        
        remaining_ratio = self.budget.get_remaining() / self.budget.max_tokens
        
        if remaining_ratio > 0.5:
            # 预算充足，不压缩
            return content
        elif remaining_ratio > 0.2:
            # 预算紧张，轻度压缩
            return await self._light_compress(content)
        else:
            # 预算严重不足，重度压缩
            if priority >= 8:
                # 高优先级，保留更多内容
                return await self._medium_compress(content)
            else:
                # 低优先级，激进压缩
                return await self._heavy_compress(content)
```

---

## 4. 其他发现

### 4.1 ✅ 已实现的优秀功能

1. **LLM服务架构**
   - 多提供商支持 (OpenAI, Qwen, MiniMax, GLM)
   - 自动回退机制
   - 指数退避重试
   - 完整的日志记录

2. **错误处理**
   - 统一的错误类型
   - 装饰器模式
   - 错误分类和恢复策略

3. **检索服务**
   - 混合搜索 (向量+BM25)
   - 重排序
   - 查询扩展
   - 缓存机制

4. **监控和日志**
   - LLM调用日志
   - 检索日志
   - 性能指标
   - 审计日志

### 4.2 ⚠️ 需要改进的地方

1. **测试覆盖率**
   - 大量property-based测试
   - 但缺少集成测试
   - 缺少性能测试

2. **文档**
   - 架构文档完善
   - 但API文档不足
   - 缺少部署文档

3. **配置管理**
   - 配置分散在多个文件
   - 缺少配置验证
   - 缺少配置热更新

---

## 5. 优先级建议

### 🔴 高优先级 (立即实施)

1. **Agent访问控制装饰器**
   - 时间: 2-3天
   - 收益: 审计追踪、防止意外修改
   - 风险: 低

2. **成本控制集成到主流程**
   - 时间: 2-3天
   - 收益: 防止预算超支
   - 风险: 低

3. **统一日志格式**
   - 时间: 1-2天
   - 收益: 更好的可观测性
   - 风险: 低

### 🟡 中优先级 (1-2周内)

4. **Function Calling基础支持**
   - 时间: 5-7天
   - 收益: 增强Agent能力
   - 风险: 中

5. **状态快照机制**
   - 时间: 2-3天
   - 收益: 调试和回滚
   - 风险: 低

6. **动态压缩策略**
   - 时间: 3-5天
   - 收益: 更智能的资源管理
   - 风险: 中

### 🟢 低优先级 (可选)

7. **完整的Agent隔离**
   - 时间: 10-15天
   - 收益: 更强的隔离保证
   - 风险: 高 (可能破坏现有架构)

8. **工具系统框架**
   - 时间: 7-10天
   - 收益: 可扩展的工具生态
   - 风险: 中

---

## 6. 实施路线图

### 第1周: 访问控制和成本管理

**目标**: 加强数据访问审计和成本控制

- [ ] 实现DataAccessControl装饰器
- [ ] 为所有Agent节点添加访问声明
- [ ] 集成CostController到LLMService
- [ ] 添加预算检查到主要节点
- [ ] 编写单元测试

### 第2周: 日志和监控增强

**目标**: 统一日志格式，改进可观测性

- [ ] 定义统一的LogAction枚举
- [ ] 重构所有日志记录点
- [ ] 添加状态快照机制
- [ ] 实现快照比较功能
- [ ] 更新文档

### 第3-4周: Function Calling支持

**目标**: 基础的工具调用能力

- [ ] 扩展LLM适配器支持tools参数
- [ ] 实现Tool基类和ToolDefinition
- [ ] 创建RetrievalTool
- [ ] 集成到Writer Agent
- [ ] 编写集成测试

### 第5-6周: 动态优化

**目标**: 智能的资源管理

- [ ] 实现AdaptiveCompressor
- [ ] 添加优先级队列
- [ ] 实现动态压缩触发
- [ ] 性能测试和优化
- [ ] 文档更新

---

## 7. 风险评估

| 风险项 | 严重程度 | 概率 | 缓解措施 |
|--------|---------|------|---------|
| 装饰器性能开销 | 低 | 30% | 性能测试，可配置开关 |
| 成本控制误判 | 中 | 40% | 保守的阈值，人工审核 |
| Function Call兼容性 | 中 | 50% | 渐进式部署，回退机制 |
| 压缩质量下降 | 中 | 40% | A/B测试，质量监控 |
| 破坏现有功能 | 高 | 20% | 完整的回归测试 |

---

## 8. 总结

### 当前项目状态: 🟡 良好但需改进

**优势**:
- ✅ 核心功能完整
- ✅ 架构设计合理
- ✅ 代码质量高
- ✅ 文档较完善

**不足**:
- ⚠️ Agent隔离未强制执行
- ⚠️ Function Call支持缺失
- ⚠️ 成本控制未集成
- ⚠️ 测试覆盖不足

### 推荐行动

1. **不要进行大规模重构** - 当前架构基本合理
2. **渐进式改进** - 通过装饰器和工具类增强
3. **优先解决高优先级问题** - 访问控制和成本管理
4. **保持向后兼容** - 所有改进都应该是可选的

### 预期成果

实施上述改进后，系统将具备:
- ✅ 完整的数据访问审计
- ✅ 有效的成本控制
- ✅ 基础的工具调用能力
- ✅ 智能的资源管理
- ✅ 更好的可观测性

**总体评估**: 项目基础扎实，通过渐进式改进可以达到生产级别的质量标准。
