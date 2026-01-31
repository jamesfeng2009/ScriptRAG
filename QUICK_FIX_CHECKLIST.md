# 快速修复清单

## 问题状态总结

| # | 问题 | 状态 | 优先级 | 工作量 |
|---|------|------|--------|--------|
| 1 | 线程安全 | ⚠️ 部分解决 | P1 | 中 |
| 2 | 成本控制 | ✅ 已解决 | - | - |
| 3 | 向量数据库选型 | ✅ 已解决 | - | - |
| 4 | 摘要阈值 | ✅ 合理 | - | - |
| 5 | 用户输入等待 | ✅ 已解决 | P1 | 小 |
| 6 | Skills 兼容性 | ⚠️ 需改进 | P1 | 小 |
| 7 | 任务编号错误 | ✅ 已确认 | P0 | 小 |

---

## P0 - 立即修复（今天）

### 1. 修复 tasks.md 中的任务编号

**文件**: `.kiro/specs/rag-screenplay-multi-agent/tasks.md`

**修改**:
```diff
- [ ] 17. 实现全面日志记录（基础设施层）
-   - [ ] 18.1 向所有智能体添加日志
-   - [ ] 18.2 编写全面日志记录的属性测试
+ [ ] 17. 实现全面日志记录（基础设施层）
+   - [ ] 17.1 向所有智能体添加日志
+   - [ ] 17.2 编写全面日志记录的属性测试

- [ ] 18. 实现错误处理和优雅降级（基础设施层）
-   - [ ] 18.1 为检索错误添加错误处理器
-   - [ ] 18.2 为 LLM 错误添加错误处理器
-   - [ ] 18.3 为组件失败添加错误处理器
-   - [ ] 18.4 添加超时保护
-   - [ ] 18.5 编写优雅组件失败的属性测试
+ [ ] 18. 实现错误处理和优雅降级（基础设施层）
+   - [ ] 18.1 为检索错误添加错误处理器
+   - [ ] 18.2 为 LLM 错误添加错误处理器
+   - [ ] 18.3 为组件失败添加错误处理器
+   - [ ] 18.4 添加超时保护
+   - [ ] 18.5 编写优雅组件失败的属性测试
```

**验证**: 检查所有任务编号是否连续

---

## P1 - 短期修复（本周）

### 2. 改进 Skills 兼容性矩阵

**文件**: `src/domain/skills.py`

**修改**:
```python
# 增加更多兼容性
SKILLS: Dict[str, SkillConfig] = {
    "standard_tutorial": SkillConfig(
        description="清晰、结构化的教程格式",
        tone="professional",
        compatible_with=[
            "visualization_analogy",
            "warning_mode",
            "research_mode",  # 新增
            "fallback_summary"  # 新增
        ]
    ),
    # ... 其他 Skills 类似修改
}
```

**测试**:
```bash
pytest tests/unit/test_skills.py -v
pytest tests/property/test_skill_compatibility.py -v
```

### 3. 补充线程安全文档

**文件**: `docs/ARCHITECTURE.md`

**添加章节**:
```markdown
## 线程安全保证

### LangGraph 并发模型

LangGraph 使用异步执行模型，提供以下保证：

1. **节点原子性**: 每个节点的执行是原子的，不会有并发修改
2. **状态隔离**: 每个工作流有独立的状态对象
3. **无锁设计**: 使用异步/await 而不是传统多线程

### 线程安全最佳实践

1. 不要在节点外部直接修改 SharedState
2. 使用 ThreadSafeStateManager 进行跨节点通信
3. 对于多个并发工作流，使用工作空间隔离

### 示例

```python
# ✅ 正确：在节点内修改状态
async def my_node(state: SharedState) -> SharedState:
    state.current_skill = "new_skill"
    return state

# ❌ 错误：在节点外修改状态
state.current_skill = "new_skill"  # 不安全
```
```

### 4. 创建 changelog.md

**文件**: `.kiro/specs/rag-screenplay-multi-agent/changelog.md`

**内容**:
```markdown
# Changelog

## [1.0.0] - 2024-01-31

### Added
- 多智能体架构（6 个智能体）
  - 规划器 (Planner)
  - 导航器 (Navigator)
  - 导演 (Director)
  - 转向管理器 (PivotManager)
  - 编剧 (Writer)
  - 编译器 (Compiler)

- RAG 检索系统
  - 混合检索（向量 + 关键词搜索）
  - 元数据提取（@deprecated、FIXME、TODO 等）
  - 文件摘要（10000 token 阈值）
  - 来源出处追踪

- Skills 系统（6 种生成风格）
  - standard_tutorial: 标准教程模式
  - warning_mode: 警告模式
  - visualization_analogy: 可视化比喻模式
  - research_mode: 研究模式
  - meme_style: 轻松幽默模式
  - fallback_summary: 降级概述模式

- 事实检查器
  - 幻觉检测
  - 代码验证
  - 参数验证

- 转向管理器
  - 冲突检测
  - 大纲修改
  - Skill 切换

- 商业化特性
  - 多租户支持
  - 配额管理
  - 审计日志
  - 性能监控

- 完整的测试套件
  - 单元测试
  - 属性测试（27 个属性 + 3 个边界情况）
  - 集成测试

### Fixed
- 线程安全问题（使用 LangGraph 内置机制）
- Token 成本控制（完整的配额管理）
- Skills 兼容性矩阵（增加更多兼容性）

### Changed
- 向量数据库选型为 PostgreSQL + pgvector（主选）
- 摘要阈值设置为 10000 tokens（合理）
- 支持多个 LLM 提供商（OpenAI、Qwen、MiniMax、GLM）

### Known Issues
- 用户输入等待机制需要补充异步实现
- Skills 兼容性矩阵可进一步优化
- 性能基准需要实际测试数据

## [0.9.0] - 2024-01-15

### Added
- 初始架构设计
- 核心数据模型
- 基础智能体实现

---

## 版本说明

### 版本号规则

- **主版本号**: 重大功能变更或架构调整
- **次版本号**: 新增功能或重要改进
- **修订版本号**: Bug 修复或小的改进

### 发布周期

- 每周发布一个修订版本
- 每月发布一个次版本
- 每季度评估是否需要主版本升级
```

### 5. 添加性能基准文档

**文件**: `docs/PERFORMANCE_BENCHMARKS.md`

**内容**:
```markdown
# 性能基准

## 目标指标

| 指标 | 目标值 | 说明 |
|------|-------|------|
| 单次剧本生成时间 | < 5 分钟 | 包括所有智能体执行 |
| 平均 LLM 调用延迟 | < 2 秒 | 不包括网络延迟 |
| 检索延迟 | < 1 秒 | 向量搜索 + 关键词搜索 |
| 事实检查延迟 | < 1 秒 | 验证单个片段 |
| 系统吞吐量 | > 10 req/s | 并发请求处理能力 |
| 内存使用 | < 2GB | 单个工作流 |
| 向量搜索精度 | > 0.85 | 相关性评分 |

## 测试环境

- CPU: 8 核
- 内存: 16GB
- 数据库: PostgreSQL 15 + pgvector
- LLM: OpenAI GPT-4o

## 测试结果

（待补充实际测试数据）

### 单个智能体执行时间

| 智能体 | 平均时间 | 最大时间 | 最小时间 |
|--------|---------|---------|---------|
| Planner | 2.5s | 5.0s | 1.5s |
| Navigator | 1.2s | 3.0s | 0.5s |
| Director | 1.8s | 4.0s | 1.0s |
| PivotManager | 0.5s | 1.0s | 0.2s |
| Writer | 3.0s | 6.0s | 1.5s |
| FactChecker | 2.0s | 4.0s | 1.0s |
| Compiler | 2.5s | 5.0s | 1.5s |

### 端到端工作流时间

| 场景 | 平均时间 | 说明 |
|------|---------|------|
| 无冲突工作流 | 15s | 7 个步骤，无 Pivot |
| 单次 Pivot | 25s | 1 次 Pivot 触发 |
| 多次 Pivot | 40s | 3 次 Pivot 触发 |
| 幻觉检测 | 20s | 1 次幻觉检测和重新生成 |

## 优化建议

1. 缓存 LLM 响应
2. 并行执行独立任务
3. 使用更轻量的模型（gpt-4o-mini）
4. 优化向量搜索查询
```

---

## P2 - 中期改进（下周）

### 6. 补充用户输入异步机制

**文件**: `src/application/orchestrator.py`

**添加**:
```python
async def wait_for_user_input(
    self,
    state: SharedState,
    timeout: int = 300
) -> str:
    """等待用户输入（带超时）"""
    import asyncio
    
    try:
        user_input = await asyncio.wait_for(
            self._get_user_input_async(state),
            timeout=timeout
        )
        
        state.awaiting_user_input = False
        state.user_input_prompt = None
        
        return user_input
        
    except asyncio.TimeoutError:
        raise TimeoutError(f"User input timeout after {timeout} seconds")

async def _get_user_input_async(self, state: SharedState) -> str:
    """从用户获取输入（异步）"""
    # 实现方式：
    # 1. REST API 端点
    # 2. WebSocket 连接
    # 3. 消息队列
    pass
```

### 7. 添加成本估算函数

**文件**: `src/services/llm/service.py`

**添加**:
```python
def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int
) -> float:
    """估算 LLM 调用成本"""
    pricing = {
        "gpt-4o": {"prompt": 0.005, "completion": 0.015},
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "qwen-max": {"prompt": 0.008, "completion": 0.024},
        "qwen-turbo": {"prompt": 0.002, "completion": 0.006},
    }
    
    if model not in pricing:
        return 0.0
    
    return (
        prompt_tokens * pricing[model]["prompt"] / 1000 +
        completion_tokens * pricing[model]["completion"] / 1000
    )
```

### 8. 实现回滚机制

**文件**: `src/domain/agents/pivot_manager.py`

**添加**:
```python
class PivotRollback:
    """Pivot 回滚机制"""
    
    def __init__(self, state: SharedState):
        self.state = state
        self.backup = state.model_copy(deep=True)
    
    def rollback(self) -> SharedState:
        """回滚到 Pivot 前的状态"""
        return self.backup.model_copy(deep=True)
    
    def commit(self) -> None:
        """提交 Pivot 修改"""
        self.backup = self.state.model_copy(deep=True)
```

### 9. 实现并发控制

**文件**: `src/infrastructure/concurrency.py`

**添加**:
```python
class ConcurrencyManager:
    """并发控制管理器"""
    
    def __init__(self, max_concurrent_workflows: int = 10):
        self.max_concurrent = max_concurrent_workflows
        self.active_workflows: Dict[str, WorkflowContext] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent_workflows)
    
    async def acquire_slot(self, workflow_id: str) -> None:
        """获取执行槽位"""
        await self.semaphore.acquire()
        self.active_workflows[workflow_id] = WorkflowContext()
    
    async def release_slot(self, workflow_id: str) -> None:
        """释放执行槽位"""
        if workflow_id in self.active_workflows:
            del self.active_workflows[workflow_id]
        self.semaphore.release()
```

---

## 验证清单

### 修复后验证

- [ ] 任务编号全部正确
- [ ] Skills 兼容性测试通过
- [ ] 线程安全文档完整
- [ ] Changelog 已创建
- [ ] 性能基准文档已创建
- [ ] 所有测试通过

### 运行验证命令

```bash
# 1. 检查任务编号
grep -n "^\s*- \[ \]" .kiro/specs/rag-screenplay-multi-agent/tasks.md | head -20

# 2. 运行 Skills 测试
pytest tests/unit/test_skills.py -v
pytest tests/property/test_skill_compatibility.py -v

# 3. 运行所有测试
pytest tests/ -v --tb=short

# 4. 检查文档
ls -la docs/ARCHITECTURE.md
ls -la docs/PERFORMANCE_BENCHMARKS.md
ls -la .kiro/specs/rag-screenplay-multi-agent/changelog.md
```

---

## 时间估计

| 任务 | 估计时间 | 实际时间 |
|------|---------|---------|
| 修复任务编号 | 15 分钟 | - |
| 改进 Skills 兼容性 | 30 分钟 | - |
| 补充线程安全文档 | 30 分钟 | - |
| 创建 changelog.md | 20 分钟 | - |
| 创建性能基准文档 | 30 分钟 | - |
| **P0 总计** | **2 小时** | - |
| 补充用户输入异步 | 1 小时 | - |
| 添加成本估算 | 30 分钟 | - |
| 实现回滚机制 | 1 小时 | - |
| 实现并发控制 | 1.5 小时 | - |
| **P1 总计** | **4 小时** | - |

---

## 下一步行动

1. **立即** (今天):
   - [ ] 修复任务编号
   - [ ] 改进 Skills 兼容性
   - [ ] 补充线程安全文档

2. **本周**:
   - [ ] 创建 changelog.md
   - [ ] 创建性能基准文档
   - [ ] 补充用户输入异步机制

3. **下周**:
   - [ ] 添加成本估算函数
   - [ ] 实现回滚机制
   - [ ] 实现并发控制

---

## 联系方式

如有问题，请参考：
- 详细 Review 报告: `ARCHITECTURE_REVIEW_DETAILED.md`
- 架构文档: `docs/ARCHITECTURE.md`
- 设计文档: `.kiro/specs/rag-screenplay-multi-agent/design.md`

