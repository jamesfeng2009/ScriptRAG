# 失败测试修复清单

## 修复进度追踪
- [ ] Hallucination Workflow (0/8)
- [ ] LLM Provider Fallback (0/8)  
- [ ] Retry Limit Workflow (0/9)
- [ ] Pivot Workflow (0/7)
- [ ] LangGraph Workflow (0/3)
- [ ] Recursion Limit (0/1)

**总进度**: 0/35 (0%)

---

## 1. Hallucination Workflow (8个测试)

### 文件: `tests/integration/test_hallucination_workflow.py`

- [ ] **test_hallucination_detected_by_fact_checker**
  - **功能**: 验证事实检查器能检测到幻觉
  - **可能问题**: Mock LLM的幻觉检测响应格式不正确
  - **相关代码**: `src/domain/agents/fact_checker.py`
  - **修复建议**: 检查fact_checker的verify_fragment方法和Mock响应

- [ ] **test_regeneration_triggered_on_hallucination**
  - **功能**: 验证检测到幻觉后触发重新生成
  - **可能问题**: 重新生成流程的触发条件不正确
  - **相关代码**: `src/application/orchestrator.py`
  - **修复建议**: 检查fact_check_and_completion节点的路由逻辑

- [ ] **test_workflow_completes_after_regeneration**
  - **功能**: 验证重新生成后工作流能正常完成
  - **可能问题**: 重新生成后的状态管理问题
  - **相关代码**: `src/application/orchestrator.py`
  - **修复建议**: 验证重新生成后current_step_index的更新

- [ ] **test_fact_checker_validation_logged**
  - **功能**: 验证事实检查验证过程被记录
  - **可能问题**: 日志记录功能缺失
  - **相关代码**: `src/domain/agents/fact_checker.py`, `src/infrastructure/logging.py`
  - **修复建议**: 在fact_checker中添加日志记录

- [ ] **test_retry_count_incremented_on_hallucination**
  - **功能**: 验证检测到幻觉时重试计数递增
  - **可能问题**: 重试计数更新逻辑问题
  - **相关代码**: `src/domain/agents/retry_protection.py`
  - **修复建议**: 检查retry_count的更新时机

- [ ] **test_no_hallucinated_content_in_final_screenplay**
  - **功能**: 验证最终剧本不包含幻觉内容
  - **可能问题**: 幻觉内容未被正确移除
  - **相关代码**: `src/domain/agents/fact_checker.py`, `src/domain/agents/compiler.py`
  - **修复建议**: 验证invalid fragment的移除逻辑

- [ ] **test_fact_checker_compares_with_retrieved_docs**
  - **功能**: 验证事实检查器与检索文档进行比较
  - **可能问题**: 检索文档未正确传递给fact_checker
  - **相关代码**: `src/domain/agents/fact_checker.py`
  - **修复建议**: 检查retrieved_docs在state中的传递

- [ ] **test_multiple_hallucinations_handled**
  - **功能**: 验证能处理多个幻觉
  - **可能问题**: 多次重新生成的循环逻辑问题
  - **相关代码**: `src/application/orchestrator.py`
  - **修复建议**: 验证多次fact_check的循环处理

---

## 2. LLM Provider Fallback (8个测试)

### 文件: `tests/integration/test_llm_provider_fallback.py`

- [ ] **test_fallback_provider_used_on_primary_failure**
  - **功能**: 验证主provider失败时使用备用provider
  - **可能问题**: Fallback机制未实现
  - **相关代码**: `src/services/llm/service.py`
  - **修复建议**: 实现LLMService的fallback逻辑

- [ ] **test_provider_switch_logged**
  - **功能**: 验证provider切换被记录
  - **可能问题**: 日志记录缺失
  - **相关代码**: `src/services/llm/service.py`, `src/infrastructure/logging.py`
  - **修复建议**: 在provider切换时添加日志

- [ ] **test_llm_call_logs_recorded**
  - **功能**: 验证LLM调用被记录
  - **可能问题**: LLM调用日志功能未实现
  - **相关代码**: `src/services/llm/service.py`
  - **修复建议**: 在每次LLM调用时记录日志

- [ ] **test_workflow_completes_with_fallback_provider**
  - **功能**: 验证使用fallback provider能完成工作流
  - **可能问题**: Fallback provider的响应处理问题
  - **相关代码**: `src/services/llm/service.py`
  - **修复建议**: 确保fallback provider返回相同格式的响应

- [ ] **test_multiple_provider_failures_handled**
  - **功能**: 验证能处理多个provider失败
  - **可能问题**: 多个provider失败的级联处理
  - **相关代码**: `src/services/llm/service.py`
  - **修复建议**: 实现provider列表的遍历和重试

- [ ] **test_provider_failure_doesnt_halt_workflow**
  - **功能**: 验证provider失败不会停止工作流
  - **可能问题**: 错误处理不当导致工作流中断
  - **相关代码**: `src/services/llm/service.py`, `src/infrastructure/error_handler.py`
  - **修复建议**: 添加异常捕获和优雅降级

- [ ] **test_response_time_logged_for_llm_calls**
  - **功能**: 验证LLM调用响应时间被记录
  - **可能问题**: 响应时间监控未实现
  - **相关代码**: `src/services/llm/service.py`, `src/infrastructure/metrics.py`
  - **修复建议**: 添加响应时间计时和记录

- [ ] **test_token_count_tracked_for_llm_calls**
  - **功能**: 验证LLM调用token数量被跟踪
  - **可能问题**: Token计数功能未实现
  - **相关代码**: `src/services/llm/service.py`
  - **修复建议**: 从LLM响应中提取和记录token数量

---

## 3. Retry Limit Workflow (9个测试)

### 文件: `tests/integration/test_retry_limit_workflow.py`

- [ ] **test_retry_limit_enforced_after_max_attempts**
  - **功能**: 验证达到最大重试次数后强制执行限制
  - **可能问题**: 重试限制检查逻辑不正确
  - **相关代码**: `src/domain/agents/retry_protection.py`
  - **修复建议**: 检查check_retry_limit方法的边界条件

- [ ] **test_forced_degradation_skips_step**
  - **功能**: 验证强制降级会跳过步骤
  - **可能问题**: 降级机制未实现
  - **相关代码**: `src/domain/agents/retry_protection.py`
  - **修复建议**: 实现force_degradation方法

- [ ] **test_workflow_continues_after_skip**
  - **功能**: 验证跳过步骤后工作流继续
  - **可能问题**: 跳过后的状态更新不正确
  - **相关代码**: `src/application/orchestrator.py`
  - **修复建议**: 确保跳过步骤后current_step_index递增

- [ ] **test_retry_attempts_logged**
  - **功能**: 验证重试尝试被记录
  - **可能问题**: 日志记录缺失
  - **相关代码**: `src/domain/agents/retry_protection.py`
  - **修复建议**: 在重试检查时添加日志

- [ ] **test_placeholder_fragment_for_skipped_step**
  - **功能**: 验证跳过的步骤有占位符片段
  - **可能问题**: 占位符片段生成逻辑缺失
  - **相关代码**: `src/domain/agents/retry_protection.py`
  - **修复建议**: 在force_degradation中生成占位符

- [ ] **test_no_infinite_loop_on_repeated_conflicts**
  - **功能**: 验证重复冲突不会导致无限循环
  - **可能问题**: 循环检测逻辑缺失
  - **相关代码**: `src/application/orchestrator.py`
  - **修复建议**: 添加循环检测和强制退出

- [ ] **test_retry_count_incremented_correctly**
  - **功能**: 验证重试计数正确递增
  - **可能问题**: 重试计数更新时机不对
  - **相关代码**: `src/domain/agents/retry_protection.py`
  - **修复建议**: 在每次重试时更新计数

- [ ] **test_degradation_action_logged**
  - **功能**: 验证降级操作被记录
  - **可能问题**: 降级日志缺失
  - **相关代码**: `src/domain/agents/retry_protection.py`
  - **修复建议**: 在降级时添加日志

- [ ] **test_final_screenplay_produced_despite_skips**
  - **功能**: 验证即使有跳过也能生成最终剧本
  - **可能问题**: 编译器处理占位符片段的问题
  - **相关代码**: `src/domain/agents/compiler.py`
  - **修复建议**: 确保编译器能处理占位符

---

## 4. Pivot Workflow (7个测试)

### 文件: `tests/integration/test_pivot_workflow.py`

- [ ] **test_pivot_triggered_on_deprecation_conflict**
  - **功能**: 验证弃用冲突触发pivot
  - **可能问题**: Pivot触发条件判断不正确
  - **相关代码**: `src/domain/agents/director.py`
  - **修复建议**: 检查director的evaluate_step方法

- [ ] **test_outline_modified_after_pivot**
  - **功能**: 验证pivot后大纲被修改
  - **可能问题**: Pivot后大纲更新逻辑缺失
  - **相关代码**: `src/domain/agents/pivot_manager.py`
  - **修复建议**: 实现pivot后的大纲修改

- [ ] **test_re_retrieval_after_pivot**
  - **功能**: 验证pivot后重新检索
  - **可能问题**: Pivot后检索逻辑未触发
  - **相关代码**: `src/domain/agents/pivot_manager.py`, `src/domain/agents/navigator.py`
  - **修复建议**: 在pivot后清空retrieved_docs并触发检索

- [ ] **test_pivot_loop_completes_successfully**
  - **功能**: 验证pivot循环成功完成
  - **可能问题**: Pivot循环的退出条件不正确
  - **相关代码**: `src/application/orchestrator.py`
  - **修复建议**: 检查pivot循环的终止条件

- [ ] **test_skill_switch_to_warning_mode**
  - **功能**: 验证切换到警告模式
  - **可能问题**: 技能切换逻辑问题
  - **相关代码**: `src/domain/skills.py`
  - **修复建议**: 检查skill_switch方法

- [ ] **test_pivot_reason_logged**
  - **功能**: 验证pivot原因被记录
  - **可能问题**: Pivot日志缺失
  - **相关代码**: `src/domain/agents/pivot_manager.py`
  - **修复建议**: 在pivot时添加日志

- [ ] **test_multiple_pivots_handled**
  - **功能**: 验证能处理多个pivot
  - **可能问题**: 多次pivot的状态管理问题
  - **相关代码**: `src/application/orchestrator.py`
  - **修复建议**: 验证多次pivot的循环处理

---

## 5. LangGraph Workflow (3个测试)

### 文件: `tests/integration/test_langgraph_workflow.py`

- [ ] **test_fact_check_and_completion_invalid**
  - **功能**: 验证事实检查失败时的路由
  - **可能问题**: 路由条件判断不正确
  - **相关代码**: `src/application/orchestrator.py` (fact_check_and_completion函数)
  - **修复建议**: 检查fact_check_passed标志的判断

- [ ] **test_fact_check_and_completion_continue**
  - **功能**: 验证事实检查通过且未完成时继续
  - **可能问题**: 完成条件判断不正确
  - **相关代码**: `src/application/orchestrator.py`
  - **修复建议**: 检查current_step_index与outline长度的比较

- [ ] **test_fact_check_and_completion_done**
  - **功能**: 验证事实检查通过且完成时结束
  - **可能问题**: 结束条件判断不正确
  - **相关代码**: `src/application/orchestrator.py`
  - **修复建议**: 检查END节点的路由条件

---

## 6. Recursion Limit (1个测试)

### 文件: `tests/property/test_recursion_limit_propagation.py`

- [ ] **test_recursion_limit_propagates_to_langgraph**
  - **功能**: 验证递归限制参数传递到LangGraph
  - **可能问题**: LangGraph配置中recursion_limit参数未传递
  - **相关代码**: `src/application/orchestrator.py` (Orchestrator.__init__)
  - **修复建议**: 在graph.compile()时传递recursion_limit参数

---

## 修复优先级建议

### 第一批 (核心功能，影响最大)
1. Hallucination Workflow (8个)
2. Retry Limit Workflow (9个)
3. LLM Provider Fallback (8个)

### 第二批 (高级功能)
4. Pivot Workflow (7个)
5. LangGraph Workflow (3个)

### 第三批 (边缘情况)
6. Recursion Limit (1个)

---

## 修复流程建议

对于每个失败的测试：

1. **理解测试意图**
   - 阅读测试代码
   - 理解测试验证的功能
   - 识别测试的断言

2. **定位问题代码**
   - 找到相关的源代码文件
   - 理解当前实现
   - 识别缺失或错误的逻辑

3. **实现修复**
   - 修改源代码
   - 添加必要的日志
   - 处理边界条件

4. **验证修复**
   - 运行单个测试
   - 运行相关测试套件
   - 运行完整测试套件

5. **更新清单**
   - 标记测试为已完成
   - 记录修复的关键点
   - 更新进度

---

## 注意事项

1. **环境问题**: 当前pytest有langsmith插件兼容性问题，需要先解决
2. **Mock数据**: 确保Mock数据足够真实，能触发所有代码路径
3. **状态管理**: 注意SharedState在各个agent间的传递和更新
4. **日志记录**: 添加足够的日志以便调试
5. **回归测试**: 每次修复后运行完整测试套件，确保没有引入新问题
