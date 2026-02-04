"""Agent 协作逻辑优化

功能：
1. Agent 协商机制 - Agent 之间显式协商
2. 并行执行器 - 并行执行独立 Agent 任务
3. 执行追踪器 - 追踪 Agent 决策链
4. 自我反思机制 - Agent 从失败中学习

解决的问题：
- Director 做决策时，Writer 无法提供反馈
- 所有 Agent 串行执行，效率低
- 难以追踪 Agent 决策链
- Agent 不会从失败中学习
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from ...services.llm.service import LLMService

logger = logging.getLogger(__name__)


class NegotiationStatus(Enum):
    """协商状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    FAILED = "failed"


class DecisionType(Enum):
    """决策类型"""
    SKILL_SWITCH = "skill_switch"
    STRATEGY_CHANGE = "strategy_change"
    PRIORITY_ADJUST = "priority_adjust"
    TASK_REORDER = "task_reorder"


@dataclass
class NegotiationContext:
    """协商上下文"""
    negotiation_id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    current_state: Dict[str, Any] = field(default_factory=dict)
    director_recommendation: Optional[str] = None
    writer_preference: Optional[str] = None
    content_analysis: Dict[str, Any] = field(default_factory=dict)
    status: NegotiationStatus = NegotiationStatus.PENDING
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    final_decision: Optional[str] = None
    decision_reason: str = ""


@dataclass
class ExecutionNode:
    """执行节点"""
    node_id: str = field(default_factory=lambda: str(uuid4()))
    agent_name: str = ""
    input_summary: Dict[str, Any] = field(default_factory=dict)
    output_summary: Dict[str, Any] = field(default_factory=dict)
    decision_reason: str = ""
    execution_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    status: str = "completed"
    error: Optional[str] = None


@dataclass
class FailureRecord:
    """失败记录"""
    failure_id: str = field(default_factory=lambda: str(uuid4()))
    agent_name: str = ""
    failure_type: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    was_recovered: bool = False
    recovery_strategy: Optional[str] = None


@dataclass
class ReflectionResult:
    """反思结果"""
    analysis: str = ""
    adjustments: List[Dict[str, Any]] = field(default_factory=list)
    confidence_change: float = 0.0
    suggested_strategy: Optional[str] = None
    should_retry: bool = False
    retry_with_different_approach: bool = False


class AgentNegotiator:
    """
    Agent 协商器
    
    功能：
    1. Director 基于内容复杂度推荐 skill
    2. Writer 基于生成历史提供偏好
    3. 综合决策（支持 LLM 仲裁）
    
    使用示例:
    ```python
    negotiator = AgentNegotiator(llm_service=llm_service)
    decision = await negotiator.negotiate_skill_switch(
        director_recommendation="detailed_explainer",
        writer_preference="concise_summary",
        content_analysis={"difficulty": "high", "topic": "architecture"}
    )
    ```
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        enable_llm_arbitration: bool = True,
        arbitration_model: str = "lightweight"
    ):
        """
        初始化协商器
        
        Args:
            llm_service: LLM 服务（用于仲裁）
            enable_llm_arbitration: 是否启用 LLM 仲裁
            arbitration_model: 仲裁使用的模型
        """
        self.llm_service = llm_service
        self.enable_llm_arbitration = enable_llm_arbitration
        self.arbitration_model = arbitration_model
        
        self.negotiation_history: List[NegotiationContext] = []
        
        self.decision_rules = {
            "high_complexity": ["detailed_explainer", "architectural_analysis"],
            "medium_complexity": ["standard_explainer", "practical_guide"],
            "low_complexity": ["concise_summary", "quick_reference"],
        }
    
    async def negotiate_skill_switch(
        self,
        director_recommendation: str,
        writer_preference: str,
        content_analysis: Dict[str, Any],
        query: str = "",
        current_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        协商技能切换
        
        Args:
            director_recommendation: Director 的推荐
            writer_preference: Writer 的偏好
            content_analysis: 内容分析结果
            query: 查询文本
            current_state: 当前状态
            
        Returns:
            (最终决策, 决策原因)
        """
        logger.info(
            f"Negotiating skill switch: director='{director_recommendation}', "
            f"writer='{writer_preference}'"
        )
        
        if director_recommendation == writer_preference:
            logger.info("No negotiation needed - recommendations match")
            return director_recommendation, "Director and Writer agreed"
        
        negotiation_context = NegotiationContext(
            query=query,
            current_state=current_state or {},
            director_recommendation=director_recommendation,
            writer_preference=writer_preference,
            content_analysis=content_analysis,
            status=NegotiationStatus.IN_PROGRESS
        )
        
        decision = await self._mediate_decision(
            director_recommendation,
            writer_preference,
            content_analysis,
            negotiation_context
        )
        
        negotiation_context.final_decision = decision
        negotiation_context.status = NegotiationStatus.RESOLVED
        negotiation_context.resolved_at = time.time()
        negotiation_context.decision_reason = self._get_decision_reason(
            director_recommendation, writer_preference, decision
        )
        
        self.negotiation_history.append(negotiation_context)
        
        logger.info(f"Negotiation resolved: decision='{decision}'")
        
        return decision, negotiation_context.decision_reason
    
    async def _mediate_decision(
        self,
        director_recommendation: str,
        writer_preference: str,
        content_analysis: Dict[str, Any],
        context: NegotiationContext
    ) -> str:
        """调解决策冲突"""
        complexity = content_analysis.get("difficulty_level", "medium")
        
        if complexity == "high":
            return director_recommendation
        
        if complexity == "low":
            return writer_preference
        
        if not self.enable_llm_arbitration or not self.llm_service:
            return self._rule_based_mediation(
                director_recommendation, writer_preference, content_analysis
            )
        
        return await self._llm_arbitration(
            director_recommendation, writer_preference, content_analysis
        )
    
    def _rule_based_mediation(
        self,
        director_recommendation: str,
        writer_preference: str,
        content_analysis: Dict[str, Any]
    ) -> str:
        """基于规则的调解"""
        complexity = content_analysis.get("difficulty_level", "medium")
        
        history = content_analysis.get("generation_history", [])
        writer_success_count = sum(
            1 for h in history if h.get("agent") == "writer" and h.get("success")
        )
        
        if writer_success_count >= 3:
            return writer_preference
        
        complexity_scores = {
            "high": {"detailed_explainer": 1.0, "concise_summary": 0.3},
            "medium": {"standard_explainer": 0.8, "practical_guide": 0.7},
            "low": {"quick_reference": 0.9, "concise_summary": 0.8}
        }
        
        scores = complexity_scores.get(complexity, complexity_scores["medium"])
        
        director_score = scores.get(director_recommendation, 0.5)
        writer_score = scores.get(writer_preference, 0.5)
        
        if director_score > writer_score:
            return director_recommendation
        elif writer_score > director_score:
            return writer_preference
        
        return director_recommendation if complexity == "high" else writer_preference
    
    async def _llm_arbitration(
        self,
        director_recommendation: str,
        writer_preference: str,
        content_analysis: Dict[str, Any]
    ) -> str:
        """LLM 仲裁"""
        analysis_str = json.dumps(content_analysis, ensure_ascii=False, indent=2)
        
        prompt = f"""你是协调者，需要在两个建议中选择最优方案。

## Director 的推荐
推荐技能: {director_recommendation}
理由: 基于内容复杂度分析认为此技能最适合当前任务

## Writer 的偏好
偏好技能: {writer_preference}
理由: 基于生成历史和当前上下文认为此技能效果更好

## 内容分析
{analysis_str}

## 要求
1. 分析两个方案的优缺点
2. 考虑内容复杂度、生成历史、技术准确性等因素
3. 输出 JSON 格式: {{"decision": "最终选择的技能", "reason": "决策原因"}}

最终决策:"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个多智能体协调专家，擅长在多个智能体的建议中选择最优方案。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await self.llm_service.chat_completion(
                messages=messages,
                task_type=self.arbitration_model,
                temperature=0.3,
                max_tokens=200
            )
            
            try:
                result = json.loads(response)
                if "decision" in result:
                    return result["decision"]
            except json.JSONDecodeError:
                pass
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"LLM arbitration failed: {str(e)}")
            return self._rule_based_mediation(
                director_recommendation, writer_preference, content_analysis
            )
    
    def _get_decision_reason(
        self,
        director_rec: str,
        writer_pref: str,
        final: str
    ) -> str:
        """生成决策原因"""
        if director_rec == final and writer_pref == final:
            return "Director 和 Writer 达成一致"
        elif director_rec == final:
            return f"采用 Director 推荐（{director_rec}），Writer 偏好被覆盖"
        elif writer_pref == final:
            return f"采用 Writer 偏好（{writer_pref}），Director 推荐被覆盖"
        else:
            return f"仲裁选择：{final}"
    
    def get_negotiation_stats(self) -> Dict[str, Any]:
        """获取协商统计"""
        total = len(self.negotiation_history)
        resolved = sum(1 for n in self.negotiation_history if n.status == NegotiationStatus.RESOLVED)
        failed = sum(1 for n in self.negotiation_history if n.status == NegotiationStatus.FAILED)
        
        return {
            "total_negotiations": total,
            "resolved": resolved,
            "failed": failed,
            "resolution_rate": resolved / total if total > 0 else 0
        }


class ParallelAgentExecutor:
    """
    并行 Agent 执行器
    
    功能：
    并行执行独立的 Agent 任务，提高执行效率
    
    可并行的任务：
    1. Navigator 检索 + Director 评估历史步骤
    2. Writer 生成 + Fact Checker 预加载模型
    
    使用示例:
    ```python
    executor = ParallelAgentExecutor(orchestrator)
    result = await executor.execute_parallel(state, [
        self.navigator_node,
        self.director_evaluate_history
    ])
    ```
    """
    
    def __init__(
        self,
        orchestrator: Any,
        max_concurrent: int = 3,
        enable_dependency_check: bool = True
    ):
        """
        初始化并行执行器
        
        Args:
            orchestrator: 编排器实例
            max_concurrent: 最大并发数
            enable_dependency_check: 是否启用依赖检查
        """
        self.orchestrator = orchestrator
        self.max_concurrent = max_concurrent
        self.enable_dependency_check = enable_dependency_check
        
        self.execution_history: List[Dict[str, Any]] = []
    
    async def execute_parallel(
        self,
        state: Dict[str, Any],
        tasks: List[Callable],
        task_names: Optional[List[str]] = None,
        dependencies: Optional[Dict[int, List[int]]] = None
    ) -> Dict[str, Any]:
        """
        并行执行多个任务
        
        Args:
            state: 当前状态
            tasks: 任务函数列表
            task_names: 任务名称列表
            dependencies: 任务依赖关系 {task_index: [dependent_task_indices]}
            
        Returns:
            合并的结果
        """
        if not tasks:
            return {}
        
        task_labels = task_names or [f"task_{i}" for i in range(len(tasks))]
        
        logger.info(f"Starting parallel execution of {len(tasks)} tasks")
        start_time = time.time()
        
        results: Dict[str, Any] = {}
        errors: Dict[str, str] = {}
        
        async def execute_task(
            task: Callable,
            task_name: str,
            task_index: int
        ) -> Tuple[str, Any]:
            """执行单个任务"""
            try:
                if asyncio.iscoroutinefunction(task):
                    result = await task(state)
                else:
                    result = task(state)
                
                return task_name, result
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Task {task_name} failed: {error_msg}")
                return task_name, {"error": error_msg}
        
        task_coroutines = [
            execute_task(task, label, i)
            for i, (task, label) in enumerate(zip(tasks, task_labels))
        ]
        
        task_results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        for i, result in enumerate(task_results):
            task_name = task_labels[i]
            
            if isinstance(result, Exception):
                errors[task_name] = str(result)
                results[task_name] = {"error": str(result)}
            else:
                name, value = result
                results[name] = value
        
        elapsed = time.time() - start_time
        
        execution_record = {
            "timestamp": start_time,
            "duration_ms": elapsed * 1000,
            "task_count": len(tasks),
            "successful_tasks": len(tasks) - len(errors),
            "failed_tasks": len(errors),
            "task_results": {k: v for k, v in results.items() if "error" not in v},
            "errors": errors
        }
        self.execution_history.append(execution_record)
        
        logger.info(
            f"Parallel execution completed in {elapsed:.2f}s: "
            f"{len(tasks) - len(errors)}/{len(tasks)} tasks succeeded"
        )
        
        merged_result = self._merge_results(results)
        merged_result["_parallel_execution_info"] = {
            "duration_ms": elapsed * 1000,
            "task_count": len(tasks),
            "success_count": len(tasks) - len(errors)
        }
        
        return merged_result
    
    def _merge_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """合并多个任务的结果"""
        merged: Dict[str, Any] = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                merged.update(value)
            else:
                merged[key] = value
        
        return merged
    
    async def execute_with_dependencies(
        self,
        state: Dict[str, Any],
        tasks: List[Callable],
        dependencies: Dict[int, List[int]],
        task_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        按依赖关系执行任务
        
        Args:
            state: 初始状态
            tasks: 任务函数列表
            dependencies: 依赖关系
            task_names: 任务名称
            
        Returns:
            合并的结果
        """
        if not self.enable_dependency_check:
            return await self.execute_parallel(state, tasks, task_names)
        
        if not dependencies:
            return await self.execute_parallel(state, tasks, task_names)
        
        task_labels = task_names or [f"task_{i}" for i in range(len(tasks))]
        
        completed = set()
        results: Dict[str, Any] = {}
        
        while len(completed) < len(tasks):
            ready_tasks = [
                i for i in range(len(tasks))
                if i not in completed and self._are_dependencies_met(i, dependencies, completed)
            ]
            
            if not ready_tasks:
                remaining = set(range(len(tasks))) - completed
                logger.warning(f"Deadlock detected with tasks: {remaining}")
                break
            
            batch = ready_tasks[:self.max_concurrent]
            
            batch_tasks = [(tasks[i], task_labels[i], i) for i in batch]
            batch_results = await self.execute_parallel(state, [t[0] for t in batch_tasks], task_labels)
            
            for i in batch:
                completed.add(i)
                task_key = task_labels[i]
                if task_key in batch_results:
                    results[task_labels[i]] = batch_results[task_key]
        
        return results
    
    def _are_dependencies_met(
        self,
        task_index: int,
        dependencies: Dict[int, List[int]],
        completed: Set[int]
    ) -> bool:
        """检查任务依赖是否满足"""
        deps = dependencies.get(task_index, [])
        return all(d in completed for d in deps)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        total_duration = sum(e["duration_ms"] for e in self.execution_history)
        total_tasks = sum(e["task_count"] for e in self.execution_history)
        successful_tasks = sum(e["successful_tasks"] for e in self.execution_history)
        
        return {
            "total_executions": len(self.execution_history),
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": total_tasks - successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "avg_duration_ms": total_duration / len(self.execution_history)
        }


class AgentExecutionTracer:
    """
    Agent 执行追踪器
    
    功能：
    追踪 Agent 执行链，支持可视化和分析
    
    使用示例:
    ```python
    tracer = AgentExecutionTracer()
    
    with tracer.trace("writer") as ctx:
        result = await writer.generate(state)
        ctx.record_output({"content_length": len(result)})
    
    print(tracer.visualize_chain())
    ```
    """
    
    def __init__(
        self,
        enable_performance_tracking: bool = True,
        max_chain_length: int = 1000
    ):
        """
        初始化追踪器
        
        Args:
            enable_performance_tracking: 是否启用性能追踪
            max_chain_length: 最大链长度
        """
        self.execution_chain: List[ExecutionNode] = []
        self.enable_performance = enable_performance_tracking
        self.max_chain_length = max_chain_length
        
        self.agent_timings: Dict[str, List[float]] = {}
        self.decision_patterns: Dict[str, int] = {}
    
    def trace_decision(
        self,
        agent_name: str,
        input_state: Dict[str, Any],
        output_state: Dict[str, Any],
        decision_reason: str,
        execution_time_ms: float,
        status: str = "completed",
        error: Optional[str] = None
    ) -> ExecutionNode:
        """
        记录决策
        
        Args:
            agent_name: Agent 名称
            input_state: 输入状态摘要
            output_state: 输出状态摘要
            decision_reason: 决策原因
            execution_time_ms: 执行时间（毫秒）
            status: 状态
            error: 错误信息
            
        Returns:
            执行节点
        """
        node = ExecutionNode(
            agent_name=agent_name,
            input_summary=self._summarize_state(input_state),
            output_summary=self._summarize_state(output_state),
            decision_reason=decision_reason,
            execution_time_ms=execution_time_ms,
            status=status,
            error=error
        )
        
        self.execution_chain.append(node)
        
        if self.enable_performance:
            if agent_name not in self.agent_timings:
                self.agent_timings[agent_name] = []
            self.agent_timings[agent_name].append(execution_time_ms)
        
        self._update_decision_patterns(agent_name, decision_reason)
        
        if len(self.execution_chain) > self.max_chain_length:
            self.execution_chain = self.execution_chain[-self.max_chain_length:]
        
        return node
    
    def _summarize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """摘要状态"""
        summary = {}
        
        keys_to_keep = [
            "current_step_index", "outline_length", "workflow_complete",
            "last_retrieved_docs_count", "director_feedback",
            "fact_check_passed", "content_length"
        ]
        
        for key in keys_to_keep:
            if key in state:
                value = state[key]
                if isinstance(value, list):
                    summary[key] = f"[{len(value)} items]"
                elif isinstance(value, dict):
                    summary[key] = f"[{len(value)} keys]"
                else:
                    summary[key] = value
        
        if len(state) > len(keys_to_keep):
            summary["_extra_keys"] = len(state) - len(keys_to_keep)
        
        return summary
    
    def _update_decision_patterns(
        self,
        agent_name: str,
        decision_reason: str
    ):
        """更新决策模式统计"""
        pattern_key = f"{agent_name}:{decision_reason[:50]}"
        self.decision_patterns[pattern_key] = (
            self.decision_patterns.get(pattern_key, 0) + 1
        )
    
    def visualize_chain(self, max_nodes: Optional[int] = None) -> str:
        """
        生成可视化的执行链
        
        Args:
            max_nodes: 最大显示节点数
            
        Returns:
            格式化的执行链
        """
        if not self.execution_chain:
            return "No execution chain recorded"
        
        nodes_to_show = self.execution_chain[-max_nodes:] if max_nodes else self.execution_chain
        
        lines = [
            "=" * 60,
            "Agent 执行链追踪",
            "=" * 60,
            f"总节点数: {len(self.execution_chain)}",
            ""
        ]
        
        for i, node in enumerate(nodes_to_show):
            status_icon = "✓" if node.status == "completed" else "✗"
            
            lines.append(
                f"{i+1}. [{status_icon}] {node.agent_name} "
                f"({node.execution_time_ms:.1f}ms)"
            )
            lines.append(f"   原因: {node.decision_reason[:100]}")
            
            if node.error:
                lines.append(f"   错误: {node.error[:100]}")
            
            lines.append("")
        
        if len(self.execution_chain) > len(nodes_to_show):
            lines.append(f"... 还有 {len(self.execution_chain) - len(nodes_to_show)} 个节点")
        
        total_time = sum(n.execution_time_ms for n in self.execution_chain)
        lines.append("")
        lines.append(f"总执行时间: {total_time:.1f}ms")
        lines.append("=" * 60)
        
        return '\n'.join(lines)
    
    def get_performance_report(self) -> str:
        """生成性能报告"""
        if not self.agent_timings:
            return "No performance data available"
        
        lines = [
            "=" * 50,
            "Agent 性能报告",
            "=" * 50,
            ""
        ]
        
        for agent, timings in self.agent_timings.items():
            if not timings:
                continue
            
            avg_time = sum(timings) / len(timings)
            max_time = max(timings)
            min_time = min(timings)
            
            lines.append(f"{agent}:")
            lines.append(f"  执行次数: {len(timings)}")
            lines.append(f"  平均时间: {avg_time:.1f}ms")
            lines.append(f"  最大时间: {max_time:.1f}ms")
            lines.append(f"  最小时间: {min_time:.1f}ms")
            lines.append("")
        
        lines.append("=" * 50)
        
        return '\n'.join(lines)
    
    def get_top_decision_patterns(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """获取最常见的决策模式"""
        sorted_patterns = sorted(
            self.decision_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_patterns[:top_n]
    
    def reset_chain(self):
        """重置执行链"""
        self.execution_chain.clear()
        self.agent_timings.clear()
        self.decision_patterns.clear()


class AgentReflection:
    """
    Agent 自我反思机制
    
    功能：
    从失败中学习，调整策略
    
    例如：
    - Writer 连续失败 3 次 → 切换到更简单的 skill
    - Fact Checker 频繁检测到幻觉 → 降低 Writer 的 temperature
    
    使用示例:
    ```python
    reflection = AgentReflection(llm_service=llm_service)
    
    adjustment = await reflection.reflect_on_failure(
        agent_name="writer",
        failure_context={"error": "content_too_long", "step": 5},
        state=current_state
    )
    
    if adjustment["switch_to_simpler"]:
        current_skill = "detailed_explainer"
        current_skill = "standard_explainer"
    ```
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        failure_history_limit: int = 10,
        reflection_threshold: int = 3
    ):
        """
        初始化反思机制
        
        Args:
            llm_service: LLM 服务（用于深度分析）
            failure_history_limit: 失败历史限制
            reflection_threshold: 反思阈值（连续失败次数）
        """
        self.llm_service = llm_service
        self.failure_history_limit = failure_history_limit
        self.reflection_threshold = reflection_threshold
        
        self.failure_history: List[FailureRecord] = []
        self.reflection_cache: Dict[str, ReflectionResult] = {}
        
        self.failure_patterns: Dict[str, List[FailureRecord]] = {}
        
        self.adjustment_strategies = {
            "content_too_long": {
                "reduce_length": True,
                "switch_to_simpler": False,
                "reduce_temperature": True
            },
            "hallucination_detected": {
                "reduce_temperature": True,
                "increase_fact_check_strictness": True
            },
            "logic_error": {
                "increase_reasoning_steps": True,
                "switch_to_simpler": True
            },
            "style_mismatch": {
                "adjust_skill_parameters": True,
                "reduce_temperature": False
            },
            "timeout": {
                "reduce_output_length": True,
                "increase_timeout": True
            }
        }
    
    async def reflect_on_failure(
        self,
        agent_name: str,
        failure_context: Dict[str, Any],
        state: Dict[str, Any],
        bypass_cache: bool = False
    ) -> ReflectionResult:
        """
        反思失败原因并调整策略
        
        Args:
            agent_name: Agent 名称
            failure_context: 失败上下文
            state: 当前状态
            bypass_cache: 绕过缓存
            
        Returns:
            反思结果
        """
        failure_type = failure_context.get("error", "unknown")
        
        cache_key = f"{agent_name}:{failure_type}"
        
        if not bypass_cache and cache_key in self.reflection_cache:
            cached = self.reflection_cache[cache_key]
            if self._is_cache_valid(cached):
                return cached
        
        failure_record = FailureRecord(
            agent_name=agent_name,
            failure_type=failure_type,
            context=failure_context,
            timestamp=time.time()
        )
        
        self.failure_history.append(failure_record)
        
        if agent_name not in self.failure_patterns:
            self.failure_patterns[agent_name] = []
        self.failure_patterns[agent_name].append(failure_record)
        
        failure_history = self._get_agent_failure_history(agent_name)
        
        if len(failure_history) >= self.reflection_threshold:
            pattern = self._analyze_failure_pattern(failure_history)
            
            if self.llm_service:
                result = await self._llm_reflection(
                    agent_name, failure_type, failure_context, state, pattern
                )
            else:
                result = self._rule_based_reflection(
                    agent_name, failure_type, pattern
                )
            
            result.adjustments = self._generate_adjustments(failure_type, pattern)
            
            self.reflection_cache[cache_key] = result
            
            return result
        
        result = ReflectionResult(
            analysis=f"单次失败: {failure_type}",
            adjustments=self._generate_adjustments(failure_type, None),
            should_retry=True
        )
        
        return result
    
    def _get_agent_failure_history(
        self,
        agent_name: str,
        since: Optional[float] = None
    ) -> List[FailureRecord]:
        """获取 Agent 的失败历史"""
        cutoff = since or (time.time() - 3600)
        
        return [
            f for f in self.failure_history
            if f.agent_name == agent_name and f.timestamp > cutoff
        ]
    
    def _analyze_failure_pattern(
        self,
        failures: List[FailureRecord]
    ) -> Dict[str, Any]:
        """分析失败模式"""
        if not failures:
            return {"pattern": "none", "frequency": 0}
        
        failure_types = [f.failure_type for f in failures]
        type_counts = {}
        for ft in failure_types:
            type_counts[ft] = type_counts.get(ft, 0) + 1
        
        most_common = max(type_counts.items(), key=lambda x: x[1])
        
        time_span = failures[-1].timestamp - failures[0].timestamp if failures else 0
        
        return {
            "pattern": most_common[0],
            "frequency": most_common[1],
            "time_span_seconds": time_span,
            "total_failures": len(failures),
            "type_distribution": type_counts
        }
    
    def _generate_adjustments(
        self,
        failure_type: str,
        pattern: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """生成调整建议"""
        base_adjustments = self.adjustment_strategies.get(failure_type, [])
        
        adjustments = []
        for adj_type, value in base_adjustments.items():
            adjustments.append({
                "type": adj_type,
                "value": value,
                "reason": f"Based on {failure_type} failure"
            })
        
        if pattern and pattern.get("frequency", 0) >= 3:
            adjustments.append({
                "type": "escalate_adjustment",
                "value": True,
                "reason": f"Repeated failures ({pattern['frequency']} times)"
            })
        
        return adjustments
    
    async def _llm_reflection(
        self,
        agent_name: str,
        failure_type: str,
        failure_context: Dict[str, Any],
        state: Dict[str, Any],
        pattern: Dict[str, Any]
    ) -> ReflectionResult:
        """LLM 深度反思"""
        context_str = json.dumps(failure_context, ensure_ascii=False, indent=2)
        state_str = json.dumps(state, ensure_ascii=False, indent=2)[:1000]
        pattern_str = json.dumps(pattern, ensure_ascii=False)
        
        prompt = f"""你是 {agent_name} 的反思系统。分析以下失败并提出改进策略。

## 失败信息
失败类型: {failure_type}
失败上下文:
{context_str}

## 失败模式
{pattern_str}

## 当前状态（部分）
{state_str}

## 分析要求
1. 分析失败的根本原因
2. 提出具体的改进策略
3. 评估是否应该重试以及重试策略

## 输出格式
JSON 格式:
{{
    "analysis": "失败原因分析",
    "suggested_strategy": "建议的新策略",
    "confidence_change": -0.1,
    "should_retry": true,
    "retry_with_different_approach": true
}}

反思结果:"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个 AI 系统反思专家，擅长分析失败原因并提出改进策略。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await self.llm_service.chat_completion(
                messages=messages,
                task_type="reasoning",
                temperature=0.5,
                max_tokens=300
            )
            
            try:
                result = json.loads(response)
                return ReflectionResult(
                    analysis=result.get("analysis", ""),
                    adjustments=self._generate_adjustments(failure_type, pattern),
                    confidence_change=result.get("confidence_change", 0.0),
                    suggested_strategy=result.get("suggested_strategy"),
                    should_retry=result.get("should_retry", True),
                    retry_with_different_approach=result.get("retry_with_different_approach", False)
                )
            except json.JSONDecodeError:
                pass
            
            return ReflectionResult(
                analysis=f"LLM analysis: {response[:200]}",
                adjustments=self._generate_adjustments(failure_type, pattern),
                should_retry=True
            )
            
        except Exception as e:
            logger.error(f"LLM reflection failed: {str(e)}")
            return ReflectionResult(
                analysis=f"Reflection failed: {str(e)}",
                adjustments=self._generate_adjustments(failure_type, pattern),
                should_retry=True
            )
    
    def _rule_based_reflection(
        self,
        agent_name: str,
        failure_type: str,
        pattern: Dict[str, Any]
    ) -> ReflectionResult:
        """基于规则的反思"""
        if failure_type == "content_too_long":
            return ReflectionResult(
                analysis="Content generation exceeded limits",
                suggested_strategy="reduce_output_length",
                should_retry=True,
                retry_with_different_approach=True
            )
        
        if failure_type == "hallucination_detected":
            return ReflectionResult(
                analysis="Fact checker detected potential hallucination",
                suggested_strategy="increase_fact_check_strictness",
                confidence_change=-0.15,
                should_retry=True
            )
        
        return ReflectionResult(
            analysis=f"Rule-based analysis for {failure_type}",
            adjustments=self._generate_adjustments(failure_type, pattern),
            should_retry=True
        )
    
    def _is_cache_valid(self, result: ReflectionResult) -> bool:
        """检查缓存是否有效"""
        return result.should_retry
    
    def get_reflection_stats(self) -> Dict[str, Any]:
        """获取反思统计"""
        total_reflections = len(self.reflection_cache)
        
        agent_stats = {}
        for agent_name, failures in self.failure_patterns.items():
            recent_failures = [
                f for f in failures
                if f.timestamp > time.time() - 3600
            ]
            agent_stats[agent_name] = {
                "total_failures": len(failures),
                "recent_failures": len(recent_failures),
                "reflections": total_reflections
            }
        
        return {
            "total_failure_records": len(self.failure_history),
            "cached_reflections": total_reflections,
            "agent_stats": agent_stats,
            "reflection_threshold": self.reflection_threshold
        }
    
    def clear_history(self, older_than: Optional[float] = None):
        """清除历史记录"""
        if older_than is None:
            self.failure_history.clear()
            self.failure_patterns.clear()
            return
        
        cutoff = time.time() - older_than
        self.failure_history = [
            f for f in self.failure_history
            if f.timestamp > cutoff
        ]
        
        for agent in self.failure_patterns:
            self.failure_patterns[agent] = [
                f for f in self.failure_patterns[agent]
                if f.timestamp > cutoff
            ]


class CollaborationManager:
    """
    协作管理器
    
    整合所有协作组件，提供统一的协作管理接口
    
    使用示例:
    ```python
    manager = CollaborationManager(
        llm_service=llm_service,
        enable_parallel=True,
        enable_tracing=True
    )
    
    # 启动并行执行
    result = await manager.execute_with_collaboration(
        state=state,
        tasks=[navigator, director],
        negotiate_skill=True
    )
    
    # 查看执行追踪
    print(manager.visualize_execution())
    ```
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化协作管理器
        
        Args:
            llm_service: LLM 服务
            config: 配置选项
        """
        self.config = config or {}
        
        self.negotiator = AgentNegotiator(
            llm_service=llm_service,
            enable_llm_arbitration=self.config.get("enable_llm_arbitration", True)
        )
        
        self.executor = ParallelAgentExecutor(
            orchestrator=None,
            max_concurrent=self.config.get("max_concurrent", 3)
        )
        
        self.tracer = AgentExecutionTracer(
            enable_performance_tracking=self.config.get("enable_performance_tracking", True)
        )
        
        self.reflection = AgentReflection(
            llm_service=llm_service,
            reflection_threshold=self.config.get("reflection_threshold", 3)
        )
    
    async def negotiate_and_execute(
        self,
        state: Dict[str, Any],
        tasks: List[Callable],
        director_recommendation: Optional[str] = None,
        writer_preference: Optional[str] = None,
        content_analysis: Optional[Dict[str, Any]] = None,
        query: str = ""
    ) -> Dict[str, Any]:
        """协商后执行"""
        if director_recommendation and writer_preference:
            decision, reason = await self.negotiator.negotiate_skill_switch(
                director_recommendation=director_recommendation,
                writer_preference=writer_preference,
                content_analysis=content_analysis or {},
                query=query,
                current_state=state
            )
            
            state["negotiated_skill"] = decision
            state["negotiation_reason"] = reason
        
        start_time = time.time()
        result = await self.executor.execute_parallel(state, tasks)
        elapsed_ms = (time.time() - start_time) * 1000
        
        self.tracer.trace_decision(
            agent_name="collaboration_manager",
            input_state={"tasks": len(tasks)},
            output_state={"result_keys": list(result.keys())},
            decision_reason=f"Parallel execution of {len(tasks)} tasks",
            execution_time_ms=elapsed_ms
        )
        
        return result
    
    async def execute_with_reflection(
        self,
        agent_name: str,
        task: Callable,
        state: Dict[str, Any],
        failure_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], bool]:
        """执行并反思"""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(task):
                result = await task(state)
            else:
                result = task(state)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            self.tracer.trace_decision(
                agent_name=agent_name,
                input_state=state,
                output_summary=self._summarize(result),
                decision_reason="Task executed successfully",
                execution_time_ms=elapsed_ms
            )
            
            return result, True
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            
            error_context = failure_context or {"error": str(e)}
            
            self.tracer.trace_decision(
                agent_name=agent_name,
                input_state=state,
                output_summary={},
                decision_reason=f"Task failed: {str(e)}",
                execution_time_ms=elapsed_ms,
                status="failed",
                error=str(e)
            )
            
            reflection = await self.reflection.reflect_on_failure(
                agent_name=agent_name,
                failure_context=error_context,
                state=state
            )
            
            return {
                "error": str(e),
                "reflection": reflection,
                "adjustments": reflection.adjustments
            }, False
    
    def _summarize(self, obj: Any) -> Dict[str, Any]:
        """摘要对象"""
        if isinstance(obj, dict):
            return {k: self._summarize(v) for k, v in list(obj.items())[:10]}
        elif isinstance(obj, list):
            return f"[{len(obj)} items]"
        else:
            return str(obj)[:100]
    
    def visualize_execution(self) -> str:
        """可视化执行"""
        return self.tracer.visualize_chain()
    
    def get_performance_report(self) -> str:
        """获取性能报告"""
        lines = [
            "=" * 60,
            "协作管理器性能报告",
            "=" * 60,
            "",
            "执行追踪:",
            self.tracer.get_performance_report(),
            "",
            "协商统计:",
            json.dumps(self.negotiator.get_negotiation_stats(), indent=2),
            "",
            "执行统计:",
            json.dumps(self.executor.get_execution_stats(), indent=2),
            "",
            "反思统计:",
            json.dumps(self.reflection.get_reflection_stats(), indent=2),
            "",
            "=" * 60
        ]
        
        return '\n'.join(lines)
