"""检索结果隔离模块

功能：
1. 检索结果每步清空（防止累积膨胀）
2. 最小化上下文获取
3. 来源追溯与引用验证

设计原则：
- Navigator 负责检索，不保留历史
- 每个 Agent 只获取当次需要的检索结果
- FactChecker 只验证当前片段，不看历史错误
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


logger = logging.getLogger(__name__)


class IsolationLevel(Enum):
    """隔离级别"""
    NONE = "none"
    STEP = "step"
    SESSION = "session"
    PERMANENT = "permanent"


@dataclass
class RetrievalRecord:
    """检索记录（用于审计）"""
    step_index: int
    query: str
    doc_count: int
    timestamp: str
    agent: str = "navigator"


@dataclass
class MinimalContext:
    """最小化上下文（用于幻觉控制）"""
    current_step_query: str
    retrieved_docs: List[Dict[str, Any]]
    doc_count: int
    sources: List[str]
    citations: List[str]


class RetrievalIsolation:
    """检索结果隔离器

    核心功能：
    1. 确保每次只保留当次检索结果
    2. 记录检索历史（审计用，不暴露给 Agent）
    3. 提供最小化上下文给 FactChecker

    与 GlobalState 的交互：
    - 写入：retrieved_docs（每步覆盖）
    - 隔离：retrieval_history（Agent 不可见）
    """

    def __init__(self, max_docs_per_step: int = 5):
        """
        初始化检索隔离器

        Args:
            max_docs_per_step: 每步最大检索文档数
        """
        self.max_docs_per_step = max_docs_per_step
        self._retrieval_history: List[RetrievalRecord] = []
        self._isolation_enabled = True

    @property
    def retrieval_history(self) -> List[RetrievalRecord]:
        """获取检索历史（Agent 不可见）"""
        return self._retrieval_history.copy()

    def prepare_retrieval(
        self,
        state: Dict[str, Any],
        query: str
    ) -> MinimalContext:
        """
        准备检索（清空上步结果，获取当前查询）

        Args:
            state: 当前状态
            query: 当前查询

        Returns:
            最小化上下文
        """
        step_index = state.get("current_step_index", 0)

        logger.info(
            f"RetrievalIsolation: Preparing step {step_index}, "
            f"query: {query[:50]}..."
        )

        return MinimalContext(
            current_step_query=query,
            retrieved_docs=[],
            doc_count=0,
            sources=[],
            citations=[]
        )

    def record_retrieval(
        self,
        query: str,
        results: List[Dict[str, Any]],
        step_index: int,
        agent: str = "navigator"
    ) -> List[Dict[str, Any]]:
        """
        记录检索结果（隔离存储）

        Args:
            query: 查询内容
            results: 检索结果
            step_index: 当前步骤索引
            agent: 执行检索的 Agent

        Returns:
            过滤后的结果（限制数量）
        """
        if not self._isolation_enabled:
            logger.warning("RetrievalIsolation is disabled!")
            return results

        filtered_results = results[:self.max_docs_per_step]

        record = RetrievalRecord(
            step_index=step_index,
            query=query,
            doc_count=len(filtered_results),
            timestamp=datetime.now().isoformat(),
            agent=agent
        )
        self._retrieval_history.append(record)

        logger.info(
            f"Recorded retrieval for step {step_index}: "
            f"{len(filtered_results)} docs (total history: {len(self._retrieval_history)})"
        )

        return filtered_results

    def get_minimal_context_for_factchecker(
        self,
        state: Dict[str, Any],
        current_fragment_index: int
    ) -> MinimalContext:
        """
        获取 FactChecker 所需的最小上下文

        设计原则：
        - FactChecker 只看当前片段的检索结果
        - 不看历史检索结果（防止错误传播）
        - 不看检索历史（隔离存储）

        Args:
            state: 当前状态
            current_fragment_index: 当前片段索引

        Returns:
            最小化上下文
        """
        step_index = state.get("current_step_index", 0)
        retrieved_docs = state.get("retrieved_docs", [])

        sources = []
        citations = []

        for doc in retrieved_docs:
            if isinstance(doc, dict):
                source = doc.get("source", doc.get("file_path", "unknown"))
                sources.append(source)
                citations.append(doc.get("citation", f"[{len(sources)}]"))

        return MinimalContext(
            current_step_query=state.get("current_step_query", ""),
            retrieved_docs=retrieved_docs,
            doc_count=len(retrieved_docs),
            sources=sources,
            citations=citations
        )

    def get_minimal_context_for_writer(
        self,
        state: Dict[str, Any],
        current_step_index: int
    ) -> MinimalContext:
        """
        获取 Writer 所需的最小上下文

        设计原则：
        - Writer 只看当前步的检索结果
        - 可以看到前一个片段（上下文连贯）

        Args:
            state: 当前状态
            current_step_index: 当前步骤索引

        Returns:
            最小化上下文
        """
        retrieved_docs = state.get("retrieved_docs", [])

        sources = []
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                source = doc.get("source", doc.get("file_path", "unknown"))
                if source not in sources:
                    sources.append(source)

        return MinimalContext(
            current_step_query=state.get("current_step_query", ""),
            retrieved_docs=retrieved_docs,
            doc_count=len(retrieved_docs),
            sources=sources,
            citations=[]
        )

    def clear_step_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        清空当前步的检索结果（准备下步检索）

        Args:
            state: 当前状态

        Returns:
            清空后的状态
        """
        new_state = state.copy()
        new_state["retrieved_docs"] = []
        new_state["current_step_query"] = ""

        logger.debug(
            f"Cleared retrieval results for step {state.get('current_step_index', 'unknown')}"
        )

        return new_state

    def get_audit_report(self) -> Dict[str, Any]:
        """获取审计报告"""
        return {
            "total_retrievals": len(self._retrieval_history),
            "retrieval_history": [
                {
                    "step": r.step_index,
                    "query": r.query[:100],
                    "doc_count": r.doc_count,
                    "timestamp": r.timestamp,
                    "agent": r.agent
                }
                for r in self._retrieval_history
            ],
            "isolation_enabled": self._isolation_enabled,
            "max_docs_per_step": self.max_docs_per_step
        }

    def enable_isolation(self):
        """启用隔离"""
        self._isolation_enabled = True
        logger.info("RetrievalIsolation: Enabled")

    def disable_isolation(self):
        """禁用隔离（仅用于调试）"""
        self._isolation_enabled = False
        logger.warning("RetrievalIsolation: Disabled (debug mode)")


class ContextMinimizer:
    """上下文最小化器

    功能：
    1. 根据 Agent 类型返回最小化上下文
    2. 过滤敏感信息
    3. 截断长文本
    """

    # Agent 上下文配置
    AGENT_CONTEXTS = {
        "fact_checker": {
            "max_docs": 3,
            "max_fragment_length": 5000,
            "include_sources": True,
            "include_citations": True
        },
        "writer": {
            "max_docs": 5,
            "max_fragment_length": 10000,
            "include_sources": True,
            "include_citations": False
        },
        "director": {
            "max_docs": 3,
            "include_sources": True,
            "include_citations": False
        },
        "navigator": {
            "max_docs": 10,
            "include_sources": True,
            "include_citations": True
        }
    }

    @classmethod
    def minimize_for_agent(
        cls,
        agent_type: str,
        state: Dict[str, Any],
        current_step_index: int
    ) -> Dict[str, Any]:
        """
        根据 Agent 类型最小化上下文

        Args:
            agent_type: Agent 类型
            state: 当前状态
            current_step_index: 当前步骤索引

        Returns:
            最小化后的上下文
        """
        config = cls.AGENT_CONTEXTS.get(agent_type, cls.AGENT_CONTEXTS["navigator"])

        minimized = {
            "user_topic": state.get("user_topic", ""),
            "project_context": state.get("project_context", ""),
            "current_step_index": current_step_index,
        }

        outline = state.get("outline", [])
        if isinstance(outline, list) and current_step_index < len(outline):
            minimized["current_step"] = outline[current_step_index]
        else:
            minimized["current_step"] = {}

        retrieved_docs = state.get("retrieved_docs", [])
        if isinstance(retrieved_docs, list):
            minimized["retrieved_docs"] = retrieved_docs[:config["max_docs"]]
            minimized["doc_count"] = len(minimized["retrieved_docs"])
        else:
            minimized["retrieved_docs"] = []
            minimized["doc_count"] = 0

        fragments = state.get("fragments", [])
        if isinstance(fragments, list) and fragments:
            current_fragment = fragments[-1]
            fragment_content = current_fragment.get("content", "")

            if config.get("max_fragment_length"):
                fragment_content = fragment_content[:config["max_fragment_length"]]

            minimized["current_fragment"] = {
                "content": fragment_content,
                "skill_used": current_fragment.get("skill_used", ""),
                "step_index": current_fragment.get("step_index", current_step_index)
            }
        else:
            minimized["current_fragment"] = None

        return minimized

    @classmethod
    def get_fact_checker_context(
        cls,
        state: Dict[str, Any],
        current_fragment_index: int
    ) -> Dict[str, Any]:
        """
        获取 FactChecker 专用上下文（最精简）

        设计原则：
        - 只包含当前片段
        - 只包含当次检索结果
        - 包含来源引用
        """
        context = cls.minimize_for_agent(
            agent_type="fact_checker",
            state=state,
            current_step_index=current_fragment_index
        )

        fragments = state.get("fragments", [])
        if fragments and current_fragment_index < len(fragments):
            context["fragment_to_verify"] = fragments[current_fragment_index]
        else:
            context["fragment_to_verify"] = None

        context["verification_scope"] = {
            "can_access_history": False,
            "can_access_retrieval_log": False,
            "max_retrieval_docs": 3,
            "require_citations": True
        }

        return context
