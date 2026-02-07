"""FactChecker 最小上下文集成

功能：
1. 集成 ContextMinimizer 为 FactChecker 提供最小化上下文
2. 限制 FactChecker 的访问范围（防止看到历史错误）
3. 只验证当前片段和当次检索结果
4. 减少 Token 使用量

设计原则：
- FactChecker 只看当前片段
- FactChecker 只看当次检索结果
- FactChecker 不看历史检索日志
- FactChecker 不看历史错误
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ...services.retrieval_isolation import ContextMinimizer
from ...infrastructure.logging import get_agent_logger
from .fact_checker import (
    build_verification_messages,
    parse_verification_response
)


logger = logging.getLogger(__name__)
agent_logger = get_agent_logger(__name__)


class FactCheckerMinimalContext:
    """FactChecker 最小上下文处理器

    集成检索隔离和上下文最小化，确保 FactChecker：
    1. 只获取当前步骤的必要信息
    2. 不累积历史检索结果
    3. 不访问执行日志（Agent 内部决策）
    4. 只验证当次检索的来源
    """

    def __init__(self):
        """初始化 FactChecker 最小上下文处理器"""
        self._context_cache = {}

    def get_verification_context(
        self,
        state: Dict[str, Any],
        current_fragment_index: int
    ) -> Dict[str, Any]:
        """
        获取 FactChecker 验证所需的最小上下文

        Args:
            state: 当前全局状态
            current_fragment_index: 当前片段索引

        Returns:
            最小化后的验证上下文
        """
        context = ContextMinimizer.get_fact_checker_context(
            state=state,
            current_fragment_index=current_fragment_index
        )

        context["_metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "context_type": "fact_checker_minimal",
            "isolation_enabled": True,
            "can_access_history": False,
            "can_access_retrieval_log": False
        }

        return context

    def should_verify(
        self,
        state: Dict[str, Any],
        fragment_index: int
    ) -> Tuple[bool, str]:
        """
        判断是否应该进行验证

        Args:
            state: 当前状态
            fragment_index: 片段索引

        Returns:
            (should_verify, reason)
        """
        fragments = state.get("fragments", [])

        if not fragments:
            return False, "no_fragments"

        if fragment_index >= len(fragments):
            return False, "fragment_index_out_of_range"

        fragment = fragments[fragment_index]

        if not fragment.get("content"):
            return False, "empty_fragment"

        if fragment.get("fact_check_passed"):
            return False, "already_verified"

        return True, "pending_verification"

    def extract_retrieved_docs_for_verification(
        self,
        state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        提取用于验证的检索文档

        设计原则：
        - 只使用当次检索结果
        - 限制文档数量（最多 3 个）
        - 过滤空文档

        Args:
            state: 当前状态

        Returns:
            过滤后的检索文档列表
        """
        retrieved_docs = state.get("retrieved_docs", [])

        if not retrieved_docs:
            logger.info("No retrieved docs for verification - research mode or missing retrieval")
            return []

        filtered_docs = []

        for doc in retrieved_docs:
            if isinstance(doc, dict):
                content = doc.get("content", "")
                source = doc.get("source", doc.get("file_path", "unknown"))

                if content and len(content.strip()) > 0:
                    filtered_docs.append({
                        "content": content[:5000],
                        "source": source,
                        "citation": doc.get("citation", f"[{len(filtered_docs) + 1}]")
                    })

            elif hasattr(doc, 'content'):
                filtered_docs.append({
                    "content": doc.content[:5000],
                    "source": doc.source,
                    "citation": getattr(doc, "citation", None) or f"[{len(filtered_docs) + 1}]"
                })

        max_docs = 3
        return filtered_docs[:max_docs]

    def format_sources_summary(
        self,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """
        格式化源文档摘要

        Args:
            retrieved_docs: 检索文档列表

        Returns:
            格式化的源文档摘要字符串
        """
        if not retrieved_docs:
            return "（无检索文档）"

        sources_summary = []

        for i, doc in enumerate(retrieved_docs):
            source = doc.get("source", f"源文档 {i+1}")
            content = doc.get("content", "")[:1000]
            citation = doc.get("citation", f"[{i+1}]")

            sources_summary.append(
                f"{citation} {source}:\n{content}..."
            )

        return "\n\n".join(sources_summary)

    def create_verification_messages(
        self,
        fragment_content: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        创建验证消息

        Args:
            fragment_content: 片段内容
            retrieved_docs: 检索文档列表

        Returns:
            格式化的消息列表
        """
        sources_summary = self.format_sources_summary(retrieved_docs)
        return build_verification_messages(fragment_content, sources_summary)

    def create_verification_record(
        self,
        fragment_index: int,
        is_valid: bool,
        hallucinations: List[str],
        doc_count: int,
        context_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        创建验证记录（用于审计）

        Args:
            fragment_index: 片段索引
            is_valid: 是否有效
            hallucinations: 幻觉列表
            doc_count: 使用的文档数量
            context_metadata: 上下文元数据

        Returns:
            验证记录
        """
        return {
            "fragment_index": fragment_index,
            "is_valid": is_valid,
            "hallucination_count": len(hallucinations),
            "hallucinations": hallucinations,
            "documents_used": doc_count,
            "isolation_enabled": context_metadata.get("isolation_enabled", True),
            "timestamp": datetime.now().isoformat(),
            "context_type": "fact_checker_minimal"
        }


class VerificationScope:
    """FactChecker 验证范围控制器

    控制 FactChecker 可以访问哪些信息，防止：
    1. 看到历史错误导致级联幻觉
    2. 访问检索日志导致信息泄露
    3. 累积过多上下文导致验证质量下降
    """

    DEFAULT_SCOPE = {
        "can_access_history": False,
        "can_access_retrieval_log": False,
        "can_access_execution_log": False,
        "can_access_other_fragments": False,
        "max_retrieval_docs": 3,
        "max_fragment_length": 10000,
        "require_citations": True,
        "verify_after_write": True
    }

    STRICT_SCOPE = {
        "can_access_history": False,
        "can_access_retrieval_log": False,
        "can_access_execution_log": False,
        "can_access_other_fragments": False,
        "max_retrieval_docs": 2,
        "max_fragment_length": 5000,
        "require_citations": True,
        "verify_after_write": True
    }

    @classmethod
    def get_scope(cls, strictness: str = "default") -> Dict[str, Any]:
        """
        获取验证范围配置

        Args:
            strictness: 严格程度 ('default' | 'strict')

        Returns:
            验证范围配置
        """
        if strictness == "strict":
            return cls.STRICT_SCOPE
        return cls.DEFAULT_SCOPE

    @classmethod
    def apply_scope(
        cls,
        scope: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        应用验证范围，清理不应当访问的信息

        Args:
            scope: 验证范围配置
            state: 原始状态

        Returns:
            清理后的状态
        """
        filtered_state = {
            "user_topic": state.get("user_topic", ""),
            "project_context": state.get("project_context", ""),
            "current_step_query": state.get("current_step_query", ""),
        }

        if scope.get("can_access_other_fragments"):
            filtered_state["fragments"] = state.get("fragments", [])
        else:
            fragments = state.get("fragments", [])
            if fragments:
                filtered_state["fragments"] = [fragments[-1]]
            else:
                filtered_state["fragments"] = []

        retrieved_docs = state.get("retrieved_docs", [])
        if isinstance(retrieved_docs, list):
            max_docs = scope.get("max_retrieval_docs", 3)
            filtered_state["retrieved_docs"] = retrieved_docs[:max_docs]
        else:
            filtered_state["retrieved_docs"] = []

        outline = state.get("outline", [])
        if isinstance(outline, list) and scope.get("can_access_history"):
            filtered_state["outline"] = outline
        else:
            current_step = state.get("current_step_index", 0)
            if outline and current_step < len(outline):
                filtered_state["current_step"] = outline[current_step]
            else:
                filtered_state["current_step"] = {}

        filtered_state["_verification_scope"] = scope

        return filtered_state
