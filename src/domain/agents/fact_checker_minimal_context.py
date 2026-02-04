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
        retrieved_docs = state.get("last_retrieved_docs", [])

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
                    "citation": doc.get("citation", f"[{len(filtered_docs) + 1}]")
                })

        max_docs = 3
        return filtered_docs[:max_docs]

    def format_sources_for_verification(
        self,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """
        格式化源文档用于验证提示

        Args:
            retrieved_docs: 检索文档列表

        Returns:
            格式化的源文档字符串
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

    def create_verification_prompt(
        self,
        fragment_content: str,
        retrieved_docs: List[Dict[str, Any]],
        verification_scope: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        创建验证提示

        设计原则：
        - 提示 FactChecker 只验证当前片段
        - 不提及其他历史片段
        - 要求标注来源

        Args:
            fragment_content: 片段内容
            retrieved_docs: 检索文档
            verification_scope: 验证范围配置

        Returns:
            格式化的消息列表
        """
        sources_formatted = self.format_sources_for_verification(retrieved_docs)

        system_message = """你是一个事实检查专家。你的任务是验证生成的内容是否与提供的源文档一致。

验证原则：
1. 只验证当前片段与源文档的一致性
2. 不要假设其他片段的内容（每个片段独立验证）
3. 检查代码示例是否存在于源文档中
4. 检查函数名、类名、参数名是否准确
5. 检查技术细节是否与源文档匹配

输出格式：
- 如果内容完全基于源文档，回答 'VALID'
- 如果发现幻觉，回答 'INVALID' 并列出幻觉，每个幻觉格式为：`- 幻觉: 描述`

注意：
- 最多检查前 5 个源文档
- 如果源文档不足以验证某个陈述，明确说明"""

        user_message = f"""源文档内容:
{sources_formatted}

待验证的片段内容:
{fragment_content}

请验证生成的片段是否与源文档一致："""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    def parse_verification_result(
        self,
        response_text: str
    ) -> Tuple[bool, List[str]]:
        """
        解析验证结果

        Args:
            response_text: LLM 响应文本

        Returns:
            (is_valid, hallucinations)
        """
        response_text = response_text.strip()

        if response_text.startswith("VALID"):
            return True, []

        elif response_text.startswith("INVALID"):
            hallucinations = []
            lines = response_text.split('\n')

            for line in lines[1:]:
                line = line.strip()
                if line.startswith('- ') or line.startswith('• '):
                    hallucination = line[2:].strip()
                    if hallucination:
                        hallucinations.append(hallucination)

            return False, hallucinations

        else:
            logger.warning(f"Cannot parse verification result: {response_text[:100]}")
            return True, []

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

        retrieved_docs = state.get("last_retrieved_docs", [])
        if isinstance(retrieved_docs, list):
            max_docs = scope.get("max_retrieval_docs", 3)
            filtered_state["last_retrieved_docs"] = retrieved_docs[:max_docs]
        else:
            filtered_state["last_retrieved_docs"] = []

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
