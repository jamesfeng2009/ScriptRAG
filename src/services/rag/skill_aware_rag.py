"""增强型 RAG 服务 - 集成技能路由

本模块提供与技能路由系统集成的增强 RAG 服务：
1. 基于查询内容的自动技能选择
2. 技能感知的上下文构建
3. 技能切换日志记录
4. 使用统计和监控
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from src.domain.skills import (
    SkillConfig,
    SKILLS,
)
from src.domain.skill_router import ContentAnalysis
from src.services.persistence.skill_routing_service import (
    SkillRoutingService,
    ContextEnricher,
    create_routing_service,
    default_routing_service,
    default_enricher,
)
from src.services.rag.rag_service import RAGService, QueryResult
from src.services.retrieval.strategies import RetrievalResult


logger = logging.getLogger(__name__)


@dataclass
class EnhancedQueryResult:
    """增强型问答结果"""
    answer: str
    sources: List[Dict[str, Any]]
    skill: str
    skill_confidence: float
    content_analysis: Dict[str, Any]
    generation_time: float
    routing_time: float


class SkillAwareRAGService:
    """技能感知 RAG 服务

    继承自 RAGService，集成技能路由功能：
    1. 自动根据查询选择最佳技能
    2. 使用技能特定的提示词生成回答
    3. 记录技能使用统计
    """

    def __init__(
        self,
        rag_service: RAGService = None,
        routing_service: SkillRoutingService = None,
        enricher: ContextEnricher = None,
        enable_skill_routing: bool = True,
        auto_switch_skill: bool = True,
        session_id: str = "default"
    ):
        """初始化技能感知 RAG 服务

        Args:
            rag_service: 基础 RAG 服务
            routing_service: 技能路由服务
            enricher: 上下文增强器
            enable_skill_routing: 是否启用技能路由
            auto_switch_skill: 是否自动切换技能
            session_id: 会话 ID
        """
        self.rag_service = rag_service or RAGService()
        self.routing_service = routing_service or default_routing_service
        self.enricher = enricher or default_enricher
        self.enable_skill_routing = enable_skill_routing
        self.auto_switch_skill = auto_switch_skill
        self.session_id = session_id

        self._current_skill: Optional[str] = "standard_tutorial"
        self._current_skill_config: Optional[SkillConfig] = None

        logger.info(
            f"SkillAwareRAGService initialized "
            f"(routing={enable_skill_routing}, auto_switch={auto_switch_skill})"
        )

    async def initialize(self) -> None:
        """初始化服务"""
        await self.rag_service.initialize()
        logger.info("SkillAwareRAGService initialized")

    async def close(self) -> None:
        """关闭服务"""
        await self.rag_service.close()
        logger.info("SkillAwareRAGService closed")

    async def query(
        self,
        question: str,
        history: List[Dict[str, str]] = None,
        force_skill: Optional[str] = None,
        track_stats: bool = True
    ) -> EnhancedQueryResult:
        """
        增强型问答查询

        Args:
            question: 用户问题
            history: 对话历史
            force_skill: 强制使用的技能
            track_stats: 是否跟踪统计

        Returns:
            EnhancedQueryResult: 增强型问答结果
        """
        start_time = time.time()
        routing_start = start_time

        routing_context = self.enricher.enrich(question, history)
        routing_context.session_id = self.session_id
        routing_context.previous_skill = self._current_skill if self.auto_switch_skill else None

        if self.enable_skill_routing and not force_skill:
            routing_result = self.routing_service.select_skill(
                context=routing_context,
                force_skill=force_skill
            )

            self._current_skill = routing_result.selected_skill
            self._current_skill_config = SKILLS.get(self._current_skill)

            if routing_result.transition and self.enable_skill_routing:
                self.routing_service.record_transition(routing_result.transition)

        elif force_skill and force_skill in SKILLS:
            self._current_skill = force_skill
            self._current_skill_config = SKILLS.get(force_skill)

        routing_time = time.time() - routing_start

        logger.info(
            f"Selected skill: {self._current_skill} "
            f"(confidence: {self._current_skill_config.tone if self._current_skill_config else 'N/A'})"
        )

        content_analysis = self._analyze_content_for_prompt(question, history)

        answer, sources, generation_time = await self._generate_with_skill(
            question=question,
            history=history,
            content_analysis=content_analysis
        )

        total_time = time.time() - start_time

        result = EnhancedQueryResult(
            answer=answer,
            sources=sources,
            skill=self._current_skill,
            skill_confidence=routing_result.confidence if self.enable_skill_routing else 1.0,
            content_analysis=asdict(content_analysis) if content_analysis else {},
            generation_time=generation_time,
            routing_time=routing_time
        )

        if track_stats and self.enable_skill_routing:
            self._update_stats(question, result)

        return result

    def _analyze_content_for_prompt(
        self,
        question: str,
        history: List[Dict[str, str]] = None
    ) -> ContentAnalysis:
        """分析内容以构建提示词"""
        return self.enricher.enrich(question, history)

    async def _generate_with_skill(
        self,
        question: str,
        history: List[Dict[str, str]] = None,
        content_analysis: ContentAnalysis = None
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """使用当前技能生成回答"""
        start_time = time.time()

        processed_query = self.rag_service._preprocess_query(question, history)

        query_emb = await self.rag_service.llm_service.embed([processed_query])
        query_emb = query_emb[0]

        results = await self.rag_service._vector_search(query_emb)

        if not results:
            return "未找到相关信息", [], time.time() - start_time

        reranked = self.rag_service._rerank(processed_query, results)

        context = self._build_skill_aware_context(reranked, content_analysis)

        prompt = self._build_skill_prompt(
            question=question,
            context=context,
            skill_config=self._current_skill_config,
            analysis=content_analysis
        )

        answer = await self.rag_service._generate(question, prompt)

        sources = [
            {
                "text": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                "score": round(r.metadata.get('rerank_score', r.similarity), 4),
                "file_path": r.file_path
            }
            for r in reranked[:self.rag_service.rerank_top_k]
        ]

        generation_time = time.time() - start_time

        logger.info(
            f"Generation complete with skill '{self._current_skill}' "
            f"(time: {generation_time:.2f}s)"
        )

        return answer, sources, generation_time

    def _build_skill_aware_context(
        self,
        results: List[RetrievalResult],
        analysis: ContentAnalysis = None
    ) -> str:
        """构建技能感知的上下文"""
        parts = []

        markers = analysis.markers if analysis else []
        if markers:
            parts.append(f"[重要标记]: {', '.join(markers)}")

        for i, r in enumerate(results):
            content = r.content

            if markers:
                for marker in markers:
                    if marker.lower() in content.lower():
                        content = f"[⚠️ {marker}] {content}"
                        break

            parts.append(f"[来源{i+1}] {content}")

        return "\n\n".join(parts)

    def _build_skill_prompt(
        self,
        question: str,
        context: str,
        skill_config: SkillConfig = None,
        analysis: ContentAnalysis = None
    ) -> str:
        """构建基于技能的提示词"""
        if skill_config:
            system_prompt = self._get_system_prompt_for_skill(skill_config)
        else:
            system_prompt = self._get_default_system_prompt()

        markers = analysis.markers if analysis else []
        tone_instructions = self._get_tone_instructions(skill_config)

        prompt = f"""{system_prompt}

## 语气要求
{tone_instructions}

## 特殊标记处理
{"注意：内容中包含以下重要标记: " + ", ".join(markers) if markers else "正常处理"}

## 参考信息
{context}

## 用户问题
{question}

## 回答
"""

        return prompt

    def _get_system_prompt_for_skill(self, skill_config: SkillConfig) -> str:
        """获取技能特定的系统提示词"""
        base_prompt = "你是一个专业的信息检索助手。"

        tone_prompts = {
            "professional": "你是一个专业的技术教程编写专家，使用清晰、正式的语调。",
            "technical": "你是一个技术专家，使用精确、专业的语言。",
            "cautionary": "你是一个技术风险评估专家，专注于识别和强调潜在问题。",
            "exploratory": "你是一个研究顾问，善于探索和承认知识边界。",
            "engaging": "你是一个善于用生动方式解释概念的教育专家。",
            "casual": "你是一个轻松幽默的技术内容创作者。",
            "neutral": "你是一个客观中立的助手，提供简洁的事实性回答。",
        }

        return tone_prompts.get(skill_config.tone, base_prompt)

    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return """你是一个专业的信息检索助手。请严格遵循以下规则：

1. 仅基于参考信息回答，不要添加外部知识
2. 如果参考信息不足以回答问题，明确说"未找到相关信息"
3. 回答要简洁准确，不要冗长
4. 所有陈述必须有来源支持"""

    def _get_tone_instructions(self, skill_config: SkillConfig = None) -> str:
        """获取语气指令"""
        if not skill_config:
            return "使用专业、客观的语调。"

        tone_instructions = {
            "professional": "- 使用正式、专业的语调\n- 结构清晰，层次分明\n- 提供详细的步骤说明",
            "technical": "- 使用精确的技术语言\n- 包含具体的参数和配置\n- 注重准确性和完整性",
            "cautionary": "- 强调潜在风险和问题\n- 提供安全建议和最佳实践\n- 使用警示性语言",
            "exploratory": "- 承认信息边界\n- 建议可能的研究方向\n- 保持开放和探索的态度",
            "engaging": "- 使用生动的语言和类比\n- 让复杂概念易于理解\n- 适当使用例子说明",
            "casual": "- 使用轻松、友好的语调\n- 可以使用日常用语\n- 让内容有趣且易于记忆",
            "neutral": "- 使用客观中立的语调\n- 提供简洁的事实性信息\n- 避免主观判断",
        }

        return tone_instructions.get(
            skill_config.tone,
            "使用专业、客观的语调。"
        )

    def _update_stats(
        self,
        question: str,
        result: EnhancedQueryResult
    ) -> None:
        """更新统计信息"""
        try:
            stats = {
                "timestamp": datetime.now().isoformat(),
                "question_length": len(question),
                "answer_length": len(result.answer),
                "skill": result.skill,
                "confidence": result.skill_confidence,
                "routing_time": result.routing_time,
                "generation_time": result.generation_time,
                "sources_count": len(result.sources),
                "has_markers": len(result.content_analysis.get("markers", [])) > 0,
            }

            logger.info(f"Skill usage stats: {json.dumps(stats, ensure_ascii=False)}")

        except Exception as e:
            logger.warning(f"Failed to update stats: {e}")

    def get_current_skill(self) -> Optional[str]:
        """获取当前技能"""
        return self._current_skill

    def get_current_skill_config(self) -> Optional[SkillConfig]:
        """获取当前技能配置"""
        return self._current_skill_config

    def switch_skill(self, skill_name: str) -> bool:
        """切换技能

        Args:
            skill_name: 技能名称

        Returns:
            是否切换成功
        """
        if skill_name not in SKILLS:
            logger.warning(f"Unknown skill: {skill_name}")
            return False

        self._current_skill = skill_name
        self._current_skill_config = SKILLS[skill_name]

        logger.info(f"Switched to skill: {skill_name}")
        return True

    def get_skill_info(self) -> Dict[str, Any]:
        """获取当前技能信息"""
        if not self._current_skill_config:
            return {"error": "No skill selected"}

        config = self._current_skill_config

        return {
            "skill": self._current_skill,
            "description": config.description,
            "tone": config.tone,
            "layer": config.layer.value,
            "compatible_with": config.compatible_with,
            "triggers": [
                {"type": t.type, "value": t.value, "priority": t.priority}
                for t in config.triggers
            ] if config.triggers else [],
        }

    async def get_usage_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        return self.routing_service.get_usage_statistics()

    async def get_transition_history(
        self,
        skill: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取技能切换历史"""
        return self.routing_service.get_transition_history(skill, limit)


class EnhancedRAGFactory:
    """增强型 RAG 服务工厂"""

    @staticmethod
    async def create(
        workspace_id: str = "default",
        enable_skill_routing: bool = True,
        routing_strategy: str = "auto",
        session_id: str = "default"
    ) -> SkillAwareRAGService:
        """创建增强型 RAG 服务"""
        from src.services.rag.rag_service import create_rag_service

        rag_service = await create_rag_service(workspace_id)

        routing_service = create_routing_service(
            strategy=routing_strategy,
            config={"enable_logging": True, "session_id": session_id}
        )

        return SkillAwareRAGService(
            rag_service=rag_service,
            routing_service=routing_service,
            enable_skill_routing=enable_skill_routing,
            session_id=session_id
        )

    @staticmethod
    def create_sync(
        workspace_id: str = "default",
        enable_skill_routing: bool = True,
        routing_strategy: str = "auto",
        session_id: str = "default"
    ) -> SkillAwareRAGService:
        """同步创建增强型 RAG 服务"""
        rag_service = RAGService(workspace_id=workspace_id)

        routing_service = create_routing_service(
            strategy=routing_strategy,
            config={"enable_logging": True, "session_id": session_id}
        )

        return SkillAwareRAGService(
            rag_service=rag_service,
            routing_service=routing_service,
            enable_skill_routing=enable_skill_routing,
            session_id=session_id
        )


async def create_enhanced_rag_service(
    workspace_id: str = "default",
    enable_skill_routing: bool = True,
    routing_strategy: str = "auto"
) -> SkillAwareRAGService:
    """工厂函数：创建增强型 RAG 服务"""
    return await EnhancedRAGFactory.create(
        workspace_id=workspace_id,
        enable_skill_routing=enable_skill_routing,
        routing_strategy=routing_strategy
    )
