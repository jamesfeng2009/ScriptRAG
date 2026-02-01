"""Enhanced Multi-Agent System for RAG-Based Screenplay Generation

This module implements an advanced multi-agent system that can dynamically adjust
screenplay direction based on RAG content analysis and skill content matching.

Architecture:
    1. RAGContentAnalyzer - Analyzes retrieved content semantically
    2. DynamicDirector - Makes direction adjustment decisions
    3. SkillRecommender - Recommends skills based on content analysis
    4. StructuredScreenplayWriter - Generates structured screenplay format

Key Features:
    - Semantic analysis of RAG content for direction detection
    - Dynamic skill switching based on content characteristics
    - Structured screenplay output (scenes, characters, dialogue)
    - Real-time direction adjustment during generation
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json

from ...services.llm.service import LLMService
from ...domain.models import SharedState, RetrievedDocument, OutlineStep
from ...infrastructure.logging import get_agent_logger


logger = logging.getLogger(__name__)
agent_logger = get_agent_logger(__name__)


class ContentType(Enum):
    """内容类型枚举 - 用于分类RAG检索到的内容"""
    TUTORIAL = "tutorial"
    API_REFERENCE = "api_reference"
    EXAMPLE_CODE = "example_code"
    BEST_PRACTICE = "best_practice"
    TROUBLESHOOTING = "troubleshooting"
    CONCEPT_EXPLANATION = "concept_explanation"
    WARNING_NOTICE = "warning_notice"
    DEPRECATED_CONTENT = "deprecated_content"
    SECURITY_NOTICE = "security_notice"


class DirectionAdjustmentType(Enum):
    """方向调整类型枚举"""
    NO_CHANGE = "no_change"
    SKILL_SWITCH = "skill_switch"
    OUTLINE_REPLAN = "outline_replan"
    EMPHASIS_SHIFT = "emphasis_shift"
    TOPIC_BRANCH = "topic_branch"
    WARNING_INSERT = "warning_insert"


class ToneStyle(Enum):
    """语调风格枚举"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    HUMOROUS = "humorous"
    ACADEMIC = "academic"
    BEGINNER_FRIENDLY = "beginner_friendly"


@dataclass
class ContentAnalysis:
    """内容分析结果"""
    content_types: List[ContentType]
    main_topic: str
    sub_topics: List[str]
    difficulty_level: float  # 0.0 - 1.0
    tone_style: ToneStyle
    key_concepts: List[str]
    warnings: List[str]
    prerequisites: List[str]
    suggested_skill: Optional[str] = None
    confidence: float = 0.0


@dataclass
class DirectionAdjustment:
    """方向调整决策"""
    adjustment_type: DirectionAdjustmentType
    reason: str
    from_skill: Optional[str] = None
    to_skill: Optional[str] = None
    new_outline_steps: List[Dict[str, Any]] = field(default_factory=list)
    emphasis_notes: List[str] = field(default_factory=list)
    warnings_to_insert: List[str] = field(default_factory=list)
    branch_topic: Optional[str] = None
    confidence: float = 0.0


@dataclass
class SceneInfo:
    """场景信息"""
    scene_id: str
    scene_type: str  # "dialogue", "narration", "action", "code_demo"
    setting: str
    characters: List[str]
    content: str
    code_snippets: List[str] = field(default_factory=list)
    visual_notes: List[str] = field(default_factory=list)


class RAGContentAnalyzer:
    """RAG内容分析器 - 语义分析检索到的内容
    
    职责：
    1. 分类内容类型（教程/API/示例/最佳实践等）
    2. 提取主题和关键概念
    3. 评估难度级别
    4. 检测警告和注意事项
    5. 推荐合适的Skill
    """
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        logger.info("RAGContentAnalyzer initialized")
    
    async def analyze(
        self,
        query: str,
        retrieved_docs: List[RetrievedDocument]
    ) -> ContentAnalysis:
        """分析检索到的内容
        
        Args:
            query: 原始查询
            retrieved_docs: 检索到的文档列表
            
        Returns:
            ContentAnalysis: 包含分析结果的对象
        """
        logger.info(f"Analyzing {len(retrieved_docs)} retrieved documents")
        
        if not retrieved_docs:
            return ContentAnalysis(
                content_types=[ContentType.CONCEPT_EXPLANATION],
                main_topic=query,
                sub_topics=[],
                difficulty_level=0.5,
                tone_style=ToneStyle.PROFESSIONAL,
                key_concepts=[],
                warnings=[],
                prerequisites=[],
                suggested_skill="standard_tutorial",
                confidence=0.5
            )
        
        # 构建内容摘要
        content_summary = self._build_content_summary(retrieved_docs)
        
        # 调用LLM进行深度分析
        analysis_prompt = self._build_analysis_prompt(query, content_summary)
        
        try:
            response = await self.llm_service.chat_completion(
                messages=analysis_prompt,
                task_type="high_performance",
                temperature=0.3,
                max_tokens=1500
            )
            
            analysis_result = self._parse_analysis_response(response)
            
            # 添加日志记录
            agent_logger.log_rag_analysis(
                query=query,
                doc_count=len(retrieved_docs),
                content_types=analysis_result.content_types,
                difficulty=analysis_result.difficulty_level,
                suggested_skill=analysis_result.suggested_skill
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return self._fallback_analysis(query, retrieved_docs)
    
    def _build_content_summary(self, docs: List[RetrievedDocument]) -> str:
        """构建内容摘要用于分析"""
        summaries = []
        for i, doc in enumerate(docs[:5]):  # 最多使用5个文档
            summary = f"""
文档 {i+1} ({doc.source}):
- 置信度: {doc.confidence:.2f}
- 内容摘要: {doc.content[:500]}
- 元数据: {json.dumps(doc.metadata, ensure_ascii=False)}
"""
            summaries.append(summary)
        return "\n".join(summaries)
    
    def _build_analysis_prompt(
        self,
        query: str,
        content_summary: str
    ) -> List[Dict[str, str]]:
        """构建分析提示"""
        return [
            {
                "role": "system",
                "content": """你是一个内容分析专家。请分析给定的技术文档内容，返回结构化的分析结果。

分析维度：
1. 内容类型（可多选）：tutorial, api_reference, example_code, best_practice, troubleshooting, concept_explanation, warning_notice, deprecated_content, security_notice
2. 主话题：用一句话概括主要内容
3. 子话题列表：列出涉及的具体主题
4. 难度级别：0.0（非常简单）到 1.0（极其复杂）
5. 推荐语调：professional, casual, humorous, academic, beginner_friendly
6. 关键概念：列出核心概念术语
7. 警告信息：列出需要注意的警告
8. 前置知识：列出理解内容需要的前提知识
9. 推荐Skill：基于内容特点推荐最合适的Skill（standard_tutorial, warning_mode, visualization_analogy, research_mode, meme_style）
10. 置信度：你对分析的信心程度（0.0-1.0）

请以JSON格式返回结果，格式如下：
```json
{
    "content_types": ["tutorial", "example_code"],
    "main_topic": "如何异步编程",
    "sub_topics": ["async/await", "任务创建", "错误处理"],
    "difficulty_level": 0.65,
    "tone_style": "professional",
    "key_concepts": ["协程", "事件循环", "Future"],
    "warnings": ["注意版本兼容性"],
    "prerequisites": ["Python基础", "同步编程理解"],
    "suggested_skill": "visualization_analogy",
    "confidence": 0.85
}
```"""
            },
            {
                "role": "user",
                "content": f"""原始查询: {query}

检索到的文档内容:
{content_summary}

请分析以上内容，返回结构化的分析结果。"""
            }
        ]
    
    def _parse_analysis_response(self, response: str) -> ContentAnalysis:
        """解析LLM返回的分析结果"""
        try:
            # 提取JSON部分
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
            
            # 转换内容类型
            content_types = []
            for ct in data.get("content_types", []):
                try:
                    content_types.append(ContentType(ct))
                except ValueError:
                    pass
            
            # 转换语调风格
            try:
                tone_style = ToneStyle(data.get("tone_style", "professional"))
            except ValueError:
                tone_style = ToneStyle.PROFESSIONAL
            
            return ContentAnalysis(
                content_types=content_types or [ContentType.CONCEPT_EXPLANATION],
                main_topic=data.get("main_topic", ""),
                sub_topics=data.get("sub_topics", []),
                difficulty_level=data.get("difficulty_level", 0.5),
                tone_style=tone_style,
                key_concepts=data.get("key_concepts", []),
                warnings=data.get("warnings", []),
                prerequisites=data.get("prerequisites", []),
                suggested_skill=data.get("suggested_skill"),
                confidence=data.get("confidence", 0.7)
            )
        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            return ContentAnalysis(
                content_types=[ContentType.CONCEPT_EXPLANATION],
                main_topic="Content analysis failed",
                sub_topics=[],
                difficulty_level=0.5,
                tone_style=ToneStyle.PROFESSIONAL,
                key_concepts=[],
                warnings=[],
                prerequisites=[],
                suggested_skill="standard_tutorial",
                confidence=0.3
            )
    
    def _fallback_analysis(
        self,
        query: str,
        docs: List[RetrievedDocument]
    ) -> ContentAnalysis:
        """回退分析（当LLM调用失败时）"""
        content_types = []
        warnings = []
        difficulty = 0.5
        
        for doc in docs:
            # 基于元数据推断
            if doc.metadata.get("has_deprecated"):
                content_types.append(ContentType.DEPRECATED_CONTENT)
            if doc.metadata.get("has_security"):
                content_types.append(ContentType.SECURITY_NOTICE)
            if doc.metadata.get("has_fixme"):
                content_types.append(ContentType.TROUBLESHOOTING)
            
            # 基于内容长度估算难度
            difficulty = min(0.5 + len(doc.content) / 10000, 0.9)
        
        if not content_types:
            content_types = [ContentType.CONCEPT_EXPLANATION]
        
        return ContentAnalysis(
            content_types=content_types,
            main_topic=query,
            sub_topics=[],
            difficulty_level=difficulty,
            tone_style=ToneStyle.PROFESSIONAL,
            key_concepts=[],
            warnings=warnings,
            prerequisites=[],
            suggested_skill="standard_tutorial",
            confidence=0.4
        )


class DynamicDirector:
    """动态导演 - 基于内容分析做出方向调整决策
    
    职责：
    1. 接收RAGContentAnalyzer的分析结果
    2. 决定是否需要调整方向
    3. 生成具体的调整策略
    4. 与SkillRecommender协作
    """
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        logger.info("DynamicDirector initialized")
    
    async def evaluate_and_adjust(
        self,
        state: SharedState,
        content_analysis: ContentAnalysis
    ) -> Tuple[SharedState, DirectionAdjustment]:
        """评估内容并做出方向调整决策
        
        Args:
            state: 当前共享状态
            content_analysis: RAG内容分析结果
            
        Returns:
            Tuple[SharedState, DirectionAdjustment]: 更新后的状态和调整决策
        """
        logger.info(f"DynamicDirector: Evaluating content analysis, confidence={content_analysis.confidence}")
        
        current_step = state.get_current_step()
        if not current_step:
            return state, DirectionAdjustment(
                adjustment_type=DirectionAdjustmentType.NO_CHANGE,
                reason="No current step",
                confidence=1.0
            )
        
        # 决策1：是否需要Skill切换
        skill_adjustment = self._evaluate_skill_switch(state, content_analysis)
        
        # 决策2：是否需要大纲调整
        outline_adjustment = self._evaluate_outline_change(state, content_analysis)
        
        # 决策3：是否需要强调重点转移
        emphasis_adjustment = self._evaluate_emphasis_shift(state, content_analysis)
        
        # 综合决策：选择最重要的调整
        adjustment = self._select_primary_adjustment(
            skill_adjustment,
            outline_adjustment,
            emphasis_adjustment,
            content_analysis
        )
        
        # 应用调整
        if adjustment.adjustment_type != DirectionAdjustmentType.NO_CHANGE:
            state = self._apply_adjustment(state, adjustment, current_step)
        
        # 记录决策日志
        agent_logger.log_direction_decision(
            step_id=current_step.step_id,
            adjustment_type=adjustment.adjustment_type.value,
            reason=adjustment.reason,
            confidence=adjustment.confidence,
            content_analysis_summary={
                "content_types": [ct.value for ct in content_analysis.content_types],
                "difficulty": content_analysis.difficulty_level,
                "suggested_skill": content_analysis.suggested_skill
            }
        )
        
        return state, adjustment
    
    def _evaluate_skill_switch(
        self,
        state: SharedState,
        analysis: ContentAnalysis
    ) -> DirectionAdjustment:
        """评估是否需要Skill切换"""
        current_skill = state.current_skill
        suggested_skill = analysis.suggested_skill
        
        if not suggested_skill or suggested_skill == current_skill:
            return DirectionAdjustment(
                adjustment_type=DirectionAdjustmentType.NO_CHANGE,
                reason="No skill switch needed",
                confidence=analysis.confidence
            )
        
        # 评估切换的必要性
        switch_reasons = []
        
        # 基于内容类型判断
        if ContentType.WARNING_NOTICE in analysis.content_types:
            switch_reasons.append("warning_content_detected")
        if ContentType.DEPRECATED_CONTENT in analysis.content_types:
            switch_reasons.append("deprecated_content_detected")
        if ContentType.SECURITY_NOTICE in analysis.content_types:
            switch_reasons.append("security_content_detected")
        if analysis.difficulty_level > 0.7:
            switch_reasons.append("high_complexity")
        if ContentType.TROUBLESHOOTING in analysis.content_types:
            switch_reasons.append("troubleshooting_content")
        
        if switch_reasons:
            return DirectionAdjustment(
                adjustment_type=DirectionAdjustmentType.SKILL_SWITCH,
                reason=f"Content analysis suggests skill switch: {', '.join(switch_reasons)}",
                from_skill=current_skill,
                to_skill=suggested_skill,
                confidence=analysis.confidence
            )
        
        return DirectionAdjustment(
            adjustment_type=DirectionAdjustmentType.NO_CHANGE,
            reason="Content analysis does not warrant skill switch",
            confidence=analysis.confidence
        )
    
    def _evaluate_outline_change(
        self,
        state: SharedState,
        analysis: ContentAnalysis
    ) -> DirectionAdjustment:
        """评估是否需要大纲重新规划"""
        # 检测主题偏离
        current_step = state.get_current_step()
        if not current_step:
            return DirectionAdjustment(
                adjustment_type=DirectionAdjustmentType.NO_CHANGE,
                reason="No current step",
                confidence=1.0
            )
        
        # 检查是否检测到分支主题
        if len(analysis.sub_topics) > 3:
            return DirectionAdjustment(
                adjustment_type=DirectionAdjustmentType.TOPIC_BRANCH,
                reason=f"Multiple sub-topics detected: {analysis.sub_topics}",
                branch_topic=analysis.sub_topics[0],
                new_outline_steps=[
                    {"description": f"主要主题: {analysis.main_topic}", "status": "in_progress"},
                    {"description": f"分支主题: {analysis.sub_topics[0]}", "status": "pending"},
                    {"description": f"分支主题: {analysis.sub_topics[1]}", "status": "pending"}
                ] if analysis.sub_topics else [],
                confidence=analysis.confidence * 0.8
            )
        
        # 检查是否需要插入警告
        if analysis.warnings:
            return DirectionAdjustment(
                adjustment_type=DirectionAdjustmentType.WARNING_INSERT,
                reason=f"Warnings detected: {analysis.warnings}",
                warnings_to_insert=analysis.warnings,
                confidence=analysis.confidence
            )
        
        return DirectionAdjustment(
            adjustment_type=DirectionAdjustmentType.NO_CHANGE,
            reason="No outline change needed",
            confidence=analysis.confidence
        )
    
    def _evaluate_emphasis_shift(
        self,
        state: SharedState,
        analysis: ContentAnalysis
    ) -> DirectionAdjustment:
        """评估是否需要重点转移"""
        emphasis_notes = []
        
        # 基于前置知识强调
        if analysis.prerequisites:
            emphasis_notes.append(f"确保读者理解: {', '.join(analysis.prerequisites)}")
        
        # 基于关键概念强调
        if analysis.key_concepts:
            emphasis_notes.append(f"重点解释概念: {', '.join(analysis.key_concepts[:3])}")
        
        # 基于难度调整
        if analysis.difficulty_level > 0.7:
            emphasis_notes.append("高难度内容，需要更多示例和解释")
        elif analysis.difficulty_level < 0.3:
            emphasis_notes.append("低难度内容，可以简化说明")
        
        if emphasis_notes:
            return DirectionAdjustment(
                adjustment_type=DirectionAdjustmentType.EMPHASIS_SHIFT,
                reason="Emphasis shift based on content analysis",
                emphasis_notes=emphasis_notes,
                confidence=analysis.confidence
            )
        
        return DirectionAdjustment(
            adjustment_type=DirectionAdjustmentType.NO_CHANGE,
            reason="No emphasis shift needed",
            confidence=analysis.confidence
        )
    
    def _select_primary_adjustment(
        self,
        skill_adj: DirectionAdjustment,
        outline_adj: DirectionAdjustment,
        emphasis_adj: DirectionAdjustment,
        analysis: ContentAnalysis
    ) -> DirectionAdjustment:
        """选择最重要的调整类型"""
        adjustments = [
            (skill_adj, 1.0 if skill_adj.confidence > 0.7 else 0.5),
            (outline_adj, 0.8 if outline_adj.confidence > 0.6 else 0.4),
            (emphasis_adj, 0.6 if emphasis_adj.confidence > 0.5 else 0.3)
        ]
        
        # 按优先级和置信度排序
        adjustments.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)
        
        primary = adjustments[0][0]
        
        if primary.adjustment_type == DirectionAdjustmentType.NO_CHANGE:
            return emphasis_adj  # 返回强调转移作为备选
        
        return primary
    
    def _apply_adjustment(
        self,
        state: SharedState,
        adjustment: DirectionAdjustment,
        current_step: OutlineStep
    ) -> SharedState:
        """应用方向调整"""
        if adjustment.adjustment_type == DirectionAdjustmentType.SKILL_SWITCH:
            if adjustment.to_skill and adjustment.to_skill != state.current_skill:
                state.switch_skill(
                    new_skill=adjustment.to_skill,
                    reason=adjustment.reason,
                    step_id=current_step.step_id
                )
                logger.info(f"DynamicDirector: Switched skill to {adjustment.to_skill}")
        
        elif adjustment.adjustment_type == DirectionAdjustmentType.WARNING_INSERT:
            for warning in adjustment.warnings_to_insert:
                if warning not in current_step.description:
                    current_step.description = f"[WARNING] {warning}\n\n{current_step.description}"
        
        elif adjustment.adjustment_type == DirectionAdjustmentType.EMPHASIS_SHIFT:
            for note in adjustment.emphasis_notes:
                state.add_log_entry(
                    agent_name="dynamic_director",
                    action="emphasis_shift",
                    details={"note": note, "step_id": current_step.step_id}
                )
        
        return state


class SkillRecommender:
    """Skill推荐器 - 基于上下文推荐最合适的Skill
    
    职责：
    1. 分析当前上下文（用户主题、检索内容）
    2. 推荐最适合的Skill
    3. 提供备选Skill列表
    4. 解释推荐原因
    """
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        logger.info("SkillRecommender initialized")
    
    async def recommend(
        self,
        topic: str,
        context: str,
        content_analysis: Optional[ContentAnalysis] = None
    ) -> Dict[str, Any]:
        """推荐最合适的Skill
        
        Args:
            topic: 用户主题
            context: 项目上下文
            content_analysis: 可选的内容分析结果
            
        Returns:
            Dict包含推荐结果和解释
        """
        logger.info(f"SkillRecommender: Analyzing topic={topic}")
        
        # 构建推荐提示
        recommend_prompt = self._build_recommend_prompt(topic, context, content_analysis)
        
        try:
            response = await self.llm_service.chat_completion(
                messages=recommend_prompt,
                task_type="high_performance",
                temperature=0.3,
                max_tokens=800
            )
            
            result = self._parse_recommend_response(response)
            
            # 记录推荐日志
            agent_logger.log_skill_recommendation(
                topic=topic,
                recommended_skill=result["recommended_skill"],
                confidence=result["confidence"],
                reasoning=result["reasoning"]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Skill recommendation failed: {e}")
            return self._fallback_recommendation(topic)
    
    def _build_recommend_prompt(
        self,
        topic: str,
        context: str,
        content_analysis: Optional[ContentAnalysis]
    ) -> List[Dict[str, str]]:
        """构建推荐提示"""
        analysis_info = ""
        if content_analysis:
            analysis_info = f"""
内容分析结果:
- 内容类型: {[ct.value for ct in content_analysis.content_types]}
- 难度级别: {content_analysis.difficulty_level}
- 关键概念: {content_analysis.key_concepts}
- 警告信息: {content_analysis.warnings}
"""
        
        return [
            {
                "role": "system",
                "content": """你是一个Skill推荐专家。根据用户主题和内容分析，推荐最合适的Skill。

可用的Skill:
1. standard_tutorial - 标准教程风格，适合大多数技术讲解
2. warning_mode - 警告模式，适合讲解危险操作或注意事项
3. visualization_analogy - 可视化类比，适合复杂概念的解释
4. research_mode - 研究模式，适合深入探索未知领域
5. meme_style - 表情包风格，适合轻松有趣的内容

推荐格式（JSON）：
```json
{
    "recommended_skill": "visualization_analogy",
    "confidence": 0.85,
    "reasoning": "内容涉及复杂概念，需要使用类比来帮助理解",
    "alternatives": ["standard_tutorial", "research_mode"]
}
```"""
            },
            {
                "role": "user",
                "content": f"""用户主题: {topic}

项目上下文: {context}

{analysis_info}

请推荐最合适的Skill。"""
            }
        ]
    
    def _parse_recommend_response(self, response: str) -> Dict[str, Any]:
        """解析推荐响应"""
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
            
            return {
                "recommended_skill": data.get("recommended_skill", "standard_tutorial"),
                "confidence": data.get("confidence", 0.5),
                "reasoning": data.get("reasoning", ""),
                "alternatives": data.get("alternatives", ["standard_tutorial"])
            }
        except Exception as e:
            logger.error(f"Failed to parse recommendation: {e}")
            return self._fallback_recommendation("")
    
    def _fallback_recommendation(self, topic: str) -> Dict[str, Any]:
        """回退推荐"""
        topic_lower = topic.lower()
        
        if "warning" in topic_lower or "安全" in topic_lower or "security" in topic_lower:
            return {
                "recommended_skill": "warning_mode",
                "confidence": 0.7,
                "reasoning": "基于主题关键词推断需要警告模式",
                "alternatives": ["standard_tutorial"]
            }
        if "async" in topic_lower or "并发" in topic_lower or "parallel" in topic_lower:
            return {
                "recommended_skill": "visualization_analogy",
                "confidence": 0.6,
                "reasoning": "基于主题关键词推断需要可视化类比",
                "alternatives": ["standard_tutorial", "research_mode"]
            }
        
        return {
            "recommended_skill": "standard_tutorial",
            "confidence": 0.5,
            "reasoning": "使用默认推荐",
            "alternatives": ["warning_mode", "visualization_analogy"]
        }


class StructuredScreenplayWriter:
    """结构化剧本 writer - 生成有结构的剧本内容
    
    输出格式：
    - 场景（Scene）：场景描述
    - 角色（Character）：说话者
    - 对话（Dialogue）：台词
    - 代码示例（Code）：代码块
    - 视觉效果说明（Visual）：建议的视觉效果呈现
    """
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        logger.info("StructuredScreenplayWriter initialized")
    
    async def write_scene(
        self,
        step_description: str,
        retrieved_docs: List[RetrievedDocument],
        current_skill: str,
        emphasis_notes: List[str] = None
    ) -> SceneInfo:
        """生成单个场景的内容
        
        Args:
            step_description: 步骤描述
            retrieved_docs: 检索到的文档
            current_skill: 当前使用的Skill
            emphasis_notes: 重点强调说明
            
        Returns:
            SceneInfo: 包含场景信息的对象
        """
        logger.info(f"StructuredScreenplayWriter: Writing scene for step")
        
        # 构建场景生成提示
        scene_prompt = self._build_scene_prompt(
            step_description,
            retrieved_docs,
            current_skill,
            emphasis_notes
        )
        
        try:
            response = await self.llm_service.chat_completion(
                messages=scene_prompt,
                task_type="creative",
                temperature=0.7,
                max_tokens=2000
            )
            
            scene_info = self._parse_scene_response(response, step_description)
            
            return scene_info
            
        except Exception as e:
            logger.error(f"Scene writing failed: {e}")
            return self._fallback_scene(step_description, retrieved_docs, current_skill)
    
    def _build_scene_prompt(
        self,
        step_description: str,
        docs: List[RetrievedDocument],
        skill: str,
        emphasis_notes: List[str]
    ) -> List[Dict[str, str]]:
        """构建场景生成提示"""
        docs_content = "\n".join([
            f"[{doc.source}]: {doc.content[:300]}..."
            for doc in docs[:3]
        ])
        
        emphasis = ""
        if emphasis_notes:
            emphasis = f"\n重点强调: {'; '.join(emphasis_notes)}"
        
        skill_instructions = {
            "standard_tutorial": "使用清晰的教学语言，逐步引导读者理解。",
            "warning_mode": "强调安全注意事项，使用警告式语言。",
            "visualization_analogy": "使用类比和可视化方式解释概念。",
            "research_mode": "深入探索，提供详细的技术背景。",
            "meme_style": "使用轻松幽默的语言，添加趣味元素。"
        }
        
        instruction = skill_instructions.get(skill, skill_instructions["standard_tutorial"])
        
        return [
            {
                "role": "system",
                "content": f"""你是一个技术剧本编剧。根据给定的步骤描述和参考资料，生成结构化的剧本内容。

当前Skill: {skill}
Skill指导: {instruction}{emphasis}

输出格式要求（JSON）：
```json
{{
    "scene_type": "dialogue",  // 或 narration, action, code_demo
    "setting": "场景设置描述",
    "characters": ["角色1", "角色2"],
    "content": "完整的剧本内容（对话或叙述）",
    "code_snippets": ["代码块1", "代码块2"],
    "visual_notes": ["视觉效果建议1", "视觉效果建议2"]
}}
```

剧本应该:
1. 结构清晰，易于理解
2. 包含适当的对话和叙述
3. 融入代码示例（如有必要）
4. 包含视觉效果建议
5. 符合当前Skill的风格要求"""
            },
            {
                "role": "user",
                "content": f"""步骤描述: {step_description}

参考资料:
{docs_content}

请生成结构化的剧本内容。"""
            }
        ]
    
    def _parse_scene_response(self, response: str, step_description: str) -> SceneInfo:
        """解析场景响应"""
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
            
            return SceneInfo(
                scene_id=f"scene_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                scene_type=data.get("scene_type", "narration"),
                setting=data.get("setting", ""),
                characters=data.get("characters", []),
                content=data.get("content", ""),
                code_snippets=data.get("code_snippets", []),
                visual_notes=data.get("visual_notes", [])
            )
        except Exception as e:
            logger.error(f"Failed to parse scene response: {e}")
            return self._fallback_scene(step_description, [], "standard_tutorial")
    
    def _fallback_scene(
        self,
        step_description: str,
        docs: List[RetrievedDocument],
        skill: str
    ) -> SceneInfo:
        """回退场景生成"""
        content_parts = [step_description]
        
        for doc in docs[:2]:
            content_parts.append(f"\n参考: {doc.content[:200]}...")
        
        return SceneInfo(
            scene_id=f"scene_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            scene_type="narration",
            setting="技术讲解场景",
            characters=[],
            content="\n".join(content_parts),
            code_snippets=[],
            visual_notes=[]
        )
    
    def compile_screenplay(self, scenes: List[SceneInfo]) -> str:
        """编译最终剧本"""
        screenplay_parts = ["# 技术剧本\n"]
        
        for i, scene in enumerate(scenes, 1):
            screenplay_parts.append(f"\n## 场景 {i}: {scene.scene_type.upper()}")
            screenplay_parts.append(f"\n**场景设置**: {scene.setting}")
            
            if scene.characters:
                screenplay_parts.append(f"\n**角色**: {', '.join(scene.characters)}")
            
            screenplay_parts.append(f"\n**内容**:\n{scene.content}")
            
            if scene.code_snippets:
                screenplay_parts.append("\n**代码示例**:")
                for j, code in enumerate(scene.code_snippets, 1):
                    screenplay_parts.append(f"\n```python\n# 代码块 {j}\n{code}\n```")
            
            if scene.visual_notes:
                screenplay_parts.append(f"\n**视觉效果**:")
                for note in scene.visual_notes:
                    screenplay_parts.append(f"- {note}")
        
        return "\n".join(screenplay_parts)
