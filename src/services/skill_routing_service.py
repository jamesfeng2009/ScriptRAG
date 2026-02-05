"""技能路由服务 - RAG 集成的智能技能选择

本模块提供与 RAG 服务集成的技能路由功能：
1. 基于查询内容的智能技能选择
2. 技能切换历史跟踪
3. 与现有 RAG 流程的无缝集成
4. 技能使用统计和监控
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from collections import deque, defaultdict

from src.domain.skills import (
    SkillLayer,
    SkillConfig,
    SKILLS,
    default_skill_manager,
    check_skill_compatibility,
)
from src.domain.skill_router import (
    SignalRouter,
    SmartSkillRouter,
    ContentAnalysis,
    default_smart_router,
)


logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """路由策略"""
    AUTO = "auto"  # 完全自动选择
    SEMI_AUTO = "semi_auto"  # 自动推荐，用户确认
    MANUAL = "manual"  # 手动指定
    CONTEXT_AWARE = "context_aware"  # 上下文感知选择


@dataclass
class SkillTransition:
    """技能切换记录"""
    timestamp: str
    from_skill: Optional[str]
    to_skill: str
    reason: str
    confidence: float
    signals: List[Dict[str, str]]
    content_hash: str


@dataclass
class RoutingContext:
    """路由上下文"""
    query: str
    history: List[Dict[str, str]] = field(default_factory=list)
    user_preference: Optional[str] = None
    domain_hint: Optional[str] = None
    complexity_level: Optional[str] = None
    previous_skill: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class RoutingResult:
    """路由结果"""
    selected_skill: str
    confidence: float
    strategy: RoutingStrategy
    analysis: Dict[str, Any]
    alternatives: List[Dict[str, str]]
    transition: Optional[SkillTransition]
    recommendations: List[str]


class SkillRoutingService:
    """技能路由服务

    功能：
    - 基于查询内容自动选择最佳技能
    - 跟踪技能切换历史
    - 提供使用统计和监控
    - 支持多种路由策略
    """

    def __init__(
        self,
        router: SmartSkillRouter = None,
        strategy: RoutingStrategy = RoutingStrategy.AUTO,
        enable_logging: bool = True,
        max_history_size: int = 1000
    ):
        """初始化技能路由服务

        Args:
            router: 智能路由实例
            strategy: 路由策略
            enable_logging: 是否启用日志记录
            max_history_size: 最大历史记录数
        """
        self.router = router or default_smart_router
        self.strategy = strategy
        self.enable_logging = enable_logging
        self.max_history_size = max_history_size

        self._transition_history: deque = deque(maxlen=max_history_size)
        self._session_skills: Dict[str, List[str]] = defaultdict(list)
        self._skill_usage_count: Dict[str, int] = defaultdict(int)
        self._signal_effectiveness: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        logger.info(f"SkillRoutingService initialized with strategy: {strategy.value}")

    def analyze_query(self, query: str) -> ContentAnalysis:
        """分析查询内容"""
        return self.router._signal_router.analyze_content(query)

    def select_skill(
        self,
        context: RoutingContext,
        force_skill: Optional[str] = None
    ) -> RoutingResult:
        """选择最佳技能

        Args:
            context: 路由上下文
            force_skill: 强制使用的技能（覆盖自动选择）

        Returns:
            RoutingResult: 路由结果
        """
        start_time = time.time()

        if force_skill and force_skill in SKILLS:
            return self._handle_forced_skill(context, force_skill, start_time)

        if self.strategy == RoutingStrategy.MANUAL:
            return self._handle_manual_strategy(context, start_time)

        result = self.router.select_skill(
            content=context.query,
            current_skill=context.previous_skill,
            desired_skill=context.user_preference,
        )

        routing_result = RoutingResult(
            selected_skill=result["selected_skill"],
            confidence=result["confidence"],
            strategy=self.strategy,
            analysis={
                "content_analysis": asdict(self.analyze_query(context.query)),
                "signals": result["signals"],
                "reasoning": result["reasoning"],
                "layer_transition": result.get("layer_transition"),
            },
            alternatives=result.get("alternatives", []),
            transition=self._create_transition(context, result),
            recommendations=self._generate_recommendations(result),
        )

        elapsed_time = time.time() - start_time
        if self.enable_logging:
            self._log_routing_result(routing_result, elapsed_time)

        self._update_statistics(context, routing_result)

        return routing_result

    def _handle_forced_skill(
        self,
        context: RoutingContext,
        force_skill: str,
        start_time: float
    ) -> RoutingResult:
        """处理强制技能选择"""
        is_compatible = True
        reason = f"强制使用技能: {force_skill}"

        if context.previous_skill and context.previous_skill != force_skill:
            is_compatible = check_skill_compatibility(context.previous_skill, force_skill)
            reason = f"强制切换到 {force_skill}" if is_compatible else f"强制切换到 {force_skill}（不兼容）"

        return RoutingResult(
            selected_skill=force_skill,
            confidence=1.0 if is_compatible else 0.5,
            strategy=RoutingStrategy.MANUAL,
            analysis={
                "forced": True,
                "compatible": is_compatible,
                "reason": reason,
            },
            alternatives=[],
            transition=self._create_transition(
                context,
                {"selected_skill": force_skill, "confidence": 1.0, "signals": [], "reasoning": [reason]}
            ),
            recommendations=[],
        )

    def _handle_manual_strategy(
        self,
        context: RoutingContext,
        start_time: float
    ) -> RoutingResult:
        """处理手动策略"""
        default_skill = context.previous_skill or "standard_tutorial"

        return RoutingResult(
            selected_skill=default_skill,
            confidence=0.5,
            strategy=RoutingStrategy.MANUAL,
            analysis={
                "manual_strategy": True,
                "reason": "手动模式，使用默认或历史技能",
            },
            alternatives=[],
            transition=None,
            recommendations=[],
        )

    def _create_transition(
        self,
        context: RoutingContext,
        result: Dict[str, Any]
    ) -> Optional[SkillTransition]:
        """创建技能切换记录"""
        if context.previous_skill == result["selected_skill"]:
            return None

        return SkillTransition(
            timestamp=datetime.now().isoformat(),
            from_skill=context.previous_skill,
            to_skill=result["selected_skill"],
            reason=", ".join(result.get("reasoning", [])),
            confidence=result.get("confidence", 0.0),
            signals=result.get("signals", []),
            content_hash=str(hash(context.query)),
        )

    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        selected = result.get("selected_skill", "")

        confidence = result.get("confidence", 0.0)
        if confidence < 0.6:
            recommendations.append("置信度较低，建议人工审核")
        if confidence < 0.4:
            recommendations.append("考虑使用更通用的技能或手动选择")

        alternatives = result.get("alternatives", [])
        if len(alternatives) > 1:
            recommendations.append(f"可选方案: {', '.join(a['skill'] for a in alternatives[:2])}")

        return recommendations

    def _update_statistics(
        self,
        context: RoutingContext,
        result: RoutingResult
    ) -> None:
        """更新统计信息"""
        self._skill_usage_count[result.selected_skill] += 1

        if context.session_id:
            self._session_skills[context.session_id].append(result.selected_skill)

        for signal in result.analysis.get("signals", []):
            skill = signal.get("skill", "")
            signal_type = signal.get("type", "")
            if skill and signal_type:
                self._signal_effectiveness[skill][signal_type] += 1

    def _log_routing_result(
        self,
        result: RoutingResult,
        elapsed_time: float
    ) -> None:
        """记录路由结果"""
        logger.info(
            f"Skill routing: {result.selected_skill} "
            f"(confidence: {result.confidence:.2f}, "
            f"time: {elapsed_time*1000:.1f}ms)"
        )

        if result.transition:
            logger.info(
                f"Skill transition: {result.transition.from_skill} -> "
                f"{result.transition.to_skill} "
                f"(reason: {result.transition.reason})"
            )

    def get_transition_history(
        self,
        skill: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取切换历史"""
        history = list(self._transition_history)

        if skill:
            history = [
                t for t in history
                if t.to_skill == skill or t.from_skill == skill
            ]

        return [asdict(t) for t in history[-limit:]]

    def get_usage_statistics(self) -> Dict[str, Any]:
        """获取使用统计"""
        total = sum(self._skill_usage_count.values())

        return {
            "total_queries": total,
            "skill_distribution": dict(self._skill_usage_count),
            "layer_distribution": self._calculate_layer_distribution(),
            "signal_effectiveness": dict(self._signal_effectiveness),
            "average_confidence": self._calculate_average_confidence(),
        }

    def _calculate_layer_distribution(self) -> Dict[str, int]:
        """计算层次分布"""
        distribution = defaultdict(int)
        for skill in self._skill_usage_count:
            if skill in SKILLS:
                layer = SKILLS[skill].layer.value
                distribution[layer] += self._skill_usage_count[skill]
        return dict(distribution)

    def _calculate_average_confidence(self) -> float:
        """计算平均置信度"""
        if not self._transition_history:
            return 0.0
        confidences = [
            t.confidence for t in self._transition_history
            if t.confidence > 0
        ]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def record_transition(self, transition: SkillTransition) -> None:
        """记录技能切换"""
        self._transition_history.append(transition)

    def get_session_history(self, session_id: str) -> List[str]:
        """获取会话技能历史"""
        return self._session_skills.get(session_id, [])

    def clear_session(self, session_id: str) -> None:
        """清除会话数据"""
        if session_id in self._session_skills:
            del self._session_skills[session_id]


class ContextEnricher:
    """上下文增强器

    功能：
    - 从查询中提取隐含信息
    - 丰富路由上下文
    - 检测用户意图
    """

    COMPLEXITY_INDICATORS = {
        "high": ["复杂", "高级", "深入", "底层", "原理", "架构", "设计模式"],
        "low": ["简单", "基础", "入门", "初级", "快速", "示例"],
    }

    DOMAIN_INDICATORS = {
        "api": ["API", "接口", "端点", "请求", "响应", "HTTP", "REST"],
        "database": ["数据库", "SQL", "查询", "表", "索引", "ORM"],
        "frontend": ["前端", "UI", "组件", "页面", "样式", "React", "Vue"],
        "backend": ["后端", "服务", "认证", "权限", "中间件"],
        "security": ["安全", "加密", "认证", "授权", "漏洞"],
        "devops": ["部署", "Docker", "CI/CD", "Kubernetes", "监控"],
    }

    INTENT_PATTERNS = {
        "how_to": ["如何", "怎么", "怎样", "步骤", "教程"],
        "explanation": ["是什么", "为什么", "解释", "概念", "原理"],
        "troubleshooting": ["错误", "问题", "Bug", "修复", "解决"],
        "comparison": ["比较", "区别", "不同", "优缺点", "对比"],
        "recommendation": ["推荐", "最好", "建议", "应该", "选择"],
    }

    def enrich(self, query: str, history: List[Dict[str, str]] = None) -> RoutingContext:
        """丰富路由上下文

        Args:
            query: 用户查询
            history: 对话历史

        Returns:
            RoutingContext: 丰富后的上下文
        """
        context = RoutingContext(query=query, history=history or [])

        context.complexity_level = self._detect_complexity(query)
        context.domain_hint = self._detect_domain(query)
        context.user_preference = self._detect_intent(query)

        if history:
            context.previous_skill = self._extract_previous_skill(history)

        return context

    def _detect_complexity(self, query: str) -> Optional[str]:
        """检测复杂度"""
        query_lower = query.lower()
        for level, indicators in self.COMPLEXITY_INDICATORS.items():
            for indicator in indicators:
                if indicator.lower() in query_lower:
                    return level
        return None

    def _detect_domain(self, query: str) -> Optional[str]:
        """检测领域"""
        query_lower = query.lower()
        for domain, indicators in self.DOMAIN_INDICATORS.items():
            for indicator in indicators:
                if indicator.lower() in query_lower:
                    return domain
        return None

    def _detect_intent(self, query: str) -> Optional[str]:
        """检测意图"""
        query_lower = query.lower()
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in query_lower:
                    return intent
        return None

    def _extract_previous_skill(self, history: List[Dict[str, str]]) -> Optional[str]:
        """从历史中提取前一个技能"""
        for message in reversed(history):
            if message.get("role") == "assistant":
                metadata = message.get("metadata", {})
                return metadata.get("skill")
        return None


def create_routing_service(
    strategy: str = "auto",
    config: Dict[str, Any] = None
) -> SkillRoutingService:
    """创建路由服务工厂函数

    Args:
        strategy: 策略字符串 ("auto", "semi_auto", "manual")
        config: 配置参数

    Returns:
        SkillRoutingService: 配置好的路由服务
    """
    strategy_map = {
        "auto": RoutingStrategy.AUTO,
        "semi_auto": RoutingStrategy.SEMI_AUTO,
        "manual": RoutingStrategy.MANUAL,
        "context_aware": RoutingStrategy.CONTEXT_AWARE,
    }

    selected_strategy = strategy_map.get(strategy, RoutingStrategy.AUTO)

    return SkillRoutingService(
        strategy=selected_strategy,
        enable_logging=config.get("enable_logging", True) if config else True,
        max_history_size=config.get("max_history_size", 1000) if config else 1000,
    )


default_routing_service = SkillRoutingService()
default_enricher = ContextEnricher()
