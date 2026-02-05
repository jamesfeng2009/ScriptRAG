"""技能路由器 - 基于信号的智能路由系统

本模块提供基于多维度信号的智能技能路由功能。
支持标记信号、关键词信号、复杂度信号和领域信号的组合匹配。
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .skills import (
    SkillLayer,
    SkillConfig,
    SKILLS,
    LAYERED_SKILL_GROUPS,
    check_skill_compatibility,
    find_skill_path,
    find_layer_transition_path,
    check_layer_compatibility,
)


class SignalType(Enum):
    """信号类型枚举"""
    MARKER = "marker"
    KEYWORD = "keyword"
    COMPLEXITY = "complexity"
    DOMAIN = "domain"


@dataclass
class RoutingSignal:
    """路由信号数据类"""
    type: SignalType
    value: str
    priority: int
    matched_skill: str


@dataclass
class ContentAnalysis:
    """内容分析结果"""
    markers: List[str]
    keywords: List[str]
    complexity: str
    domain: str
    raw_text: str


class SignalRouter:
    """基于信号的技能路由器"""

    DEFAULT_MARKERS = [
        "@deprecated",
        "FIXME",
        "TODO",
        "Security",
        "WARNING",
        "HACK",
        "BUG",
        "NOTE",
        "IMPORTANT",
    ]

    COMPLEXITY_KEYWORDS = {
        "high": [
            "复杂", "高级", "深入", "底层", "原理",
            "architecture", "design", "implementation",
        ],
        "medium": [
            "中等", "进阶", "应用", "实践",
            "usage", "application", "practice",
        ],
        "low": [
            "简单", "基础", "入门", "初级",
            "basic", "beginner", "simple", "hello",
        ],
    }

    DOMAIN_KEYWORDS = {
        "api": ["API", "接口", "端点", "endpoint", "rest", "http"],
        "database": ["数据库", "DB", "sql", "query", "table"],
        "frontend": ["前端", "frontend", "react", "vue", "html", "css"],
        "backend": ["后端", "backend", "server", "api"],
        "security": ["安全", "security", "权限", "auth", "token"],
        "testing": ["测试", "test", "unit", "integration"],
        "devops": ["部署", "docker", "kubernetes", "ci/cd", "pipeline"],
        "ml": ["机器学习", "ML", "AI", "模型", "训练"],
        "fintech": ["金融", "fintech", "交易", "支付", "风控"],
    }

    def __init__(self):
        self._routing_rules: Dict[str, List[Dict]] = {}
        self._marker_patterns = self._compile_marker_patterns()

    def _compile_marker_patterns(self) -> Dict[str, re.Pattern]:
        """编译标记正则表达式"""
        patterns = {}
        for marker in self.DEFAULT_MARKERS:
            patterns[marker] = re.compile(re.escape(marker), re.IGNORECASE)
        return patterns

    def analyze_content(self, text: str) -> ContentAnalysis:
        """分析文本内容，提取各类信号"""
        markers = self._extract_markers(text)
        keywords = self._extract_keywords(text)
        complexity = self._detect_complexity(text)
        domain = self._detect_domain(text)

        return ContentAnalysis(
            markers=markers,
            keywords=keywords,
            complexity=complexity,
            domain=domain,
            raw_text=text,
        )

    def _extract_markers(self, text: str) -> List[str]:
        """提取标记信号"""
        found = []
        for marker, pattern in self._marker_patterns.items():
            if pattern.search(text):
                found.append(marker)
        return found

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词信号"""
        found = []
        text_lower = text.lower()
        for keyword in self.DEFAULT_MARKERS:
            if keyword.lower() in text_lower:
                found.append(keyword)
        return found

    def _detect_complexity(self, text: str) -> str:
        """检测复杂度信号"""
        text_lower = text.lower()

        for complexity, keywords in self.COMPLEXITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return complexity

        return "medium"

    def _detect_domain(self, text: str) -> str:
        """检测领域信号"""
        text_lower = text.lower()

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return domain

        return ""

    def route(
        self,
        content: str,
        current_skill: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[RoutingSignal]:
        """根据内容信号路由到匹配的技能

        Args:
            content: 待分析的文本内容
            current_skill: 当前使用的技能（用于兼容性检查）
            context: 额外的上下文信息

        Returns:
            按优先级排序的匹配技能列表
        """
        analysis = self.analyze_content(content)
        signals = []

        for skill_name, config in SKILLS.items():
            priority = self._calculate_skill_priority(skill_name, analysis)

            if priority > 0:
                signal = RoutingSignal(
                    type=self._get_signal_type(skill_name),
                    value=self._get_matched_value(skill_name, analysis),
                    priority=priority,
                    matched_skill=skill_name,
                )
                signals.append(signal)

        signals.sort(key=lambda x: x.priority)

        if current_skill:
            filtered_signals = self._filter_by_compatibility(
                signals, current_skill
            )
            if filtered_signals:
                return filtered_signals

        return signals

    def _calculate_skill_priority(
        self,
        skill_name: str,
        analysis: ContentAnalysis
    ) -> int:
        """计算技能匹配优先级"""
        config = SKILLS.get(skill_name)
        if not config:
            return 0

        if not config.triggers:
            return 10

        highest_priority = 100

        for trigger in config.triggers:
            trigger_priority = self._match_trigger(trigger, analysis)
            if trigger_priority > 0:
                if trigger.priority < highest_priority:
                    highest_priority = trigger.priority

        return highest_priority if highest_priority < 100 else 0

    def _match_trigger(self, trigger: Any, analysis: ContentAnalysis) -> int:
        """匹配单个触发器"""
        trigger_type = trigger.type
        trigger_value = trigger.value

        if trigger_type == "marker":
            for marker in analysis.markers:
                if trigger_value in marker or marker in trigger_value:
                    return trigger.priority

        elif trigger_type == "keyword":
            text_lower = analysis.raw_text.lower()
            trigger_lower = trigger_value.lower()
            if trigger_lower in text_lower:
                return trigger.priority

        elif trigger_type == "complexity":
            if trigger_value == analysis.complexity:
                return trigger.priority

        elif trigger_type == "domain":
            if trigger_value == analysis.domain:
                return trigger.priority

        return 0

    def _get_signal_type(self, skill_name: str) -> SignalType:
        """获取技能的主要信号类型"""
        config = SKILLS.get(skill_name)
        if not config or not config.triggers:
            return SignalType.KEYWORD

        trigger_types = [t.type for t in config.triggers]

        if "marker" in trigger_types:
            return SignalType.MARKER
        elif "domain" in trigger_types:
            return SignalType.DOMAIN
        elif "complexity" in trigger_types:
            return SignalType.COMPLEXITY
        else:
            return SignalType.KEYWORD

    def _get_matched_value(
        self,
        skill_name: str,
        analysis: ContentAnalysis
    ) -> str:
        """获取匹配的值"""
        config = SKILLS.get(skill_name)
        if not config or not config.triggers:
            return ""

        for trigger in config.triggers:
            if self._match_trigger(trigger, analysis) > 0:
                return trigger.value

        return ""

    def _filter_by_compatibility(
        self,
        signals: List[RoutingSignal],
        current_skill: str
    ) -> List[RoutingSignal]:
        """过滤出与当前技能兼容的匹配"""
        compatible_signals = []

        for signal in signals:
            if signal.matched_skill == current_skill:
                continue

            if check_skill_compatibility(current_skill, signal.matched_skill):
                compatible_signals.append(signal)
            else:
                path = find_skill_path(
                    current_skill,
                    signal.matched_skill,
                    max_hops=2
                )
                if path and len(path) > 0:
                    compatible_signals.append(signal)

        return compatible_signals

    def route_with_layer_awareness(
        self,
        content: str,
        target_layer: Optional[SkillLayer] = None,
        current_skill: Optional[str] = None
    ) -> List[RoutingSignal]:
        """支持层次感知的路由

        Args:
            content: 待分析的文本内容
            target_layer: 目标层次（可选）
            current_skill: 当前使用的技能

        Returns:
            按优先级排序的匹配技能列表
        """
        signals = self.route(content, current_skill)

        if target_layer:
            filtered_signals = [
                s for s in signals
                if SKILLS[s.matched_skill].layer == target_layer
            ]
            if filtered_signals:
                return filtered_signals

        return signals

    def get_recommended_skill(
        self,
        content: str,
        current_skill: Optional[str] = None,
        preferred_tone: Optional[str] = None,
        preferred_layer: Optional[SkillLayer] = None
    ) -> Optional[str]:
        """获取推荐的单个技能

        Args:
            content: 待分析的文本内容
            current_skill: 当前使用的技能
            preferred_tone: 偏好的语气
            preferred_layer: 偏好的层次

        Returns:
            推荐的技能名称，未找到返回 None
        """
        signals = self.route(content, current_skill)

        if not signals:
            return current_skill

        candidates = [s.matched_skill for s in signals]

        if preferred_tone:
            tone_candidates = [
                s for s in candidates
                if SKILLS[s].tone == preferred_tone
            ]
            if tone_candidates:
                candidates = tone_candidates

        if preferred_layer:
            layer_candidates = [
                s for s in candidates
                if SKILLS[s].layer == preferred_layer
            ]
            if layer_candidates:
                candidates = layer_candidates

        return candidates[0] if candidates else None


class SmartSkillRouter:
    """智能技能路由器 - 结合兼容性和信号路由"""

    def __init__(self):
        self._signal_router = SignalRouter()

    def select_skill(
        self,
        content: str,
        current_skill: Optional[str] = None,
        desired_skill: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """智能选择最佳技能

        Args:
            content: 待分析的文本内容
            current_skill: 当前使用的技能
            desired_skill: 用户期望的技能
            context: 额外上下文

        Returns:
            包含选择结果和建议的字典
        """
        result = {
            "selected_skill": current_skill or "standard_tutorial",
            "confidence": 0.0,
            "signals": [],
            "alternatives": [],
            "layer_transition": None,
            "reasoning": [],
        }

        analysis = self._signal_router.analyze_content(content)
        signals = self._signal_router.route(content, current_skill)
        result["signals"] = [
            {"type": s.type.value, "value": s.value, "skill": s.matched_skill}
            for s in signals
        ]

        if desired_skill and desired_skill in SKILLS:
            result = self._handle_desired_skill(
                result, desired_skill, current_skill, analysis
            )
        elif signals:
            result = self._handle_signal_based_selection(
                result, signals, current_skill, analysis
            )
        else:
            result["reasoning"].append("使用默认技能")

        if current_skill and result["selected_skill"] != current_skill:
            transition = self._calculate_layer_transition(
                current_skill, result["selected_skill"]
            )
            result["layer_transition"] = transition

        result["alternatives"] = self._get_alternatives(
            result["selected_skill"], signals
        )

        return result

    def _handle_desired_skill(
        self,
        result: Dict,
        desired_skill: str,
        current_skill: Optional[str],
        analysis: ContentAnalysis
    ) -> Dict:
        """处理用户期望的技能选择"""
        if current_skill and current_skill != desired_skill:
            if check_skill_compatibility(current_skill, desired_skill):
                result["selected_skill"] = desired_skill
                result["confidence"] = 0.9
                result["reasoning"].append(
                    f"直接切换到期望技能 '{desired_skill}'"
                )
            else:
                path = find_skill_path(current_skill, desired_skill, max_hops=2)
                if path and len(path) > 1:
                    next_skill = path[1]
                    result["selected_skill"] = next_skill
                    result["confidence"] = 0.7
                    result["reasoning"].append(
                        f"无法直接切换，通过 '{next_skill}' 过渡到 '{desired_skill}'"
                    )
                else:
                    closest = self._find_closest_skill(
                        current_skill, desired_skill, analysis
                    )
                    result["selected_skill"] = closest
                    result["confidence"] = 0.5
                    result["reasoning"].append(
                        f"使用最接近期望的技能 '{closest}'"
                    )
        else:
            result["selected_skill"] = desired_skill
            result["confidence"] = 0.95
            result["reasoning"].append(f"使用指定的技能 '{desired_skill}'")

        return result

    def _handle_signal_based_selection(
        self,
        result: Dict,
        signals: List[RoutingSignal],
        current_skill: Optional[str],
        analysis: ContentAnalysis
    ) -> Dict:
        """处理基于信号的技能选择"""
        best_signal = signals[0]
        result["selected_skill"] = best_signal.matched_skill
        result["confidence"] = 1.0 - (best_signal.priority / 100)
        result["reasoning"].append(
            f"根据信号 '{best_signal.value}' 推荐技能 '{best_signal.matched_skill}'"
        )

        if current_skill and current_skill != best_signal.matched_skill:
            if check_skill_compatibility(current_skill, best_signal.matched_skill):
                result["reasoning"].append("技能切换兼容")
            else:
                result["reasoning"].append("技能不兼容，建议通过兼容技能过渡")

        if analysis.markers:
            result["reasoning"].append(f"检测到标记: {', '.join(analysis.markers)}")

        return result

    def _find_closest_skill(
        self,
        current_skill: str,
        desired_skill: str,
        analysis: ContentAnalysis
    ) -> str:
        """查找最接近的技能"""
        compatible = SKILLS[current_skill].compatible_with

        for skill in compatible:
            if SKILLS[skill].tone == SKILLS[desired_skill].tone:
                return skill

        return compatible[0] if compatible else current_skill

    def _calculate_layer_transition(
        self,
        from_skill: str,
        to_skill: str
    ) -> Optional[Dict]:
        """计算层次转换"""
        from_layer = SKILLS[from_skill].layer
        to_layer = SKILLS[to_skill].layer

        if from_layer == to_layer:
            return None

        path = find_layer_transition_path(from_layer, to_layer)

        if path:
            return {
                "from": from_layer.value,
                "to": to_layer.value,
                "path": [layer.value for layer in path],
                "description": f"从 {from_layer.value} 层过渡到 {to_layer.value} 层",
            }

        return None

    def _get_alternatives(
        self,
        selected_skill: str,
        signals: List[RoutingSignal]
    ) -> List[Dict]:
        """获取备选技能"""
        alternatives = []
        selected_config = SKILLS[selected_skill]

        for skill_name in selected_config.compatible_with:
            alt_signals = [s for s in signals if s.matched_skill == skill_name]
            alternatives.append({
                "skill": skill_name,
                "reason": "兼容技能",
                "priority": alt_signals[0].priority if alt_signals else 50,
            })

        return alternatives[:3]


default_signal_router = SignalRouter()
default_smart_router = SmartSkillRouter()
