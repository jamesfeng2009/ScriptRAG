"""技能管理器 - 管理生成风格模式

本模块定义 RAG 剧本生成系统的技能系统。
技能是生成风格模式，用于调整剧本片段的生成方式。
支持分层认知框架和信号路由系统。
"""

import logging
from typing import Dict, List, Set, Optional, Literal, Any
from collections import deque
from enum import Enum
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class SkillLayer(Enum):
    """技能层次枚举

    Layer 1 (FOUNDATION): 基础层 - 标准格式、代码示例
    Layer 2 (SEMANTIC): 语义层 - 概念解释、原理说明
    Layer 3 (DOMAIN): 领域层 - 特定场景、专业领域
    """
    FOUNDATION = "foundation"
    SEMANTIC = "semantic"
    DOMAIN = "domain"


class SkillTrigger(BaseModel):
    """技能触发器配置"""
    type: Literal["marker", "keyword", "complexity", "domain"] = Field(..., description="触发器类型")
    value: str = Field(..., description="触发器值")
    priority: int = Field(default=10, description="优先级，数值越小优先级越高")


class SkillConfig(BaseModel):
    """单个技能模式的配置"""
    description: str = Field(..., description="技能用途的描述")
    tone: str = Field(..., description="技能的语气/风格")
    layer: SkillLayer = Field(default=SkillLayer.FOUNDATION, description="技能所属层次")
    triggers: List[SkillTrigger] = Field(default_factory=list, description="触发器列表")
    compatible_with: List[str] = Field(default_factory=list, description="兼容的技能名称列表")
    prompt_config: Dict[str, Any] = Field(default_factory=dict, description="LLM 生成配置")


class LayeredSkillGroup(BaseModel):
    """分层技能组配置"""
    layer: SkillLayer = Field(..., description="层次")
    description: str = Field(..., description="层次描述")
    compatible_layers: List[SkillLayer] = Field(default_factory=list, description="兼容的其他层次")


class RetrievalConfig(BaseModel):
    """RAG 检索策略配置"""

    class VectorSearchConfig(BaseModel):
        """向量搜索配置"""
        top_k: int = Field(default=5, description="返回的顶部结果数量")
        similarity_threshold: float = Field(default=0.7, description="最小相似度分数")
        embedding_model: str = Field(default="text-embedding-3-large", description="使用的嵌入模型")

    class KeywordSearchConfig(BaseModel):
        """关键词搜索配置"""
        markers: List[str] = Field(
            default_factory=lambda: ["@deprecated", "FIXME", "TODO", "Security", "WARNING", "HACK"],
            description="要搜索的敏感标记"
        )
        boost_factor: float = Field(default=1.5, description="关键词匹配的增强因子")

    class HybridMergeConfig(BaseModel):
        """混合搜索合并配置"""
        vector_weight: float = Field(default=0.6, description="向量搜索结果的权重")
        keyword_weight: float = Field(default=0.4, description="关键词搜索结果的权重")
        keyword_boost_factor: float = Field(default=1.5, description="敏感标记命中的增强因子")
        dedup_threshold: float = Field(default=0.9, description="去重相似度阈值")

    class SummarizationConfig(BaseModel):
        """摘要配置"""
        max_tokens: int = Field(default=10000, description="摘要前的最大 token 数")
        chunk_size: int = Field(default=2000, description="处理的分块大小")
        overlap: int = Field(default=200, description="分块之间的重叠量")

    vector_search: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    keyword_search: KeywordSearchConfig = Field(default_factory=KeywordSearchConfig)
    hybrid_merge: HybridMergeConfig = Field(default_factory=HybridMergeConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)


LAYERED_SKILL_GROUPS: Dict[SkillLayer, LayeredSkillGroup] = {
    SkillLayer.FOUNDATION: LayeredSkillGroup(
        layer=SkillLayer.FOUNDATION,
        description="基础层：标准格式、代码示例、教程结构",
        compatible_layers=[SkillLayer.SEMANTIC, SkillLayer.DOMAIN]
    ),
    SkillLayer.SEMANTIC: LayeredSkillGroup(
        layer=SkillLayer.SEMANTIC,
        description="语义层：概念解释、原理说明、类比分析",
        compatible_layers=[SkillLayer.FOUNDATION, SkillLayer.DOMAIN]
    ),
    SkillLayer.DOMAIN: LayeredSkillGroup(
        layer=SkillLayer.DOMAIN,
        description="领域层：特定场景、专业领域、API文档",
        compatible_layers=[SkillLayer.FOUNDATION, SkillLayer.SEMANTIC]
    ),
}


SKILL_TRIGGERS: Dict[str, List[SkillTrigger]] = {
    "warning_mode": [
        SkillTrigger(type="marker", value="@deprecated", priority=1),
        SkillTrigger(type="marker", value="FIXME", priority=1),
        SkillTrigger(type="marker", value="Security", priority=1),
        SkillTrigger(type="marker", value="WARNING", priority=2),
    ],
    "research_mode": [
        SkillTrigger(type="marker", value="TODO", priority=1),
        SkillTrigger(type="keyword", value="如何设计", priority=5),
        SkillTrigger(type="keyword", value="架构", priority=5),
    ],
    "visualization_analogy": [
        SkillTrigger(type="complexity", value="high", priority=3),
        SkillTrigger(type="keyword", value="原理", priority=4),
        SkillTrigger(type="keyword", value="解释", priority=4),
    ],
    "api_documentation": [
        SkillTrigger(type="domain", value="api", priority=1),
        SkillTrigger(type="keyword", value="API", priority=2),
        SkillTrigger(type="keyword", value="接口", priority=2),
    ],
    "code_example": [
        SkillTrigger(type="keyword", value="代码", priority=3),
        SkillTrigger(type="keyword", value="示例", priority=3),
        SkillTrigger(type="complexity", value="low", priority=5),
    ],
}


SKILLS: Dict[str, SkillConfig] = {
    "standard_tutorial": SkillConfig(
        description="清晰、结构化的教程格式",
        tone="professional",
        layer=SkillLayer.FOUNDATION,
        triggers=[
            SkillTrigger(type="keyword", value="教程", priority=5),
            SkillTrigger(type="keyword", value="入门", priority=5),
        ],
        compatible_with=[
            "visualization_analogy",
            "warning_mode",
            "research_mode",
            "fallback_summary"
        ]
    ),
    "warning_mode": SkillConfig(
        description="突出显示废弃/风险内容",
        tone="cautionary",
        layer=SkillLayer.DOMAIN,
        triggers=[
            SkillTrigger(type="marker", value="@deprecated", priority=1),
            SkillTrigger(type="marker", value="FIXME", priority=1),
            SkillTrigger(type="marker", value="Security", priority=1),
            SkillTrigger(type="marker", value="WARNING", priority=2),
        ],
        compatible_with=[
            "standard_tutorial",
            "research_mode",
            "fallback_summary"
        ]
    ),
    "visualization_analogy": SkillConfig(
        description="使用类比和可视化解释复杂概念",
        tone="engaging",
        layer=SkillLayer.SEMANTIC,
        triggers=[
            SkillTrigger(type="complexity", value="high", priority=3),
            SkillTrigger(type="keyword", value="原理", priority=4),
            SkillTrigger(type="keyword", value="解释", priority=4),
            SkillTrigger(type="keyword", value="理解", priority=4),
        ],
        compatible_with=[
            "standard_tutorial",
            "meme_style",
            "research_mode"
        ]
    ),
    "research_mode": SkillConfig(
        description="承认信息缺口并建议研究方向",
        tone="exploratory",
        layer=SkillLayer.SEMANTIC,
        triggers=[
            SkillTrigger(type="marker", value="TODO", priority=1),
            SkillTrigger(type="keyword", value="设计", priority=5),
            SkillTrigger(type="keyword", value="架构", priority=5),
            SkillTrigger(type="keyword", value="最佳实践", priority=5),
        ],
        compatible_with=[
            "standard_tutorial",
            "warning_mode",
            "visualization_analogy",
            "fallback_summary"
        ]
    ),
    "meme_style": SkillConfig(
        description="轻松幽默的呈现方式",
        tone="casual",
        layer=SkillLayer.SEMANTIC,
        triggers=[
            SkillTrigger(type="keyword", value="轻松", priority=6),
            SkillTrigger(type="keyword", value="有趣", priority=6),
        ],
        compatible_with=[
            "visualization_analogy",
            "fallback_summary",
            "standard_tutorial"
        ]
    ),
    "fallback_summary": SkillConfig(
        description="详情不可用时的高层概述",
        tone="neutral",
        layer=SkillLayer.FOUNDATION,
        triggers=[
            SkillTrigger(type="complexity", value="medium", priority=4),
        ],
        compatible_with=[
            "standard_tutorial",
            "research_mode",
            "warning_mode",
            "meme_style"
        ]
    ),
    "code_example": SkillConfig(
        description="以代码示例为主体的教学模式",
        tone="technical",
        layer=SkillLayer.FOUNDATION,
        triggers=[
            SkillTrigger(type="keyword", value="代码", priority=3),
            SkillTrigger(type="keyword", value="示例", priority=3),
            SkillTrigger(type="keyword", value="实现", priority=4),
            SkillTrigger(type="complexity", value="low", priority=5),
        ],
        compatible_with=[
            "standard_tutorial",
            "visualization_analogy",
            "api_documentation"
        ]
    ),
    "api_documentation": SkillConfig(
        description="API接口文档生成模式",
        tone="technical",
        layer=SkillLayer.DOMAIN,
        triggers=[
            SkillTrigger(type="domain", value="api", priority=1),
            SkillTrigger(type="keyword", value="API", priority=2),
            SkillTrigger(type="keyword", value="接口", priority=2),
            SkillTrigger(type="keyword", value="端点", priority=3),
            SkillTrigger(type="keyword", value="参数", priority=3),
        ],
        compatible_with=[
            "standard_tutorial",
            "code_example",
            "warning_mode"
        ]
    ),
    "security_audit": SkillConfig(
        description="安全审计和风险评估模式",
        tone="cautionary",
        layer=SkillLayer.DOMAIN,
        triggers=[
            SkillTrigger(type="marker", value="Security", priority=1),
            SkillTrigger(type="marker", value="HACK", priority=1),
            SkillTrigger(type="keyword", value="安全", priority=2),
            SkillTrigger(type="keyword", value="漏洞", priority=2),
            SkillTrigger(type="keyword", value="权限", priority=3),
        ],
        compatible_with=[
            "warning_mode",
            "api_documentation",
            "standard_tutorial"
        ]
    ),
}


RETRIEVAL_CONFIG = RetrievalConfig()


def get_skills_by_layer(layer: SkillLayer) -> Dict[str, SkillConfig]:
    """获取指定层次的所有技能"""
    return {
        name: config
        for name, config in SKILLS.items()
        if config.layer == layer
    }


def get_skills_by_triggers(
    marker: Optional[str] = None,
    keyword: Optional[str] = None,
    complexity: Optional[str] = None,
    domain: Optional[str] = None
) -> List[str]:
    """根据触发条件查找匹配的技能"""
    matched_skills = []

    for skill_name, config in SKILLS.items():
        for trigger in config.triggers:
            if trigger.type == "marker" and marker and trigger.value in marker:
                matched_skills.append((skill_name, trigger.priority))
                continue
            if trigger.type == "keyword" and keyword and trigger.value in keyword:
                matched_skills.append((skill_name, trigger.priority))
                continue
            if trigger.type == "complexity" and complexity and trigger.value == complexity:
                matched_skills.append((skill_name, trigger.priority))
                continue
            if trigger.type == "domain" and domain and trigger.value == domain:
                matched_skills.append((skill_name, trigger.priority))
                continue

    matched_skills.sort(key=lambda x: x[1])
    return [skill for skill, _ in matched_skills]


def find_skill_path(
    current_skill: str,
    desired_skill: str,
    max_hops: int = 2
) -> Optional[List[str]]:
    """使用 BFS 查找从当前 Skill 到目标 Skill 的最短路径"""
    if current_skill == desired_skill:
        return [current_skill]

    if current_skill not in SKILLS or desired_skill not in SKILLS:
        return None

    queue = deque([(current_skill, [current_skill])])
    visited = {current_skill}

    while queue:
        skill, path = queue.popleft()

        if len(path) - 1 > max_hops:
            continue

        compatible = SKILLS[skill].compatible_with

        for next_skill in compatible:
            if next_skill == desired_skill:
                return path + [next_skill]

            if next_skill not in visited:
                visited.add(next_skill)
                queue.append((next_skill, path + [next_skill]))

    return None


def find_layer_transition_path(
    current_layer: SkillLayer,
    target_layer: SkillLayer
) -> Optional[List[SkillLayer]]:
    """查找从当前层次到目标层次的最短转换路径"""
    if current_layer == target_layer:
        return [current_layer]

    queue = deque([(current_layer, [current_layer])])
    visited = {current_layer}

    while queue:
        layer, path = queue.popleft()

        group = LAYERED_SKILL_GROUPS.get(layer)
        if not group:
            continue

        for compatible_layer in group.compatible_layers:
            if compatible_layer == target_layer:
                return path + [compatible_layer]

            if compatible_layer not in visited:
                visited.add(compatible_layer)
                queue.append((compatible_layer, path + [compatible_layer]))

    return None


def check_skill_compatibility(current_skill: str, target_skill: str) -> bool:
    """检查两个技能是否兼容切换"""
    if current_skill not in SKILLS:
        raise ValueError(f"Invalid current skill: {current_skill}")
    if target_skill not in SKILLS:
        raise ValueError(f"Invalid target skill: {target_skill}")

    if current_skill == target_skill:
        return True

    return target_skill in SKILLS[current_skill].compatible_with


def check_layer_compatibility(
    current_layer: SkillLayer,
    target_layer: SkillLayer
) -> bool:
    """检查两个层次是否兼容切换"""
    if current_layer == target_layer:
        return True

    group = LAYERED_SKILL_GROUPS.get(current_layer)
    if not group:
        return False

    return target_layer in group.compatible_layers


def get_compatible_skills(skill_name: str) -> List[str]:
    """获取与给定技能兼容的技能列表"""
    if skill_name not in SKILLS:
        raise ValueError(f"Invalid skill: {skill_name}")

    return SKILLS[skill_name].compatible_with.copy()


def get_compatible_layers(layer: SkillLayer) -> List[SkillLayer]:
    """获取与给定层次兼容的层次列表"""
    group = LAYERED_SKILL_GROUPS.get(layer)
    if not group:
        return []

    return group.compatible_layers.copy()


def find_closest_compatible_skill(
    current_skill: str,
    desired_skill: str,
    global_tone: Optional[str] = None,
    allow_multi_hop: bool = True
) -> str:
    """当无法直接切换时，找到最接近的兼容技能"""
    if current_skill not in SKILLS:
        raise ValueError(f"Invalid current skill: {current_skill}")
    if desired_skill not in SKILLS:
        raise ValueError(f"Invalid desired skill: {desired_skill}")

    if check_skill_compatibility(current_skill, desired_skill):
        return desired_skill

    if allow_multi_hop:
        path = find_skill_path(current_skill, desired_skill, max_hops=2)
        if path and len(path) > 1:
            return path[1]

    compatible = get_compatible_skills(current_skill)

    if not compatible:
        return current_skill

    if global_tone:
        tone_matches = [
            skill for skill in compatible
            if SKILLS[skill].tone == global_tone
        ]
        if tone_matches:
            return tone_matches[0]

    for skill in compatible:
        if check_skill_compatibility(skill, desired_skill):
            return skill

    return compatible[0]


class SkillManager:
    """用于动态技能加载和扩展的管理器"""

    def __init__(
        self,
        custom_skills: Optional[Dict[str, SkillConfig]] = None,
        config_path: Optional[str] = None,
        enable_hot_reload: bool = False
    ):
        """初始化技能管理器"""
        self._skills: Dict[str, SkillConfig] = SKILLS.copy()
        self._config_path = config_path
        self._hot_reload_enabled = enable_hot_reload
        self._loader = None

        if config_path:
            self.reload_from_config(config_path)

        if custom_skills:
            self.register_skills(custom_skills)

        if enable_hot_reload and config_path:
            self.enable_hot_reload()

    def register_skill(self, name: str, config: SkillConfig) -> None:
        """Register a new skill or update an existing one"""
        for compatible_skill in config.compatible_with:
            if compatible_skill not in self._skills and compatible_skill != name:
                raise ValueError(
                    f"Compatible skill '{compatible_skill}' not found in registered skills"
                )

        self._skills[name] = config

    def register_skills(self, skills: Dict[str, SkillConfig]) -> None:
        """一次性注册多个技能"""
        for name, config in skills.items():
            self.register_skill(name, config)

    def get_skill(self, name: str) -> SkillConfig:
        """根据名称获取技能配置"""
        if name not in self._skills:
            raise ValueError(f"Skill not found: {name}")
        return self._skills[name]

    def list_skills(self) -> List[str]:
        """Get list of all registered skill names"""
        return list(self._skills.keys())

    def list_skills_by_layer(self, layer: SkillLayer) -> List[str]:
        """获取指定层次的所有技能名称"""
        return [
            name for name, config in self._skills.items()
            if config.layer == layer
        ]

    def get_skill_layers(self) -> List[SkillLayer]:
        """获取所有可用的技能层次"""
        return list(LAYERED_SKILL_GROUPS.keys())

    def check_compatibility(self, current_skill: str, target_skill: str) -> bool:
        """检查两个技能是否兼容"""
        if current_skill not in self._skills:
            raise ValueError(f"Invalid current skill: {current_skill}")
        if target_skill not in self._skills:
            raise ValueError(f"Invalid target skill: {target_skill}")

        if current_skill == target_skill:
            return True

        return target_skill in self._skills[current_skill].compatible_with

    def get_compatible_skills(self, skill_name: str) -> List[str]:
        """获取兼容技能列表"""
        if skill_name not in self._skills:
            raise ValueError(f"Invalid skill: {skill_name}")

        return self._skills[skill_name].compatible_with.copy()

    def find_compatible_skill(
        self,
        current_skill: str,
        desired_skill: str,
        global_tone: Optional[str] = None,
        allow_multi_hop: bool = True
    ) -> str:
        """找到切换时最接近的兼容技能"""
        if current_skill not in self._skills:
            raise ValueError(f"Invalid current skill: {current_skill}")
        if desired_skill not in self._skills:
            raise ValueError(f"Invalid desired skill: {desired_skill}")

        if self.check_compatibility(current_skill, desired_skill):
            return desired_skill

        if allow_multi_hop:
            path = find_skill_path(current_skill, desired_skill, max_hops=2)
            if path and len(path) > 1:
                return path[1]

        compatible = self.get_compatible_skills(current_skill)

        if not compatible:
            return current_skill

        if global_tone:
            tone_matches = [
                skill for skill in compatible
                if self._skills[skill].tone == global_tone
            ]
            if tone_matches:
                return tone_matches[0]

        for skill in compatible:
            if self.check_compatibility(skill, desired_skill):
                return skill

        return compatible[0]

    def get_skill_by_tone(self, tone: str) -> List[str]:
        """Get all skills with a specific tone"""
        return [
            name for name, config in self._skills.items()
            if config.tone == tone
        ]

    def find_skills_by_trigger(
        self,
        marker: Optional[str] = None,
        keyword: Optional[str] = None,
        complexity: Optional[str] = None,
        domain: Optional[str] = None
    ) -> List[str]:
        """根据触发条件查找匹配的技能"""
        return get_skills_by_triggers(marker, keyword, complexity, domain)

    def validate_skill_graph(self) -> bool:
        """验证技能兼容性图是否格式正确"""
        for skill_name, config in self._skills.items():
            for compatible_skill in config.compatible_with:
                if compatible_skill not in self._skills:
                    return False
        return True

    def reload_from_config(self, config_path: str):
        """Reload skills from configuration file"""
        from .skill_loader import SkillConfigLoader

        loader = SkillConfigLoader(config_path)
        new_skills = loader.load_from_yaml()

        self._skills = new_skills
        self._config_path = config_path

        logger.info(f"Reloaded {len(new_skills)} skills from configuration")

    def enable_hot_reload(self):
        """Enable hot-reloading of configuration file"""
        if not self._config_path:
            raise ValueError("Cannot enable hot-reload without config_path")

        from .skill_loader import SkillConfigLoader

        if self._loader is None:
            self._loader = SkillConfigLoader(self._config_path)
            self._loader.register_reload_callback(self._on_config_reload)
            self._loader.start_watching()
            self._hot_reload_enabled = True
            logger.info("Hot-reload enabled for skills configuration")
        else:
            logger.warning("Hot-reload already enabled")

    def disable_hot_reload(self):
        """禁用配置文件的热重载"""
        if self._loader is not None:
            self._loader.stop_watching()
            self._loader = None
            self._hot_reload_enabled = False
            logger.info("Hot-reload disabled for skills configuration")

    def _on_config_reload(self, new_skills: Dict[str, SkillConfig]):
        """Callback when configuration is reloaded"""
        logger.info(f"Configuration reloaded with {len(new_skills)} skills")
        self._skills = new_skills

    def export_to_config(self, output_path: str):
        """将当前技能导出到配置文件"""
        from .skill_loader import SkillConfigLoader
        from pathlib import Path

        loader = SkillConfigLoader()
        loader.export_to_yaml(self._skills, Path(output_path))
        logger.info(f"Exported skills to: {output_path}")

    def get_config_path(self) -> Optional[str]:
        """Get the current configuration file path"""
        return self._config_path

    def is_hot_reload_enabled(self) -> bool:
        """Check if hot-reload is enabled"""
        return self._hot_reload_enabled


default_skill_manager = SkillManager()
