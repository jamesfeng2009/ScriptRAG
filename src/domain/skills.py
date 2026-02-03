"""技能管理器 - 管理生成风格模式

本模块定义 RAG 剧本生成系统的技能系统。
技能是生成风格模式，用于调整剧本片段的生成方式。
"""

import logging
from typing import Dict, List, Set, Optional, Literal, Any
from collections import deque
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class SkillConfig(BaseModel):
    """单个技能模式的配置"""
    description: str = Field(..., description="技能用途的描述")
    tone: str = Field(..., description="技能的语气/风格")
    compatible_with: List[str] = Field(default_factory=list, description="兼容的技能名称列表")
    prompt_config: Dict[str, Any] = Field(default_factory=dict, description="LLM 生成配置")


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


# 技能配置字典
SKILLS: Dict[str, SkillConfig] = {
    "standard_tutorial": SkillConfig(
        description="清晰、结构化的教程格式",
        tone="professional",
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
        compatible_with=[
            "standard_tutorial",
            "research_mode",
            "fallback_summary"
        ]
    ),
    "visualization_analogy": SkillConfig(
        description="使用类比和可视化解释复杂概念",
        tone="engaging",
        compatible_with=[
            "standard_tutorial",
            "meme_style",
            "research_mode"
        ]
    ),
    "research_mode": SkillConfig(
        description="承认信息缺口并建议研究方向",
        tone="exploratory",
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
        compatible_with=[
            "visualization_analogy",
            "fallback_summary",
            "standard_tutorial"
        ]
    ),
    "fallback_summary": SkillConfig(
        description="详情不可用时的高层概述",
        tone="neutral",
        compatible_with=[
            "standard_tutorial",
            "research_mode",
            "warning_mode",
            "meme_style"
        ]
    )
}

# Retrieval configuration
RETRIEVAL_CONFIG = RetrievalConfig()


def find_skill_path(
    current_skill: str,
    desired_skill: str,
    max_hops: int = 2
) -> Optional[List[str]]:
    """
    使用 BFS 查找从当前 Skill 到目标 Skill 的最短路径
    
    Args:
        current_skill: 当前 Skill
        desired_skill: 目标 Skill
        max_hops: 最大跳转次数
        
    Returns:
        Skill 路径列表，如果找不到返回 None
        
    示例:
        find_skill_path("meme_style", "warning_mode")
        → ["meme_style", "visualization_analogy", "standard_tutorial", "warning_mode"]
    """
    if current_skill == desired_skill:
        return [current_skill]
    
    if current_skill not in SKILLS or desired_skill not in SKILLS:
        return None
    
    # BFS 查找
    queue = deque([(current_skill, [current_skill])])
    visited = {current_skill}
    
    while queue:
        skill, path = queue.popleft()
        
        # 检查是否超过最大跳转次数
        if len(path) - 1 > max_hops:
            continue
        
        # 获取兼容 Skills
        compatible = SKILLS[skill].compatible_with
        
        for next_skill in compatible:
            if next_skill == desired_skill:
                # 找到目标
                return path + [next_skill]
            
            if next_skill not in visited:
                visited.add(next_skill)
                queue.append((next_skill, path + [next_skill]))
    
    # 找不到路径
    return None


def check_skill_compatibility(current_skill: str, target_skill: str) -> bool:
    """检查两个技能是否兼容切换
    
    Args:
        current_skill: 当前活动技能名称
        target_skill: 要切换到的目标技能
        
    Returns:
        技能兼容返回 True，否则返回 False
        
    Raises:
        ValueError: 如果技能名称无效
    """
    if current_skill not in SKILLS:
        raise ValueError(f"Invalid current skill: {current_skill}")
    if target_skill not in SKILLS:
        raise ValueError(f"Invalid target skill: {target_skill}")
    
    # Same skill is always compatible
    if current_skill == target_skill:
        return True
    
    # Check if target skill is in current skill's compatible list
    return target_skill in SKILLS[current_skill].compatible_with


def get_compatible_skills(skill_name: str) -> List[str]:
    """获取与给定技能兼容的技能列表
    
    Args:
        skill_name: 技能名称
        
    Returns:
        兼容技能名称列表
        
    Raises:
        ValueError: 如果技能名称无效
    """
    if skill_name not in SKILLS:
        raise ValueError(f"Invalid skill: {skill_name}")
    
    return SKILLS[skill_name].compatible_with.copy()


def find_closest_compatible_skill(
    current_skill: str,
    desired_skill: str,
    global_tone: Optional[str] = None,
    allow_multi_hop: bool = True
) -> str:
    """当无法直接切换时，找到最接近的兼容技能
    
    Args:
        current_skill: 当前活动技能
        desired_skill: 期望的目标技能
        global_tone: 可选的全局语气偏好
        allow_multi_hop: 是否允许多跳路径
        
    Returns:
        最接近的兼容技能名称
        
    Raises:
        ValueError: 如果技能名称无效
    """
    if current_skill not in SKILLS:
        raise ValueError(f"Invalid current skill: {current_skill}")
    if desired_skill not in SKILLS:
        raise ValueError(f"Invalid desired skill: {desired_skill}")
    
    # If directly compatible, return desired skill
    if check_skill_compatibility(current_skill, desired_skill):
        return desired_skill
    
    # If multi-hop allowed, try to find a path
    if allow_multi_hop:
        path = find_skill_path(current_skill, desired_skill, max_hops=2)
        if path and len(path) > 1:
            return path[1]  # Return next step
    
    # Get compatible skills
    compatible = get_compatible_skills(current_skill)
    
    if not compatible:
        # No compatible skills, stay with current
        return current_skill
    
    # If global tone is specified, prefer skills with matching tone
    if global_tone:
        tone_matches = [
            skill for skill in compatible
            if SKILLS[skill].tone == global_tone
        ]
        if tone_matches:
            return tone_matches[0]
    
    # Check if any compatible skill is compatible with desired skill
    for skill in compatible:
        if check_skill_compatibility(skill, desired_skill):
            return skill
    
    # Return first compatible skill as fallback
    return compatible[0]


class SkillManager:
    """用于动态技能加载和扩展的管理器
    
    本类提供管理技能、检查兼容性和支持动态技能注册的集中式接口。
    
    功能：
    - 从配置文件加载技能
    - 热重载配置更改
    - 动态技能注册
    - 兼容性验证
    """
    
    def __init__(
        self,
        custom_skills: Optional[Dict[str, SkillConfig]] = None,
        config_path: Optional[str] = None,
        enable_hot_reload: bool = False
    ):
        """初始化技能管理器
        
        Args:
            custom_skills: 可选的要注册的自定义技能字典
            config_path: 可选的技能配置文件路径
            enable_hot_reload: 是否启用配置热重载
        """
        self._skills: Dict[str, SkillConfig] = SKILLS.copy()
        self._config_path = config_path
        self._hot_reload_enabled = enable_hot_reload
        self._loader = None
        
        # Load from config file if provided
        if config_path:
            self.reload_from_config(config_path)
        
        # Register custom skills
        if custom_skills:
            self.register_skills(custom_skills)
        
        # Enable hot-reload if requested
        if enable_hot_reload and config_path:
            self.enable_hot_reload()
    
    def register_skill(self, name: str, config: SkillConfig) -> None:
        """Register a new skill or update an existing one
        
        Args:
            name: Name of the skill
            config: Skill configuration
            
        Raises:
            ValueError: If skill configuration is invalid
        """
        # Validate that compatible_with references valid skills
        for compatible_skill in config.compatible_with:
            if compatible_skill not in self._skills and compatible_skill != name:
                raise ValueError(
                    f"Compatible skill '{compatible_skill}' not found in registered skills"
                )
        
        self._skills[name] = config
    
    def register_skills(self, skills: Dict[str, SkillConfig]) -> None:
        """一次性注册多个技能
        
        Args:
            skills: 技能名称到配置的字典
        """
        for name, config in skills.items():
            self.register_skill(name, config)
    
    def get_skill(self, name: str) -> SkillConfig:
        """根据名称获取技能配置
        
        Args:
            name: 技能名称
            
        Returns:
            技能配置
            
        Raises:
            ValueError: 如果技能未找到
        """
        if name not in self._skills:
            raise ValueError(f"Skill not found: {name}")
        return self._skills[name]
    
    def list_skills(self) -> List[str]:
        """Get list of all registered skill names
        
        Returns:
            List of skill names
        """
        return list(self._skills.keys())
    
    def check_compatibility(self, current_skill: str, target_skill: str) -> bool:
        """检查两个技能是否兼容
        
        Args:
            current_skill: 当前活动技能
            target_skill: 要切换到的目标技能
            
        Returns:
            兼容返回 True，否则返回 False
        """
        if current_skill not in self._skills:
            raise ValueError(f"Invalid current skill: {current_skill}")
        if target_skill not in self._skills:
            raise ValueError(f"Invalid target skill: {target_skill}")
        
        if current_skill == target_skill:
            return True
        
        return target_skill in self._skills[current_skill].compatible_with
    
    def get_compatible_skills(self, skill_name: str) -> List[str]:
        """获取兼容技能列表
        
        Args:
            skill_name: 技能名称
            
        Returns:
            兼容技能名称列表
        """
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
        """找到切换时最接近的兼容技能
        
        Args:
            current_skill: 当前活动技能
            desired_skill: 期望的目标技能
            global_tone: 可选的全局语气偏好
            allow_multi_hop: 是否允许多跳路径
            
        Returns:
            最接近的兼容技能名称
        """
        if current_skill not in self._skills:
            raise ValueError(f"Invalid current skill: {current_skill}")
        if desired_skill not in self._skills:
            raise ValueError(f"Invalid desired skill: {desired_skill}")
        
        # If directly compatible, return desired skill
        if self.check_compatibility(current_skill, desired_skill):
            return desired_skill
        
        # If multi-hop allowed, try to find a path
        if allow_multi_hop:
            path = find_skill_path(current_skill, desired_skill, max_hops=2)
            if path and len(path) > 1:
                return path[1]  # Return next step
        
        # Get compatible skills
        compatible = self.get_compatible_skills(current_skill)
        
        if not compatible:
            return current_skill
        
        # Prefer skills with matching tone
        if global_tone:
            tone_matches = [
                skill for skill in compatible
                if self._skills[skill].tone == global_tone
            ]
            if tone_matches:
                return tone_matches[0]
        
        # Check if any compatible skill can reach desired skill
        for skill in compatible:
            if self.check_compatibility(skill, desired_skill):
                return skill
        
        # Return first compatible skill
        return compatible[0]
    
    def get_skill_by_tone(self, tone: str) -> List[str]:
        """Get all skills with a specific tone
        
        Args:
            tone: Tone to filter by
            
        Returns:
            List of skill names with matching tone
        """
        return [
            name for name, config in self._skills.items()
            if config.tone == tone
        ]
    
    def validate_skill_graph(self) -> bool:
        """验证技能兼容性图是否格式正确
        
        Returns:
            有效返回 True，否则返回 False
        """
        for skill_name, config in self._skills.items():
            for compatible_skill in config.compatible_with:
                if compatible_skill not in self._skills:
                    return False
        return True
    
    def reload_from_config(self, config_path: str):
        """Reload skills from configuration file
        
        Args:
            config_path: Path to skills configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        from .skill_loader import SkillConfigLoader
        
        loader = SkillConfigLoader(config_path)
        new_skills = loader.load_from_yaml()
        
        # Replace current skills with loaded skills
        self._skills = new_skills
        self._config_path = config_path
        
        logger.info(f"Reloaded {len(new_skills)} skills from configuration")
    
    def enable_hot_reload(self):
        """Enable hot-reloading of configuration file
        
        Requires config_path to be set during initialization.
        """
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
        """Callback when configuration is reloaded
        
        Args:
            new_skills: New skills loaded from configuration
        """
        logger.info(f"Configuration reloaded with {len(new_skills)} skills")
        self._skills = new_skills
    
    def export_to_config(self, output_path: str):
        """将当前技能导出到配置文件
        
        Args:
            output_path: 保存配置的路径
        """
        from .skill_loader import SkillConfigLoader
        from pathlib import Path
        
        loader = SkillConfigLoader()
        loader.export_to_yaml(self._skills, Path(output_path))
        logger.info(f"Exported skills to: {output_path}")
    
    def get_config_path(self) -> Optional[str]:
        """Get the current configuration file path
        
        Returns:
            Configuration file path or None
        """
        return self._config_path
    
    def is_hot_reload_enabled(self) -> bool:
        """Check if hot-reload is enabled
        
        Returns:
            True if hot-reload is enabled
        """
        return self._hot_reload_enabled


# Create default skill manager instance
default_skill_manager = SkillManager()
