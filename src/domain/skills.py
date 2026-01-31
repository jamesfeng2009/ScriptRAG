"""Skill Manager - Manages generation style modes

This module defines the Skills system for the RAG screenplay generation system.
Skills are generation style modes that adjust how screenplay fragments are generated.
"""

import logging
from typing import Dict, List, Set, Optional, Literal
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class SkillConfig(BaseModel):
    """Configuration for a single Skill mode"""
    description: str = Field(..., description="Description of the skill's purpose")
    tone: str = Field(..., description="Tone/style of the skill")
    compatible_with: List[str] = Field(default_factory=list, description="List of compatible skill names")


class RetrievalConfig(BaseModel):
    """Configuration for RAG retrieval strategies"""
    
    class VectorSearchConfig(BaseModel):
        """Vector search configuration"""
        top_k: int = Field(default=5, description="Number of top results to return")
        similarity_threshold: float = Field(default=0.7, description="Minimum similarity score")
        embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model to use")
    
    class KeywordSearchConfig(BaseModel):
        """Keyword search configuration"""
        markers: List[str] = Field(
            default_factory=lambda: ["@deprecated", "FIXME", "TODO", "Security", "WARNING", "HACK"],
            description="Sensitive markers to search for"
        )
        boost_factor: float = Field(default=1.5, description="Boost factor for keyword matches")
    
    class HybridMergeConfig(BaseModel):
        """Hybrid search merge configuration"""
        vector_weight: float = Field(default=0.6, description="Weight for vector search results")
        keyword_weight: float = Field(default=0.4, description="Weight for keyword search results")
        keyword_boost_factor: float = Field(default=1.5, description="Boost factor for sensitive marker hits")
        dedup_threshold: float = Field(default=0.9, description="Deduplication similarity threshold")
    
    class SummarizationConfig(BaseModel):
        """Summarization configuration"""
        max_tokens: int = Field(default=10000, description="Maximum tokens before summarization")
        chunk_size: int = Field(default=2000, description="Chunk size for processing")
        overlap: int = Field(default=200, description="Overlap between chunks")
    
    vector_search: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    keyword_search: KeywordSearchConfig = Field(default_factory=KeywordSearchConfig)
    hybrid_merge: HybridMergeConfig = Field(default_factory=HybridMergeConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)


# Skills configuration dictionary
SKILLS: Dict[str, SkillConfig] = {
    "standard_tutorial": SkillConfig(
        description="清晰、结构化的教程格式",
        tone="professional",
        compatible_with=["visualization_analogy", "warning_mode"]
    ),
    "warning_mode": SkillConfig(
        description="突出显示废弃/风险内容",
        tone="cautionary",
        compatible_with=["standard_tutorial", "research_mode"]
    ),
    "visualization_analogy": SkillConfig(
        description="使用类比和可视化解释复杂概念",
        tone="engaging",
        compatible_with=["standard_tutorial", "meme_style"]
    ),
    "research_mode": SkillConfig(
        description="承认信息缺口并建议研究方向",
        tone="exploratory",
        compatible_with=["standard_tutorial", "warning_mode"]
    ),
    "meme_style": SkillConfig(
        description="轻松幽默的呈现方式",
        tone="casual",
        compatible_with=["visualization_analogy", "fallback_summary"]
    ),
    "fallback_summary": SkillConfig(
        description="详情不可用时的高层概述",
        tone="neutral",
        compatible_with=["standard_tutorial", "research_mode"]
    )
}

# Retrieval configuration
RETRIEVAL_CONFIG = RetrievalConfig()


def check_skill_compatibility(current_skill: str, target_skill: str) -> bool:
    """Check if two skills are compatible for switching
    
    Args:
        current_skill: Current active skill name
        target_skill: Target skill to switch to
        
    Returns:
        True if skills are compatible, False otherwise
        
    Raises:
        ValueError: If skill names are invalid
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
    """Get list of skills compatible with the given skill
    
    Args:
        skill_name: Name of the skill
        
    Returns:
        List of compatible skill names
        
    Raises:
        ValueError: If skill name is invalid
    """
    if skill_name not in SKILLS:
        raise ValueError(f"Invalid skill: {skill_name}")
    
    return SKILLS[skill_name].compatible_with.copy()


def find_closest_compatible_skill(
    current_skill: str,
    desired_skill: str,
    global_tone: Optional[str] = None
) -> str:
    """Find the closest compatible skill when direct switch is not possible
    
    Args:
        current_skill: Current active skill
        desired_skill: Desired target skill
        global_tone: Optional global tone preference
        
    Returns:
        Name of the closest compatible skill
        
    Raises:
        ValueError: If skill names are invalid
    """
    if current_skill not in SKILLS:
        raise ValueError(f"Invalid current skill: {current_skill}")
    if desired_skill not in SKILLS:
        raise ValueError(f"Invalid desired skill: {desired_skill}")
    
    # If directly compatible, return desired skill
    if check_skill_compatibility(current_skill, desired_skill):
        return desired_skill
    
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
    """Manager for dynamic skill loading and extension
    
    This class provides a centralized interface for managing skills,
    checking compatibility, and supporting dynamic skill registration.
    
    Features:
    - Load skills from configuration files
    - Hot-reload configuration changes
    - Dynamic skill registration
    - Compatibility validation
    """
    
    def __init__(
        self,
        custom_skills: Optional[Dict[str, SkillConfig]] = None,
        config_path: Optional[str] = None,
        enable_hot_reload: bool = False
    ):
        """Initialize the Skill Manager
        
        Args:
            custom_skills: Optional dictionary of custom skills to register
            config_path: Optional path to skills configuration file
            enable_hot_reload: Whether to enable hot-reloading of configuration
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
        """Register multiple skills at once
        
        Args:
            skills: Dictionary of skill names to configurations
        """
        for name, config in skills.items():
            self.register_skill(name, config)
    
    def get_skill(self, name: str) -> SkillConfig:
        """Get skill configuration by name
        
        Args:
            name: Name of the skill
            
        Returns:
            Skill configuration
            
        Raises:
            ValueError: If skill not found
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
        """Check if two skills are compatible
        
        Args:
            current_skill: Current active skill
            target_skill: Target skill to switch to
            
        Returns:
            True if compatible, False otherwise
        """
        if current_skill not in self._skills:
            raise ValueError(f"Invalid current skill: {current_skill}")
        if target_skill not in self._skills:
            raise ValueError(f"Invalid target skill: {target_skill}")
        
        if current_skill == target_skill:
            return True
        
        return target_skill in self._skills[current_skill].compatible_with
    
    def get_compatible_skills(self, skill_name: str) -> List[str]:
        """Get list of compatible skills
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            List of compatible skill names
        """
        if skill_name not in self._skills:
            raise ValueError(f"Invalid skill: {skill_name}")
        
        return self._skills[skill_name].compatible_with.copy()
    
    def find_compatible_skill(
        self,
        current_skill: str,
        desired_skill: str,
        global_tone: Optional[str] = None
    ) -> str:
        """Find closest compatible skill for switching
        
        Args:
            current_skill: Current active skill
            desired_skill: Desired target skill
            global_tone: Optional global tone preference
            
        Returns:
            Name of the closest compatible skill
        """
        if current_skill not in self._skills:
            raise ValueError(f"Invalid current skill: {current_skill}")
        if desired_skill not in self._skills:
            raise ValueError(f"Invalid desired skill: {desired_skill}")
        
        # If directly compatible, return desired skill
        if self.check_compatibility(current_skill, desired_skill):
            return desired_skill
        
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
        """Validate that the skill compatibility graph is well-formed
        
        Returns:
            True if valid, False otherwise
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
        """Disable hot-reloading of configuration file"""
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
        """Export current skills to configuration file
        
        Args:
            output_path: Path where to save the configuration
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
