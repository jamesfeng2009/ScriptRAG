"""提示词管理器 - 管理技能的提示词配置

本模块提供集中式接口，用于管理从技能配置文件加载的提示词配置。
"""

import logging
from typing import Dict, Optional
from pathlib import Path

from .skill_loader import SkillConfigLoader, PromptConfig


logger = logging.getLogger(__name__)


class PromptManager:
    """
    提示词管理器
    
    管理所有技能的提示词配置，提供：
    - 从配置加载提示词
    - 使用上下文格式化提示词
    - 热重载支持
    - 回退到默认提示词
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        enable_hot_reload: bool = False
    ):
        """
        初始化提示词管理器
        
        Args:
            config_path: 技能配置文件的路径
            enable_hot_reload: 是否启用热重载
        """
        self.config_path = config_path or "config/skills.yaml"
        self._prompt_configs: Dict[str, PromptConfig] = {}
        self._loader: Optional[SkillConfigLoader] = None
        self._hot_reload_enabled = False
        
        # Load initial configuration
        self._load_prompts()
        
        # Enable hot-reload if requested
        if enable_hot_reload:
            self.enable_hot_reload()
    
    def _load_prompts(self):
        """从文件加载提示词配置"""
        try:
            if not Path(self.config_path).exists():
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return
            
            loader = SkillConfigLoader(self.config_path)
            self._prompt_configs = loader.load_prompt_configs()
            
            logger.info(f"Loaded {len(self._prompt_configs)} prompt configurations")
            
        except Exception as e:
            logger.error(f"Failed to load prompt configurations: {str(e)}")
            # Continue with empty configs, will use defaults
    
    def get_prompt_config(self, skill_name: str) -> Optional[PromptConfig]:
        """
        获取技能的提示词配置
        
        Args:
            skill_name: 技能名称
            
        Returns:
            找到返回 PromptConfig，否则返回 None
        """
        return self._prompt_configs.get(skill_name)
    
    def format_messages(
        self,
        skill_name: str,
        step_description: str,
        retrieved_content: str
    ) -> list:
        """
        Format messages for LLM using skill's prompt configuration
        
        Args:
            skill_name: Name of the skill
            step_description: Description of the current step
            retrieved_content: Retrieved content from RAG
            
        Returns:
            List of message dictionaries for LLM
        """
        prompt_config = self.get_prompt_config(skill_name)
        
        if not prompt_config:
            logger.warning(f"No prompt config found for skill: {skill_name}, using default")
            return self._get_default_messages(skill_name, step_description, retrieved_content)
        
        # Format user template with context
        user_content = prompt_config.user_template.format(
            step_description=step_description,
            retrieved_content=retrieved_content
        )
        
        messages = [
            {
                "role": "system",
                "content": prompt_config.system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        return messages
    
    def get_temperature(self, skill_name: str) -> float:
        """
        获取技能的 temperature 设置
        
        Args:
            skill_name: 技能名称
            
        Returns:
            Temperature 值（默认：0.7）
        """
        prompt_config = self.get_prompt_config(skill_name)
        return prompt_config.temperature if prompt_config else 0.7
    
    def get_max_tokens(self, skill_name: str) -> int:
        """
        获取技能的 max_tokens 设置
        
        Args:
            skill_name: 技能名称
            
        Returns:
            Max tokens 值（默认：2000）
        """
        prompt_config = self.get_prompt_config(skill_name)
        return prompt_config.max_tokens if prompt_config else 2000
    
    def _get_default_messages(
        self,
        skill_name: str,
        step_description: str,
        retrieved_content: str
    ) -> list:
        """
        Get default messages when no config is found
        
        Args:
            skill_name: Name of the skill
            step_description: Description of the current step
            retrieved_content: Retrieved content
            
        Returns:
            Default message list
        """
        return [
            {
                "role": "system",
                "content": f"你是一个专业的内容生成助手，当前使用 {skill_name} 模式。"
            },
            {
                "role": "user",
                "content": f"步骤描述: {step_description}\n\n检索内容:\n{retrieved_content}\n\n请生成内容。"
            }
        ]
    
    def reload_prompts(self):
        """Reload prompt configurations from file"""
        logger.info("Reloading prompt configurations")
        self._load_prompts()
    
    def enable_hot_reload(self):
        """启用提示词配置的热重载"""
        if self._hot_reload_enabled:
            logger.warning("Hot-reload already enabled")
            return
        
        try:
            self._loader = SkillConfigLoader(self.config_path)
            self._loader.register_reload_callback(self._on_config_reload)
            self._loader.start_watching()
            self._hot_reload_enabled = True
            
            logger.info("Hot-reload enabled for prompt configurations")
            
        except Exception as e:
            logger.error(f"Failed to enable hot-reload: {str(e)}")
    
    def disable_hot_reload(self):
        """禁用提示词配置的热重载"""
        if self._loader is not None:
            self._loader.stop_watching()
            self._loader = None
            self._hot_reload_enabled = False
            
            logger.info("Hot-reload disabled for prompt configurations")
    
    def _on_config_reload(self, new_skills):
        """
        Callback when configuration is reloaded
        
        Args:
            new_skills: New skills loaded from configuration
        """
        logger.info("Prompt configurations reloaded")
        self._load_prompts()
    
    def is_hot_reload_enabled(self) -> bool:
        """检查是否启用了热重载"""
        return self._hot_reload_enabled
    
    def list_available_skills(self) -> list:
        """获取有提示词配置的技能列表"""
        return list(self._prompt_configs.keys())


# 创建默认提示词管理器实例
default_prompt_manager = PromptManager()
