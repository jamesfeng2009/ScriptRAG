"""Skill Configuration Loader

This module provides functionality to load skills from YAML configuration files,
validate them, and support hot-reloading.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Optional, List, Callable, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from pydantic import BaseModel, Field, field_validator

from .skills import SkillConfig, SkillManager


logger = logging.getLogger(__name__)


class PromptConfig(BaseModel):
    """Prompt configuration for a skill"""
    system_prompt: str = Field(..., description="System prompt template")
    user_template: str = Field(..., description="User prompt template with placeholders")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="LLM temperature")
    max_tokens: int = Field(default=2000, gt=0, description="Maximum tokens to generate")
    
    @field_validator('system_prompt', 'user_template')
    @classmethod
    def validate_prompts(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v


class SkillConfigYAML(BaseModel):
    """YAML skill configuration model"""
    description: str = Field(..., description="Skill description")
    tone: str = Field(..., description="Skill tone/style")
    compatible_with: List[str] = Field(default_factory=list, description="Compatible skills")
    prompt_config: PromptConfig = Field(..., description="Prompt configuration")
    enabled: bool = Field(default=True, description="Whether skill is enabled")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SkillsConfigFile(BaseModel):
    """Complete skills configuration file model"""
    version: str = Field(..., description="Configuration version")
    skills: Dict[str, SkillConfigYAML] = Field(..., description="Skills dictionary")
    
    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        if not v.startswith('1.'):
            raise ValueError(f"Unsupported configuration version: {v}")
        return v


class SkillConfigLoader:
    """
    Skill configuration loader
    
    Features:
    - Load skills from YAML files
    - Validate configuration
    - Convert to SkillConfig objects
    - Support hot-reloading
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize skill config loader
        
        Args:
            config_path: Path to skills configuration file
        """
        self.config_path = Path(config_path) if config_path else Path("config/skills.yaml")
        self.observer: Optional[Observer] = None
        self.reload_callbacks: List[Callable] = []
        
        logger.info(f"SkillConfigLoader initialized with config: {self.config_path}")
    
    def load_from_yaml(self, path: Optional[Path] = None) -> Dict[str, SkillConfig]:
        """
        Load skills from YAML file
        
        Args:
            path: Path to YAML file (uses default if not provided)
            
        Returns:
            Dictionary of skill name to SkillConfig
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_file = path or self.config_path
        
        if not config_file.exists():
            raise FileNotFoundError(f"Skills configuration file not found: {config_file}")
        
        logger.info(f"Loading skills from: {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            # Validate configuration structure
            config = SkillsConfigFile(**raw_config)
            
            # Convert to SkillConfig objects
            skills = {}
            for skill_name, skill_yaml in config.skills.items():
                if not skill_yaml.enabled:
                    logger.info(f"Skipping disabled skill: {skill_name}")
                    continue
                
                # Convert YAML config to SkillConfig
                skill_config = SkillConfig(
                    description=skill_yaml.description,
                    tone=skill_yaml.tone,
                    compatible_with=skill_yaml.compatible_with
                )
                
                skills[skill_name] = skill_config
            
            logger.info(f"Successfully loaded {len(skills)} skills from configuration")
            return skills
            
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML configuration: {str(e)}")
            raise ValueError(f"Invalid YAML configuration: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load skills configuration: {str(e)}")
            raise
    
    def load_prompt_configs(self, path: Optional[Path] = None) -> Dict[str, PromptConfig]:
        """
        Load prompt configurations from YAML file
        
        Args:
            path: Path to YAML file (uses default if not provided)
            
        Returns:
            Dictionary of skill name to PromptConfig
        """
        config_file = path or self.config_path
        
        if not config_file.exists():
            raise FileNotFoundError(f"Skills configuration file not found: {config_file}")
        
        logger.info(f"Loading prompt configs from: {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            config = SkillsConfigFile(**raw_config)
            
            # Extract prompt configs
            prompt_configs = {}
            for skill_name, skill_yaml in config.skills.items():
                if skill_yaml.enabled:
                    prompt_configs[skill_name] = skill_yaml.prompt_config
            
            logger.info(f"Successfully loaded {len(prompt_configs)} prompt configurations")
            return prompt_configs
            
        except Exception as e:
            logger.error(f"Failed to load prompt configurations: {str(e)}")
            raise
    
    def validate_config(self, config_dict: dict) -> bool:
        """
        Validate skills configuration
        
        Args:
            config_dict: Raw configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate structure
            config = SkillsConfigFile(**config_dict)
            
            # Validate compatibility references
            skill_names = set(config.skills.keys())
            for skill_name, skill_config in config.skills.items():
                for compatible_skill in skill_config.compatible_with:
                    if compatible_skill not in skill_names:
                        logger.error(
                            f"Skill '{skill_name}' references unknown compatible skill: '{compatible_skill}'"
                        )
                        return False
            
            # Validate prompt templates
            for skill_name, skill_config in config.skills.items():
                required_placeholders = ['{step_description}', '{retrieved_content}']
                template = skill_config.prompt_config.user_template
                
                for placeholder in required_placeholders:
                    if placeholder not in template:
                        logger.warning(
                            f"Skill '{skill_name}' user_template missing placeholder: {placeholder}"
                        )
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def export_to_yaml(self, skills: Dict[str, SkillConfig], output_path: Path):
        """
        Export skills to YAML file
        
        Args:
            skills: Dictionary of skills to export
            output_path: Output file path
        """
        logger.info(f"Exporting {len(skills)} skills to: {output_path}")
        
        # Convert SkillConfig to YAML format
        yaml_skills = {}
        for skill_name, skill_config in skills.items():
            yaml_skills[skill_name] = {
                'description': skill_config.description,
                'tone': skill_config.tone,
                'compatible_with': skill_config.compatible_with,
                'prompt_config': {
                    'system_prompt': 'TODO: Add system prompt',
                    'user_template': 'TODO: Add user template',
                    'temperature': 0.7,
                    'max_tokens': 2000
                },
                'enabled': True,
                'metadata': {}
            }
        
        config_dict = {
            'version': '1.0',
            'skills': yaml_skills
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Successfully exported skills to: {output_path}")
    
    def register_reload_callback(self, callback: Callable):
        """
        Register a callback to be called when configuration is reloaded
        
        Args:
            callback: Callback function to register
        """
        self.reload_callbacks.append(callback)
        logger.info(f"Registered reload callback: {callback.__name__}")
    
    def start_watching(self):
        """
        Start watching configuration file for changes (hot-reload)
        """
        if self.observer is not None:
            logger.warning("File watcher already started")
            return
        
        event_handler = SkillConfigFileHandler(self)
        self.observer = Observer()
        self.observer.schedule(
            event_handler,
            str(self.config_path.parent),
            recursive=False
        )
        self.observer.start()
        
        logger.info(f"Started watching configuration file: {self.config_path}")
    
    def stop_watching(self):
        """
        Stop watching configuration file
        """
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Stopped watching configuration file")
    
    def _trigger_reload(self):
        """
        Trigger reload callbacks
        """
        logger.info("Configuration file changed, triggering reload callbacks")
        
        try:
            # Reload configuration
            new_skills = self.load_from_yaml()
            
            # Call all registered callbacks
            for callback in self.reload_callbacks:
                try:
                    callback(new_skills)
                except Exception as e:
                    logger.error(f"Reload callback failed: {callback.__name__}, error: {str(e)}")
            
            logger.info("Configuration reload completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {str(e)}")


class SkillConfigFileHandler(FileSystemEventHandler):
    """
    File system event handler for skill configuration hot-reload
    """
    
    def __init__(self, loader: SkillConfigLoader):
        self.loader = loader
        self.config_filename = loader.config_path.name
    
    def on_modified(self, event):
        """
        Handle file modification event
        """
        if isinstance(event, FileModifiedEvent) and event.src_path.endswith(self.config_filename):
            logger.info(f"Configuration file modified: {event.src_path}")
            self.loader._trigger_reload()


def create_default_config(output_path: Path):
    """
    Create a default skills configuration file
    
    Args:
        output_path: Path where to create the config file
    """
    from .skills import SKILLS
    
    loader = SkillConfigLoader()
    loader.export_to_yaml(SKILLS, output_path)
    logger.info(f"Created default configuration at: {output_path}")
