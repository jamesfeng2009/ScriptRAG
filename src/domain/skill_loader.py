"""技能配置加载器

本模块提供从 YAML 配置文件加载技能、验证配置和支持热重载的功能。
支持分层认知框架和信号路由系统的配置。
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Optional, List, Callable, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from pydantic import BaseModel, Field, field_validator

from .skills import SkillConfig, SkillManager, SkillLayer, SkillTrigger


logger = logging.getLogger(__name__)


class PromptConfig(BaseModel):
    """技能的提示词配置"""
    system_prompt: str = Field(..., description="系统提示词模板")
    user_template: str = Field(..., description="带占位符的用户提示词模板")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="LLM temperature")
    max_tokens: int = Field(default=2000, gt=0, description="生成的最大 token 数")

    @field_validator('system_prompt', 'user_template')
    @classmethod
    def validate_prompts(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v


class SkillTriggerYAML(BaseModel):
    """YAML 技能触发器配置"""
    type: str = Field(..., description="触发器类型 (marker/keyword/complexity/domain)")
    value: str = Field(..., description="触发器值")
    priority: int = Field(default=10, description="优先级，数值越小优先级越高")


class SkillConfigYAML(BaseModel):
    """YAML skill configuration model"""
    description: str = Field(..., description="Skill description")
    tone: str = Field(..., description="Skill tone/style")
    layer: str = Field(default="foundation", description="Skill layer (foundation/semantic/domain)")
    triggers: List[SkillTriggerYAML] = Field(default_factory=list, description="Skill triggers")
    compatible_with: List[str] = Field(default_factory=list, description="Compatible skills")
    prompt_config: PromptConfig = Field(..., description="Prompt configuration")
    enabled: bool = Field(default=True, description="Whether skill is enabled")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('layer')
    @classmethod
    def validate_layer(cls, v):
        valid_layers = ["foundation", "semantic", "domain"]
        if v not in valid_layers:
            raise ValueError(f"Invalid layer: {v}. Must be one of {valid_layers}")
        return v


class LayerConfig(BaseModel):
    """层次配置模型"""
    description: str = Field(..., description="层次描述")
    compatible_layers: List[str] = Field(default_factory=list, description="兼容的层次")


class SkillsConfigFile(BaseModel):
    """完整的技能配置文件模型 (v2.0)"""
    version: str = Field(..., description="配置版本")
    layers: Dict[str, LayerConfig] = Field(default_factory=dict, description="层次定义")
    skills: Dict[str, SkillConfigYAML] = Field(..., description="技能字典")
    routing: Dict[str, Any] = Field(default_factory=dict, description="路由规则配置")
    compatibility: Dict[str, Any] = Field(default_factory=dict, description="兼容性规则配置")

    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        if v not in ["1.0", "2.0"]:
            raise ValueError(f"Unsupported configuration version: {v}. Must be 1.0 or 2.0")
        return v


class SkillConfigLoader:
    """
    技能配置加载器

    功能：
    - 从 YAML 文件加载技能
    - 验证配置
    - 转换为 SkillConfig 对象
    - 支持分层和触发器配置
    - 支持热重载
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化技能配置加载器

        Args:
            config_path: 技能配置文件的路径
        """
        self.config_path = Path(config_path) if config_path else Path("config/skills.yaml")
        self.observer: Optional[Observer] = None
        self.reload_callbacks: List[Callable] = []

        logger.info(f"SkillConfigLoader initialized with config: {self.config_path}")

    def load_from_yaml(self, path: Optional[Path] = None) -> Dict[str, SkillConfig]:
        """
        从 YAML 文件加载技能

        Args:
            path: YAML 文件路径（如果未提供则使用默认路径）

        Returns:
            技能名称到 SkillConfig 的字典

        Raises:
            FileNotFoundError: 如果配置文件不存在
            ValueError: 如果配置无效
        """
        config_file = path or self.config_path

        if not config_file.exists():
            raise FileNotFoundError(f"Skills configuration file not found: {config_file}")

        logger.info(f"Loading skills from: {config_file}")

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)

            config = SkillsConfigFile(**raw_config)

            skills = {}
            for skill_name, skill_yaml in config.skills.items():
                if not skill_yaml.enabled:
                    logger.info(f"Skipping disabled skill: {skill_name}")
                    continue

                triggers = [
                    SkillTrigger(
                        type=trigger.type,
                        value=trigger.value,
                        priority=trigger.priority
                    )
                    for trigger in skill_yaml.triggers
                ]

                layer = SkillLayer(skill_yaml.layer)

                skill_config = SkillConfig(
                    description=skill_yaml.description,
                    tone=skill_yaml.tone,
                    layer=layer,
                    triggers=triggers,
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
        从 YAML 文件加载提示词配置

        Args:
            path: YAML 文件路径（如果未提供则使用默认路径）

        Returns:
            技能名称到 PromptConfig 的字典
        """
        config_file = path or self.config_path

        if not config_file.exists():
            raise FileNotFoundError(f"Skills configuration file not found: {config_file}")

        logger.info(f"Loading prompt configs from: {config_file}")

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)

            config = SkillsConfigFile(**raw_config)

            prompt_configs = {}
            for skill_name, skill_yaml in config.skills.items():
                if skill_yaml.enabled:
                    prompt_configs[skill_name] = skill_yaml.prompt_config

            logger.info(f"Successfully loaded {len(prompt_configs)} prompt configurations")
            return prompt_configs

        except Exception as e:
            logger.error(f"Failed to load prompt configurations: {str(e)}")
            raise

    def load_layers(self, path: Optional[Path] = None) -> Dict[str, LayerConfig]:
        """
        从 YAML 文件加载层次配置

        Args:
            path: YAML 文件路径（如果未提供则使用默认路径）

        Returns:
            层次名称到 LayerConfig 的字典
        """
        config_file = path or self.config_path

        if not config_file.exists():
            raise FileNotFoundError(f"Skills configuration file not found: {config_file}")

        logger.info(f"Loading layers from: {config_file}")

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)

            config = SkillsConfigFile(**raw_config)

            logger.info(f"Successfully loaded {len(config.layers)} layers")
            return config.layers

        except Exception as e:
            logger.error(f"Failed to load layers configuration: {str(e)}")
            raise

    def load_routing_config(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """
        从 YAML 文件加载路由规则配置

        Args:
            path: YAML 文件路径（如果未提供则使用默认路径）

        Returns:
            路由规则配置字典
        """
        config_file = path or self.config_path

        if not config_file.exists():
            raise FileNotFoundError(f"Skills configuration file not found: {config_file}")

        logger.info(f"Loading routing config from: {config_file}")

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)

            config = SkillsConfigFile(**raw_config)

            logger.info("Successfully loaded routing configuration")
            return config.routing

        except Exception as e:
            logger.error(f"Failed to load routing configuration: {str(e)}")
            raise

    def validate_config(self, config_dict: dict) -> bool:
        """
        验证技能配置

        Args:
            config_dict: 原始配置字典

        Returns:
            有效返回 True，无效返回 False
        """
        try:
            config = SkillsConfigFile(**config_dict)

            skill_names = set(config.skills.keys())
            for skill_name, skill_config in config.skills.items():
                for compatible_skill in skill_config.compatible_with:
                    if compatible_skill not in skill_names:
                        logger.error(
                            f"Skill '{skill_name}' references unknown compatible skill: '{compatible_skill}'"
                        )
                        return False

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

        yaml_skills = {}
        for skill_name, skill_config in skills.items():
            yaml_skills[skill_name] = {
                'description': skill_config.description,
                'tone': skill_config.tone,
                'layer': skill_config.layer.value,
                'triggers': [
                    {
                        'type': trigger.type,
                        'value': trigger.value,
                        'priority': trigger.priority
                    }
                    for trigger in skill_config.triggers
                ],
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
            'version': '2.0',
            'layers': {},
            'skills': yaml_skills,
            'routing': {},
            'compatibility': {}
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Successfully exported skills to: {output_path}")

    def register_reload_callback(self, callback: Callable):
        """
        注册在配置重新加载时调用的回调函数

        Args:
            callback: 要注册的回调函数
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
        触发重新加载回调
        """
        logger.info("Configuration file changed, triggering reload callbacks")

        try:
            new_skills = self.load_from_yaml()

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
    创建默认技能配置文件

    Args:
        output_path: 创建配置文件的路径
    """
    from .skills import SKILLS

    loader = SkillConfigLoader()
    loader.export_to_yaml(SKILLS, output_path)
    logger.info(f"Created default configuration at: {output_path}")
