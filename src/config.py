"""集中配置管理模块"""
import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv(override=True)


@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str
    port: int
    database: str
    user: str
    password: str
    echo: bool = False
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """从环境变量加载数据库配置"""
        return cls(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', 5433)),
            database=os.getenv('POSTGRES_DB', 'Screenplay'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', '123456'),
            echo=os.getenv('DATABASE_ECHO', 'false').lower() == 'true'
        )
    
    @property
    def url(self) -> str:
        """生成数据库连接 URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class LLMConfig:
    """LLM API 配置"""
    glm_api_key: Optional[str]
    openai_api_key: Optional[str]
    qwen_api_key: Optional[str]
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量加载 LLM 配置"""
        return cls(
            glm_api_key=os.getenv('GLM_API_KEY'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            qwen_api_key=os.getenv('QWEN_API_KEY')
        )
    
    def get_available_provider(self) -> Optional[str]:
        """获取可用的 LLM 提供商"""
        if self.glm_api_key:
            return 'glm'
        if self.openai_api_key:
            return 'openai'
        if self.qwen_api_key:
            return 'qwen'
        return None


@dataclass
class AppConfig:
    """应用配置"""
    log_level: str
    config_path: str
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """从环境变量加载应用配置"""
        return cls(
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            config_path=os.getenv('CONFIG_PATH', 'config.yaml')
        )


def get_database_config() -> DatabaseConfig:
    """获取数据库配置（每次都从环境变量读取）"""
    return DatabaseConfig.from_env()


def get_llm_config() -> LLMConfig:
    """获取 LLM 配置（每次都从环境变量读取）"""
    return LLMConfig.from_env()


def get_app_config() -> AppConfig:
    """获取应用配置（每次都从环境变量读取）"""
    return AppConfig.from_env()


def get_config() -> tuple[DatabaseConfig, LLMConfig, AppConfig]:
    """获取所有配置"""
    return (
        get_database_config(),
        get_llm_config(),
        get_app_config()
    )
