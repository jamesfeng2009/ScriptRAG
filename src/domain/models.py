"""
核心数据模型（领域层）

本模块定义了基于 RAG 的剧本生成多智能体系统的核心数据模型。
所有模型使用 Pydantic 进行数据验证和序列化。
"""

from typing import List, Dict, Optional, Literal, Any, TYPE_CHECKING
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime


if TYPE_CHECKING:
    pass


class IntentAnalysis(BaseModel):
    """意图分析结果
    
    表示意图解析智能体输出的分析结果，包含主要意图、关键词和建议数据源。
    """
    primary_intent: str = Field(
        ...,
        description="主要意图描述",
        min_length=1
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="用于检索的关键词列表"
    )
    search_sources: List[str] = Field(
        default_factory=list,
        description="建议的检索数据源，如 ['rag', 'mysql', 'web']"
    )
    confidence: float = Field(
        default=0.8,
        description="分析置信度",
        ge=0.0,
        le=1.0
    )
    alternative_intents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="备选意图列表"
    )
    intent_type: str = Field(
        default="informational",
        description="意图类型：informational/navigational/transactional/computational"
    )
    language: str = Field(
        default="zh",
        description="查询语言：zh/en/mixed"
    )
    
    def model_dump_for_logging(self) -> Dict[str, Any]:
        """转换为日志友好的格式"""
        return {
            "primary_intent": self.primary_intent,
            "keywords": self.keywords,
            "search_sources": self.search_sources,
            "confidence": self.confidence,
            "intent_type": self.intent_type,
            "language": self.language,
            "alternative_intents_count": len(self.alternative_intents)
        }


class RetrievedDocument(BaseModel):
    """检索的文档
    
    表示从 RAG 系统检索到的单个文档，包含内容、来源、置信度和元数据。
    """
    content: str = Field(..., description="文档内容", min_length=1)
    source: str = Field(..., description="文档来源路径", min_length=1)
    confidence: float = Field(..., description="置信度分数", ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    summary: Optional[str] = Field(None, description="文档摘要（用于大文件）")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """验证内容不为空白字符串"""
        if not v.strip():
            raise ValueError("文档内容不能为空白字符串")
        return v
    
    @field_validator('source')
    @classmethod
    def validate_source(cls, v: str) -> str:
        """验证来源路径不为空白字符串"""
        if not v.strip():
            raise ValueError("文档来源不能为空白字符串")
        return v


class OutlineStep(BaseModel):
    """大纲步骤
    
    表示剧本大纲中的单个步骤，包含描述、状态和重试计数。
    """
    step_id: int = Field(..., description="步骤 ID", ge=0)
    title: str = Field(default="", description="步骤标题", min_length=0)
    description: str = Field(..., description="步骤描述", min_length=1)
    status: Literal["pending", "in_progress", "completed", "skipped"] = Field(
        default="pending",
        description="步骤状态"
    )
    retry_count: int = Field(default=0, description="重试次数", ge=0)
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v: str) -> str:
        """验证描述不为空白字符串"""
        if not v.strip():
            raise ValueError("步骤描述不能为空白字符串")
        return v
    
    @field_validator('retry_count')
    @classmethod
    def validate_retry_count(cls, v: int) -> int:
        """验证重试次数不超过合理范围"""
        if v > 10:  # 设置一个合理的上限
            raise ValueError("重试次数不能超过 10 次")
        return v


class ScreenplayFragment(BaseModel):
    """剧本片段
    
    表示为特定大纲步骤生成的剧本段落，包含内容、使用的 Skill 和来源。
    """
    step_id: int = Field(..., description="对应的步骤 ID", ge=0)
    content: str = Field(..., description="剧本片段内容", min_length=1)
    skill_used: str = Field(..., description="使用的 Skill 模式", min_length=1)
    sources: List[str] = Field(default_factory=list, description="引用的来源列表")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """验证内容不为空白字符串"""
        if not v.strip():
            raise ValueError("剧本片段内容不能为空白字符串")
        return v
    
    @field_validator('skill_used')
    @classmethod
    def validate_skill_used(cls, v: str) -> str:
        """验证 Skill 名称有效"""
        valid_skills = {
            "standard_tutorial",
            "warning_mode",
            "visualization_analogy",
            "research_mode",
            "meme_style",
            "fallback_summary"
        }
        if v not in valid_skills:
            raise ValueError(
                f"无效的 Skill 模式: {v}. "
                f"有效的 Skill 包括: {', '.join(valid_skills)}"
            )
        return v


class SharedState(BaseModel):
    """共享状态
    
    所有智能体共享的全局状态对象，包含用户输入、大纲、检索内容、
    生成的片段和控制信号。
    """
    # 用户输入
    user_topic: str = Field(..., description="用户主题", min_length=1)
    project_context: str = Field(default="", description="项目上下文")
    
    # 大纲管理
    outline: List[OutlineStep] = Field(default_factory=list, description="剧本大纲")
    current_step_index: int = Field(default=0, description="当前步骤索引", ge=0)
    
    # RAG 检索
    retrieved_docs: List[RetrievedDocument] = Field(
        default_factory=list,
        description="检索到的文档列表"
    )
    
    # 意图解析（Agentic RAG）
    current_intent: Optional["IntentAnalysis"] = Field(
        None,
        description="当前步骤的意图分析结果"
    )
    
    # 生成
    fragments: List[ScreenplayFragment] = Field(
        default_factory=list,
        description="生成的剧本片段列表"
    )
    current_skill: str = Field(
        default="standard_tutorial",
        description="当前使用的 Skill 模式"
    )
    skill_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Skill 切换历史记录"
    )
    global_tone: str = Field(default="professional", description="全局语调")
    
    # 控制信号
    pivot_triggered: bool = Field(default=False, description="是否触发转向")
    pivot_reason: Optional[str] = Field(None, description="转向原因")
    max_retries: int = Field(default=3, description="最大重试次数", ge=1, le=10)
    fact_check_passed: bool = Field(default=True, description="事实检查是否通过")
    
    # 用户交互
    awaiting_user_input: bool = Field(default=False, description="是否等待用户输入")
    user_input_prompt: Optional[str] = Field(None, description="用户输入提示信息")
    
    # 日志
    execution_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="执行日志"
    )
    
    # 元数据
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    @field_validator('user_topic')
    @classmethod
    def validate_user_topic(cls, v: str) -> str:
        """验证用户主题不为空白字符串"""
        if not v.strip():
            raise ValueError("用户主题不能为空白字符串")
        return v
    
    @field_validator('current_skill')
    @classmethod
    def validate_current_skill(cls, v: str) -> str:
        """验证当前 Skill 有效"""
        from .skills import SKILLS
        valid_skills = set(SKILLS.keys())
        if v not in valid_skills:
            raise ValueError(
                f"无效的 Skill 模式: {v}. "
                f"有效的 Skill 包括: {', '.join(valid_skills)}"
            )
        return v
    
    @field_validator('current_step_index')
    @classmethod
    def validate_current_step_index(cls, v: int) -> int:
        """验证当前步骤索引非负"""
        if v < 0:
            raise ValueError("当前步骤索引不能为负数")
        return v
    
    @model_validator(mode='after')
    def validate_step_index_in_range(self) -> 'SharedState':
        """验证当前步骤索引在大纲范围内
        
        允许 current_step_index == len(outline) 表示所有步骤已完成
        """
        if self.outline and self.current_step_index > len(self.outline):
            raise ValueError(
                f"当前步骤索引 {self.current_step_index} "
                f"超出大纲范围 (0-{len(self.outline)})"
            )
        return self
    
    @model_validator(mode='after')
    def validate_pivot_reason_when_triggered(self) -> 'SharedState':
        """验证转向触发时必须提供原因"""
        if self.pivot_triggered and not self.pivot_reason:
            raise ValueError("转向触发时必须提供转向原因")
        return self
    
    @model_validator(mode='after')
    def validate_user_input_prompt_when_awaiting(self) -> 'SharedState':
        """验证等待用户输入时必须提供提示信息"""
        if self.awaiting_user_input and not self.user_input_prompt:
            raise ValueError("等待用户输入时必须提供提示信息")
        return self
    
    def add_log_entry(
        self,
        agent_name: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """添加日志条目
        
        Args:
            agent_name: 智能体名称
            action: 执行的动作
            details: 详细信息
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "action": action,
            "details": details or {}
        }
        self.execution_log.append(log_entry)
        self.updated_at = datetime.now()
    
    def get_current_step(self) -> Optional[OutlineStep]:
        """获取当前步骤
        
        Returns:
            当前步骤对象，如果索引超出范围则返回 None
        """
        if 0 <= self.current_step_index < len(self.outline):
            return self.outline[self.current_step_index]
        return None
    
    def advance_step(self) -> bool:
        """前进到下一步
        
        Returns:
            是否成功前进（如果已到达末尾则返回 False）
        """
        if self.current_step_index < len(self.outline) - 1:
            self.current_step_index += 1
            self.updated_at = datetime.now()
            return True
        return False
    
    def is_complete(self) -> bool:
        """检查是否所有步骤都已完成
        
        Returns:
            是否所有步骤都已完成或跳过
        """
        if not self.outline:
            return False
        return all(
            step.status in ["completed", "skipped"]
            for step in self.outline
        )
    
    def switch_skill(
        self,
        new_skill: str,
        reason: str,
        step_id: Optional[int] = None
    ) -> None:
        """切换 Skill 并记录历史
        
        Args:
            new_skill: 新的 Skill 模式
            reason: 切换原因
            step_id: 触发切换的步骤 ID（可选）
        
        Raises:
            ValueError: 如果 Skill 名称无效
        """
        from .skills import SKILLS
        valid_skills = set(SKILLS.keys())
        
        if new_skill not in valid_skills:
            raise ValueError(
                f"无效的 Skill 模式: {new_skill}. "
                f"有效的 Skill 包括: {', '.join(valid_skills)}"
            )
        
        # 如果 Skill 没有变化，不记录
        if new_skill == self.current_skill:
            return
        
        # 记录 Skill 切换历史
        skill_change = {
            "timestamp": datetime.now().isoformat(),
            "from_skill": self.current_skill,
            "to_skill": new_skill,
            "reason": reason,
            "step_id": step_id or self.current_step_index
        }
        self.skill_history.append(skill_change)
        
        # 更新当前 Skill
        self.current_skill = new_skill
        self.updated_at = datetime.now()
        
        # 添加日志条目
        self.add_log_entry(
            agent_name="system",
            action="skill_switch",
            details=skill_change
        )
    
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "user_topic": "如何使用 Python 实现 REST API",
                "project_context": "使用 FastAPI 框架",
                "outline": [
                    {
                        "step_id": 0,
                        "description": "介绍 FastAPI 框架",
                        "status": "pending",
                        "retry_count": 0
                    }
                ],
                "current_step_index": 0,
                "current_skill": "standard_tutorial",
                "global_tone": "professional",
                "max_retries": 3
            }
        }
    }
