"""工具定义（Function Calling Schemas）"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RetrieveArgs(BaseModel):
    """检索工具参数"""
    query: str = Field(..., description="检索查询")
    top_k: int = Field(default=5, description="返回结果数量")
    filters: Optional[Dict] = Field(default=None, description="过滤条件")


class WriteFragmentArgs(BaseModel):
    """写作工具参数"""
    step_id: int = Field(..., description="步骤ID")
    outline_index: int = Field(..., description="大纲索引")
    skill: str = Field(default="standard_tutorial", description="使用的技能")


class AddStepArgs(BaseModel):
    """动态添加步骤参数"""
    after_step_id: int = Field(..., description="在哪个步骤后添加")
    title: str = Field(..., description="新步骤标题")
    description: str = Field(..., description="步骤描述")
    skill: str = Field(default="standard_tutorial", description="使用的技能")


class UserInputArgs(BaseModel):
    """用户输入工具参数"""
    prompt: str = Field(..., description="向用户展示的提示")
    choices: List[str] = Field(..., description="选项列表")


class ModifyStepArgs(BaseModel):
    """修改步骤参数"""
    step_index: int = Field(..., description="要修改的步骤索引")
    new_title: Optional[str] = Field(default=None, description="新步骤标题")
    new_description: Optional[str] = Field(default=None, description="新步骤描述")
    new_skill: Optional[str] = Field(default=None, description="新使用的技能")


class DeleteStepArgs(BaseModel):
    """删除步骤参数"""
    step_index: int = Field(..., description="要删除的步骤索引")


class RegenerateFragmentArgs(BaseModel):
    """重新生成片段参数"""
    step_index: int = Field(..., description="要重新生成的步骤索引")
    reason: Optional[str] = Field(default=None, description="重新生成的原因")


class GetStatusArgs(BaseModel):
    """获取状态参数"""
    pass


# 工具 Schema 定义（OpenAI 格式）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "根据查询检索相关文档",
            "parameters": RetrieveArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_fragment",
            "description": "生成剧本片段",
            "parameters": WriteFragmentArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_step",
            "description": "动态添加新步骤到剧本大纲",
            "parameters": AddStepArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modify_step",
            "description": "修改现有步骤的标题、描述或技能",
            "parameters": ModifyStepArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_step",
            "description": "删除指定步骤",
            "parameters": DeleteStepArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "regenerate_fragment",
            "description": "重新生成某个步骤的剧本片段",
            "parameters": RegenerateFragmentArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_status",
            "description": "获取当前工作流状态",
            "parameters": GetStatusArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_user_input",
            "description": "暂停并等待用户输入",
            "parameters": UserInputArgs.model_json_schema()
        }
    }
]


# 工具名称常量
TOOL_RETRIEVE = "retrieve"
TOOL_WRITE_FRAGMENT = "write_fragment"
TOOL_ADD_STEP = "add_step"
TOOL_MODIFY_STEP = "modify_step"
TOOL_DELETE_STEP = "delete_step"
TOOL_REGENERATE_FRAGMENT = "regenerate_fragment"
TOOL_GET_STATUS = "get_current_status"
TOOL_REQUEST_USER_INPUT = "request_user_input"
