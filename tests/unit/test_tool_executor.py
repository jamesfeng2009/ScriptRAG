"""Unit tests for Function Calling tools"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock

from src.domain.tools.tool_definitions import (
    TOOLS,
    TOOL_RETRIEVE,
    TOOL_WRITE_FRAGMENT,
    TOOL_ADD_STEP,
    TOOL_MODIFY_STEP,
    TOOL_DELETE_STEP,
    TOOL_REGENERATE_FRAGMENT,
    TOOL_GET_STATUS,
    TOOL_REQUEST_USER_INPUT,
    RetrieveArgs,
    WriteFragmentArgs,
    AddStepArgs,
    ModifyStepArgs,
    DeleteStepArgs,
    RegenerateFragmentArgs,
    UserInputArgs,
)
from src.domain.tools.tool_executor import ToolExecutor
from src.domain.tools.tool_service import ToolService


class TestToolDefinitions:
    """测试工具定义"""

    def test_tool_constants_exist(self):
        """测试工具常量是否存在"""
        assert TOOL_RETRIEVE == "retrieve"
        assert TOOL_WRITE_FRAGMENT == "write_fragment"
        assert TOOL_ADD_STEP == "add_step"
        assert TOOL_MODIFY_STEP == "modify_step"
        assert TOOL_DELETE_STEP == "delete_step"
        assert TOOL_REGENERATE_FRAGMENT == "regenerate_fragment"
        assert TOOL_GET_STATUS == "get_current_status"
        assert TOOL_REQUEST_USER_INPUT == "request_user_input"

    def test_retrieve_args_schema(self):
        """测试检索工具参数 Schema"""
        args = RetrieveArgs(query="test query", top_k=5, filters={"category": "docs"})
        assert args.query == "test query"
        assert args.top_k == 5
        assert args.filters == {"category": "docs"}

    def test_add_step_args_schema(self):
        """测试添加步骤工具参数 Schema"""
        args = AddStepArgs(
            after_step_id=1,
            title="New Step",
            description="Step description",
            skill="standard_tutorial"
        )
        assert args.after_step_id == 1
        assert args.title == "New Step"
        assert args.description == "Step description"
        assert args.skill == "standard_tutorial"

    def test_modify_step_args_schema(self):
        """测试修改步骤工具参数 Schema"""
        args = ModifyStepArgs(
            step_index=2,
            new_title="Updated Title",
            new_description="Updated Description"
        )
        assert args.step_index == 2
        assert args.new_title == "Updated Title"
        assert args.new_description == "Updated Description"

    def test_delete_step_args_schema(self):
        """测试删除步骤工具参数 Schema"""
        args = DeleteStepArgs(step_index=3)
        assert args.step_index == 3

    def test_regenerate_fragment_args_schema(self):
        """测试重新生成片段工具参数 Schema"""
        args = RegenerateFragmentArgs(step_index=1, reason="内容不满意")
        assert args.step_index == 1
        assert args.reason == "内容不满意"

    def test_user_input_args_schema(self):
        """测试用户输入工具参数 Schema"""
        args = UserInputArgs(
            prompt="请选择下一步操作",
            choices=["继续", "暂停", "修改"]
        )
        assert args.prompt == "请选择下一步操作"
        assert len(args.choices) == 3

    def test_tools_list_contains_all_tools(self):
        """测试工具列表包含所有工具"""
        tool_names = [tool["function"]["name"] for tool in TOOLS]
        assert "retrieve" in tool_names
        assert "write_fragment" in tool_names
        assert "add_step" in tool_names
        assert "modify_step" in tool_names
        assert "delete_step" in tool_names
        assert "regenerate_fragment" in tool_names
        assert "get_current_status" in tool_names
        assert "request_user_input" in tool_names

    def test_each_tool_has_required_fields(self):
        """测试每个工具都有必需字段"""
        for tool in TOOLS:
            assert "type" in tool
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]


class TestToolExecutor:
    """测试工具执行器"""

    @pytest.fixture
    def mock_dependencies(self):
        """创建模拟依赖"""
        llm_service = MagicMock()
        retrieval_service = MagicMock()
        node_factory = MagicMock()
        return llm_service, retrieval_service, node_factory

    @pytest.fixture
    def tool_executor(self, mock_dependencies):
        """创建工具执行器实例"""
        llm_service, retrieval_service, node_factory = mock_dependencies
        return ToolExecutor(
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            node_factory=node_factory
                    )

    @pytest.fixture
    def sample_state(self):
        """创建示例状态"""
        return {
            "outline": [
                {"step_id": 0, "title": "Step 1", "description": "Description 1", "skill": "standard_tutorial"},
                {"step_id": 1, "title": "Step 2", "description": "Description 2", "skill": "standard_tutorial"},
                {"step_id": 2, "title": "Step 3", "description": "Description 3", "skill": "standard_tutorial"},
            ],
            "fragments": [
                {"step_id": 0, "content": "Fragment 1 content", "skill_used": "standard_tutorial"}
            ],
            "last_retrieved_docs": [],
            "current_step_index": 0,
            "awaiting_user_input": False,
            "human_intervention": None
        }

    @pytest.mark.asyncio
    async def test_execute_add_step(self, tool_executor, sample_state):
        """测试执行添加步骤"""
        call = {
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "add_step",
                "arguments": json.dumps({
                    "after_step_id": 1,
                    "title": "New Step",
                    "description": "New Description",
                    "skill": "standard_tutorial"
                })
            }
        }

        result = await tool_executor.execute_tool_call(call, sample_state)

        assert result["success"] is True
        assert result["name"] == "add_step"
        assert "outline" in result["content"]
        assert len(result["content"]["outline"]) == 4

    @pytest.mark.asyncio
    async def test_execute_modify_step(self, tool_executor, sample_state):
        """测试执行修改步骤"""
        call = {
            "id": "call_002",
            "type": "function",
            "function": {
                "name": "modify_step",
                "arguments": json.dumps({
                    "step_index": 1,
                    "new_title": "Updated Title"
                })
            }
        }

        result = await tool_executor.execute_tool_call(call, sample_state)

        assert result["success"] is True
        assert result["name"] == "modify_step"
        assert result["content"]["outline"][1]["title"] == "Updated Title"

    @pytest.mark.asyncio
    async def test_execute_delete_step(self, tool_executor, sample_state):
        """测试执行删除步骤"""
        call = {
            "id": "call_003",
            "type": "function",
            "function": {
                "name": "delete_step",
                "arguments": json.dumps({
                    "step_index": 1
                })
            }
        }

        result = await tool_executor.execute_tool_call(call, sample_state)

        assert result["success"] is True
        assert result["name"] == "delete_step"
        assert len(result["content"]["outline"]) == 2

    @pytest.mark.asyncio
    async def test_execute_get_status(self, tool_executor, sample_state):
        """测试执行获取状态"""
        call = {
            "id": "call_004",
            "type": "function",
            "function": {
                "name": "get_current_status",
                "arguments": "{}"
            }
        }

        result = await tool_executor.execute_tool_call(call, sample_state)

        assert result["success"] is True
        assert result["name"] == "get_current_status"
        assert "total_steps" in result["content"]
        assert "completed_steps" in result["content"]
        assert "progress_percentage" in result["content"]

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, tool_executor, sample_state):
        """测试执行未知工具"""
        call = {
            "id": "call_999",
            "type": "function",
            "function": {
                "name": "unknown_tool",
                "arguments": "{}"
            }
        }

        result = await tool_executor.execute_tool_call(call, sample_state)

        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    def test_merge_tool_result_add_step(self, tool_executor, sample_state):
        """测试合并添加步骤结果"""
        result = {
            "name": "add_step",
            "content": {
                "outline": [
                    {"step_id": 0, "title": "Step 1"},
                    {"step_id": 1, "title": "New Step"},
                    {"step_id": 2, "title": "Step 2"},
                    {"step_id": 3, "title": "Step 3"},
                ]
            }
        }

        updated_state = tool_executor._merge_tool_result(sample_state, result)

        assert len(updated_state["outline"]) == 4

    def test_merge_tool_result_modify_step(self, tool_executor, sample_state):
        """测试合并修改步骤结果"""
        result = {
            "name": "modify_step",
            "content": {
                "outline": [
                    {"step_id": 0, "title": "Step 1"},
                    {"step_id": 1, "title": "Updated Title"},
                    {"step_id": 2, "title": "Step 3"},
                ]
            }
        }

        updated_state = tool_executor._merge_tool_result(sample_state, result)

        assert updated_state["outline"][1]["title"] == "Updated Title"


class TestToolService:
    """测试工具服务"""

    @pytest.fixture
    def mock_llm_service(self):
        """创建模拟 LLM 服务"""
        llm_service = MagicMock()
        adapter = MagicMock()
        adapter.chat_completion_with_tools = AsyncMock(return_value={
            "content": "我已经完成了修改。",
            "tool_calls": [],
            "finish_reason": "stop"
        })
        adapter.chat_completion = AsyncMock(return_value="好的，我来帮你修改。")
        llm_service._get_adapter = MagicMock(return_value=adapter)
        llm_service._get_model_name = MagicMock(return_value="gpt-4o")
        return llm_service

    @pytest.fixture
    def mock_tool_executor(self):
        """创建模拟工具执行器"""
        tool_executor = MagicMock()
        tool_executor.execute_tool_call = AsyncMock(return_value={
            "name": "add_step",
            "content": {"outline": []},
            "success": True,
            "error": None
        })
        tool_executor._merge_tool_result = MagicMock(return_value={})
        return tool_executor

    @pytest.fixture
    def tool_service(self, mock_llm_service, mock_tool_executor):
        """创建工具服务实例"""
        return ToolService(
            llm_service=mock_llm_service,
            tool_executor=mock_tool_executor,
            max_iterations=10
        )

    def test_get_tool_schemas(self, tool_service):
        """测试获取工具 Schema"""
        schemas = tool_service.get_tool_schemas()
        assert len(schemas) == 8
        tool_names = [s["function"]["name"] for s in schemas]
        assert "add_step" in tool_names
        assert "modify_step" in tool_names
        assert "delete_step" in tool_names

    def test_create_system_prompt(self, tool_service):
        """测试创建系统提示词"""
        prompt = tool_service.create_system_prompt(include_tools=True)
        assert "剧本生成" in prompt
        assert "工具" in prompt or "add_step" in prompt

    @pytest.mark.asyncio
    async def test_execute_single_turn_no_tools(self, tool_service, mock_llm_service):
        """测试单轮对话无工具调用"""
        mock_llm_service._get_adapter.return_value.chat_completion_with_tools.return_value = {
            "content": "好的，我来帮你创建剧本大纲。",
            "tool_calls": [],
            "finish_reason": "stop"
        }

        result = await tool_service.execute_single_turn(
            user_message="帮我创建一个剧本",
            state={},
            chat_history=[]
        )

        assert "response" in result
        assert "tool_calls_executed" in result
        assert len(result["tool_calls_executed"]) == 0

    @pytest.mark.asyncio
    async def test_execute_single_turn_with_tools(self, tool_service, mock_llm_service, mock_tool_executor):
        """测试单轮对话有工具调用"""
        mock_llm_service._get_adapter.return_value.chat_completion_with_tools.return_value = {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "add_step",
                        "arguments": json.dumps({
                            "after_step_id": 0,
                            "title": "New Step",
                            "description": "Description"
                        })
                    }
                }
            ],
            "finish_reason": "tool_calls"
        }

        result = await tool_service.execute_single_turn(
            user_message="添加一个新步骤",
            state={},
            chat_history=[]
        )

        assert "response" in result
        assert len(result["tool_calls_executed"]) > 0
        mock_tool_executor.execute_tool_call.assert_called()


class TestIntegration:
    """集成测试"""

    def test_tool_definitions_to_executor_flow(self):
        """测试工具定义到执行器的流程"""
        outline = [
            {"step_id": 0, "title": "Step 1", "description": "Desc 1", "skill": "standard_tutorial"}
        ]

        args = AddStepArgs(
            after_step_id=0,
            title="Step 2",
            description="Desc 2",
            skill="standard_tutorial"
        )

        new_step = {
            "step_id": len(outline),
            "title": args.title,
            "description": args.description,
            "skill": args.skill,
            "status": "pending",
            "dynamically_added": True
        }

        updated_outline = outline[:args.after_step_id + 1] + [new_step] + outline[args.after_step_id + 1:]

        for i, step in enumerate(updated_outline):
            step["step_id"] = i

        assert len(updated_outline) == 2
        assert updated_outline[1]["title"] == "Step 2"
        assert updated_outline[1]["dynamically_added"] is True

    def test_modify_step_updates_correctly(self):
        """测试修改步骤正确更新"""
        outline = [
            {"step_id": 0, "title": "Step 1", "description": "Desc 1"},
            {"step_id": 1, "title": "Step 2", "description": "Desc 2"},
            {"step_id": 2, "title": "Step 3", "description": "Desc 3"},
        ]

        modify_args = ModifyStepArgs(
            step_index=1,
            new_title="Updated Step 2",
            new_description="Updated Description"
        )

        step = outline[modify_args.step_index]
        if modify_args.new_title:
            step["title"] = modify_args.new_title
        if modify_args.new_description:
            step["description"] = modify_args.new_description

        assert outline[1]["title"] == "Updated Step 2"
        assert outline[1]["description"] == "Updated Description"

    def test_delete_step_renumbers_correctly(self):
        """测试删除步骤后重新编号"""
        outline = [
            {"step_id": 0, "title": "Step 1"},
            {"step_id": 1, "title": "Step 2"},
            {"step_id": 2, "title": "Step 3"},
        ]

        delete_index = 1
        deleted_step = outline[delete_index]
        updated_outline = outline[:delete_index] + outline[delete_index + 1:]

        for i, step in enumerate(updated_outline):
            step["step_id"] = i

        assert len(updated_outline) == 2
        assert updated_outline[0]["step_id"] == 0
        assert updated_outline[1]["step_id"] == 1
        assert deleted_step["title"] == "Step 2"
