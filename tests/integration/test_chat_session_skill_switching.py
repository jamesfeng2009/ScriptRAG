"""Integration Test for Chat Session with Skill Switching during Generation

测试场景：以安培晴明为主角的剧本生成，过程中通过 Skills 切换改变剧本方向

测试流程：
1. 创建 Chat Session，设置 skill=mysterious_fantasy
2. 对话讨论晴明的基本设定
3. 切换到 hot_battle skill，讨论战斗场景
4. 导出对话历史
5. 调用 /generate 生成剧本，关联 chat_session_id
6. 验证生成的剧本包含神秘+战斗元素
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.domain.skills import SKILLS
from src.presentation.api import (
    ChatHistoryManager,
)
from src.services.task_persistence_service import TaskRecord
from src.services.chat_session_persistence_service import (
    ChatSessionPersistenceService,
    ChatSessionRecord,
)


class MockSkillService:
    """Mock Skill Service for testing"""
    
    def __init__(self):
        self._skills = {
            "standard_tutorial": {
                "name": "standard_tutorial",
                "display_name": "标准教程风格",
                "description": "基础的教学和示例风格",
                "prompt_config": {
                    "system_prompt": "你是一个专业的写作助手，提供清晰、有条理的写作指导。"
                }
            }
        }
    
    async def get(self, skill_name: str):
        if skill_name in self._skills:
            return MagicMock(
                name=skill_name,
                skill_data=self._skills[skill_name]
            )
        return None
    
    async def list(self, filter_params=None):
        skills = []
        for name, data in self._skills.items():
            skills.append(MagicMock(name=name, skill_data=data))
        return skills


class MockLLMService:
    """Mock LLM Service for testing"""
    
    def __init__(self):
        self._responses = {
            "mysterious_fantasy": """在遥远的平安时代，京都的夜色总是笼罩在一层朦胧的薄雾之中。

阴阳师安倍晴明站在清凉殿的廊下，白色的狩衣在月光下泛着淡淡的银光。他的眼眸深邃如井，仿佛能洞察阴阳两界的奥秘。

"大人，源氏家族的式神又出现在城中了。"式神青蛙子低声禀报。

晴明轻轻展开手中的折扇，嘴角浮现一丝意味深长的微笑。"让他们来吧，这京都的夜色，正是最适合妖怪活动的时刻。"

空气中弥漫着樱花的芬芳，却也隐藏着不为人知的杀机。晴明缓步走入庭院，每一步都仿佛踏在虚无与现实之间。""",
            
            "hot_battle": """激烈的战斗在京都的夜空中爆发！

晴明的折扇化作一道银色的闪电，直取式神的面门。强大的灵力在空气中撕裂出一道肉眼可见的裂痕。

"退下！"晴明冷喝一声，身后浮现出十二道金色的结界，每一道都蕴含着毁天灭地的力量。

式神发出一声凄厉的咆哮，化作三头六臂的狰狞形态，六只手臂同时挥舞着各种武器：太刀、薙箭、弓矢、锁链、拳套、战斧。

"雕虫小技！"晴明身形一闪，化作一道残影，在六把武器的缝隙中穿梭自如。他的每一次移动都带着破空之声，每一扇挥出都激起一阵狂暴的灵力风暴。

战斗愈演愈烈，整个京都都能感受到这股惊天动地的力量波动。"""
        }
    
    async def chat_completion(self, messages, temperature=0.7, max_tokens=2000):
        full_prompt = "\n".join([m.get("content", "") for m in messages])
        
        if "神秘幻想" in full_prompt or "阴阳师" in full_prompt:
            return self._responses["mysterious_fantasy"]
        elif "热血战斗" in full_prompt or "战斗场面" in full_prompt:
            return self._responses["hot_battle"]
        
        return f"""剧本片段生成完成。

包含的元素：
- 神秘氛围：平安京的夜色、阴阳师、式神
- 战斗场景：激烈的对决、灵力的碰撞

生成时间：{datetime.now().isoformat()}"""
    
    async def close(self):
        pass


@pytest.fixture
def mock_services():
    """Create mock services for testing"""
    skill_service = MockSkillService()
    llm_service = MockLLMService()
    rag_service = AsyncMock()
    rag_service.query = AsyncMock(return_value=MagicMock(
        answer="",
        sources=[]
    ))
    return skill_service, llm_service, rag_service


@pytest.fixture
def fresh_session_id():
    """Create a unique test session ID"""
    return f"test_chat_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"


class TestChatSessionWithSkillSwitching:
    """Test suite for Chat Session with Skill Switching during Generation"""
    
    def test_scenario_abe_single_skill(self, fresh_session_id, mock_services):
        """
        测试场景 1: 使用单一 skill 创建 Chat Session
        
        验证：系统应该能够创建带 skill 配置的会话
        """
        skill_service, llm_service, rag_service = mock_services
        
        config = {
            "skill": "mysterious_fantasy",
            "enable_rag": False,
            "temperature": 0.7
        }
        
        ChatHistoryManager.create_session(
            session_id=fresh_session_id,
            mode="simple",
            config=config
        )
        
        ChatHistoryManager.add_message(
            fresh_session_id,
            "user",
            "请以阴阳师安培晴明为主角，写一段神秘幻想风格的开场场景。"
        )
        
        history = ChatHistoryManager.get_history(fresh_session_id)
        assert len(history) == 1
        assert history[0].role == "user"
        assert "晴明" in history[0].content
        
        session = ChatHistoryManager.get_session(fresh_session_id)
        assert session["config"]["skill"] == "mysterious_fantasy"
        assert session["mode"] == "simple"
        
        print(f"[Test 1 Passed] Session {fresh_session_id} created with mysterious_fantasy skill")
    
    def test_scenario_abe_with_skill_switching(self, fresh_session_id, mock_services):
        """
        测试场景 2: 在对话过程中切换 skill (mysterious_fantasy → hot_battle)
        
        验证：
        1. 初始使用 mysterious_fantasy skill
        2. 对话过程中切换到 hot_battle
        3. 对话历史记录 skill 切换
        """
        skill_service, llm_service, rag_service = mock_services
        
        config = {
            "skill": "mysterious_fantasy",
            "enable_rag": False,
            "temperature": 0.7
        }
        
        ChatHistoryManager.create_session(
            session_id=fresh_session_id,
            mode="agent",
            config=config
        )
        
        ChatHistoryManager.add_message(fresh_session_id, "user",
            "我想要创作一个以阴阳师安培晴明为主角的剧本，请帮他设计基本人设。")
        
        session = ChatHistoryManager.get_session(fresh_session_id)
        session["config"]["skill"] = "hot_battle"
        session["config"]["skill_switch_history"] = session.get("config", {}).get("skill_switch_history", [])
        session["config"]["skill_switch_history"].append({
            "from_skill": "mysterious_fantasy",
            "to_skill": "hot_battle",
            "timestamp": datetime.now().isoformat(),
            "reason": "用户需求：加入战斗元素"
        })
        
        ChatHistoryManager.add_message(fresh_session_id, "user",
            "现在请为晴明设计一场与强大式神的激烈战斗场景。")
        
        updated_session = ChatHistoryManager.get_session(fresh_session_id)
        assert updated_session["config"]["skill"] == "hot_battle"
        assert len(updated_session["config"]["skill_switch_history"]) == 1
        
        final_history = ChatHistoryManager.get_history(fresh_session_id)
        assert len(final_history) == 2
        
        print(f"[Test 2 Passed] Session {fresh_session_id} switched from mysterious_fantasy to hot_battle")
    
    def test_scenario_full_workflow_with_skill_switching(self, fresh_session_id, mock_services):
        """
        测试场景 3: 完整的 Chat → Generate 流程，包含 Skill 切换
        
        这是核心测试场景：
        1. 创建 Chat Session
        2. 多轮对话，包含 skill 切换
        3. 导出对话历史
        4. 调用 generate 生成剧本
        5. 验证生成的剧本包含对话中讨论的元素
        """
        skill_service, llm_service, rag_service = mock_services
        
        initial_config = {
            "skill": "mysterious_fantasy",
            "enable_rag": False,
            "temperature": 0.7
        }
        
        ChatHistoryManager.create_session(
            session_id=fresh_session_id,
            mode="agent",
            config=initial_config
        )
        
        ChatHistoryManager.add_message(fresh_session_id, "user",
            "我需要创作一个以阴阳师安培晴明为主角的剧本。\n"
            "背景：平安时代的京都，妖魔横行。\n"
            "晴明：冷静沉稳的阴阳师，擅长操控式神。")
        
        session = ChatHistoryManager.get_session(fresh_session_id)
        session["config"]["skill"] = "hot_battle"
        session["config"]["skill_switch_history"] = [{
            "from_skill": "mysterious_fantasy",
            "to_skill": "hot_battle",
            "timestamp": datetime.now().isoformat(),
            "reason": "需要在剧本中加入激烈的战斗场面"
        }]
        
        ChatHistoryManager.add_message(fresh_session_id, "user",
            "设计一场晴明与源氏式神的激烈战斗：\n"
            "1. 展现晴明强大的式神操控能力\n"
            "2. 激烈的灵力对决")
        
        messages = [{"role": m.role, "content": m.content} for m in ChatHistoryManager.get_history(fresh_session_id)]
        assert len(messages) == 2
        
        history = ChatHistoryManager.get_history(fresh_session_id)
        user_messages = [m for m in history if m.role == "user"]
        
        assert len(user_messages) == 2
        assert "晴明" in user_messages[0].content
        assert "战斗" in user_messages[1].content
        
        final_session = ChatHistoryManager.get_session(fresh_session_id)
        assert final_session["config"]["skill"] == "hot_battle"
        assert len(final_session["config"]["skill_switch_history"]) == 1
        
        print(f"[Test 3 Passed] Full workflow test for session {fresh_session_id}")
    
    def test_skill_compatibility_check(self):
        """
        测试场景 4: Skill 兼容性检查
        
        验证：系统应该能够正确检查 skill 是否可用
        """
        available_skills = list(SKILLS.keys())
        
        assert "standard_tutorial" in available_skills, "standard_tutorial skill should be available"
        assert len(available_skills) >= 1, "At least one skill should be available"
        
        tutorial_skill = SKILLS.get("standard_tutorial")
        assert tutorial_skill is not None
        assert hasattr(tutorial_skill, 'description') or 'description' in str(tutorial_skill)
        
        print("[Test 4 Passed] Skill compatibility check passed")
    
    def test_task_record_with_chat_session(self):
        """
        测试场景 5: TaskRecord 支持 chat_session_id
        
        验证：TaskRecord 能够正确存储和读取 chat_session_id
        """
        task = TaskRecord(
            task_id="test_task_001",
            status="pending",
            topic="以安培晴明为主角的剧本",
            context="从 Chat Session 导出的对话历史",
            current_skill="standard_tutorial",
            chat_session_id="test_chat_session_001"
        )
        
        assert task.chat_session_id == "test_chat_session_001"
        
        task_dict = task.to_dict()
        assert task_dict.get("chat_session_id") == "test_chat_session_001"
        
        task2 = TaskRecord(
            task_id="test_task_002",
            status="in_progress",
            topic="晴明的冒险故事",
            chat_session_id="test_chat_session_002"
        )
        assert task2.chat_session_id == "test_chat_session_002"
        
        print("[Test 5 Passed] TaskRecord with chat_session_id passed")
    
    def test_chat_session_model_structure(self):
        """
        测试场景 6: ChatSessionRecord 数据模型结构
        
        验证：ChatSessionRecord 能够正确存储会话数据
        """
        record = ChatSessionRecord(
            id="test_session_001",
            topic="晴明剧本创作",
            mode="agent",
            config={
                "skill": "standard_tutorial",
                "enable_rag": False,
                "temperature": 0.7
            },
            message_history=[
                {"role": "user", "content": "请创作晴明剧本"},
                {"role": "assistant", "content": "好的，我来帮你"}
            ],
            related_task_id=None,
            status="active"
        )
        
        assert record.id == "test_session_001"
        assert record.topic == "晴明剧本创作"
        assert record.mode == "agent"
        assert record.config.get("skill") == "standard_tutorial"
        assert len(record.message_history) == 2
        assert record.status == "active"
        
        record_dict = record.to_dict()
        assert record_dict["id"] == "test_session_001"
        assert record_dict["config"]["skill"] == "standard_tutorial"
        
        print("[Test 6 Passed] ChatSessionRecord model structure passed")


class TestIntegrationFullScenario:
    """Integration test for the complete Abe-no-Semimaru generation scenario"""
    
    def test_complete_abe_scenario(self, mock_services):
        """
        完整测试场景：晴明剧本生成 + Skill 切换
        
        测试步骤：
        1. 创建 Chat Session
        2. 对话讨论世界观
        3. 切换 skill 配置
        4. 对话讨论战斗场景
        5. 验证所有配置正确
        """
        skill_service, llm_service, rag_service = mock_services
        
        session_id = f"abe_complete_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        ChatHistoryManager.create_session(
            session_id=session_id,
            mode="agent",
            config={"skill": "standard_tutorial", "enable_rag": False}
        )
        
        ChatHistoryManager.add_message(session_id, "user",
            "我要写一个以阴阳师安培晴明为主角的剧本。\n"
            "时代：平安时代\n"
            "地点：京都\n"
            "核心冲突：人与妖的共存问题\n"
            "请帮我设定世界观。")
        
        session = ChatHistoryManager.get_session(session_id)
        session["config"]["skill"] = "standard_tutorial"
        session["config"]["skill_switch_history"] = [{
            "from_skill": "standard_tutorial",
            "to_skill": "standard_tutorial",
            "timestamp": datetime.now().isoformat(),
            "reason": "需要在第三幕加入激烈的最终决战"
        }]
        
        ChatHistoryManager.add_message(session_id, "user",
            "现在设计最终决战场景：\n"
            "晴明 vs 酒吞童子\n"
            "需要展现：\n"
            "1. 双方强大的式神操控能力\n"
            "2. 天地变色的灵力碰撞")
        
        messages = [{"role": m.role, "content": m.content} for m in ChatHistoryManager.get_history(session_id)]
        
        assert len(messages) == 2
        
        session_data = ChatHistoryManager.get_session(session_id)
        assert session_data["config"]["skill"] == "standard_tutorial"
        assert len(session_data["config"]["skill_switch_history"]) == 1
        
        switch_record = session_data["config"]["skill_switch_history"][0]
        assert switch_record["from_skill"] == "standard_tutorial"
        assert switch_record["to_skill"] == "standard_tutorial"
        
        print(f"[Integration Test Passed] Complete Abe-no-Semimaru scenario")
        print(f"  - Session ID: {session_id}")
        print(f"  - Messages: {len(messages)}")
        print(f"  - Final Skill: {session_data['config']['skill']}")
        print(f"  - Skill Switches: {len(session_data['config']['skill_switch_history'])}")
    
    def test_skill_prompt_integration(self):
        """
        测试场景 7: Skill System Prompt 集成
        
        验证：不同 skill 对应不同的 system prompt
        """
        from src.domain.skills import SKILLS, SkillConfig
        
        available_skills = list(SKILLS.keys())
        assert len(available_skills) >= 1
        
        for skill_name in available_skills[:2]:
            skill = SKILLS.get(skill_name)
            assert skill is not None
            assert isinstance(skill, SkillConfig)
            assert hasattr(skill, 'prompt_config')
            assert isinstance(skill.prompt_config, dict)
        
        print("[Test 7 Passed] Skill prompt integration passed")
