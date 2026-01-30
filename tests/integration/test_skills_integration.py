"""Integration tests for Skills system with SharedState"""

import pytest
from src.domain.models import SharedState, OutlineStep
from src.domain.skills import (
    SKILLS,
    RETRIEVAL_CONFIG,
    SkillManager,
    check_skill_compatibility,
    default_skill_manager
)


class TestSkillsWithSharedState:
    """Test Skills system integration with SharedState"""
    
    def test_shared_state_with_default_skill(self):
        """Test that SharedState can use default skill"""
        state = SharedState(
            user_topic="Test topic",
            project_context="Test context",
            outline=[
                OutlineStep(step_id=1, description="Step 1", status="pending")
            ]
        )
        
        assert state.current_skill == "standard_tutorial"
        assert state.current_skill in SKILLS
    
    def test_shared_state_skill_switching(self):
        """Test skill switching in SharedState"""
        state = SharedState(
            user_topic="Test topic",
            project_context="Test context",
            outline=[
                OutlineStep(step_id=1, description="Step 1", status="pending")
            ],
            current_skill="standard_tutorial"
        )
        
        # Switch to compatible skill
        new_skill = "warning_mode"
        assert check_skill_compatibility(state.current_skill, new_skill)
        
        state.current_skill = new_skill
        assert state.current_skill == "warning_mode"
    
    def test_shared_state_with_global_tone(self):
        """Test SharedState with global_tone matching skill tone"""
        state = SharedState(
            user_topic="Test topic",
            project_context="Test context",
            outline=[
                OutlineStep(step_id=1, description="Step 1", status="pending")
            ],
            current_skill="standard_tutorial",
            global_tone="professional"
        )
        
        skill_config = SKILLS[state.current_skill]
        assert skill_config.tone == state.global_tone
    
    def test_skill_manager_with_state_workflow(self):
        """Test using SkillManager in a workflow scenario"""
        manager = SkillManager()
        
        # Initial state
        state = SharedState(
            user_topic="Deprecated API usage",
            project_context="Legacy codebase",
            outline=[
                OutlineStep(step_id=1, description="Explain deprecated API", status="pending")
            ],
            current_skill="standard_tutorial"
        )
        
        # Detect deprecation conflict, need to switch to warning_mode
        desired_skill = "warning_mode"
        
        # Check if we can switch
        if manager.check_compatibility(state.current_skill, desired_skill):
            state.current_skill = desired_skill
        else:
            # Find closest compatible skill
            state.current_skill = manager.find_compatible_skill(
                state.current_skill,
                desired_skill,
                state.global_tone
            )
        
        assert state.current_skill == "warning_mode"
        assert SKILLS[state.current_skill].tone == "cautionary"
    
    def test_retrieval_config_with_workflow(self):
        """Test using RETRIEVAL_CONFIG in a workflow scenario"""
        # Simulate retrieval configuration usage
        vector_config = RETRIEVAL_CONFIG.vector_search
        keyword_config = RETRIEVAL_CONFIG.keyword_search
        
        # Verify configuration is accessible and valid
        assert vector_config.top_k > 0
        assert vector_config.similarity_threshold > 0
        assert len(keyword_config.markers) > 0
        assert keyword_config.boost_factor > 1.0
    
    def test_complex_skill_switching_scenario(self):
        """Test complex skill switching scenario with multiple transitions"""
        manager = SkillManager()
        
        state = SharedState(
            user_topic="Complex algorithm explanation",
            project_context="Advanced codebase",
            outline=[
                OutlineStep(step_id=1, description="Explain algorithm", status="pending")
            ],
            current_skill="standard_tutorial",
            global_tone="professional"
        )
        
        # Scenario 1: Content is too complex, switch to visualization_analogy
        if manager.check_compatibility(state.current_skill, "visualization_analogy"):
            state.current_skill = "visualization_analogy"
        
        assert state.current_skill == "visualization_analogy"
        
        # Scenario 2: Need to make it more casual, try meme_style
        if manager.check_compatibility(state.current_skill, "meme_style"):
            state.current_skill = "meme_style"
        
        assert state.current_skill == "meme_style"
        
        # Scenario 3: Information is missing, need research_mode
        # meme_style is not compatible with research_mode, find path
        closest = manager.find_compatible_skill(
            state.current_skill,
            "research_mode",
            state.global_tone
        )
        
        # Should find fallback_summary as intermediate
        assert closest in manager.get_compatible_skills(state.current_skill)


class TestSkillManagerExtensibility:
    """Test SkillManager extensibility features"""
    
    def test_register_custom_skill_for_domain(self):
        """Test registering a custom skill for specific domain"""
        from src.domain.skills import SkillConfig
        
        manager = SkillManager()
        
        # Register a custom skill for technical documentation
        custom_skill = SkillConfig(
            description="Technical API documentation style",
            tone="technical",
            compatible_with=["standard_tutorial", "warning_mode"]
        )
        
        manager.register_skill("api_documentation", custom_skill)
        
        assert "api_documentation" in manager.list_skills()
        # Check compatibility from the new skill to existing skills
        assert manager.check_compatibility("api_documentation", "standard_tutorial")
        assert manager.check_compatibility("api_documentation", "warning_mode")
    
    def test_skill_manager_isolation(self):
        """Test that different SkillManager instances are isolated"""
        manager1 = SkillManager()
        manager2 = SkillManager()
        
        from src.domain.skills import SkillConfig
        
        custom_skill = SkillConfig(
            description="Custom skill 1",
            tone="custom",
            compatible_with=["standard_tutorial"]
        )
        
        manager1.register_skill("custom1", custom_skill)
        
        assert "custom1" in manager1.list_skills()
        assert "custom1" not in manager2.list_skills()
