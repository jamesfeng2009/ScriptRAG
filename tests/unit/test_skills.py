"""Unit tests for Skills system configuration"""

import pytest
from src.domain.skills import (
    SKILLS,
    RETRIEVAL_CONFIG,
    SkillConfig,
    RetrievalConfig,
    SkillManager,
    check_skill_compatibility,
    get_compatible_skills,
    find_closest_compatible_skill,
    default_skill_manager
)


class TestSkillsConfiguration:
    """Test the SKILLS dictionary configuration"""
    
    def test_all_six_skills_defined(self):
        """Test that all six required skills are defined"""
        expected_skills = {
            "standard_tutorial",
            "warning_mode",
            "visualization_analogy",
            "research_mode",
            "meme_style",
            "fallback_summary"
        }
        assert set(SKILLS.keys()) == expected_skills
    
    def test_skill_has_required_attributes(self):
        """Test that each skill has description, tone, and compatible_with"""
        for skill_name, skill_config in SKILLS.items():
            assert isinstance(skill_config, SkillConfig)
            assert skill_config.description
            assert skill_config.tone
            assert isinstance(skill_config.compatible_with, list)
    
    def test_standard_tutorial_config(self):
        """Test standard_tutorial skill configuration"""
        skill = SKILLS["standard_tutorial"]
        assert skill.tone == "professional"
        assert "visualization_analogy" in skill.compatible_with
        assert "warning_mode" in skill.compatible_with
    
    def test_warning_mode_config(self):
        """Test warning_mode skill configuration"""
        skill = SKILLS["warning_mode"]
        assert skill.tone == "cautionary"
        assert "standard_tutorial" in skill.compatible_with
        assert "research_mode" in skill.compatible_with
    
    def test_visualization_analogy_config(self):
        """Test visualization_analogy skill configuration"""
        skill = SKILLS["visualization_analogy"]
        assert skill.tone == "engaging"
        assert "standard_tutorial" in skill.compatible_with
        assert "meme_style" in skill.compatible_with
    
    def test_research_mode_config(self):
        """Test research_mode skill configuration"""
        skill = SKILLS["research_mode"]
        assert skill.tone == "exploratory"
        assert "standard_tutorial" in skill.compatible_with
        assert "warning_mode" in skill.compatible_with
    
    def test_meme_style_config(self):
        """Test meme_style skill configuration"""
        skill = SKILLS["meme_style"]
        assert skill.tone == "casual"
        assert "visualization_analogy" in skill.compatible_with
        assert "fallback_summary" in skill.compatible_with
    
    def test_fallback_summary_config(self):
        """Test fallback_summary skill configuration"""
        skill = SKILLS["fallback_summary"]
        assert skill.tone == "neutral"
        assert "standard_tutorial" in skill.compatible_with
        assert "research_mode" in skill.compatible_with


class TestRetrievalConfiguration:
    """Test the RETRIEVAL_CONFIG configuration"""
    
    def test_retrieval_config_exists(self):
        """Test that RETRIEVAL_CONFIG is properly initialized"""
        assert isinstance(RETRIEVAL_CONFIG, RetrievalConfig)
    
    def test_vector_search_config(self):
        """Test vector search configuration"""
        config = RETRIEVAL_CONFIG.vector_search
        assert config.top_k == 5
        assert config.similarity_threshold == 0.7
        assert config.embedding_model == "text-embedding-3-large"
    
    def test_keyword_search_config(self):
        """Test keyword search configuration"""
        config = RETRIEVAL_CONFIG.keyword_search
        assert "@deprecated" in config.markers
        assert "FIXME" in config.markers
        assert "TODO" in config.markers
        assert "Security" in config.markers
        assert config.boost_factor == 1.5
    
    def test_hybrid_merge_config(self):
        """Test hybrid merge configuration"""
        config = RETRIEVAL_CONFIG.hybrid_merge
        assert config.vector_weight == 0.6
        assert config.keyword_weight == 0.4
        assert config.keyword_boost_factor == 1.5
        assert config.dedup_threshold == 0.9
    
    def test_summarization_config(self):
        """Test summarization configuration"""
        config = RETRIEVAL_CONFIG.summarization
        assert config.max_tokens == 10000
        assert config.chunk_size == 2000
        assert config.overlap == 200


class TestSkillCompatibilityFunctions:
    """Test skill compatibility checking functions"""
    
    def test_check_compatibility_same_skill(self):
        """Test that a skill is compatible with itself"""
        assert check_skill_compatibility("standard_tutorial", "standard_tutorial")
    
    def test_check_compatibility_valid_pair(self):
        """Test compatibility between valid skill pairs"""
        assert check_skill_compatibility("standard_tutorial", "visualization_analogy")
        assert check_skill_compatibility("warning_mode", "research_mode")
        assert check_skill_compatibility("meme_style", "fallback_summary")
    
    def test_check_compatibility_invalid_pair(self):
        """Test incompatibility between invalid skill pairs"""
        assert not check_skill_compatibility("standard_tutorial", "meme_style")
        assert not check_skill_compatibility("warning_mode", "visualization_analogy")
    
    def test_check_compatibility_invalid_skill_raises(self):
        """Test that invalid skill names raise ValueError"""
        with pytest.raises(ValueError, match="Invalid current skill"):
            check_skill_compatibility("invalid_skill", "standard_tutorial")
        
        with pytest.raises(ValueError, match="Invalid target skill"):
            check_skill_compatibility("standard_tutorial", "invalid_skill")
    
    def test_get_compatible_skills(self):
        """Test getting list of compatible skills"""
        compatible = get_compatible_skills("standard_tutorial")
        assert "visualization_analogy" in compatible
        assert "warning_mode" in compatible
        assert "research_mode" in compatible
        assert "fallback_summary" in compatible
        assert len(compatible) == 4
    
    def test_get_compatible_skills_invalid_raises(self):
        """Test that invalid skill name raises ValueError"""
        with pytest.raises(ValueError, match="Invalid skill"):
            get_compatible_skills("invalid_skill")
    
    def test_find_closest_compatible_direct(self):
        """Test finding closest compatible skill when directly compatible"""
        result = find_closest_compatible_skill(
            "standard_tutorial",
            "visualization_analogy"
        )
        assert result == "visualization_analogy"
    
    def test_find_closest_compatible_indirect(self):
        """Test finding closest compatible skill when not directly compatible"""
        # standard_tutorial -> research_mode (directly compatible)
        result = find_closest_compatible_skill(
            "standard_tutorial",
            "research_mode"
        )
        # Should return research_mode directly since it's directly compatible
        assert result == "research_mode"
    
    def test_find_closest_compatible_with_tone(self):
        """Test finding closest compatible skill with tone preference"""
        result = find_closest_compatible_skill(
            "standard_tutorial",
            "meme_style",
            global_tone="professional"
        )
        # Should prefer professional tone if available
        assert result in get_compatible_skills("standard_tutorial")
    
    def test_find_closest_compatible_no_path(self):
        """Test finding closest compatible when no path exists"""
        result = find_closest_compatible_skill(
            "warning_mode",
            "meme_style"
        )
        # Should return a compatible skill from warning_mode
        assert result in ["standard_tutorial", "research_mode", "fallback_summary"]


class TestSkillManager:
    """Test the SkillManager class"""
    
    def test_skill_manager_initialization(self):
        """Test SkillManager initialization with default skills"""
        manager = SkillManager()
        assert len(manager.list_skills()) == 6
        assert "standard_tutorial" in manager.list_skills()
    
    def test_skill_manager_with_custom_skills(self):
        """Test SkillManager initialization with custom skills"""
        custom_skills = {
            "custom_skill": SkillConfig(
                description="Custom skill",
                tone="custom",
                compatible_with=["standard_tutorial"]
            )
        }
        manager = SkillManager(custom_skills=custom_skills)
        assert "custom_skill" in manager.list_skills()
        assert len(manager.list_skills()) == 7
    
    def test_register_skill(self):
        """Test registering a new skill"""
        manager = SkillManager()
        new_skill = SkillConfig(
            description="New skill",
            tone="new",
            compatible_with=["standard_tutorial"]
        )
        manager.register_skill("new_skill", new_skill)
        assert "new_skill" in manager.list_skills()
        assert manager.get_skill("new_skill") == new_skill
    
    def test_register_skill_invalid_compatibility_raises(self):
        """Test that registering skill with invalid compatibility raises error"""
        manager = SkillManager()
        invalid_skill = SkillConfig(
            description="Invalid skill",
            tone="invalid",
            compatible_with=["nonexistent_skill"]
        )
        with pytest.raises(ValueError, match="not found in registered skills"):
            manager.register_skill("invalid_skill", invalid_skill)
    
    def test_register_multiple_skills(self):
        """Test registering multiple skills at once"""
        manager = SkillManager()
        new_skills = {
            "skill1": SkillConfig(
                description="Skill 1",
                tone="tone1",
                compatible_with=["standard_tutorial"]
            ),
            "skill2": SkillConfig(
                description="Skill 2",
                tone="tone2",
                compatible_with=["skill1"]
            )
        }
        manager.register_skills(new_skills)
        assert "skill1" in manager.list_skills()
        assert "skill2" in manager.list_skills()
    
    def test_get_skill(self):
        """Test getting skill configuration"""
        manager = SkillManager()
        skill = manager.get_skill("standard_tutorial")
        assert isinstance(skill, SkillConfig)
        assert skill.tone == "professional"
    
    def test_get_skill_not_found_raises(self):
        """Test that getting nonexistent skill raises ValueError"""
        manager = SkillManager()
        with pytest.raises(ValueError, match="Skill not found"):
            manager.get_skill("nonexistent_skill")
    
    def test_check_compatibility(self):
        """Test checking skill compatibility through manager"""
        manager = SkillManager()
        assert manager.check_compatibility("standard_tutorial", "visualization_analogy")
        assert not manager.check_compatibility("standard_tutorial", "meme_style")
    
    def test_get_compatible_skills_manager(self):
        """Test getting compatible skills through manager"""
        manager = SkillManager()
        compatible = manager.get_compatible_skills("standard_tutorial")
        assert "visualization_analogy" in compatible
        assert "warning_mode" in compatible
    
    def test_find_compatible_skill(self):
        """Test finding compatible skill through manager"""
        manager = SkillManager()
        result = manager.find_compatible_skill(
            "standard_tutorial",
            "visualization_analogy"
        )
        assert result == "visualization_analogy"
    
    def test_get_skill_by_tone(self):
        """Test getting skills by tone"""
        manager = SkillManager()
        professional_skills = manager.get_skill_by_tone("professional")
        assert "standard_tutorial" in professional_skills
        
        casual_skills = manager.get_skill_by_tone("casual")
        assert "meme_style" in casual_skills
    
    def test_validate_skill_graph(self):
        """Test validating skill compatibility graph"""
        manager = SkillManager()
        assert manager.validate_skill_graph()
    
    def test_validate_skill_graph_invalid(self):
        """Test that invalid skill graph fails validation"""
        manager = SkillManager()
        # Manually add invalid skill to bypass registration validation
        manager._skills["invalid"] = SkillConfig(
            description="Invalid",
            tone="invalid",
            compatible_with=["nonexistent"]
        )
        assert not manager.validate_skill_graph()


class TestDefaultSkillManager:
    """Test the default skill manager instance"""
    
    def test_default_manager_exists(self):
        """Test that default skill manager is initialized"""
        assert isinstance(default_skill_manager, SkillManager)
    
    def test_default_manager_has_all_skills(self):
        """Test that default manager has all six skills"""
        assert len(default_skill_manager.list_skills()) == 6
    
    def test_default_manager_is_valid(self):
        """Test that default manager has valid skill graph"""
        assert default_skill_manager.validate_skill_graph()


class TestSkillCompatibilityGraph:
    """Test the skill compatibility graph structure"""
    
    def test_standard_tutorial_connections(self):
        """Test standard_tutorial compatibility connections"""
        compatible = get_compatible_skills("standard_tutorial")
        assert set(compatible) == {"visualization_analogy", "warning_mode", "research_mode", "fallback_summary"}
    
    def test_warning_mode_connections(self):
        """Test warning_mode compatibility connections"""
        compatible = get_compatible_skills("warning_mode")
        assert set(compatible) == {"standard_tutorial", "research_mode", "fallback_summary"}
    
    def test_visualization_analogy_connections(self):
        """Test visualization_analogy compatibility connections"""
        compatible = get_compatible_skills("visualization_analogy")
        assert set(compatible) == {"standard_tutorial", "meme_style", "research_mode"}
    
    def test_research_mode_connections(self):
        """Test research_mode compatibility connections"""
        compatible = get_compatible_skills("research_mode")
        assert set(compatible) == {"standard_tutorial", "warning_mode", "visualization_analogy", "fallback_summary"}
    
    def test_meme_style_connections(self):
        """Test meme_style compatibility connections"""
        compatible = get_compatible_skills("meme_style")
        assert set(compatible) == {"visualization_analogy", "fallback_summary", "standard_tutorial"}
    
    def test_fallback_summary_connections(self):
        """Test fallback_summary compatibility connections"""
        compatible = get_compatible_skills("fallback_summary")
        assert set(compatible) == {"standard_tutorial", "research_mode", "warning_mode", "meme_style"}
    
    def test_all_skills_have_at_least_one_compatible(self):
        """Test that every skill has at least one compatible skill"""
        for skill_name in SKILLS.keys():
            compatible = get_compatible_skills(skill_name)
            assert len(compatible) >= 1, f"{skill_name} has no compatible skills"
    
    def test_compatibility_is_bidirectional_where_expected(self):
        """Test that certain compatibility relationships are bidirectional"""
        # standard_tutorial <-> warning_mode
        assert check_skill_compatibility("standard_tutorial", "warning_mode")
        assert check_skill_compatibility("warning_mode", "standard_tutorial")
        
        # standard_tutorial <-> visualization_analogy
        assert check_skill_compatibility("standard_tutorial", "visualization_analogy")
        assert check_skill_compatibility("visualization_analogy", "standard_tutorial")
