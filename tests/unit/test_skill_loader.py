"""Tests for Skill Configuration Loader"""

import pytest
import yaml
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from src.domain.skill_loader import (
    SkillConfigLoader,
    PromptConfig,
    SkillConfigYAML,
    SkillsConfigFile,
    create_default_config
)
from src.domain.skills import SkillConfig, SkillManager


class TestPromptConfig:
    """Test PromptConfig model"""
    
    def test_valid_prompt_config(self):
        """Test creating valid prompt config"""
        config = PromptConfig(
            system_prompt="You are a helpful assistant",
            user_template="Question: {step_description}\nContext: {retrieved_content}",
            temperature=0.7,
            max_tokens=2000
        )
        
        assert config.system_prompt == "You are a helpful assistant"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
    
    def test_empty_prompt_raises_error(self):
        """Test that empty prompts raise validation error"""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            PromptConfig(
                system_prompt="",
                user_template="test",
                temperature=0.7,
                max_tokens=2000
            )
    
    def test_invalid_temperature_raises_error(self):
        """Test that invalid temperature raises error"""
        with pytest.raises(ValueError):
            PromptConfig(
                system_prompt="test",
                user_template="test",
                temperature=1.5,  # Invalid: > 1.0
                max_tokens=2000
            )


class TestSkillConfigLoader:
    """Test SkillConfigLoader"""
    
    @pytest.fixture
    def sample_config_file(self):
        """Create a sample configuration file"""
        config_data = {
            'version': '1.0',
            'skills': {
                'test_skill': {
                    'description': 'Test skill',
                    'tone': 'professional',
                    'compatible_with': [],
                    'prompt_config': {
                        'system_prompt': 'You are a test assistant',
                        'user_template': 'Question: {step_description}\nContext: {retrieved_content}',
                        'temperature': 0.7,
                        'max_tokens': 2000
                    },
                    'enabled': True,
                    'metadata': {'category': 'test'}
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        yield temp_path
        
        # Cleanup
        temp_path.unlink()
    
    def test_load_from_yaml(self, sample_config_file):
        """Test loading skills from YAML file"""
        loader = SkillConfigLoader(str(sample_config_file))
        skills = loader.load_from_yaml()
        
        assert len(skills) == 1
        assert 'test_skill' in skills
        assert skills['test_skill'].description == 'Test skill'
        assert skills['test_skill'].tone == 'professional'
    
    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises error"""
        loader = SkillConfigLoader('nonexistent.yaml')
        
        with pytest.raises(FileNotFoundError):
            loader.load_from_yaml()
    
    def test_load_invalid_yaml_raises_error(self):
        """Test that invalid YAML raises error"""
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:")
            temp_path = Path(f.name)
        
        try:
            loader = SkillConfigLoader(str(temp_path))
            with pytest.raises(ValueError, match="Invalid YAML"):
                loader.load_from_yaml()
        finally:
            temp_path.unlink()
    
    def test_load_prompt_configs(self, sample_config_file):
        """Test loading prompt configurations"""
        loader = SkillConfigLoader(str(sample_config_file))
        prompt_configs = loader.load_prompt_configs()
        
        assert len(prompt_configs) == 1
        assert 'test_skill' in prompt_configs
        assert prompt_configs['test_skill'].system_prompt == 'You are a test assistant'
        assert prompt_configs['test_skill'].temperature == 0.7
    
    def test_validate_config_valid(self, sample_config_file):
        """Test validating valid configuration"""
        loader = SkillConfigLoader(str(sample_config_file))
        
        with open(sample_config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        assert loader.validate_config(config_dict) is True
    
    def test_validate_config_invalid_compatibility(self):
        """Test that invalid compatibility reference fails validation"""
        config_data = {
            'version': '1.0',
            'skills': {
                'test_skill': {
                    'description': 'Test skill',
                    'tone': 'professional',
                    'compatible_with': ['nonexistent_skill'],  # Invalid reference
                    'prompt_config': {
                        'system_prompt': 'test',
                        'user_template': 'test {step_description} {retrieved_content}',
                        'temperature': 0.7,
                        'max_tokens': 2000
                    },
                    'enabled': True,
                    'metadata': {}
                }
            }
        }
        
        loader = SkillConfigLoader()
        assert loader.validate_config(config_data) is False
    
    def test_export_to_yaml(self):
        """Test exporting skills to YAML"""
        skills = {
            'test_skill': SkillConfig(
                description='Test skill',
                tone='professional',
                compatible_with=[]
            )
        }
        
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'exported_skills.yaml'
            loader = SkillConfigLoader()
            loader.export_to_yaml(skills, output_path)
            
            assert output_path.exists()
            
            # Verify exported content
            with open(output_path, 'r') as f:
                exported = yaml.safe_load(f)
            
            assert 'version' in exported
            assert 'skills' in exported
            assert 'test_skill' in exported['skills']
    
    def test_disabled_skill_not_loaded(self):
        """Test that disabled skills are not loaded"""
        config_data = {
            'version': '1.0',
            'skills': {
                'enabled_skill': {
                    'description': 'Enabled skill',
                    'tone': 'professional',
                    'compatible_with': [],
                    'prompt_config': {
                        'system_prompt': 'test',
                        'user_template': 'test {step_description} {retrieved_content}',
                        'temperature': 0.7,
                        'max_tokens': 2000
                    },
                    'enabled': True,
                    'metadata': {}
                },
                'disabled_skill': {
                    'description': 'Disabled skill',
                    'tone': 'casual',
                    'compatible_with': [],
                    'prompt_config': {
                        'system_prompt': 'test',
                        'user_template': 'test {step_description} {retrieved_content}',
                        'temperature': 0.7,
                        'max_tokens': 2000
                    },
                    'enabled': False,  # Disabled
                    'metadata': {}
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            loader = SkillConfigLoader(str(temp_path))
            skills = loader.load_from_yaml()
            
            assert len(skills) == 1
            assert 'enabled_skill' in skills
            assert 'disabled_skill' not in skills
        finally:
            temp_path.unlink()


class TestSkillManagerWithConfig:
    """Test SkillManager with configuration loading"""
    
    @pytest.fixture
    def sample_config_file(self):
        """Create a sample configuration file"""
        config_data = {
            'version': '1.0',
            'skills': {
                'config_skill': {
                    'description': 'Skill from config',
                    'tone': 'technical',
                    'compatible_with': [],
                    'prompt_config': {
                        'system_prompt': 'test',
                        'user_template': 'test {step_description} {retrieved_content}',
                        'temperature': 0.7,
                        'max_tokens': 2000
                    },
                    'enabled': True,
                    'metadata': {}
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        yield temp_path
        
        temp_path.unlink()
    
    def test_skill_manager_load_from_config(self, sample_config_file):
        """Test SkillManager loading from config file"""
        manager = SkillManager(config_path=str(sample_config_file))
        
        assert 'config_skill' in manager.list_skills()
        skill = manager.get_skill('config_skill')
        assert skill.description == 'Skill from config'
        assert skill.tone == 'technical'
    
    def test_skill_manager_reload_from_config(self, sample_config_file):
        """Test SkillManager reloading from config"""
        manager = SkillManager()
        
        # Initially no config_skill
        assert 'config_skill' not in manager.list_skills()
        
        # Reload from config
        manager.reload_from_config(str(sample_config_file))
        
        # Now config_skill should be present
        assert 'config_skill' in manager.list_skills()
    
    def test_skill_manager_export_to_config(self):
        """Test SkillManager exporting to config"""
        manager = SkillManager()
        
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'exported.yaml'
            manager.export_to_config(str(output_path))
            
            assert output_path.exists()
            
            # Verify can be loaded back
            new_manager = SkillManager(config_path=str(output_path))
            assert len(new_manager.list_skills()) > 0
    
    def test_skill_manager_get_config_path(self, sample_config_file):
        """Test getting config path from manager"""
        manager = SkillManager(config_path=str(sample_config_file))
        
        assert manager.get_config_path() == str(sample_config_file)
    
    def test_skill_manager_without_config_path(self):
        """Test manager without config path"""
        manager = SkillManager()
        
        assert manager.get_config_path() is None
        assert not manager.is_hot_reload_enabled()


class TestCreateDefaultConfig:
    """Test creating default configuration"""
    
    def test_create_default_config(self):
        """Test creating default configuration file"""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'default_skills.yaml'
            create_default_config(output_path)
            
            assert output_path.exists()
            
            # Verify it's valid
            loader = SkillConfigLoader(str(output_path))
            skills = loader.load_from_yaml()
            
            assert len(skills) > 0
            assert 'standard_tutorial' in skills
