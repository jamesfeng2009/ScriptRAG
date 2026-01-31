"""Tests for Prompt Manager"""

import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
import yaml

from src.domain.prompt_manager import PromptManager
from src.domain.skill_loader import PromptConfig


class TestPromptManager:
    """Test PromptManager"""
    
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
                        'temperature': 0.8,
                        'max_tokens': 1500
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
    
    def test_prompt_manager_initialization(self, sample_config_file):
        """Test PromptManager initialization"""
        manager = PromptManager(config_path=str(sample_config_file))
        
        assert 'test_skill' in manager.list_available_skills()
    
    def test_get_prompt_config(self, sample_config_file):
        """Test getting prompt configuration"""
        manager = PromptManager(config_path=str(sample_config_file))
        
        config = manager.get_prompt_config('test_skill')
        
        assert config is not None
        assert config.system_prompt == 'You are a test assistant'
        assert config.temperature == 0.8
        assert config.max_tokens == 1500
    
    def test_get_nonexistent_prompt_config(self, sample_config_file):
        """Test getting nonexistent prompt config returns None"""
        manager = PromptManager(config_path=str(sample_config_file))
        
        config = manager.get_prompt_config('nonexistent_skill')
        
        assert config is None
    
    def test_format_messages(self, sample_config_file):
        """Test formatting messages with prompt config"""
        manager = PromptManager(config_path=str(sample_config_file))
        
        messages = manager.format_messages(
            skill_name='test_skill',
            step_description='Test step',
            retrieved_content='Test content'
        )
        
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == 'You are a test assistant'
        assert messages[1]['role'] == 'user'
        assert 'Test step' in messages[1]['content']
        assert 'Test content' in messages[1]['content']
    
    def test_format_messages_with_nonexistent_skill(self, sample_config_file):
        """Test formatting messages with nonexistent skill uses default"""
        manager = PromptManager(config_path=str(sample_config_file))
        
        messages = manager.format_messages(
            skill_name='nonexistent_skill',
            step_description='Test step',
            retrieved_content='Test content'
        )
        
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert 'nonexistent_skill' in messages[0]['content']
    
    def test_get_temperature(self, sample_config_file):
        """Test getting temperature from config"""
        manager = PromptManager(config_path=str(sample_config_file))
        
        temp = manager.get_temperature('test_skill')
        
        assert temp == 0.8
    
    def test_get_temperature_default(self, sample_config_file):
        """Test getting temperature returns default for nonexistent skill"""
        manager = PromptManager(config_path=str(sample_config_file))
        
        temp = manager.get_temperature('nonexistent_skill')
        
        assert temp == 0.7  # Default
    
    def test_get_max_tokens(self, sample_config_file):
        """Test getting max_tokens from config"""
        manager = PromptManager(config_path=str(sample_config_file))
        
        max_tokens = manager.get_max_tokens('test_skill')
        
        assert max_tokens == 1500
    
    def test_get_max_tokens_default(self, sample_config_file):
        """Test getting max_tokens returns default for nonexistent skill"""
        manager = PromptManager(config_path=str(sample_config_file))
        
        max_tokens = manager.get_max_tokens('nonexistent_skill')
        
        assert max_tokens == 2000  # Default
    
    def test_reload_prompts(self, sample_config_file):
        """Test reloading prompt configurations"""
        manager = PromptManager(config_path=str(sample_config_file))
        
        # Initial state
        assert 'test_skill' in manager.list_available_skills()
        
        # Reload
        manager.reload_prompts()
        
        # Should still have the skill
        assert 'test_skill' in manager.list_available_skills()
    
    def test_prompt_manager_with_nonexistent_file(self):
        """Test PromptManager with nonexistent config file"""
        manager = PromptManager(config_path='nonexistent.yaml')
        
        # Should not crash, just use defaults
        assert manager.list_available_skills() == []
    
    def test_hot_reload_enable_disable(self, sample_config_file):
        """Test enabling and disabling hot-reload"""
        manager = PromptManager(config_path=str(sample_config_file))
        
        # Initially disabled
        assert not manager.is_hot_reload_enabled()
        
        # Enable
        manager.enable_hot_reload()
        assert manager.is_hot_reload_enabled()
        
        # Disable
        manager.disable_hot_reload()
        assert not manager.is_hot_reload_enabled()


class TestPromptManagerIntegration:
    """Test PromptManager integration with Writer Agent"""
    
    def test_writer_agent_uses_prompt_manager(self):
        """Test that Writer Agent can use PromptManager"""
        from src.domain.agents.writer import get_prompt_manager, set_prompt_manager
        from src.domain.models import OutlineStep, RetrievedDocument
        
        # Get default prompt manager
        manager = get_prompt_manager()
        
        assert manager is not None
        assert isinstance(manager, PromptManager)
    
    def test_apply_skill_with_config(self):
        """Test apply_skill function with configuration"""
        from src.domain.agents.writer import apply_skill
        from src.domain.models import OutlineStep, RetrievedDocument
        
        step = OutlineStep(
            step_id=0,
            description="Test step",
            status="pending",
            retry_count=0
        )
        
        retrieved_docs = [
            RetrievedDocument(
                content="Test content",
                source="test.py",
                confidence=0.9,
                metadata={}
            )
        ]
        
        # Apply skill
        result = apply_skill(
            skill_name="standard_tutorial",
            step=step,
            retrieved_docs=retrieved_docs,
            llm_service=None  # Not used in apply_skill
        )
        
        assert 'messages' in result
        assert 'metadata' in result
        assert 'temperature' in result
        assert 'max_tokens' in result
        assert len(result['messages']) == 2
    
    def test_apply_skill_with_unknown_skill(self):
        """Test apply_skill with unknown skill falls back gracefully"""
        from src.domain.agents.writer import apply_skill
        from src.domain.models import OutlineStep, RetrievedDocument
        
        step = OutlineStep(
            step_id=0,
            description="Test step",
            status="pending",
            retry_count=0
        )
        
        retrieved_docs = []
        
        # Apply unknown skill
        result = apply_skill(
            skill_name="unknown_skill_xyz",
            step=step,
            retrieved_docs=retrieved_docs,
            llm_service=None
        )
        
        # Should fallback to standard_tutorial
        assert 'messages' in result
        assert result['metadata']['skill_used'] == 'standard_tutorial'
