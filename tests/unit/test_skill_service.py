"""Unit tests for SkillService

Note: Tests for SkillDatabaseService with real database connections
are in tests/integration/test_skill_service_integration.py
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from src.services.persistence.skill_persistence_service import (
    SkillRecord,
    SkillDatabaseService,
    SkillService
)
from src.domain.skills import SkillConfig, SKILLS


class TestSkillRecord:
    """Test SkillRecord data class"""

    def test_skill_record_creation(self):
        """Test creating a SkillRecord"""
        record = SkillRecord(
            skill_name="standard_tutorial",
            description="Test description",
            tone="professional",
            compatible_with=["warning_mode"],
            prompt_config={"temperature": 0.7, "max_tokens": 2000},
            is_enabled=True,
            is_default=True
        )

        assert record.skill_name == "standard_tutorial"
        assert record.description == "Test description"
        assert record.tone == "professional"
        assert record.compatible_with == ["warning_mode"]
        assert record.prompt_config == {"temperature": 0.7, "max_tokens": 2000}
        assert record.is_enabled is True
        assert record.is_default is True

    def test_skill_record_defaults(self):
        """Test SkillRecord default values"""
        record = SkillRecord(
            skill_name="test_skill",
            description="Test description",
            tone="casual"
        )

        assert record.compatible_with == []
        assert record.prompt_config == {}
        assert record.is_enabled is True
        assert record.is_default is False
        assert record.extra_data == {}


class TestSkillService:
    """Test SkillService with mocked database service"""

    @pytest.fixture
    def mock_db_service(self):
        """Create a mock database service"""
        mock = MagicMock(spec=SkillDatabaseService)
        return mock

    @pytest.fixture
    def skill_service(self, mock_db_service):
        """Create SkillService with mocked database service"""
        return SkillService(mock_db_service, enable_cache=True)

    def test_service_initialization(self, skill_service):
        """Test SkillService initialization"""
        assert skill_service.enable_cache is True
        assert skill_service._cache == {}

    def test_get_cache_key(self, skill_service):
        """Test cache key generation"""
        key = skill_service._get_cache_key('all')
        assert key == "skills:all"

    def test_invalidate_all_cache(self, skill_service):
        """Test cache invalidation"""
        skill_service._cache["skills:all"] = []
        skill_service._cache["skills:enabled"] = []

        skill_service._invalidate_all_cache()

        assert "skills:all" not in skill_service._cache
        assert "skills:enabled" not in skill_service._cache

    @pytest.mark.asyncio
    async def test_get_all_with_cache(self, skill_service, mock_db_service):
        """Test retrieving all skills with cache hit"""
        mock_records = [
            SkillRecord(
                skill_name='standard_tutorial',
                description='Test',
                tone='professional'
            )
        ]

        skill_service._cache["skills:all"] = mock_records

        result = await skill_service.get_all()

        assert len(result) == 1
        assert result[0].skill_name == 'standard_tutorial'
        mock_db_service.get_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_all_without_cache(self, skill_service, mock_db_service):
        """Test retrieving all skills without cache hit"""
        mock_records = [
            SkillRecord(
                skill_name='warning_mode',
                description='Warning',
                tone='cautionary'
            )
        ]

        mock_db_service.get_all = AsyncMock(return_value=mock_records)

        result = await skill_service.get_all(use_cache=False)

        assert len(result) == 1
        mock_db_service.get_all.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_get_skill_config_from_database(self, skill_service, mock_db_service):
        """Test getting skill config from database record"""
        mock_record = SkillRecord(
            skill_name='standard_tutorial',
            description='Test description',
            tone='professional',
            compatible_with=['warning_mode'],
            prompt_config={'temperature': 0.7, 'max_tokens': 2000}
        )

        mock_db_service.get = AsyncMock(return_value=mock_record)

        result = await skill_service.get_skill_config('standard_tutorial')

        assert result is not None
        assert isinstance(result, SkillConfig)
        assert result.description == 'Test description'
        assert result.tone == 'professional'
        assert result.compatible_with == ['warning_mode']
        assert result.prompt_config == {'temperature': 0.7, 'max_tokens': 2000}

    @pytest.mark.asyncio
    async def test_get_skill_config_from_defaults(self, skill_service, mock_db_service):
        """Test getting skill config from defaults when not in database"""
        mock_db_service.get = AsyncMock(return_value=None)

        result = await skill_service.get_skill_config('standard_tutorial')

        assert result is not None
        assert isinstance(result, SkillConfig)
        assert result.description == SKILLS['standard_tutorial'].description
        assert result.tone == SKILLS['standard_tutorial'].tone
        assert result.compatible_with == SKILLS['standard_tutorial'].compatible_with
        assert result.prompt_config == SKILLS['standard_tutorial'].prompt_config

    @pytest.mark.asyncio
    async def test_get_skill_config_not_found(self, skill_service, mock_db_service):
        """Test getting non-existent skill config"""
        mock_db_service.get = AsyncMock(return_value=None)

        result = await skill_service.get_skill_config('non_existent')

        assert result is None

    @pytest.mark.asyncio
    async def test_get_available_skills_from_database(self, skill_service, mock_db_service):
        """Test getting available skills from database"""
        mock_records = [
            SkillRecord(
                skill_name='meme_style',
                description='Meme style',
                tone='casual',
                compatible_with=[],
                prompt_config={'temperature': 0.9}
            )
        ]

        mock_db_service.get_enabled = AsyncMock(return_value=mock_records)

        result = await skill_service.get_available_skills()

        assert 'meme_style' in result
        skill = result['meme_style']
        assert skill.description == 'Meme style'
        assert skill.tone == 'casual'
        assert skill.prompt_config == {'temperature': 0.9}

    @pytest.mark.asyncio
    async def test_get_available_skills_fallback_to_defaults(self, skill_service, mock_db_service):
        """Test fallback to default skills when database is empty"""
        mock_db_service.get_enabled = AsyncMock(return_value=[])

        result = await skill_service.get_available_skills()

        assert len(result) == len(SKILLS)
        for skill_name in SKILLS:
            assert skill_name in result

    @pytest.mark.asyncio
    async def test_create_invalidates_cache(self, skill_service, mock_db_service):
        """Test that creating a skill invalidates cache"""
        mock_record = SkillRecord(
            skill_name='new_skill',
            description='New skill',
            tone='professional'
        )

        mock_db_service.create = AsyncMock(return_value=mock_record)

        skill_service._cache["skills:all"] = []

        await skill_service.create(mock_record)

        mock_db_service.create.assert_called_once()
        assert "skills:all" not in skill_service._cache

    @pytest.mark.asyncio
    async def test_update_invalidates_cache(self, skill_service, mock_db_service):
        """Test that updating a skill invalidates cache"""
        mock_updated_record = SkillRecord(
            skill_name='standard_tutorial',
            description='Updated description',
            tone='professional'
        )

        mock_db_service.update = AsyncMock(return_value=mock_updated_record)

        skill_service._cache["skills:all"] = []

        await skill_service.update('standard_tutorial', description='Updated')

        mock_db_service.update.assert_called_once()
        assert "skills:all" not in skill_service._cache

    @pytest.mark.asyncio
    async def test_delete_invalidates_cache(self, skill_service, mock_db_service):
        """Test that deleting a skill invalidates cache"""
        mock_db_service.delete = AsyncMock(return_value=True)

        skill_service._cache["skills:all"] = []

        result = await skill_service.delete('standard_tutorial')

        assert result is True
        mock_db_service.delete.assert_called_once()
        assert "skills:all" not in skill_service._cache


class TestSkillConfigWithPromptConfig:
    """Test SkillConfig with prompt_config field"""

    def test_skill_config_with_prompt_config(self):
        """Test creating SkillConfig with prompt_config"""
        config = SkillConfig(
            description="Test skill",
            tone="professional",
            compatible_with=["skill_a", "skill_b"],
            prompt_config={"temperature": 0.7, "max_tokens": 1500}
        )

        assert config.prompt_config == {"temperature": 0.7, "max_tokens": 1500}

    def test_skill_config_without_prompt_config(self):
        """Test creating SkillConfig without prompt_config uses default"""
        config = SkillConfig(
            description="Test skill",
            tone="casual",
            compatible_with=[]
        )

        assert config.prompt_config == {}

    def test_skill_config_prompt_config_in_model_dump(self):
        """Test that prompt_config is included in model_dump"""
        config = SkillConfig(
            description="Test skill",
            tone="professional",
            compatible_with=["skill_a"],
            prompt_config={"temperature": 0.8}
        )

        dumped = config.model_dump()

        assert "prompt_config" in dumped
        assert dumped["prompt_config"] == {"temperature": 0.8}


class TestSkillServiceAsyncContextManager:
    """Test SkillService async context manager"""

    @pytest.fixture
    def mock_db_service(self):
        """Create a mock database service"""
        mock = MagicMock(spec=SkillDatabaseService)
        mock.disconnect = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_db_service):
        """Test using SkillService as async context manager"""
        service = SkillService(mock_db_service, enable_cache=True)

        async with service as s:
            assert s is service

        mock_db_service.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_exception(self, mock_db_service):
        """Test async context manager handles exceptions"""
        service = SkillService(mock_db_service, enable_cache=True)

        with pytest.raises(ValueError):
            async with service as s:
                raise ValueError("Test exception")

        mock_db_service.disconnect.assert_called_once()
