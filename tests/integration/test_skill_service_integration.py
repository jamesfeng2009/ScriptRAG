"""Integration tests for SkillService with real database"""

import pytest
import asyncio
from src.services.persistence.skill_persistence_service import (
    SkillRecord,
    SkillDatabaseService,
    SkillService
)
from src.domain.skills import SkillConfig, SKILLS


pytestmark = pytest.mark.integration


class TestSkillServiceIntegration:
    """Integration tests for SkillService with PostgreSQL"""

    @pytest.fixture
    def db_service(self):
        """Create SkillDatabaseService for testing"""
        service = SkillDatabaseService(
            host="localhost",
            port=5433,
            database="Screenplay",
            user="postgres",
            password="123456",
            echo=False
        )
        return service

    @pytest.fixture
    def skill_service(self, db_service):
        """Create SkillService for testing"""
        return SkillService(db_service, enable_cache=True)

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self, db_service):
        """Setup database connection and table before each test"""
        try:
            await db_service.connect()
            await db_service.create_table()
            yield
            try:
                await db_service.delete_all()
            except Exception:
                pass
            try:
                await db_service.disconnect()
            except Exception:
                pass
        except Exception:
            yield

    @pytest.mark.asyncio
    async def test_create_and_get_skill(self, db_service, skill_service):
        """Test creating and retrieving a skill"""
        record = SkillRecord(
            skill_name="test_skill_1",
            description="Test skill description",
            tone="professional",
            compatible_with=["standard_tutorial"],
            prompt_config={"temperature": 0.7, "max_tokens": 2000},
            is_enabled=True,
            is_default=False
        )

        created = await skill_service.create(record)
        assert created is not None
        assert created.skill_name == "test_skill_1"

        retrieved = await skill_service.get("test_skill_1")
        assert retrieved is not None
        assert retrieved.description == "Test skill description"
        assert retrieved.tone == "professional"
        assert retrieved.prompt_config == {"temperature": 0.7, "max_tokens": 2000}

    @pytest.mark.asyncio
    async def test_update_skill(self, db_service, skill_service):
        """Test updating a skill"""
        record = SkillRecord(
            skill_name="test_skill_2",
            description="Original description",
            tone="casual",
            compatible_with=[],
            prompt_config={"temperature": 0.5}
        )
        await skill_service.create(record)

        updated = await skill_service.update(
            "test_skill_2",
            description="Updated description",
            tone="professional",
            prompt_config={"temperature": 0.8}
        )

        assert updated is not None
        assert updated.description == "Updated description"
        assert updated.tone == "professional"
        assert updated.prompt_config == {"temperature": 0.8}

    @pytest.mark.asyncio
    async def test_delete_skill(self, db_service, skill_service):
        """Test deleting a skill"""
        record = SkillRecord(
            skill_name="test_skill_3",
            description="To be deleted",
            tone="professional"
        )
        await skill_service.create(record)

        result = await skill_service.delete("test_skill_3")
        assert result is True

        retrieved = await skill_service.get("test_skill_3")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_skill_config_from_database(self, db_service, skill_service):
        """Test getting skill config from database"""
        record = SkillRecord(
            skill_name="test_skill_4",
            description="Config test skill",
            tone="professional",
            compatible_with=["warning_mode"],
            prompt_config={"temperature": 0.7, "max_tokens": 1500}
        )
        await skill_service.create(record)

        config = await skill_service.get_skill_config("test_skill_4")

        assert config is not None
        assert isinstance(config, SkillConfig)
        assert config.description == "Config test skill"
        assert config.tone == "professional"
        assert config.compatible_with == ["warning_mode"]
        assert config.prompt_config == {"temperature": 0.7, "max_tokens": 1500}

    @pytest.mark.asyncio
    async def test_get_skill_config_fallback_to_defaults(self, skill_service):
        """Test fallback to default skills when database is empty"""
        config = await skill_service.get_skill_config("standard_tutorial")

        assert config is not None
        assert config.description == SKILLS["standard_tutorial"].description
        assert config.tone == SKILLS["standard_tutorial"].tone

    @pytest.mark.asyncio
    async def test_get_all_skills(self, db_service, skill_service):
        """Test getting all skills"""
        records = [
            SkillRecord(
                skill_name="custom_skill_1",
                description="Custom skill 1",
                tone="professional",
                compatible_with=[],
                prompt_config={"temperature": 0.6}
            ),
            SkillRecord(
                skill_name="custom_skill_2",
                description="Custom skill 2",
                tone="casual",
                compatible_with=[],
                prompt_config={"temperature": 0.9}
            )
        ]

        for record in records:
            await skill_service.create(record)

        skills = await skill_service.get_all()

        assert len(skills) >= 2
        skill_names = [s.skill_name for s in skills]
        assert "custom_skill_1" in skill_names
        assert "custom_skill_2" in skill_names

    @pytest.mark.asyncio
    async def test_get_available_skills(self, db_service, skill_service):
        """Test getting all available skills"""
        records = [
            SkillRecord(
                skill_name="custom_skill_1",
                description="Custom skill 1",
                tone="professional",
                compatible_with=[],
                prompt_config={"temperature": 0.6}
            ),
            SkillRecord(
                skill_name="custom_skill_2",
                description="Custom skill 2",
                tone="casual",
                compatible_with=[],
                prompt_config={"temperature": 0.9}
            )
        ]

        for record in records:
            await skill_service.create(record)

        skills = await skill_service.get_available_skills()

        assert "custom_skill_1" in skills
        assert "custom_skill_2" in skills
        assert skills["custom_skill_1"].prompt_config == {"temperature": 0.6}
        assert skills["custom_skill_2"].prompt_config == {"temperature": 0.9}

    @pytest.mark.asyncio
    async def test_get_available_skills_fallback(self, skill_service):
        """Test fallback to all default skills when no custom skills exist"""
        skills = await skill_service.get_available_skills()

        assert len(skills) == len(SKILLS)
        for skill_name in SKILLS:
            assert skill_name in skills

    @pytest.mark.asyncio
    async def test_ensure_default_skills(self, db_service, skill_service):
        """Test ensuring default skills are created"""
        records = await skill_service.ensure_default_skills()

        assert len(records) == len(SKILLS)

        for skill_name in SKILLS:
            skill = await skill_service.get(skill_name)
            assert skill is not None
            assert skill.is_enabled is True

    @pytest.mark.asyncio
    async def test_caching_behavior(self, db_service, skill_service):
        """Test that caching works correctly"""
        record = SkillRecord(
            skill_name="cache_test_skill",
            description="Cache test",
            tone="professional"
        )
        await skill_service.create(record)

        await skill_service.get_all()  # This populates the cache
        cache_key = skill_service._get_cache_key("all")
        assert cache_key in skill_service._cache

        await skill_service.update("cache_test_skill", description="Updated")
        assert cache_key not in skill_service._cache

        await skill_service.get_all()  # This repopulates the cache
        assert cache_key in skill_service._cache

    @pytest.mark.asyncio
    async def test_disabled_skills_not_returned(self, db_service, skill_service):
        """Test that disabled skills are not returned in available skills"""
        enabled_record = SkillRecord(
            skill_name="enabled_skill",
            description="Enabled skill",
            tone="professional",
            is_enabled=True
        )
        disabled_record = SkillRecord(
            skill_name="disabled_skill",
            description="Disabled skill",
            tone="casual",
            is_enabled=False
        )

        await skill_service.create(enabled_record)
        await skill_service.create(disabled_record)

        skills = await skill_service.get_available_skills()

        assert "enabled_skill" in skills
        assert "disabled_skill" not in skills

    @pytest.mark.asyncio
    async def test_default_skill_functionality(self, db_service, skill_service):
        """Test setting and getting default skill"""
        record = SkillRecord(
            skill_name="default_skill",
            description="Default skill",
            tone="professional",
            is_default=True
        )
        await skill_service.create(record)

        default = await skill_service.get_default()

        assert default is not None
        assert default.skill_name == "default_skill"
        assert default.is_default is True

    @pytest.mark.asyncio
    async def test_async_context_manager(self, db_service, skill_service):
        """Test using SkillService as async context manager"""
        async with skill_service as service:
            assert service is skill_service
            assert db_service._pool is not None

        assert db_service._pool is None

    @pytest.mark.asyncio
    async def test_skill_with_extra_data(self, db_service, skill_service):
        """Test storing and retrieving extra_data"""
        record = SkillRecord(
            skill_name="extra_data_skill",
            description="Extra data test",
            tone="professional",
            extra_data={"custom_field": "value", "metadata": {"key": "value"}}
        )
        await skill_service.create(record)

        retrieved = await skill_service.get("extra_data_skill")

        assert retrieved is not None
        assert retrieved.extra_data == {"custom_field": "value", "metadata": {"key": "value"}}


class TestSkillServicePromptConfigIntegration:
    """Integration tests specifically for prompt_config field"""

    @pytest.fixture
    def db_service(self):
        """Create SkillDatabaseService for testing"""
        service = SkillDatabaseService(
            host="localhost",
            port=5433,
            database="Screenplay",
            user="postgres",
            password="123456",
            echo=False
        )
        return service

    @pytest.fixture
    def skill_service(self, db_service):
        """Create SkillService for testing"""
        return SkillService(db_service, enable_cache=True)

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self, db_service):
        """Setup database connection and table before each test"""
        await db_service.connect()
        await db_service.create_table()
        yield
        await db_service.delete_all()
        await db_service.disconnect()

    @pytest.mark.asyncio
    async def test_store_and_retrieve_complex_prompt_config(self, skill_service):
        """Test storing and retrieving complex prompt_config"""
        complex_prompt_config = {
            "temperature": 0.7,
            "max_tokens": 2000,
            "system_prompt": "You are a helpful assistant.",
            "user_template": "Please explain {topic}.",
            "stop_sequences": ["END", "STOP"],
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.0
        }

        record = SkillRecord(
            skill_name="complex_prompt_skill",
            description="Complex prompt config test",
            tone="professional",
            prompt_config=complex_prompt_config
        )
        await skill_service.create(record)

        retrieved = await skill_service.get("complex_prompt_skill")
        assert retrieved is not None
        assert retrieved.prompt_config == complex_prompt_config

        config = await skill_service.get_skill_config("complex_prompt_skill")
        assert config is not None
        assert config.prompt_config == complex_prompt_config

    @pytest.mark.asyncio
    async def test_update_prompt_config(self, skill_service):
        """Test updating prompt_config"""
        record = SkillRecord(
            skill_name="update_prompt_skill",
            description="Update prompt test",
            tone="professional",
            prompt_config={"temperature": 0.5}
        )
        await skill_service.create(record)

        new_prompt_config = {"temperature": 0.8, "max_tokens": 3000}
        await skill_service.update(
            "update_prompt_skill",
            prompt_config=new_prompt_config
        )

        retrieved = await skill_service.get("update_prompt_skill")
        assert retrieved.prompt_config == new_prompt_config

    @pytest.mark.asyncio
    async def test_empty_prompt_config(self, skill_service):
        """Test skill with empty prompt_config"""
        record = SkillRecord(
            skill_name="empty_prompt_skill",
            description="Empty prompt config test",
            tone="professional",
            prompt_config={}
        )
        await skill_service.create(record)

        retrieved = await skill_service.get("empty_prompt_skill")
        assert retrieved is not None
        assert retrieved.prompt_config == {}

        config = await skill_service.get_skill_config("empty_prompt_skill")
        assert config is not None
        assert config.prompt_config == {}
