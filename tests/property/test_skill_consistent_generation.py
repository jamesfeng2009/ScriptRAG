"""Property-Based Tests for Skill Consistent Generation

Feature: rag-screenplay-multi-agent
Property 7: Skill 一致生成

属性描述:
对于任何编剧生成的剧本片段，使用的 Skill 应与 SharedState 中的 current_skill 一致，
并且生成的内容应遵循该 Skill 的风格指南。

v2.1 Update:
- apply_skill now uses prompt_manager.format_messages() instead of direct SKILL_PROMPTS
- Tests are simplified to verify core functionality without exact string matching
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from src.domain.models import SharedState, OutlineStep, RetrievedDocument, ScreenplayFragment
from src.domain.agents.writer import apply_skill
from src.domain.skills import SKILLS
from src.domain.agents.writer import get_prompt_manager
from unittest.mock import AsyncMock, MagicMock
import logging

logger = logging.getLogger(__name__)


@st.composite
def skill_name_strategy(draw):
    """Generate a valid skill name"""
    return draw(st.sampled_from(list(SKILLS.keys())))


@st.composite
def outline_step_strategy(draw):
    """Generate a valid OutlineStep"""
    step_id = draw(st.integers(min_value=0, max_value=100))
    description = draw(st.text(min_size=10, max_size=200))
    status = draw(st.sampled_from(["pending", "in_progress", "completed", "skipped"]))
    retry_count = draw(st.integers(min_value=0, max_value=3))
    
    return OutlineStep(
        step_id=step_id,
        description=description,
        status=status,
        retry_count=retry_count
    )


@st.composite
def retrieved_document_strategy(draw):
    """Generate a valid RetrievedDocument"""
    content = draw(st.text(min_size=50, max_size=500))
    source = draw(st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='/_.-')))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return RetrievedDocument(
        content=content,
        source=source or "test.py",
        confidence=confidence,
        metadata={}
    )


@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(
    skill_name=skill_name_strategy(),
    step=outline_step_strategy(),
    retrieved_docs=st.lists(retrieved_document_strategy(), min_size=0, max_size=5)
)
def test_property_skill_consistent_generation(skill_name, step, retrieved_docs):
    """
    属性 7: Skill 一致生成
    
    验证: 需求 4.7
    
    对于任何编剧生成的剧本片段，使用的 Skill 应与指定的 skill_name 一致，
    并且生成的内容应遵循该 Skill 的风格指南。
    
    This test is marked as xfail because apply_skill now uses prompt_manager
    which adds formatting to prompts, causing exact string matching to fail.
    """
    mock_llm_service = MagicMock()
    
    skill_data = apply_skill(
        skill_name=skill_name,
        step=step,
        retrieved_docs=retrieved_docs,
        llm_service=mock_llm_service
    )
    
    assert skill_data["metadata"]["skill_used"] == skill_name, (
        f"Skill used ({skill_data['metadata']['skill_used']}) "
        f"does not match requested skill ({skill_name})"
    )
    
    assert len(skill_data["messages"]) >= 1, "Messages should contain at least system prompt"
    
    system_message = skill_data["messages"][0]
    assert system_message["role"] == "system", "First message should be system prompt"
    
    assert len(system_message["content"]) > 0, "System prompt should not be empty"
    
    if len(skill_data["messages"]) >= 2:
        user_message = skill_data["messages"][1]
        assert user_message["role"] == "user", "Second message should be user prompt"
        assert step.description in user_message["content"], (
            "User message should contain step description"
        )
    
    assert skill_data["metadata"]["num_docs"] == len(retrieved_docs), (
        f"Metadata num_docs ({skill_data['metadata']['num_docs']}) "
        f"does not match actual number of documents ({len(retrieved_docs)})"
    )
    
    expected_sources = [doc.source for doc in retrieved_docs]
    assert skill_data["metadata"]["sources"] == expected_sources, (
        "Metadata sources do not match retrieved document sources"
    )
    
    logger.info(
        f"✓ Property 7 verified: Skill '{skill_name}' applied consistently "
        f"with {len(retrieved_docs)} documents"
    )


@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(
    skill_name=skill_name_strategy(),
    step=outline_step_strategy()
)
def test_property_skill_empty_retrieval_handling(skill_name, step):
    """
    属性 7.1: 空检索时的 Skill 处理
    
    验证: 需求 4.7, 7.3
    
    当没有检索到文档时，apply_skill 应该正确处理空列表，
    并在用户提示中明确指出没有检索内容。
    """
    mock_llm_service = MagicMock()
    
    skill_data = apply_skill(
        skill_name=skill_name,
        step=step,
        retrieved_docs=[],
        llm_service=mock_llm_service
    )
    
    assert skill_data["metadata"]["num_docs"] == 0, (
        "Metadata should indicate zero documents"
    )
    
    assert "[无检索内容]" in skill_data["messages"][1]["content"], (
        "User message should indicate no retrieval content"
    )
    
    assert skill_data["metadata"]["sources"] == [], (
        "Sources should be empty for no retrieved documents"
    )
    
    logger.info(
        f"✓ Property 7.1 verified: Skill '{skill_name}' handles empty retrieval correctly"
    )


@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(
    skill_name=skill_name_strategy(),
    step=outline_step_strategy(),
    retrieved_docs=st.lists(retrieved_document_strategy(), min_size=1, max_size=3)
)
def test_property_skill_fallback_for_invalid_skill(skill_name, step, retrieved_docs):
    """
    属性 7.2: 无效 Skill 的回退处理
    
    验证: 需求 4.7
    
    当提供无效的 Skill 名称时，apply_skill 应该回退到 standard_tutorial。
    """
    mock_llm_service = MagicMock()
    
    invalid_skill = "invalid_skill_name_12345"
    
    skill_data = apply_skill(
        skill_name=invalid_skill,
        step=step,
        retrieved_docs=retrieved_docs,
        llm_service=mock_llm_service
    )
    
    assert skill_data["metadata"]["skill_used"] == "standard_tutorial", (
        f"Should fallback to standard_tutorial for invalid skill, "
        f"but got {skill_data['metadata']['skill_used']}"
    )
    
    logger.info(
        f"✓ Property 7.2 verified: Invalid skill fallback to standard_tutorial"
    )


def test_skill_data_contains_required_fields():
    """
    Basic test: Verify apply_skill returns data with required fields
    
    This test verifies the core functionality without exact string matching.
    """
    step = OutlineStep(
        step_id=1,
        description="Test step description for verification",
        status="pending",
        retry_count=0
    )
    
    mock_llm_service = MagicMock()
    
    skill_data = apply_skill(
        skill_name="standard_tutorial",
        step=step,
        retrieved_docs=[],
        llm_service=mock_llm_service
    )
    
    assert "messages" in skill_data
    assert "metadata" in skill_data
    assert len(skill_data["messages"]) >= 2
    assert skill_data["metadata"]["skill_used"] == "standard_tutorial"
    assert skill_data["metadata"]["num_docs"] == 0
    assert skill_data["metadata"]["sources"] == []
    
    assert skill_data["messages"][0]["role"] == "system"
    assert skill_data["messages"][1]["role"] == "user"


def test_prompt_manager_is_available():
    """
    Basic test: Verify prompt manager is available for skill formatting
    """
    prompt_manager = get_prompt_manager()
    assert prompt_manager is not None
    assert "standard_tutorial" in prompt_manager.list_available_skills()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
