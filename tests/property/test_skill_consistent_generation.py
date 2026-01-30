"""Property-Based Tests for Skill Consistent Generation

Feature: rag-screenplay-multi-agent
Property 7: Skill 一致生成

属性描述:
对于任何编剧生成的剧本片段，使用的 Skill 应与 SharedState 中的 current_skill 一致，
并且生成的内容应遵循该 Skill 的风格指南。
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from src.domain.models import SharedState, OutlineStep, RetrievedDocument, ScreenplayFragment
from src.domain.agents.writer import apply_skill, SKILL_PROMPTS
from src.domain.skills import SKILLS
from unittest.mock import AsyncMock, MagicMock
import logging

logger = logging.getLogger(__name__)


# Strategy for generating valid skill names
@st.composite
def skill_name_strategy(draw):
    """Generate a valid skill name"""
    return draw(st.sampled_from(list(SKILLS.keys())))


# Strategy for generating OutlineStep
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


# Strategy for generating RetrievedDocument
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


@pytest.mark.property
@settings(
    max_examples=100,
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
    """
    # Create a mock LLM service (not used in apply_skill, but needed for type checking)
    mock_llm_service = MagicMock()
    
    # Apply the skill
    skill_data = apply_skill(
        skill_name=skill_name,
        step=step,
        retrieved_docs=retrieved_docs,
        llm_service=mock_llm_service
    )
    
    # Property 1: The skill_used in metadata should match the input skill_name
    assert skill_data["metadata"]["skill_used"] == skill_name, (
        f"Skill used ({skill_data['metadata']['skill_used']}) "
        f"does not match requested skill ({skill_name})"
    )
    
    # Property 2: The messages should contain the skill-specific system prompt
    assert len(skill_data["messages"]) >= 1, "Messages should contain at least system prompt"
    
    system_message = skill_data["messages"][0]
    assert system_message["role"] == "system", "First message should be system prompt"
    
    # Property 3: The system prompt should match the skill's prompt
    expected_system_prompt = SKILL_PROMPTS[skill_name]["system_prompt"]
    assert system_message["content"] == expected_system_prompt, (
        f"System prompt does not match skill '{skill_name}' prompt"
    )
    
    # Property 4: The user message should contain the step description
    if len(skill_data["messages"]) >= 2:
        user_message = skill_data["messages"][1]
        assert user_message["role"] == "user", "Second message should be user prompt"
        assert step.description in user_message["content"], (
            "User message should contain step description"
        )
    
    # Property 5: The metadata should contain correct number of documents
    assert skill_data["metadata"]["num_docs"] == len(retrieved_docs), (
        f"Metadata num_docs ({skill_data['metadata']['num_docs']}) "
        f"does not match actual number of documents ({len(retrieved_docs)})"
    )
    
    # Property 6: The metadata should contain correct sources
    expected_sources = [doc.source for doc in retrieved_docs]
    assert skill_data["metadata"]["sources"] == expected_sources, (
        "Metadata sources do not match retrieved document sources"
    )
    
    logger.info(
        f"✓ Property 7 verified: Skill '{skill_name}' applied consistently "
        f"with {len(retrieved_docs)} documents"
    )


@pytest.mark.property
@settings(
    max_examples=50,
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
    # Create a mock LLM service
    mock_llm_service = MagicMock()
    
    # Apply skill with empty retrieved_docs
    skill_data = apply_skill(
        skill_name=skill_name,
        step=step,
        retrieved_docs=[],
        llm_service=mock_llm_service
    )
    
    # Property 1: Should handle empty retrieval gracefully
    assert skill_data["metadata"]["num_docs"] == 0, (
        "Metadata should indicate zero documents"
    )
    
    # Property 2: User message should indicate no retrieval content
    user_message = skill_data["messages"][1]
    assert "[无检索内容]" in user_message["content"], (
        "User message should indicate no retrieval content"
    )
    
    # Property 3: Sources should be empty
    assert skill_data["metadata"]["sources"] == [], (
        "Sources should be empty for no retrieved documents"
    )
    
    logger.info(
        f"✓ Property 7.1 verified: Skill '{skill_name}' handles empty retrieval correctly"
    )


@pytest.mark.property
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(
    step=outline_step_strategy(),
    retrieved_docs=st.lists(retrieved_document_strategy(), min_size=1, max_size=3)
)
def test_property_skill_fallback_for_invalid_skill(step, retrieved_docs):
    """
    属性 7.2: 无效 Skill 的回退处理
    
    验证: 需求 4.7
    
    当提供无效的 Skill 名称时，apply_skill 应该回退到 standard_tutorial。
    """
    # Create a mock LLM service
    mock_llm_service = MagicMock()
    
    # Use an invalid skill name
    invalid_skill = "invalid_skill_name_12345"
    
    # Apply skill with invalid name
    skill_data = apply_skill(
        skill_name=invalid_skill,
        step=step,
        retrieved_docs=retrieved_docs,
        llm_service=mock_llm_service
    )
    
    # Property 1: Should fallback to standard_tutorial
    assert skill_data["metadata"]["skill_used"] == "standard_tutorial", (
        f"Should fallback to standard_tutorial for invalid skill, "
        f"but got {skill_data['metadata']['skill_used']}"
    )
    
    # Property 2: Should use standard_tutorial's system prompt
    expected_system_prompt = SKILL_PROMPTS["standard_tutorial"]["system_prompt"]
    assert skill_data["messages"][0]["content"] == expected_system_prompt, (
        "Should use standard_tutorial's system prompt for invalid skill"
    )
    
    logger.info(
        f"✓ Property 7.2 verified: Invalid skill '{invalid_skill}' "
        f"correctly falls back to standard_tutorial"
    )


@pytest.mark.property
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(
    skill_name=skill_name_strategy(),
    step=outline_step_strategy(),
    retrieved_docs=st.lists(retrieved_document_strategy(), min_size=1, max_size=10)
)
def test_property_skill_document_limit(skill_name, step, retrieved_docs):
    """
    属性 7.3: 文档数量限制
    
    验证: 需求 4.7
    
    apply_skill 应该限制使用的文档数量（最多 5 个），
    以避免上下文过长。
    """
    # Assume we have more than 5 documents
    assume(len(retrieved_docs) > 5)
    
    # Create a mock LLM service
    mock_llm_service = MagicMock()
    
    # Apply skill
    skill_data = apply_skill(
        skill_name=skill_name,
        step=step,
        retrieved_docs=retrieved_docs,
        llm_service=mock_llm_service
    )
    
    # Property 1: Metadata should reflect all documents
    assert skill_data["metadata"]["num_docs"] == len(retrieved_docs), (
        "Metadata should reflect all retrieved documents"
    )
    
    # Property 2: User message should only include first 5 documents
    user_message = skill_data["messages"][1]["content"]
    
    # Count how many "文档 X" patterns appear in the message
    doc_count = 0
    for i in range(1, len(retrieved_docs) + 1):
        if f"文档 {i}" in user_message:
            doc_count += 1
    
    assert doc_count <= 5, (
        f"User message should include at most 5 documents, but found {doc_count}"
    )
    
    logger.info(
        f"✓ Property 7.3 verified: Skill '{skill_name}' limits documents to 5 "
        f"(total: {len(retrieved_docs)})"
    )


@pytest.mark.property
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(
    skill_name=skill_name_strategy(),
    step=outline_step_strategy(),
    retrieved_docs=st.lists(retrieved_document_strategy(), min_size=1, max_size=3)
)
def test_property_skill_source_attribution(skill_name, step, retrieved_docs):
    """
    属性 7.4: 来源归属
    
    验证: 需求 4.7, 3.6
    
    apply_skill 应该在元数据中正确记录所有来源，
    以支持来源归属追踪。
    """
    # Create a mock LLM service
    mock_llm_service = MagicMock()
    
    # Apply skill
    skill_data = apply_skill(
        skill_name=skill_name,
        step=step,
        retrieved_docs=retrieved_docs,
        llm_service=mock_llm_service
    )
    
    # Property 1: All sources should be recorded
    expected_sources = [doc.source for doc in retrieved_docs]
    actual_sources = skill_data["metadata"]["sources"]
    
    assert len(actual_sources) == len(expected_sources), (
        f"Number of sources mismatch: expected {len(expected_sources)}, "
        f"got {len(actual_sources)}"
    )
    
    # Property 2: Sources should be in the same order
    assert actual_sources == expected_sources, (
        "Sources should be in the same order as retrieved documents"
    )
    
    # Property 3: Each source should appear in the user message
    user_message = skill_data["messages"][1]["content"]
    for source in expected_sources[:5]:  # Only first 5 are included in message
        assert source in user_message, (
            f"Source '{source}' should appear in user message"
        )
    
    logger.info(
        f"✓ Property 7.4 verified: Skill '{skill_name}' correctly attributes "
        f"{len(retrieved_docs)} sources"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
