"""
属性测试：废弃冲突检测

属性 9: 废弃冲突检测
对于任何在检索文档中提到标记为 @deprecated 的功能的大纲步骤，
导演应触发原因为 "deprecation_conflict" 的转向。

Feature: rag-screenplay-multi-agent
Property 9: 废弃冲突检测
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from src.domain.models import (
    SharedState,
    OutlineStep,
    RetrievedDocument
)
from src.domain.agents.director import detect_conflicts


# 自定义策略生成器
@st.composite
def outline_step_strategy(draw):
    """生成有效的 OutlineStep"""
    description = draw(st.text(min_size=1, max_size=200).filter(lambda x: x.strip()))
    return OutlineStep(
        step_id=draw(st.integers(min_value=0, max_value=100)),
        description=description,
        status=draw(st.sampled_from(["pending", "in_progress", "completed", "skipped"])),
        retry_count=draw(st.integers(min_value=0, max_value=10))
    )


@st.composite
def retrieved_document_with_deprecated_strategy(draw):
    """生成带有废弃标记的 RetrievedDocument"""
    content = draw(st.text(min_size=1, max_size=500).filter(lambda x: x.strip()))
    source = draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    
    # 确保有废弃标记
    has_deprecated = True
    
    return RetrievedDocument(
        content=content,
        source=source,
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        metadata={
            'has_deprecated': has_deprecated,
            'has_fixme': draw(st.booleans()),
            'has_todo': draw(st.booleans()),
            'has_security': draw(st.booleans())
        }
    )


@st.composite
def retrieved_document_without_deprecated_strategy(draw):
    """生成不带废弃标记的 RetrievedDocument"""
    content = draw(st.text(min_size=1, max_size=500).filter(lambda x: x.strip()))
    source = draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    
    # 确保没有废弃标记
    has_deprecated = False
    
    return RetrievedDocument(
        content=content,
        source=source,
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        metadata={
            'has_deprecated': has_deprecated,
            'has_fixme': draw(st.booleans()),
            'has_todo': draw(st.booleans()),
            'has_security': draw(st.booleans())
        }
    )


@st.composite
def retrieved_document_with_security_strategy(draw):
    """生成带有安全标记的 RetrievedDocument"""
    content = draw(st.text(min_size=1, max_size=500).filter(lambda x: x.strip()))
    source = draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    
    return RetrievedDocument(
        content=content,
        source=source,
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        metadata={
            'has_deprecated': False,
            'has_fixme': draw(st.booleans()),
            'has_todo': draw(st.booleans()),
            'has_security': True
        }
    )


@st.composite
def retrieved_document_with_fixme_strategy(draw):
    """生成带有 FIXME 标记的 RetrievedDocument"""
    content = draw(st.text(min_size=1, max_size=500).filter(lambda x: x.strip()))
    source = draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    
    return RetrievedDocument(
        content=content,
        source=source,
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        metadata={
            'has_deprecated': False,
            'has_fixme': True,
            'has_todo': draw(st.booleans()),
            'has_security': False
        }
    )


class TestDeprecationConflictDetection:
    """测试废弃冲突检测属性"""
    
    @given(
        step=outline_step_strategy(),
        deprecated_docs=st.lists(
            retrieved_document_with_deprecated_strategy(),
            min_size=1,
            max_size=5
        ),
        other_docs=st.lists(
            retrieved_document_without_deprecated_strategy(),
            max_size=3
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_9_deprecation_conflict_detected(
        self,
        step: OutlineStep,
        deprecated_docs: list,
        other_docs: list
    ):
        """
        属性 9: 废弃冲突检测 - 检测到废弃标记时触发冲突
        
        当检索文档中包含废弃标记时，应检测到冲突。
        """
        # 合并文档列表
        all_docs = deprecated_docs + other_docs
        
        # 调用冲突检测
        has_conflict, conflict_type, conflict_details = detect_conflicts(step, all_docs)
        
        # 验证检测到冲突
        assert has_conflict is True
        assert conflict_type == "deprecation_conflict"
        assert conflict_details is not None
        assert "废弃" in conflict_details or "deprecated" in conflict_details.lower()
        
        # 验证冲突详情包含步骤 ID
        assert str(step.step_id) in conflict_details
        
        # 验证冲突详情包含至少一个废弃文档的来源
        deprecated_sources = [doc.source for doc in deprecated_docs]
        assert any(source in conflict_details for source in deprecated_sources)
    
    @given(
        step=outline_step_strategy(),
        docs=st.lists(
            retrieved_document_without_deprecated_strategy(),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_9_no_conflict_without_deprecated(
        self,
        step: OutlineStep,
        docs: list
    ):
        """
        属性 9: 废弃冲突检测 - 没有废弃标记时不触发冲突
        
        当检索文档中不包含废弃标记时，不应检测到废弃冲突。
        """
        # 确保所有文档都没有废弃标记
        for doc in docs:
            assert doc.metadata.get('has_deprecated', False) is False
        
        # 调用冲突检测
        has_conflict, conflict_type, conflict_details = detect_conflicts(step, docs)
        
        # 如果检测到冲突，应该不是废弃冲突
        if has_conflict:
            assert conflict_type != "deprecation_conflict"
    
    @given(
        step=outline_step_strategy(),
        security_docs=st.lists(
            retrieved_document_with_security_strategy(),
            min_size=1,
            max_size=3
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_9_security_issue_detected(
        self,
        step: OutlineStep,
        security_docs: list
    ):
        """
        属性 9: 废弃冲突检测 - 检测到安全标记时触发冲突
        
        当检索文档中包含安全标记时，应检测到安全问题冲突。
        """
        # 调用冲突检测
        has_conflict, conflict_type, conflict_details = detect_conflicts(step, security_docs)
        
        # 验证检测到冲突
        assert has_conflict is True
        assert conflict_type == "security_issue"
        assert conflict_details is not None
        assert "安全" in conflict_details or "security" in conflict_details.lower()
    
    @given(
        step=outline_step_strategy(),
        fixme_docs=st.lists(
            retrieved_document_with_fixme_strategy(),
            min_size=1,
            max_size=3
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_9_fixme_issue_detected(
        self,
        step: OutlineStep,
        fixme_docs: list
    ):
        """
        属性 9: 废弃冲突检测 - 检测到 FIXME 标记时触发冲突
        
        当检索文档中包含 FIXME 标记时，应检测到需要修复的问题。
        """
        # 调用冲突检测
        has_conflict, conflict_type, conflict_details = detect_conflicts(step, fixme_docs)
        
        # 验证检测到冲突
        assert has_conflict is True
        assert conflict_type == "fixme_issue"
        assert conflict_details is not None
        assert "FIXME" in conflict_details or "fixme" in conflict_details.lower()
    
    @given(step=outline_step_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_9_empty_docs_no_conflict(self, step: OutlineStep):
        """
        属性 9: 废弃冲突检测 - 空文档列表不触发冲突
        
        当检索文档列表为空时，不应检测到冲突。
        """
        # 调用冲突检测
        has_conflict, conflict_type, conflict_details = detect_conflicts(step, [])
        
        # 验证没有检测到冲突
        assert has_conflict is False
        assert conflict_type is None
        assert conflict_details is None
    
    @given(
        step=outline_step_strategy(),
        deprecated_docs=st.lists(
            retrieved_document_with_deprecated_strategy(),
            min_size=1,
            max_size=3
        ),
        security_docs=st.lists(
            retrieved_document_with_security_strategy(),
            min_size=1,
            max_size=2
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_9_multiple_conflicts_priority(
        self,
        step: OutlineStep,
        deprecated_docs: list,
        security_docs: list
    ):
        """
        属性 9: 废弃冲突检测 - 多个冲突时的优先级
        
        当检索文档中包含多种冲突标记时，应优先检测废弃冲突。
        """
        # 合并文档列表
        all_docs = deprecated_docs + security_docs
        
        # 调用冲突检测
        has_conflict, conflict_type, conflict_details = detect_conflicts(step, all_docs)
        
        # 验证检测到冲突
        assert has_conflict is True
        
        # 废弃冲突应该优先于安全问题
        assert conflict_type == "deprecation_conflict"
    
    @given(
        step=outline_step_strategy(),
        docs=st.lists(
            retrieved_document_without_deprecated_strategy(),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_9_conflict_details_format(
        self,
        step: OutlineStep,
        docs: list
    ):
        """
        属性 9: 废弃冲突检测 - 冲突详情格式正确
        
        当检测到冲突时，冲突详情应包含必要的信息。
        """
        # 添加一个废弃文档
        deprecated_doc = RetrievedDocument(
            content="Test content",
            source="test_deprecated.py",
            confidence=0.8,
            metadata={'has_deprecated': True}
        )
        docs.append(deprecated_doc)
        
        # 调用冲突检测
        has_conflict, conflict_type, conflict_details = detect_conflicts(step, docs)
        
        # 验证冲突详情格式
        assert has_conflict is True
        assert isinstance(conflict_details, str)
        assert len(conflict_details) > 0
        
        # 验证包含关键信息
        assert str(step.step_id) in conflict_details
        assert deprecated_doc.source in conflict_details

