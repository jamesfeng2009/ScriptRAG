"""
属性测试：幻觉检测

属性 14: 幻觉检测
对于任何编剧生成的剧本片段，事实检查器应根据检索文档进行验证，
并检测对不存在的代码、函数或参数的引用。


Feature: rag-screenplay-multi-agent
Property 14: 幻觉检测
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import AsyncMock, MagicMock

from src.domain.models import (
    ScreenplayFragment,
    RetrievedDocument
)
from src.domain.agents.fact_checker import verify_fragment, _heuristic_verification


# 自定义策略生成器
@st.composite
def screenplay_fragment_strategy(draw):
    """生成有效的 ScreenplayFragment"""
    step_id = draw(st.integers(min_value=0, max_value=100))
    
    # 生成包含代码引用的内容
    content_templates = [
        "使用 `{func}()` 函数来处理数据。",
        "调用 `{func}()` 方法实现功能。",
        "```python\ndef {func}():\n    pass\n```",
        "`{cls}` 类提供了核心功能。",
        "通过 `{func}()` 和 `{param}` 参数来配置。"
    ]
    
    template = draw(st.sampled_from(content_templates))
    func_name = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll')),
        min_size=3,
        max_size=20
    ).filter(lambda x: x.isidentifier()))
    
    content = template.format(
        func=func_name,
        cls=func_name.capitalize(),
        param=func_name.lower()
    )
    
    return ScreenplayFragment(
        step_id=step_id,
        content=content,
        skill_used=draw(st.sampled_from([
            "standard_tutorial",
            "warning_mode",
            "visualization_analogy",
            "research_mode",
            "meme_style",
            "fallback_summary"
        ])),
        sources=draw(st.lists(
            st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            max_size=5
        ))
    )


@st.composite
def retrieved_document_strategy(draw):
    """生成有效的 RetrievedDocument"""
    content = draw(st.text(min_size=10, max_size=1000).filter(lambda x: x.strip()))
    source = draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    
    return RetrievedDocument(
        content=content,
        source=source,
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        metadata=draw(st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(st.booleans(), st.text(max_size=50), st.floats()),
            max_size=5
        ))
    )


@st.composite
def matching_fragment_and_docs_strategy(draw):
    """生成匹配的片段和文档（无幻觉）"""
    # 生成函数名
    func_name = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll')),
        min_size=3,
        max_size=20
    ).filter(lambda x: x.isidentifier()))
    
    # 生成包含该函数的文档
    doc_content = f"这是一个示例文档。\n\ndef {func_name}():\n    pass\n\n使用 {func_name} 函数。"
    
    doc = RetrievedDocument(
        content=doc_content,
        source="test_source.py",
        confidence=0.9,
        metadata={}
    )
    
    # 生成引用该函数的片段
    fragment_content = f"使用 `{func_name}()` 函数来处理数据。"
    
    fragment = ScreenplayFragment(
        step_id=draw(st.integers(min_value=0, max_value=100)),
        content=fragment_content,
        skill_used="standard_tutorial",
        sources=["test_source.py"]
    )
    
    return fragment, [doc]


@st.composite
def hallucinated_fragment_and_docs_strategy(draw):
    """生成包含幻觉的片段和文档"""
    # 生成两个不同的函数名
    real_func = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll')),
        min_size=3,
        max_size=20
    ).filter(lambda x: x.isidentifier()))
    
    fake_func = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll')),
        min_size=3,
        max_size=20
    ).filter(lambda x: x.isidentifier() and x != real_func))
    
    # 确保两个函数名不同
    assume(real_func != fake_func)
    
    # 生成只包含真实函数的文档
    doc_content = f"这是一个示例文档。\n\ndef {real_func}():\n    pass\n\n使用 {real_func} 函数。"
    
    doc = RetrievedDocument(
        content=doc_content,
        source="test_source.py",
        confidence=0.9,
        metadata={}
    )
    
    # 生成引用虚假函数的片段（幻觉）
    fragment_content = f"使用 `{fake_func}()` 函数来处理数据。"
    
    fragment = ScreenplayFragment(
        step_id=draw(st.integers(min_value=0, max_value=100)),
        content=fragment_content,
        skill_used="standard_tutorial",
        sources=["test_source.py"]
    )
    
    return fragment, [doc], fake_func


class TestHallucinationDetection:
    """测试幻觉检测属性"""
    
    @given(data=matching_fragment_and_docs_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_14_no_hallucination_detected(self, data):
        """
        属性 14: 幻觉检测 - 无幻觉时验证通过
        
        当片段内容与源文档一致时，应该通过验证。
        """
        fragment, docs = data
        
        # 使用启发式验证方法（不需要 LLM）
        is_valid, hallucinations = _heuristic_verification(fragment, docs)
        
        # 验证通过
        assert is_valid is True
        assert len(hallucinations) == 0
    
    @given(data=hallucinated_fragment_and_docs_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_14_hallucination_detected(self, data):
        """
        属性 14: 幻觉检测 - 检测到幻觉时验证失败
        
        当片段包含源文档中不存在的代码引用时，应该检测到幻觉。
        """
        fragment, docs, fake_func = data
        
        # 使用启发式验证方法
        is_valid, hallucinations = _heuristic_verification(fragment, docs)
        
        # 验证失败
        assert is_valid is False
        assert len(hallucinations) > 0
        
        # 验证幻觉列表包含虚假函数名
        hallucination_text = " ".join(hallucinations)
        assert fake_func in hallucination_text
    
    @given(fragment=screenplay_fragment_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_14_empty_docs_validation(self, fragment):
        """
        属性 14: 幻觉检测 - 空文档列表时的处理
        
        当没有源文档时，验证应该通过（可能是 research_mode）。
        """
        # 使用启发式验证方法
        is_valid, hallucinations = _heuristic_verification(fragment, [])
        
        # 空文档时应该通过验证
        assert is_valid is True
        assert len(hallucinations) == 0
    
    @given(
        fragment=screenplay_fragment_strategy(),
        docs=st.lists(retrieved_document_strategy(), min_size=1, max_size=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_14_hallucination_list_format(self, fragment, docs):
        """
        属性 14: 幻觉检测 - 幻觉列表格式正确
        
        当检测到幻觉时，幻觉列表应该是字符串列表。
        """
        # 使用启发式验证方法
        is_valid, hallucinations = _heuristic_verification(fragment, docs)
        
        # 验证返回值类型
        assert isinstance(is_valid, bool)
        assert isinstance(hallucinations, list)
        
        # 验证幻觉列表中的每个元素都是字符串
        for hallucination in hallucinations:
            assert isinstance(hallucination, str)
            assert len(hallucination) > 0
    
    def test_property_14_function_definition_hallucination(self):
        """
        属性 14: 幻觉检测 - 检测函数定义幻觉
        
        当片段包含源文档中不存在的函数定义时，应该检测到幻觉。
        """
        # 创建包含函数定义的片段
        fragment = ScreenplayFragment(
            step_id=1,
            content="```python\ndef fake_function():\n    pass\n```",
            skill_used="standard_tutorial",
            sources=["test.py"]
        )
        
        # 创建不包含该函数的文档（避免在描述中提到函数名）
        doc = RetrievedDocument(
            content="这是一个示例文档，包含其他内容但没有相关函数定义。",
            source="test.py",
            confidence=0.9,
            metadata={}
        )
        
        # 验证
        is_valid, hallucinations = _heuristic_verification(fragment, [doc])
        
        # 应该检测到幻觉
        assert is_valid is False
        assert len(hallucinations) > 0
        assert "fake_function" in " ".join(hallucinations)
    
    def test_property_14_function_call_hallucination(self):
        """
        属性 14: 幻觉检测 - 检测函数调用幻觉
        
        当片段包含源文档中不存在的函数调用时，应该检测到幻觉。
        """
        # 创建包含函数调用的片段
        fragment = ScreenplayFragment(
            step_id=1,
            content="使用 `nonexistent_func()` 函数来处理数据。",
            skill_used="standard_tutorial",
            sources=["test.py"]
        )
        
        # 创建不包含该函数的文档（避免在描述中提到函数名）
        doc = RetrievedDocument(
            content="这是一个示例文档，包含其他函数但没有相关的处理函数。",
            source="test.py",
            confidence=0.9,
            metadata={}
        )
        
        # 验证
        is_valid, hallucinations = _heuristic_verification(fragment, [doc])
        
        # 应该检测到幻觉
        assert is_valid is False
        assert len(hallucinations) > 0
        assert "nonexistent_func" in " ".join(hallucinations)
    
    def test_property_14_class_reference_hallucination(self):
        """
        属性 14: 幻觉检测 - 检测类引用幻觉
        
        当片段包含源文档中不存在的类引用时，应该检测到幻觉。
        """
        # 创建包含类引用的片段
        fragment = ScreenplayFragment(
            step_id=1,
            content="`FakeClass` 类提供了核心功能。",
            skill_used="standard_tutorial",
            sources=["test.py"]
        )
        
        # 创建不包含该类的文档（避免在描述中提到类名）
        doc = RetrievedDocument(
            content="这是一个示例文档，包含其他类但没有相关的核心类。",
            source="test.py",
            confidence=0.9,
            metadata={}
        )
        
        # 验证
        is_valid, hallucinations = _heuristic_verification(fragment, [doc])
        
        # 应该检测到幻觉
        assert is_valid is False
        assert len(hallucinations) > 0
        assert "FakeClass" in " ".join(hallucinations)
    
    def test_property_14_multiple_hallucinations(self):
        """
        属性 14: 幻觉检测 - 检测多个幻觉
        
        当片段包含多个幻觉时，应该全部检测出来。
        """
        # 创建包含多个幻觉的片段
        fragment = ScreenplayFragment(
            step_id=1,
            content=(
                "使用 `fake_func1()` 和 `fake_func2()` 函数。\n"
                "`FakeClass` 类提供支持。"
            ),
            skill_used="standard_tutorial",
            sources=["test.py"]
        )
        
        # 创建不包含这些引用的文档
        doc = RetrievedDocument(
            content="这是一个示例文档，不包含任何上述引用。",
            source="test.py",
            confidence=0.9,
            metadata={}
        )
        
        # 验证
        is_valid, hallucinations = _heuristic_verification(fragment, [doc])
        
        # 应该检测到多个幻觉
        assert is_valid is False
        assert len(hallucinations) >= 2  # 至少检测到 2 个幻觉
    
    def test_property_14_valid_references(self):
        """
        属性 14: 幻觉检测 - 有效引用不被误判
        
        当片段包含源文档中存在的引用时，不应该被误判为幻觉。
        """
        # 创建包含有效引用的片段
        fragment = ScreenplayFragment(
            step_id=1,
            content="使用 `real_function()` 函数来处理数据。",
            skill_used="standard_tutorial",
            sources=["test.py"]
        )
        
        # 创建包含该函数的文档
        doc = RetrievedDocument(
            content=(
                "这是一个示例文档。\n"
                "def real_function():\n"
                "    pass\n"
                "使用 real_function 来处理数据。"
            ),
            source="test.py",
            confidence=0.9,
            metadata={}
        )
        
        # 验证
        is_valid, hallucinations = _heuristic_verification(fragment, [doc])
        
        # 应该通过验证
        assert is_valid is True
        assert len(hallucinations) == 0
    
    @given(
        fragment=screenplay_fragment_strategy(),
        docs=st.lists(retrieved_document_strategy(), min_size=1, max_size=3)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_14_verification_consistency(self, fragment, docs):
        """
        属性 14: 幻觉检测 - 验证结果一致性
        
        对同一片段和文档的多次验证应该返回一致的结果。
        """
        # 第一次验证
        is_valid_1, hallucinations_1 = _heuristic_verification(fragment, docs)
        
        # 第二次验证
        is_valid_2, hallucinations_2 = _heuristic_verification(fragment, docs)
        
        # 结果应该一致
        assert is_valid_1 == is_valid_2
        assert len(hallucinations_1) == len(hallucinations_2)
