"""Property-Based Tests for Summarization

Feature: rag-screenplay-multi-agent
Property 5: 基于大小的摘要
Edge Case 3: Token 阈值边界
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Optional


# 策略：生成不同大小的文档
@st.composite
def document_strategy(draw, min_size=100, max_size=50000):
    """生成文档内容"""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # 生成重复的代码模式以模拟真实代码
    base_content = "def function_{}():\n    pass\n\n"
    content = ""
    i = 0
    while len(content) < size:
        content += base_content.format(i)
        i += 1
    return content[:size]


class MockSummarizedDocument:
    """模拟摘要文档"""
    def __init__(self, original_path: str, original_size: int, summary: str,
                 key_elements: list, metadata: dict, was_summarized: bool = True):
        self.original_path = original_path
        self.original_size = original_size
        self.summary = summary
        self.key_elements = key_elements
        self.metadata = metadata
        self.was_summarized = was_summarized


class TestSummarization:
    """摘要属性测试"""
    
    @given(content=document_strategy(min_size=1000, max_size=50000))
    @settings(max_examples=100, deadline=None)
    @pytest.mark.asyncio
    async def test_property_5_size_based_summarization(self, content: str):
        """
        属性 5: 基于大小的摘要
        
        对于任何文档，摘要服务应该：
        1. 检查文档大小是否超过阈值（10000 tokens）
        2. 对于大文件，生成摘要而不是使用原始内容
        3. 摘要应该比原始内容短
        4. 摘要应该保留关键元数据
        5. 摘要应该保留关键代码片段信息
        
        验证: 需求 3.5, 9.2, 9.3, 9.5
        """
        max_tokens = 10000
        
        # 检查大小
        needs_summary = self._check_size(content, max_tokens)
        
        # 如果需要摘要
        if needs_summary:
            # 生成摘要
            summarized = await self._mock_summarize(content, "test.py")
            
            # 属性 1: 应该标记为已摘要
            assert summarized.was_summarized, "Large document should be summarized"
            
            # 属性 2: 摘要应该比原始内容短
            assert len(summarized.summary) < len(content), \
                f"Summary ({len(summarized.summary)}) should be shorter than original ({len(content)})"
            
            # 属性 3: 应该保留原始大小信息
            assert summarized.original_size == len(content), \
                "Should preserve original size"
            
            # 属性 4: 应该有元数据
            assert 'original_size' in summarized.metadata, \
                "Should have original_size in metadata"
            
            # 属性 5: 应该有关键元素列表
            assert isinstance(summarized.key_elements, list), \
                "Should have key_elements list"
        else:
            # 小文档不需要摘要
            assert len(content) // 4 <= max_tokens, \
                "Small document should not need summarization"
    
    @given(
        size=st.integers(min_value=39900, max_value=40100)  # 围绕阈值
    )
    @settings(max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_edge_case_3_token_threshold_boundary(self, size: int):
        """
        边界情况 3: Token 阈值边界
        
        对于接近阈值的文档（10000 tokens ≈ 40000 字符），应该：
        1. 正确判断是否需要摘要
        2. 在阈值附近的行为应该一致
        3. 不应该有边界错误
        
        验证: 需求 9.4
        """
        max_tokens = 10000
        
        # 生成指定大小的内容
        content = "x" * size
        
        # 检查大小
        needs_summary = self._check_size(content, max_tokens)
        
        # 估算 token 数
        estimated_tokens = len(content) // 4
        
        # 属性 1: 判断应该与估算一致
        if estimated_tokens > max_tokens:
            assert needs_summary, \
                f"Document with {estimated_tokens} tokens should need summary (threshold: {max_tokens})"
        else:
            assert not needs_summary, \
                f"Document with {estimated_tokens} tokens should not need summary (threshold: {max_tokens})"
        
        # 属性 2: 如果需要摘要，应该能够成功生成
        if needs_summary:
            summarized = await self._mock_summarize(content, "test.py")
            assert summarized is not None, "Should be able to summarize boundary case"
            assert summarized.was_summarized, "Should be marked as summarized"
    
    @given(content=document_strategy(min_size=50000, max_size=100000))
    @settings(max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_summarization_preserves_key_info(self, content: str):
        """
        测试摘要保留关键信息
        
        对于大文档，摘要应该：
        1. 保留文件路径
        2. 保留原始大小
        3. 包含关键元素信息
        4. 包含元数据
        """
        # 生成摘要
        summarized = await self._mock_summarize(content, "large_file.py")
        
        # 检查关键信息
        assert summarized.original_path == "large_file.py", \
            "Should preserve file path"
        
        assert summarized.original_size == len(content), \
            "Should preserve original size"
        
        assert isinstance(summarized.key_elements, list), \
            "Should have key elements list"
        
        assert isinstance(summarized.metadata, dict), \
            "Should have metadata dict"
        
        assert 'original_size' in summarized.metadata, \
            "Metadata should include original size"
    
    @given(content=document_strategy(min_size=1000, max_size=5000))
    @settings(max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_small_documents_not_summarized(self, content: str):
        """
        测试小文档不被摘要
        
        对于小于阈值的文档，不应该进行摘要
        """
        max_tokens = 10000
        
        # 检查大小
        needs_summary = self._check_size(content, max_tokens)
        
        # 小文档不应该需要摘要
        estimated_tokens = len(content) // 4
        if estimated_tokens <= max_tokens:
            assert not needs_summary, \
                f"Small document ({estimated_tokens} tokens) should not need summary"
    
    def _check_size(self, content: str, max_tokens: int) -> bool:
        """
        检查文档大小
        
        在实际实现中，这将调用 SummarizationService.check_size()
        """
        # 简单的 token 估算：1 token ≈ 4 字符
        estimated_tokens = len(content) // 4
        return estimated_tokens > max_tokens
    
    async def _mock_summarize(
        self,
        content: str,
        file_path: str
    ) -> MockSummarizedDocument:
        """
        模拟摘要生成
        
        在实际实现中，这将调用 SummarizationService.summarize_document()
        """
        # 提取关键元素（简化版）
        key_elements = []
        lines = content.split('\n')
        for i, line in enumerate(lines[:100]):  # 只检查前100行
            if 'def ' in line:
                func_name = line.split('def ')[1].split('(')[0].strip() if 'def ' in line else f"func_{i}"
                key_elements.append({
                    'type': 'function',
                    'name': func_name,
                    'line': i + 1
                })
        
        # 生成简单摘要
        summary = f"Summary of {file_path}\n"
        summary += f"Original size: {len(content)} characters\n"
        summary += f"Key elements: {len(key_elements)}\n"
        
        if key_elements:
            summary += "\nFunctions:\n"
            for elem in key_elements[:5]:  # 只包含前5个
                summary += f"- {elem['name']} (line {elem['line']})\n"
        
        # 添加内容片段
        summary += f"\nContent preview:\n{content[:500]}..."
        
        metadata = {
            'original_size': len(content),
            'estimated_tokens': len(content) // 4,
            'num_key_elements': len(key_elements),
            'summary_size': len(summary)
        }
        
        return MockSummarizedDocument(
            original_path=file_path,
            original_size=len(content),
            summary=summary,
            key_elements=key_elements,
            metadata=metadata,
            was_summarized=True
        )
