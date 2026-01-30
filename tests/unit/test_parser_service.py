"""Unit tests for Parser Service"""

import pytest
from src.services.parser.tree_sitter_parser import TreeSitterParser, CodeElementType


def test_detect_language():
    """测试语言检测"""
    parser = TreeSitterParser()
    
    assert parser._detect_language("test.py") == "python"
    assert parser._detect_language("test.js") == "javascript"
    assert parser._detect_language("test.ts") == "typescript"
    assert parser._detect_language("test.java") == "java"
    assert parser._detect_language("test.unknown") is None


def test_detect_markers():
    """测试标记检测"""
    parser = TreeSitterParser()
    
    # 测试 deprecated 标记
    content_deprecated = """
    # @deprecated This function is deprecated
    def old_function():
        pass
    """
    markers = parser.detect_markers(content_deprecated)
    assert markers['has_deprecated'] is True
    assert markers['has_fixme'] is False
    
    # 测试 FIXME 标记
    content_fixme = """
    # FIXME: This needs to be fixed
    def buggy_function():
        pass
    """
    markers = parser.detect_markers(content_fixme)
    assert markers['has_fixme'] is True
    assert markers['has_deprecated'] is False
    
    # 测试 TODO 标记
    content_todo = """
    # TODO: Implement this feature
    def incomplete_function():
        pass
    """
    markers = parser.detect_markers(content_todo)
    assert markers['has_todo'] is True
    
    # 测试 Security 标记
    content_security = """
    # Security: Check for SQL injection
    def query_database(user_input):
        pass
    """
    markers = parser.detect_markers(content_security)
    assert markers['has_security'] is True


def test_fallback_parse():
    """测试回退解析"""
    parser = TreeSitterParser()
    
    content = """
    # This is a comment
    def test_function():
        # Another comment
        pass
    """
    
    parsed = parser._fallback_parse("test.py", content, "python")
    
    assert parsed.file_path == "test.py"
    assert parsed.language == "python"
    assert parsed.raw_content == content
    assert len(parsed.elements) > 0


def test_extract_comments():
    """测试注释提取"""
    parser = TreeSitterParser()
    
    content = """
    # Comment 1
    def test():
        # Comment 2
        pass
    """
    
    parsed = parser.parse("test.py", content)
    comments = parser.extract_comments(parsed)
    
    assert len(comments) > 0
    assert all(c.type == CodeElementType.COMMENT for c in comments)
