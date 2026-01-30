"""Unit tests for Vector DB Service"""

import pytest
from src.services.database.vector_db import VectorSearchResult


def test_vector_search_result_creation():
    """测试向量搜索结果创建"""
    result = VectorSearchResult(
        id="test-id",
        file_path="test.py",
        content="def test(): pass",
        similarity=0.95,
        has_deprecated=False,
        has_fixme=True,
        has_todo=False,
        has_security=False
    )
    
    assert result.id == "test-id"
    assert result.file_path == "test.py"
    assert result.similarity == 0.95
    assert result.has_fixme is True
    assert result.has_deprecated is False


def test_vector_search_result_with_metadata():
    """测试带元数据的向量搜索结果"""
    result = VectorSearchResult(
        id="test-id",
        file_path="test.py",
        content="def test(): pass",
        similarity=0.85,
        metadata={"language": "python", "author": "test"}
    )
    
    assert result.metadata["language"] == "python"
    assert result.metadata["author"] == "test"
