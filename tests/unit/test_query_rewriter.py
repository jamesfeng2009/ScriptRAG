"""Unit Tests for Query Rewriter"""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.query_rewriter import (
    QueryRewriter,
    QueryNormalizer,
    QueryContext,
    RewriteResult,
    QueryType
)


class TestQueryRewriter:
    """Test cases for QueryRewriter"""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service"""
        service = MagicMock()
        service.chat_completion = AsyncMock(return_value="在 Python 中使用 FastAPI 实现异步编程")
        return service
    
    @pytest.fixture
    def rewriter(self, mock_llm_service):
        """Create QueryRewriter instance"""
        return QueryRewriter(mock_llm_service)
    
    def test_detect_query_type_terminology(self, rewriter):
        """测试术语查询类型检测"""
        result = rewriter._detect_query_type("什么是异步编程")
        assert result == QueryType.TERMINOLOGY
    
    def test_detect_query_type_usage(self, rewriter):
        """测试用法查询类型检测"""
        result = rewriter._detect_query_type("如何使用 FastAPI")
        assert result == QueryType.USAGE
    
    def test_detect_query_type_implementation(self, rewriter):
        """测试实现查询类型检测"""
        result = rewriter._detect_query_type("如何实现用户认证")
        assert result == QueryType.IMPLEMENTATION
    
    def test_detect_query_type_comparison(self, rewriter):
        """测试比较查询类型检测"""
        result = rewriter._detect_query_type("FastAPI 和 Flask 的区别")
        assert result == QueryType.COMPARISON
    
    def test_detect_query_type_debugging(self, rewriter):
        """测试调试查询类型检测"""
        result = rewriter._detect_query_type("FastAPI 报错怎么处理")
        assert result == QueryType.DEBUGGING
    
    def test_detect_language_python(self, rewriter):
        """测试 Python 语言检测"""
        result = rewriter._detect_language_framework("如何在 Python 中使用 async", None)
        assert result['language'] == 'python'
    
    def test_detect_language_javascript(self, rewriter):
        """测试 JavaScript 语言检测"""
        result = rewriter._detect_language_framework("如何用 Node.js 处理请求", None)
        assert result['language'] == 'javascript'
    
    def test_detect_framework_fastapi(self, rewriter):
        """测试 FastAPI 框架检测"""
        result = rewriter._detect_language_framework("FastAPI 路由如何配置", None)
        assert result['framework'] == 'fastapi'
    
    def test_detect_framework_django(self, rewriter):
        """测试 Django 框架检测"""
        result = rewriter._detect_language_framework("Django ORM 如何使用", None)
        assert result['framework'] == 'django'
    
    def test_is_complex_query_short(self, rewriter):
        """短查询不算复杂"""
        assert not rewriter._is_complex_query("如何使用 async")
    
    def test_is_complex_query_with_conjunction(self, rewriter):
        """包含连词的查询是复杂查询"""
        assert rewriter._is_complex_query("用户认证和权限验证的实现")
    
    def test_is_complex_query_long(self, rewriter):
        """长查询是复杂查询"""
        assert rewriter._is_complex_query("如何在 FastAPI 中实现用户认证流程，包括登录注册和 JWT 令牌验证")
    
    def test_normalize_query(self, rewriter):
        """查询标准化"""
        query = rewriter._normalize_query("  HTTP REQUEST  和  API  ")
        assert "http request" in query.lower()
    
    def test_extract_context(self, rewriter):
        """上下文提取"""
        context = QueryContext(
            project_type="web",
            detected_language="python",
            file_path="src/api.py"
        )
        
        added = rewriter._extract_context("测试查询", context)
        
        assert len(added) >= 2
        assert any("web" in c for c in added)


class TestQueryNormalizer:
    """Test cases for QueryNormalizer"""
    
    @pytest.fixture
    def normalizer(self):
        """Create QueryNormalizer instance"""
        return QueryNormalizer()
    
    def test_normalize_basic(self, normalizer):
        """基础标准化"""
        query = normalizer.normalize("  multiple   spaces  ")
        assert query == "multiple spaces"
    
    def test_normalize_abbreviations(self, normalizer):
        """缩写规范化"""
        query = normalizer.normalize("REST API")
        assert "REST API" in query
    
    def test_expand_synonyms(self, normalizer):
        """同义词扩展"""
        expanded = normalizer.expand_synonyms("async function")
        
        assert "async function" in expanded
        assert any("异步" in e for e in expanded)
        assert any("asynchronous" in e for e in expanded)
    
    def test_expand_synonyms_empty(self, normalizer):
        """空查询"""
        expanded = normalizer.expand_synonyms("")
        assert expanded == [""]


class TestQueryContext:
    """Test cases for QueryContext"""
    
    def test_default_context(self):
        """默认上下文"""
        context = QueryContext()
        
        assert context.project_type == "general"
        assert context.detected_language == ""
        assert context.recent_queries == []
    
    def test_custom_context(self):
        """自定义上下文"""
        context = QueryContext(
            project_type="web",
            detected_language="python",
            detected_framework="fastapi",
            file_path="src/main.py",
            recent_queries=["如何启动", "如何部署"]
        )
        
        assert context.project_type == "web"
        assert context.detected_language == "python"
        assert context.detected_framework == "fastapi"
        assert len(context.recent_queries) == 2


class TestRewriteResult:
    """Test cases for RewriteResult"""
    
    def test_result_creation(self):
        """结果创建"""
        result = RewriteResult(
            original_query="测试查询",
            rewritten_query="在 Python 中测试查询",
            query_type=QueryType.USAGE,
            confidence=0.85,
            added_context=["语言: python"],
            sub_queries=["子查询1", "子查询2"]
        )
        
        assert result.original_query == "测试查询"
        assert result.confidence == 0.85
        assert len(result.sub_queries) == 2


class TestQueryType:
    """Test cases for QueryType enum"""
    
    def test_query_types(self):
        """所有查询类型"""
        assert QueryType.TERMINOLOGY.value == "terminology"
        assert QueryType.USAGE.value == "usage"
        assert QueryType.IMPLEMENTATION.value == "implementation"
        assert QueryType.COMPARISON.value == "comparison"
        assert QueryType.DEBUGGING.value == "debugging"
        assert QueryType.ARCHITECTURE.value == "architecture"
        assert QueryType.GENERAL.value == "general"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
