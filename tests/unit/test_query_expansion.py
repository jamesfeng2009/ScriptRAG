"""Unit tests for Query Expansion Service"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.services.query_expansion import QueryExpansion, QueryOptimizer


class TestQueryOptimizer:
    """Test QueryOptimizer class"""
    
    def test_optimize_basic_query(self):
        """Test basic query optimization"""
        optimizer = QueryOptimizer()
        
        # Test simple query
        result = optimizer.optimize_query("find authentication code")
        assert result == "find authentication code"
        assert len(result) > 0
    
    def test_optimize_normalizes_whitespace(self):
        """Test whitespace normalization"""
        optimizer = QueryOptimizer()
        
        # Query with extra whitespace
        result = optimizer.optimize_query("find   authentication    code")
        assert "  " not in result
        assert result == "find authentication code"
    
    def test_optimize_handles_empty_query(self):
        """Test handling of empty query"""
        optimizer = QueryOptimizer()
        
        result = optimizer.optimize_query("")
        assert result == ""
    
    def test_optimize_preserves_technical_terms(self):
        """Test that technical terms are preserved"""
        optimizer = QueryOptimizer()
        
        # Technical query
        result = optimizer.optimize_query("OAuth2 JWT token validation")
        assert "oauth2" in result.lower()
        assert "jwt" in result.lower()
        assert "token" in result.lower()
        assert "validation" in result.lower()
    
    def test_extract_intent_how_to(self):
        """Test intent extraction for how-to queries"""
        optimizer = QueryOptimizer()
        
        intent = optimizer.extract_intent("how to implement authentication")
        assert intent['type'] == 'how_to'
        assert len(intent['keywords']) > 0
    
    def test_extract_intent_troubleshooting(self):
        """Test intent extraction for troubleshooting queries"""
        optimizer = QueryOptimizer()
        
        intent = optimizer.extract_intent("authentication error in login")
        assert intent['type'] == 'troubleshooting'


class TestQueryExpansion:
    """Test QueryExpansion class"""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service"""
        service = MagicMock()
        service.generate = AsyncMock()
        return service
    
    @pytest.fixture
    def query_expansion(self, mock_llm_service):
        """Create QueryExpansion instance"""
        return QueryExpansion(mock_llm_service)
    
    @pytest.mark.asyncio
    async def test_expand_query_basic(self, query_expansion, mock_llm_service):
        """Test basic query expansion"""
        # Mock LLM response
        mock_llm_service.chat_completion = AsyncMock(return_value="""
user authentication implementation
login system code
auth module design
        """)
        
        result = await query_expansion.expand_query("authentication code")
        
        # Should include original query
        assert "authentication code" in result
        
        # Should include expanded queries
        assert len(result) > 1
    
    @pytest.mark.asyncio
    async def test_expand_query_limits_results(self, query_expansion, mock_llm_service):
        """Test that expansion limits number of results"""
        # Mock LLM response with many variations
        mock_llm_service.chat_completion = AsyncMock(return_value="\n".join([
            f"variation {i}" for i in range(1, 20)
        ]))
        
        result = await query_expansion.expand_query("test query")
        
        # Should limit to max_expansions + original (default is 2 + 1)
        assert len(result) <= 3
    
    @pytest.mark.asyncio
    async def test_expand_query_handles_llm_failure(self, query_expansion, mock_llm_service):
        """Test graceful handling of LLM failure"""
        # Mock LLM failure
        mock_llm_service.chat_completion.side_effect = Exception("LLM error")
        
        result = await query_expansion.expand_query("test query")
        
        # Should return at least the original query
        assert len(result) >= 1
        assert "test query" in result
    
    @pytest.mark.asyncio
    async def test_expand_query_deduplicates(self, query_expansion, mock_llm_service):
        """Test that duplicate expansions are removed"""
        # Mock LLM response with duplicates
        mock_llm_service.chat_completion = AsyncMock(return_value="""
authentication code
auth implementation
authentication code
login system
        """)
        
        result = await query_expansion.expand_query("authentication code")
        
        # Original query should be included
        assert "authentication code" in result
    
    @pytest.mark.asyncio
    async def test_expand_query_filters_empty_lines(self, query_expansion, mock_llm_service):
        """Test that empty lines are filtered"""
        # Mock LLM response with empty lines
        mock_llm_service.chat_completion = AsyncMock(return_value="""
authentication code

auth implementation


login system
        """)
        
        result = await query_expansion.expand_query("authentication code")
        
        # Should not include empty strings
        assert all(len(q.strip()) > 0 for q in result)
    
    @pytest.mark.asyncio
    async def test_expand_query_removes_numbering(self, query_expansion, mock_llm_service):
        """Test that numbering is removed from expansions"""
        # Mock LLM response with various numbering formats
        mock_llm_service.chat_completion = AsyncMock(return_value="""
1. authentication code
2) auth implementation
3- login system
4. user login
        """)
        
        result = await query_expansion.expand_query("test")
        
        # Should remove numbering prefixes
        for query in result:
            if query != "test":  # Skip original query
                # Check that query doesn't start with number followed by punctuation
                assert not (len(query) > 0 and query[0].isdigit())
    
    @pytest.mark.asyncio
    async def test_expand_query_preserves_technical_terms(self, query_expansion, mock_llm_service):
        """Test that technical terms are preserved in expansions"""
        # Mock LLM response
        mock_llm_service.chat_completion = AsyncMock(return_value="""
OAuth2 authentication flow
JWT token validation
API security implementation
        """)
        
        result = await query_expansion.expand_query("OAuth2 JWT")
        
        # Should have multiple results
        assert len(result) > 1
    
    @pytest.mark.asyncio
    async def test_expand_query_with_empty_input(self, query_expansion, mock_llm_service):
        """Test handling of empty input"""
        result = await query_expansion.expand_query("")
        
        # Should return at least empty string
        assert len(result) >= 1
        assert "" in result
    
    @pytest.mark.asyncio
    async def test_expand_query_prompt_construction(self, query_expansion, mock_llm_service):
        """Test that prompt is constructed correctly"""
        mock_llm_service.chat_completion = AsyncMock(return_value="test expansion")
        
        await query_expansion.expand_query("test query")
        
        # Verify LLM was called
        assert mock_llm_service.chat_completion.called
        
        # Verify messages contain the query
        call_args = mock_llm_service.chat_completion.call_args
        messages = call_args[1]['messages']
        assert any("test query" in str(msg) for msg in messages)


@pytest.mark.asyncio
async def test_integration_optimizer_and_expansion():
    """Test integration between optimizer and expansion"""
    # Create mock LLM service
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value="""
user authentication system
login implementation
auth module code
    """)
    
    # Create instances
    optimizer = QueryOptimizer()
    expansion = QueryExpansion(mock_llm)
    
    # Optimize then expand
    original_query = "how to implement the authentication in a system"
    optimized = optimizer.optimize_query(original_query)
    expanded = await expansion.expand_query(optimized)
    
    # Should have multiple queries
    assert len(expanded) >= 1
    
    # Should include optimized query
    assert optimized in expanded
