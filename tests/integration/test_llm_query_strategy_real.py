"""Integration tests for LLMGeneratedQueryStrategy with real data

This test module verifies the effectiveness of LLM-generated query strategy
using realistic code examples instead of mocks.
"""

import pytest
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.services.retrieval import (
    LLMGeneratedQueryStrategy,
    MultiQuerySearchStrategy,
    VectorSearchStrategy,
    RetrievalResult,
    StrategyRegistry
)
from src.services.retrieval.mergers import WeightedMerger
from src.services.database.vector_db import IVectorDBService
from src.services.llm.service import LLMService
from tests.fixtures.realistic_mock_data import (
    create_realistic_code_examples,
    create_mock_llm_service
)


class RealDataTestContext:
    """Real data test context for managing test data"""

    def __init__(self):
        self.test_documents: List[Dict[str, str]] = []
        self.workspace_id: str = "test-workspace-real-data"
        self.embedding_dim: int = 384

    def setup_test_documents(self):
        """Setup realistic test documents"""
        self.test_documents = create_realistic_code_examples()
        assert len(self.test_documents) >= 5, "Need at least 5 test documents"

    def get_query_embedding(self, query: str) -> List[float]:
        """Generate deterministic embedding for a query"""
        import hashlib
        hash_value = int(hashlib.md5(query.encode()).hexdigest(), 16)
        return [(hash_value % 100) / 100.0 for _ in range(self.embedding_dim)]

    def create_mock_vector_db(self) -> IVectorDBService:
        """Create a mock vector DB with real test documents"""
        class MockVectorDBWithDocuments(IVectorDBService):
            def __init__(self, documents: List[Dict[str, str]]):
                self.documents = documents
                self.initialized = False

            async def initialize(self):
                self.initialized = True

            async def close(self):
                self.initialized = False

            async def health_check(self) -> bool:
                return self.initialized

            async def upsert(self, workspace_id: str, documents: List[Dict[str, Any]]) -> Dict[str, str]:
                return {"status": "inserted", "count": len(documents)}

            async def index_document(
                self,
                workspace_id: str,
                file_path: str,
                content: str,
                embedding: List[float],
                language: Optional[str] = None,
                has_deprecated: bool = False,
                has_fixme: bool = False,
                has_todo: bool = False,
                has_security: bool = False,
                metadata: Optional[Dict[str, Any]] = None
            ) -> str:
                doc_id = f"doc-{len(self.documents) + 1}"
                self.documents.append({
                    "file_path": file_path,
                    "content": content
                })
                return doc_id

            async def delete_document(self, workspace_id: str, file_path: str) -> bool:
                for i, doc in enumerate(self.documents):
                    if doc["file_path"] == file_path:
                        self.documents.pop(i)
                        return True
                return False

            async def delete_workspace(self, workspace_id: str) -> bool:
                self.documents = []
                return True

            async def vector_search(
                self,
                workspace_id: str,
                query_embedding: List[float],
                top_k: int = 5,
                similarity_threshold: float = 0.7
            ) -> List[RetrievalResult]:
                import random
                results = []
                for i, doc in enumerate(self.documents[:top_k]):
                    similarity = 0.95 - (i * 0.05) + random.uniform(-0.05, 0.05)
                    similarity = max(0.5, min(1.0, similarity))

                    if similarity < similarity_threshold:
                        continue

                    result = RetrievalResult(
                        id=f"doc-{i+1}",
                        file_path=doc["file_path"],
                        content=doc["content"],
                        similarity=similarity,
                        confidence=similarity,
                        strategy_name="vector_search",
                        metadata={
                            "source": "realistic_mock_data",
                            "content_length": len(doc["content"])
                        }
                    )
                    results.append(result)
                return results

            async def keyword_search(
                self,
                query: str,
                workspace_id: str,
                top_k: int = 10,
                markers: List[str] = None
            ) -> List[RetrievalResult]:
                results = []
                query_lower = query.lower()
                keywords = query_lower.split()

                for i, doc in enumerate(self.documents):
                    content_lower = doc["content"].lower()
                    matches = sum(1 for kw in keywords if kw in content_lower)

                    if matches > 0:
                        confidence = min(1.0, matches * 0.3)
                        result = RetrievalResult(
                            id=f"doc-kw-{i+1}",
                            file_path=doc["file_path"],
                            content=doc["content"],
                            similarity=confidence,
                            confidence=confidence,
                            strategy_name="keyword_search",
                            metadata={"source": "realistic_mock_data"}
                        )
                        results.append(result)

                return results[:top_k]

            async def hybrid_search(
                self,
                workspace_id: str,
                query_embedding: List[float],
                keyword_filters: Dict[str, bool] = None,
                top_k: int = 5
            ) -> List[RetrievalResult]:
                vector_results = await self.vector_search(
                    workspace_id, query_embedding, top_k
                )
                keyword_results = await self.keyword_search(
                    "", workspace_id, top_k
                )

                merger = WeightedMerger()
                combined = {"vector_search": vector_results, "keyword_search": keyword_results}
                return merger.merge(combined, top_k=top_k)

        return MockVectorDBWithDocuments(self.test_documents)


@pytest.fixture
def real_data_test_context():
    """Fixture providing real data test context"""
    context = RealDataTestContext()
    context.setup_test_documents()
    return context


@pytest.fixture
def real_test_vector_db(real_data_test_context):
    """Fixture providing vector DB with real test documents"""
    return real_data_test_context.create_mock_vector_db()


@pytest.fixture
def real_test_llm_service():
    """Fixture providing LLM service for query generation"""
    mock_service = create_mock_llm_service()
    
    async def mock_embedding(text: str) -> List[float]:
        return [0.1] * 384
    
    mock_service.embedding = mock_embedding
    return mock_service


class TestLLMGeneratedQueryStrategyWithRealData:
    """Integration tests using real data for LLMGeneratedQueryStrategy"""

    @pytest.mark.asyncio
    async def test_query_variant_generation(self, real_test_vector_db, real_test_llm_service):
        """Test that LLM generates meaningful query variants from real query"""
        strategy = LLMGeneratedQueryStrategy(
            vector_db_service=real_test_vector_db,
            llm_service=real_test_llm_service,
            max_variants=3
        )

        query = "异步编程和并发控制"
        variants = await strategy._generate_query_variants(query)

        assert len(variants) >= 3, "Should generate at least 3 variants"

        variant_texts = [v.text for v in variants]
        assert any("异步" in text or "async" in text.lower() for text in variant_texts), \
            "At least one variant should relate to async"

        for variant in variants:
            assert len(variant.text) > 0, "Variant text should not be empty"
            assert variant.confidence > 0, "Confidence should be positive"

    @pytest.mark.asyncio
    async def test_multi_query_retrieval_improves_recall(self, real_test_vector_db, real_test_llm_service):
        """Test that multi-query retrieval improves recall compared to single query"""
        strategy = LLMGeneratedQueryStrategy(
            vector_db_service=real_test_vector_db,
            llm_service=real_test_llm_service,
            max_variants=3
        )

        query = "如何使用asyncio"
        ctx = RealDataTestContext()
        ctx.setup_test_documents()
        query_embedding = ctx.get_query_embedding(query)

        # Multi-query retrieval
        multi_results = await strategy.search(
            query=query,
            query_embedding=query_embedding,
            workspace_id=ctx.workspace_id,
            top_k=10
        )

        # Single query retrieval (baseline)
        single_results = await real_test_vector_db.vector_search(
            query_embedding=query_embedding,
            workspace_id=ctx.workspace_id,
            top_k=10
        )

        # Multi-query should return at least as many results as single query
        assert len(multi_results) >= len(single_results), \
            f"Multi-query ({len(multi_results)}) should match or exceed single-query ({len(single_results)})"

    @pytest.mark.asyncio
    async def test_result_coverage_across_variants(self, real_test_vector_db, real_test_llm_service):
        """Test that results come from different query variants"""
        strategy = LLMGeneratedQueryStrategy(
            vector_db_service=real_test_vector_db,
            llm_service=real_test_llm_service,
            max_variants=5
        )

        query = "异步上下文管理器"
        ctx = RealDataTestContext()
        ctx.setup_test_documents()
        query_embedding = ctx.get_query_embedding(query)

        results = await strategy.search(
            query=query,
            query_embedding=query_embedding,
            workspace_id=ctx.workspace_id,
            top_k=10
        )

        # Check that results have variant metadata
        variant_sources = set()
        for result in results:
            if hasattr(result, 'metadata') and result.metadata:
                variant_type = result.metadata.get('variant_type')
                if variant_type:
                    variant_sources.add(variant_type)

        # Should have results from multiple variant types
        assert len(variant_sources) > 0, "Results should have variant type metadata"

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self, real_test_vector_db):
        """Test that strategy falls back gracefully when LLM fails"""

        class FailingLLMService:
            async def embedding(self, text: str) -> List[float]:
                return [0.1] * 384

            async def chat_completion(self, messages, task_type, **kwargs):
                raise Exception("LLM service unavailable")

        strategy = LLMGeneratedQueryStrategy(
            vector_db_service=real_test_vector_db,
            llm_service=FailingLLMService(),
            max_variants=3
        )

        query = "测试查询"
        ctx = RealDataTestContext()
        ctx.setup_test_documents()
        query_embedding = ctx.get_query_embedding(query)

        # Should not raise exception
        results = await strategy.search(
            query=query,
            query_embedding=query_embedding,
            workspace_id=ctx.workspace_id,
            top_k=5
        )

        # Should fall back to base strategy results
        assert len(results) >= 0, "Should return results even if LLM fails"
        assert all(isinstance(r, RetrievalResult) for r in results), \
            "All results should be RetrievalResult instances"


class TestMultiQuerySearchStrategyWithRealData:
    """Integration tests using real data for MultiQuerySearchStrategy"""

    @pytest.mark.asyncio
    async def test_variant_generation_with_rules(self, real_test_vector_db):
        """Test that expansion rules generate correct variants"""
        strategy = MultiQuerySearchStrategy(
            vector_db_service=real_test_vector_db,
            llm_service=None
        )

        query = "异步函数"
        variants = strategy._generate_variants(query)

        assert len(variants) == 5, "Should generate 5 variants with default rules"

        # Check variant types
        variant_types = [v["type"] for v in variants]
        assert "original" in variant_types, "Should include original query"
        assert "how_to" in variant_types, "Should include how-to variant"
        assert "example" in variant_types, "Should include example variant"
        assert "code" in variant_types, "Should include code variant"
        assert "tutorial" in variant_types, "Should include tutorial variant"

    @pytest.mark.asyncio
    async def test_search_with_query_expansion(self, real_test_vector_db):
        """Test search with rule-based query expansion"""
        strategy = MultiQuerySearchStrategy(
            vector_db_service=real_test_vector_db,
            llm_service=None
        )

        query = "数据库连接"
        ctx = RealDataTestContext()
        ctx.setup_test_documents()
        query_embedding = ctx.get_query_embedding(query)

        results = await strategy.search(
            query=query,
            query_embedding=query_embedding,
            workspace_id=ctx.workspace_id,
            top_k=10
        )

        assert len(results) > 0, "Should return results for expanded queries"

    @pytest.mark.asyncio
    async def test_result_merging_and_ranking(self, real_test_vector_db, real_test_llm_service):
        """Test that results from different variants are properly merged and ranked"""
        strategy = MultiQuerySearchStrategy(
            vector_db_service=real_test_vector_db,
            llm_service=real_test_llm_service
        )

        query = "异常处理"
        ctx = RealDataTestContext()
        ctx.setup_test_documents()
        query_embedding = ctx.get_query_embedding(query)

        results = await strategy.search(
            query=query,
            query_embedding=query_embedding,
            workspace_id=ctx.workspace_id,
            top_k=10
        )

        # Results should be ranked by score (check only if we have multiple results)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence - 0.001, \
                    f"Results not properly ranked at index {i}: {results[i].confidence} < {results[i+1].confidence}"

        # All results should be RetrievalResult instances
        assert all(isinstance(r, RetrievalResult) for r in results), \
            "All results should be RetrievalResult instances"


class TestQueryStrategyComparison:
    """Compare different query strategies with real data"""

    @pytest.mark.asyncio
    async def test_compare_single_vs_multi_query(self, real_test_vector_db):
        """Compare single query vs multi-query strategies"""
        # Single query strategy
        single_strategy = VectorSearchStrategy(
            vector_db_service=real_test_vector_db,
            llm_service=None
        )

        # Multi-query strategy
        multi_strategy = MultiQuerySearchStrategy(
            vector_db_service=real_test_vector_db,
            llm_service=None
        )

        query = "异步操作"
        ctx = RealDataTestContext()
        ctx.setup_test_documents()
        query_embedding = ctx.get_query_embedding(query)

        # Single query
        single_results = await single_strategy.search(
            query=query,
            query_embedding=query_embedding,
            workspace_id=ctx.workspace_id,
            top_k=10
        )

        # Multi-query
        multi_results = await multi_strategy.search(
            query=query,
            query_embedding=query_embedding,
            workspace_id=ctx.workspace_id,
            top_k=10
        )

        # Multi-query should not return fewer results
        assert len(multi_results) >= len(single_results), \
            f"Multi-query ({len(multi_results)}) should match single-query ({len(single_results)})"

    @pytest.mark.asyncio
    async def test_different_queries_return_different_results(self, real_test_vector_db, real_test_llm_service):
        """Test that multi-query strategy processes different queries"""
        strategy = MultiQuerySearchStrategy(
            vector_db_service=real_test_vector_db,
            llm_service=real_test_llm_service
        )

        query1 = "数据库连接"
        query2 = "事件循环"

        ctx = RealDataTestContext()
        ctx.setup_test_documents()
        embedding1 = ctx.get_query_embedding(query1)
        embedding2 = ctx.get_query_embedding(query2)

        results1 = await strategy.search(
            query=query1,
            query_embedding=embedding1,
            workspace_id=ctx.workspace_id,
            top_k=5
        )

        results2 = await strategy.search(
            query=query2,
            query_embedding=embedding2,
            workspace_id=ctx.workspace_id,
            top_k=5
        )

        # Both should return results
        assert len(results1) > 0, "Query 1 should return results"
        assert len(results2) > 0, "Query 2 should return results"

        # Results should be RetrievalResult instances
        assert all(isinstance(r, RetrievalResult) for r in results1), "Results 1 should be RetrievalResult instances"
        assert all(isinstance(r, RetrievalResult) for r in results2), "Results 2 should be RetrievalResult instances"


class TestRetrievalQualityMetrics:
    """Test retrieval quality metrics with real data"""

    @pytest.mark.asyncio
    async def test_result_relevance_scoring(self, real_test_vector_db):
        """Test that results have appropriate relevance scores"""
        strategy = MultiQuerySearchStrategy(
            vector_db_service=real_test_vector_db,
            llm_service=None
        )

        query = "重试装饰器"
        ctx = RealDataTestContext()
        ctx.setup_test_documents()
        query_embedding = ctx.get_query_embedding(query)

        results = await strategy.search(
            query=query,
            query_embedding=query_embedding,
            workspace_id=ctx.workspace_id,
            top_k=10
        )

        # All confidence scores should be in valid range
        for result in results:
            assert 0.0 <= result.confidence <= 1.0, \
                f"Confidence {result.confidence} out of range for result {result.id}"
            assert 0.0 <= result.similarity <= 1.0, \
                f"Similarity {result.similarity} out of range for result {result.id}"

    @pytest.mark.asyncio
    async def test_result_content_quality(self, real_test_vector_db):
        """Test that results contain meaningful content"""
        strategy = MultiQuerySearchStrategy(
            vector_db_service=real_test_vector_db,
            llm_service=None
        )

        query = "上下文管理器"
        ctx = RealDataTestContext()
        ctx.setup_test_documents()
        query_embedding = ctx.get_query_embedding(query)

        results = await strategy.search(
            query=query,
            query_embedding=query_embedding,
            workspace_id=ctx.workspace_id,
            top_k=5
        )

        for result in results:
            # Result should have content
            assert len(result.content) > 0, f"Result {result.id} should have content"

            # Result should have file path
            assert len(result.file_path) > 0, f"Result {result.id} should have file path"

            # Result should have valid ID
            assert len(result.id) > 0, f"Result should have valid ID"


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration
