"""Tests for retrieval strategies and mergers"""

import pytest
import asyncio
from typing import List
from unittest.mock import Mock, AsyncMock, patch

from src.services.retrieval import (
    RetrievalConfig,
    RetrievalStrategyConfig,
    MergerConfig,
    RetrievalResult,
    VectorSearchStrategy,
    KeywordSearchStrategy,
    HybridSearchStrategy,
    WeightedMerger,
    ReciprocalRankMerger,
    FusionMerger,
    RoundRobinMerger,
    StrategyRegistry,
    MergerRegistry,
    LLMGeneratedQueryStrategy,
    LLMQueryGenerationConfig,
    QueryVariant,
    MultiQuerySearchStrategy
)


class TestRetrievalConfig:
    """Tests for RetrievalConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = RetrievalConfig()

        assert config.version == "1.0"
        assert config.strategy.name.value == "hybrid_search"
        assert config.strategy.vector.enabled is True
        assert config.strategy.vector.top_k == 5
        assert config.strategy.vector.similarity_threshold == 0.7
        assert config.strategy.keyword.enabled is True
        assert len(config.strategy.keyword.markers) == 4
        assert config.merger.type.value == "weighted_merge"

    def test_add_custom_marker(self):
        """Test adding custom markers"""
        config = RetrievalConfig()

        assert config.add_custom_marker("@important") is True
        assert "@important" in config.custom_markers
        assert config.add_custom_marker("@important") is False  # Already exists
        assert len(config.custom_markers) == 1

    def test_remove_custom_marker(self):
        """Test removing custom markers"""
        config = RetrievalConfig()
        config.add_custom_marker("@custom")

        assert config.remove_custom_marker("@custom") is True
        assert "@custom" not in config.custom_markers
        assert config.remove_custom_marker("@nonexistent") is False

    def test_get_all_markers(self):
        """Test getting all markers including custom"""
        config = RetrievalConfig()
        config.add_custom_marker("@custom1")
        config.add_custom_marker("@custom2")

        all_markers = config.get_all_markers()

        assert "@custom1" in all_markers
        assert "@custom2" in all_markers
        assert "@deprecated" in all_markers  # Default marker
        assert len(all_markers) == 6  # 4 default + 2 custom

    def test_custom_strategy_registration(self):
        """Test registering custom strategies"""
        config = RetrievalConfig()

        result = config.register_custom_strategy(
            "llm_query",
            {"enabled": True, "type": "custom"}
        )

        assert result is True
        assert "llm_query" in config.custom_strategies
        assert config.unregister_custom_strategy("llm_query") is True
        assert "llm_query" not in config.custom_strategies


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass"""

    def test_create_result(self):
        """Test creating a retrieval result"""
        result = RetrievalResult(
            id="test-id",
            file_path="/test/file.py",
            content="Test content",
            similarity=0.8,
            confidence=0.75,
            strategy_name="vector_search"
        )

        assert result.id == "test-id"
        assert result.similarity == 0.8
        assert result.confidence == 0.75
        assert result.strategy_name == "vector_search"
        assert result.metadata is not None

    def test_result_with_markers(self):
        """Test result with keyword markers"""
        result = RetrievalResult(
            id="test-id",
            file_path="/test/file.py",
            content="Test content",
            similarity=0.8,
            confidence=0.75,
            strategy_name="keyword_search",
            has_deprecated=True,
            has_fixme=False,
            has_todo=True,
            has_security=False
        )

        assert result.has_deprecated is True
        assert result.has_todo is True
        assert result.has_security is False


class TestVectorSearchStrategy:
    """Tests for VectorSearchStrategy"""

    @pytest.fixture
    def mock_services(self):
        """Create mock services"""
        vector_db = Mock()
        llm_service = Mock()

        vector_db.vector_search = AsyncMock(return_value=[])
        llm_service.embedding = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        return vector_db, llm_service

    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_services):
        """Test that search returns results"""
        vector_db, llm_service = mock_services

        mock_db_results = [
            Mock(
                id="1",
                file_path="/test/file1.py",
                content="Content 1",
                similarity=0.9,
                has_deprecated=False,
                has_fixme=False,
                has_todo=False,
                has_security=False,
                metadata={}
            )
        ]
        vector_db.vector_search = AsyncMock(return_value=mock_db_results)

        strategy = VectorSearchStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service,
            similarity_threshold=0.7
        )

        results = await strategy.search(
            query="test query",
            query_embedding=[0.1, 0.2, 0.3],
            workspace_id="workspace-1",
            top_k=5
        )

        assert len(results) == 1
        assert results[0].id == "1"
        assert results[0].strategy_name == "vector_search"

    @pytest.mark.asyncio
    async def test_search_empty_query(self, mock_services):
        """Test that empty query returns empty results"""
        vector_db, llm_service = mock_services

        strategy = VectorSearchStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        results = await strategy.search(
            query="",
            query_embedding=[0.1, 0.2, 0.3],
            workspace_id="workspace-1",
            top_k=5
        )

        assert len(results) == 0

    def test_supported_markers(self):
        """Test that vector search supports no markers"""
        strategy = VectorSearchStrategy(
            vector_db_service=Mock(),
            llm_service=Mock()
        )

        assert len(strategy.supported_markers) == 0


class TestKeywordSearchStrategy:
    """Tests for KeywordSearchStrategy"""

    @pytest.fixture
    def mock_services(self):
        """Create mock services"""
        vector_db = Mock()
        llm_service = Mock()

        vector_db.hybrid_search = AsyncMock(return_value=[])
        llm_service.embedding = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        return vector_db, llm_service

    def test_custom_markers(self):
        """Test custom marker configuration"""
        strategy = KeywordSearchStrategy(
            vector_db_service=Mock(),
            llm_service=Mock(),
            markers=["@custom", "@test"]
        )

        assert "@custom" in strategy.markers
        assert "@test" in strategy.markers

    @pytest.mark.asyncio
    async def test_search_with_custom_markers(self, mock_services):
        """Test search with custom markers"""
        vector_db, llm_service = mock_services

        mock_db_results = [
            Mock(
                id="1",
                file_path="/test/file1.py",
                content="Content 1",
                similarity=0.85,
                has_deprecated=False,
                has_fixme=True,
                has_todo=False,
                has_security=False,
                metadata={}
            )
        ]
        vector_db.hybrid_search = AsyncMock(return_value=mock_db_results)

        strategy = KeywordSearchStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service,
            markers=["@deprecated", "FIXME", "TODO", "Security"]
        )

        results = await strategy.search(
            query="test query",
            query_embedding=[0.1, 0.2, 0.3],
            workspace_id="workspace-1",
            top_k=5,
            custom_markers=["@custom"]
        )

        assert len(results) == 1
        assert results[0].strategy_name == "keyword_search"

    def test_boost_factors(self):
        """Test boost factors are applied correctly"""
        strategy = KeywordSearchStrategy(
            vector_db_service=Mock(),
            llm_service=Mock()
        )

        assert strategy.boost_factors["@deprecated"] == 1.5
        assert strategy.boost_factors["FIXME"] == 1.3


class TestWeightedMerger:
    """Tests for WeightedMerger"""

    def test_merge_single_strategy(self):
        """Test merging results from single strategy"""
        results = {
            "vector_search": [
                RetrievalResult(
                    id="1", file_path="f1", content="c1",
                    similarity=0.9, confidence=0.9, strategy_name="vector_search"
                ),
                RetrievalResult(
                    id="2", file_path="f2", content="c2",
                    similarity=0.8, confidence=0.8, strategy_name="vector_search"
                )
            ]
        }

        merger = WeightedMerger()
        merged = merger.merge(results, top_k=5)

        assert len(merged) == 2
        assert merged[0].id == "1"

    def test_merge_multiple_strategies(self):
        """Test merging results from multiple strategies"""
        results = {
            "vector_search": [
                RetrievalResult(
                    id="1", file_path="f1", content="c1",
                    similarity=0.9, confidence=0.9, strategy_name="vector_search"
                )
            ],
            "keyword_search": [
                RetrievalResult(
                    id="2", file_path="f2", content="c2",
                    similarity=0.8, confidence=0.8, strategy_name="keyword_search"
                )
            ]
        }

        weights = {"vector_search": 0.6, "keyword_search": 0.4}

        merger = WeightedMerger()
        merged = merger.merge(results, weights=weights, top_k=5)

        assert len(merged) == 2

    def test_merge_empty_results(self):
        """Test merging empty results"""
        merger = WeightedMerger()
        merged = merger.merge({}, top_k=5)

        assert len(merged) == 0

    def test_merge_with_deduplication(self):
        """Test that duplicate results are deduplicated"""
        results = {
            "vector_search": [
                RetrievalResult(
                    id="1", file_path="f1", content="same content",
                    similarity=0.9, confidence=0.9, strategy_name="vector_search"
                )
            ],
            "keyword_search": [
                RetrievalResult(
                    id="1", file_path="f1", content="same content",
                    similarity=0.8, confidence=0.8, strategy_name="keyword_search"
                )
            ]
        }

        weights = {"vector_search": 0.6, "keyword_search": 0.4}

        merger = WeightedMerger(dedup_threshold=0.95)
        merged = merger.merge(results, weights=weights, top_k=5)

        assert len(merged) == 1  # Only one result (deduplicated)


class TestReciprocalRankMerger:
    """Tests for ReciprocalRankMerger (RRF)"""

    def test_rrf_merge(self):
        """Test RRF merging"""
        results = {
            "vector_search": [
                RetrievalResult(
                    id="1", file_path="f1", content="c1",
                    similarity=0.9, confidence=0.9, strategy_name="vector_search"
                ),
                RetrievalResult(
                    id="2", file_path="f2", content="c2",
                    similarity=0.8, confidence=0.8, strategy_name="vector_search"
                )
            ],
            "keyword_search": [
                RetrievalResult(
                    id="1", file_path="f1", content="c1",
                    similarity=0.85, confidence=0.85, strategy_name="keyword_search"
                )
            ]
        }

        merger = ReciprocalRankMerger(k=60)
        merged = merger.merge(results, top_k=5)

        assert len(merged) == 2
        # ID 1 should be first (appears in both lists at rank 1)
        assert merged[0].id == "1"

    def test_rrf_custom_k(self):
        """Test RRF with custom k value"""
        merger = ReciprocalRankMerger(k=100)
        assert merger.k == 100


class TestFusionMerger:
    """Tests for FusionMerger"""

    def test_fusion_merge(self):
        """Test fusion merging"""
        results = {
            "vector_search": [
                RetrievalResult(
                    id="1", file_path="f1", content="c1",
                    similarity=0.9, confidence=0.9, strategy_name="vector_search"
                )
            ],
            "keyword_search": [
                RetrievalResult(
                    id="2", file_path="f2", content="c2",
                    similarity=0.8, confidence=0.8, strategy_name="keyword_search"
                )
            ]
        }

        merger = FusionMerger(alpha=0.7)
        merged = merger.merge(results, top_k=5)

        assert len(merged) == 2

    def test_fusion_alpha_parameter(self):
        """Test fusion alpha parameter"""
        merger = FusionMerger(alpha=0.3)
        assert merger.alpha == 0.3


class TestRoundRobinMerger:
    """Tests for RoundRobinMerger"""

    def test_round_robin_merge(self):
        """Test round-robin merging"""
        results = {
            "strategy1": [
                RetrievalResult(
                    id=f"{i}", file_path=f"f{i}", content=f"c{i}",
                    similarity=0.9, confidence=0.9, strategy_name="strategy1"
                ) for i in range(5)
            ],
            "strategy2": [
                RetrievalResult(
                    id=f"{i+10}", file_path=f"f{i+10}", content=f"c{i+10}",
                    similarity=0.8, confidence=0.8, strategy_name="strategy2"
                ) for i in range(5)
            ]
        }

        merger = RoundRobinMerger(max_per_strategy=3)
        merged = merger.merge(results, top_k=6)

        assert len(merged) == 6


class TestStrategyRegistry:
    """Tests for StrategyRegistry"""

    def test_register_strategy(self):
        """Test registering a custom strategy"""
        @StrategyRegistry.register("custom_strategy")
        class CustomStrategy:
            name = "custom_strategy"

            def __init__(self, **kwargs):
                pass

        assert "custom_strategy" in StrategyRegistry.list_strategies()

    def test_create_strategy(self):
        """Test creating a strategy from registry"""
        strategy = StrategyRegistry.create_strategy(
            "vector_search",
            vector_db_service=Mock(),
            llm_service=Mock(),
            config={"similarity_threshold": 0.8}
        )

        assert strategy is not None
        assert isinstance(strategy, VectorSearchStrategy)


class TestMergerRegistry:
    """Tests for MergerRegistry"""

    def test_list_mergers(self):
        """Test listing available mergers"""
        mergers = MergerRegistry.list_mergers()

        assert "weighted_merge" in mergers
        assert "rrf_merge" in mergers
        assert "fusion_merge" in mergers

    def test_create_merger(self):
        """Test creating a merger from registry"""
        merger = MergerRegistry.create_merger(
            "weighted_merge",
            config={"dedup_threshold": 0.8}
        )

        assert merger is not None
        assert isinstance(merger, WeightedMerger)

    def test_create_unknown_merger(self):
        """Test creating unknown merger returns None"""
        merger = MergerRegistry.create_merger("unknown_merger")

        assert merger is None


@pytest.mark.asyncio
class TestAsyncRetrieval:
    """Async tests for retrieval operations"""

    async def test_concurrent_strategy_search(self):
        """Test concurrent search with multiple strategies"""
        vector_db = Mock()
        llm_service = Mock()

        vector_db.vector_search = AsyncMock(return_value=[])
        vector_db.hybrid_search = AsyncMock(return_value=[])
        llm_service.embedding = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        vector_strategy = VectorSearchStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        keyword_strategy = KeywordSearchStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        tasks = [
            vector_strategy.search(
                query="test", query_embedding=[0.1],
                workspace_id="ws", top_k=5
            ),
            keyword_strategy.search(
                query="test", query_embedding=[0.1],
                workspace_id="ws", top_k=5
            )
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 2
        assert isinstance(results[0], list)
        assert isinstance(results[1], list)


class TestQueryVariant:
    """Tests for QueryVariant dataclass"""

    def test_create_variant(self):
        """Test creating a query variant"""
        variant = QueryVariant(
            text="用户登录流程",
            type="term_expansion",
            confidence=0.9,
            description="扩展为具体流程"
        )

        assert variant.text == "用户登录流程"
        assert variant.type == "term_expansion"
        assert variant.confidence == 0.9
        assert variant.description == "扩展为具体流程"

    def test_default_values(self):
        """Test default values"""
        variant = QueryVariant(text="测试查询")

        assert variant.type == "semantic_variant"
        assert variant.confidence == 1.0
        assert variant.description == ""


class TestLLMQueryGenerationConfig:
    """Tests for LLMQueryGenerationConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = LLMQueryGenerationConfig()

        assert config.max_variants == 5
        assert config.include_synonyms is True
        assert config.include_expansion is True
        assert config.include_abstraction is True
        assert config.include_question_form is True
        assert config.temperature == 0.7

    def test_custom_config(self):
        """Test custom configuration"""
        config = LLMQueryGenerationConfig(
            max_variants=3,
            include_synonyms=False,
            temperature=0.5
        )

        assert config.max_variants == 3
        assert config.include_synonyms is False
        assert config.temperature == 0.5


class TestLLMGeneratedQueryStrategy:
    """Tests for LLMGeneratedQueryStrategy"""

    @pytest.fixture
    def mock_services(self):
        """Create mock services"""
        vector_db = Mock()
        llm_service = Mock()

        vector_db.vector_search = AsyncMock(return_value=[])
        llm_service.embedding = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        llm_service.chat_completion = AsyncMock(
            return_value='[{"text": "查询变体1", "type": "semantic_variant", "confidence": 0.9, "description": "测试"}]'
        )

        return vector_db, llm_service

    def test_strategy_registration(self):
        """Test that strategy is registered"""
        assert "llm_generated_query" in StrategyRegistry.list_strategies()

    def test_init_with_defaults(self, mock_services):
        """Test initialization with default values"""
        vector_db, llm_service = mock_services

        strategy = LLMGeneratedQueryStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        assert strategy.max_variants == 5
        assert strategy.include_synonyms is True
        assert strategy.include_expansion is True
        assert strategy.temperature == 0.7

    def test_init_with_custom_config(self, mock_services):
        """Test initialization with custom configuration"""
        vector_db, llm_service = mock_services

        strategy = LLMGeneratedQueryStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service,
            max_variants=3,
            include_synonyms=False,
            temperature=0.5
        )

        assert strategy.max_variants == 3
        assert strategy.include_synonyms is False
        assert strategy.temperature == 0.5

    def test_validate_query(self, mock_services):
        """Test query validation"""
        vector_db, llm_service = mock_services

        strategy = LLMGeneratedQueryStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        assert strategy._validate_query("用户认证") is True
        assert strategy._validate_query("") is False
        assert strategy._validate_query("  ") is False
        assert strategy._validate_query("a") is False

    def test_parse_variants_from_response(self, mock_services):
        """Test parsing variants from LLM response"""
        vector_db, llm_service = mock_services

        strategy = LLMGeneratedQueryStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        response = '''[
            {"text": "用户登录", "type": "semantic_variant", "confidence": 0.9, "description": "登录流程"},
            {"text": "身份验证", "type": "term_expansion", "confidence": 0.85, "description": "技术术语"}
        ]'''

        variants = strategy._parse_variants_from_response(response)

        assert len(variants) == 2
        assert variants[0].text == "用户登录"
        assert variants[0].type == "semantic_variant"
        assert variants[1].text == "身份验证"
        assert variants[1].type == "term_expansion"

    def test_parse_invalid_json(self, mock_services):
        """Test parsing invalid JSON response"""
        vector_db, llm_service = mock_services

        strategy = LLMGeneratedQueryStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        variants = strategy._parse_variants_from_response("invalid json")
        assert len(variants) == 0

    def test_generate_default_variants(self, mock_services):
        """Test default variants generation"""
        vector_db, llm_service = mock_services

        strategy = LLMGeneratedQueryStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        variants = strategy._generate_default_variants("用户认证")

        assert len(variants) == 3
        assert variants[0].text == "用户认证"
        assert variants[0].type == "original"
        assert variants[1].text == "如何实现 用户认证"
        assert variants[1].type == "question_form"

    def test_annotate_result(self, mock_services):
        """Test result annotation with variant info"""
        vector_db, llm_service = mock_services

        strategy = LLMGeneratedQueryStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        result = RetrievalResult(
            id="1",
            file_path="/test/file.py",
            content="test content",
            similarity=0.9,
            confidence=0.9,
            strategy_name="vector_search"
        )

        variant = QueryVariant(
            text="查询变体",
            type="semantic_variant",
            confidence=0.9,
            description="测试变体"
        )

        annotated = strategy._annotate_result(result, variant)

        assert annotated.metadata["variant_text"] == "查询变体"
        assert annotated.metadata["variant_type"] == "semantic_variant"
        assert annotated.metadata["variant_confidence"] == 0.9
        assert annotated.metadata["source"] == "llm_generated_query"

    def test_deduplicate_results(self, mock_services):
        """Test result deduplication"""
        vector_db, llm_service = mock_services

        strategy = LLMGeneratedQueryStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        results = [
            RetrievalResult(
                id="1",
                file_path="/test/file1.py",
                content="content 1",
                similarity=0.9,
                confidence=0.9,
                strategy_name="vs1"
            ),
            RetrievalResult(
                id="2",
                file_path="/test/file2.py",
                content="content 2",
                similarity=0.8,
                confidence=0.8,
                strategy_name="vs2"
            ),
            RetrievalResult(
                id="1",
                file_path="/test/file1.py",
                content="content 1",
                similarity=0.85,
                confidence=0.85,
                strategy_name="vs3"
            )
        ]

        unique = strategy._deduplicate_results(results, top_k=5)

        assert len(unique) == 2
        assert unique[0].id == "1"
        assert unique[1].id == "2"

    @pytest.mark.asyncio
    async def test_fallback_search(self, mock_services):
        """Test fallback to base strategy"""
        vector_db, llm_service = mock_services

        vector_db.vector_search = AsyncMock(return_value=[
            Mock(
                id="1",
                file_path="/test/file.py",
                content="content",
                similarity=0.9,
                has_deprecated=False,
                has_fixme=False,
                has_todo=False,
                has_security=False,
                metadata={}
            )
        ])

        strategy = LLMGeneratedQueryStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        results = await strategy._fallback_search(
            query="测试查询",
            query_embedding=[0.1, 0.2, 0.3],
            workspace_id="ws-1",
            top_k=5
        )

        assert len(results) == 1
        assert results[0].id == "1"

    def test_supported_markers(self, mock_services):
        """Test supported markers property"""
        vector_db, llm_service = mock_services

        strategy = LLMGeneratedQueryStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        markers = strategy.supported_markers
        assert isinstance(markers, set)


class TestMultiQuerySearchStrategy:
    """Tests for MultiQuerySearchStrategy"""

    def test_strategy_registration(self):
        """Test that strategy is registered"""
        assert "multi_query_search" in StrategyRegistry.list_strategies()

    @pytest.fixture
    def mock_services(self):
        """Create mock services"""
        vector_db = Mock()
        llm_service = Mock()

        vector_db.vector_search = AsyncMock(return_value=[])
        llm_service.embedding = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        return vector_db, llm_service

    def test_init_with_defaults(self, mock_services):
        """Test initialization with defaults"""
        vector_db, llm_service = mock_services

        strategy = MultiQuerySearchStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        assert len(strategy.expansion_rules) == 5
        assert strategy.expansion_rules[0]["type"] == "original"

    def test_validate_query(self, mock_services):
        """Test query validation"""
        vector_db, llm_service = mock_services

        strategy = MultiQuerySearchStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        assert strategy._validate_query("测试查询") is True
        assert strategy._validate_query("") is False
        assert strategy._validate_query("  ") is False

    def test_generate_variants(self, mock_services):
        """Test variant generation"""
        vector_db, llm_service = mock_services

        strategy = MultiQuerySearchStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        variants = strategy._generate_variants("用户认证")

        assert len(variants) == 5
        assert variants[0]["text"] == "用户认证"
        assert variants[1]["text"] == "如何实现 用户认证"
        assert variants[2]["text"] == "用户认证 示例"

    def test_merge_and_rank(self, mock_services):
        """Test result merging and ranking"""
        vector_db, llm_service = mock_services

        strategy = MultiQuerySearchStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        results = [
            RetrievalResult(
                id="1",
                file_path="/test/file1.py",
                content="content",
                similarity=0.9,
                confidence=0.9,
                strategy_name="vs"
            ),
            RetrievalResult(
                id="2",
                file_path="/test/file2.py",
                content="content",
                similarity=0.8,
                confidence=0.8,
                strategy_name="vs"
            )
        ]

        merged = strategy._merge_and_rank(results, top_k=5)

        assert len(merged) == 2
        assert merged[0].id == "1"

    @pytest.mark.asyncio
    async def test_search_with_variants(self, mock_services):
        """Test search with multiple query variants"""
        vector_db, llm_service = mock_services

        vector_db.vector_search = AsyncMock(return_value=[
            Mock(
                id="1",
                file_path="/test/file.py",
                content="content",
                similarity=0.9,
                has_deprecated=False,
                has_fixme=False,
                has_todo=False,
                has_security=False,
                metadata={}
            )
        ])

        strategy = MultiQuerySearchStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        results = await strategy.search(
            query="用户认证",
            query_embedding=[0.1, 0.2, 0.3],
            workspace_id="ws-1",
            top_k=5
        )

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_empty_query(self, mock_services):
        """Test search with empty query"""
        vector_db, llm_service = mock_services

        strategy = MultiQuerySearchStrategy(
            vector_db_service=vector_db,
            llm_service=llm_service
        )

        results = await strategy.search(
            query="",
            query_embedding=[0.1, 0.2, 0.3],
            workspace_id="ws-1",
            top_k=5
        )

        assert len(results) == 0
