"""Unit Tests for Hybrid Search with RRF"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.hybrid_search import (
    HybridSearchService,
    RRFEngine,
    BM25KeywordSearch,
    HybridSearchPipeline,
    HybridSearchConfig,
    FusionResult
)


class TestHybridSearchConfig:
    """Test cases for HybridSearchConfig"""
    
    def test_default_config(self):
        """默认配置"""
        config = HybridSearchConfig()
        
        assert config.vector_weight == 0.6
        assert config.keyword_weight == 0.4
        assert config.top_k_vector == 20
        assert config.top_k_keyword == 20
        assert config.rrf_k == 60
        assert config.enable_fusion
        assert config.min_score_threshold == 0.3
    
    def test_custom_config(self):
        """自定义配置"""
        config = HybridSearchConfig(
            vector_weight=0.7,
            keyword_weight=0.3,
            top_k_vector=30,
            top_k_keyword=30,
            rrf_k=100,
            min_score_threshold=0.4
        )
        
        assert config.vector_weight == 0.7
        assert config.keyword_weight == 0.3
        assert config.rrf_k == 100


class TestFusionResult:
    """Test cases for FusionResult"""
    
    def test_result_creation(self):
        """结果创建"""
        result = FusionResult(
            id="doc1",
            file_path="test.py",
            content="Test content",
            vector_score=0.85,
            keyword_score=0.72,
            rrf_score=0.15,
            fused_score=0.78,
            rank_vector=1,
            rank_keyword=3
        )
        
        assert result.id == "doc1"
        assert result.vector_score == 0.85
        assert result.rrf_score == 0.15
    
    def test_to_dict(self):
        """转换为字典"""
        result = FusionResult(
            id="doc1",
            file_path="test.py",
            content="A" * 300,
            vector_score=0.85,
            keyword_score=0.72,
            rrf_score=0.15,
            fused_score=0.78,
            rank_vector=1,
            rank_keyword=3
        )
        
        d = result.to_dict()
        
        assert d["id"] == "doc1"
        assert d["vector_score"] == 0.85
        assert "..." in d["content"]


class TestRRFEngine:
    """Test cases for RRFEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create RRFEngine instance"""
        return RRFEngine(k=60)
    
    @pytest.fixture
    def mock_rankings(self):
        """Create mock rankings"""
        ranking1 = []
        ranking2 = []
        
        for i in range(5):
            result1 = MagicMock()
            result1.id = f"doc{i}"
            ranking1.append(result1)
            
            result2 = MagicMock()
            result2.id = f"doc{5-i}"  # 逆序
            ranking2.append(result2)
        
        return [ranking1, ranking2]
    
    def test_fuse_basic(self, engine, mock_rankings):
        """基础融合"""
        results = engine.fuse(*mock_rankings)
        
        assert len(results) == 6
        # 应该按分数排序
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_fuse_with_weights(self, engine, mock_rankings):
        """带权重融合"""
        weights = [0.7, 0.3]
        results = engine.fuse(*mock_rankings, weights=weights)
        
        assert len(results) == 6
        # 第一排名列表的 doc0 应该有更高分数
        first_score = next(score for doc, score in results if doc == "doc0")
        
    def test_fuse_empty(self, engine):
        """空列表"""
        results = engine.fuse()
        assert results == []
    
    def test_fuse_single_list(self, engine):
        """单个列表"""
        ranking = []
        for i in range(3):
            result = MagicMock()
            result.id = f"doc{i}"
            ranking.append(result)
        
        results = engine.fuse(ranking)
        
        assert len(results) == 3
        assert results[0][0] == "doc0"  # 按原始顺序
    
    def test_fuse_with_scores(self, engine):
        """带分数融合"""
        # 创建结果
        vector_results = []
        keyword_results = []
        
        for i in range(3):
            v_result = MagicMock()
            v_result.id = f"doc{i}"
            v_result.similarity = 0.9 - i * 0.1
            v_result.content = f"Content {i}"
            v_result.file_path = f"file{i}.py"
            vector_results.append(v_result)
            
            k_result = MagicMock()
            k_result.id = f"doc{i}"
            k_result.similarity = 0.8 - i * 0.1
            k_result.content = f"Content {i}"
            k_result.file_path = f"file{i}.py"
            k_result.metadata = {}
            keyword_results.append(k_result)
        
        results = engine.fuse_with_scores(
            vector_results,
            keyword_results,
            vector_weight=0.6,
            keyword_weight=0.4
        )
        
        assert len(results) == 3
        assert all(isinstance(r, FusionResult) for r in results)


class TestBM25KeywordSearch:
    """Test cases for BM25KeywordSearch"""
    
    @pytest.fixture
    def searcher(self):
        """Create BM25KeywordSearch instance"""
        return BM25KeywordSearch(k1=1.5, b=0.75)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for indexing"""
        return [
            {
                "id": "doc1",
                "content": "Python async programming with asyncio",
                "file_path": "async.py"
            },
            {
                "id": "doc2",
                "content": "FastAPI REST API development with Python",
                "file_path": "fastapi.py"
            },
            {
                "id": "doc3",
                "content": "Database connection and query execution",
                "file_path": "database.py"
            }
        ]
    
    def test_index_documents(self, searcher, sample_documents):
        """文档索引"""
        searcher.index_documents(sample_documents)
        
        assert searcher._doc_count == 3
        assert searcher._avg_doc_length > 0
    
    def test_search_basic(self, searcher, sample_documents):
        """基础搜索"""
        searcher.index_documents(sample_documents)
        
        results = searcher.search("Python async", top_k=2)
        
        assert len(results) <= 2
        # doc1 应该排名靠前
        assert any(r.id == "doc1" for r in results)
    
    def test_search_no_match(self, searcher, sample_documents):
        """无匹配"""
        searcher.index_documents(sample_documents)
        
        results = searcher.search("xyznonexistent", top_k=5)
        
        assert len(results) == 0
    
    def test_search_all_match(self, searcher, sample_documents):
        """全部匹配"""
        searcher.index_documents(sample_documents)
        
        # 使用通用词汇
        results = searcher.search("and", top_k=10)
        
        # 可能返回部分结果
    
    def test_search_top_k_limit(self, searcher, sample_documents):
        """Top-k 限制"""
        searcher.index_documents(sample_documents)
        
        results = searcher.search("Python", top_k=1)
        
        assert len(results) == 1
    
    def test_search_empty_index(self, searcher):
        """空索引"""
        results = searcher.search("query", top_k=5)
        
        assert results == []


class TestHybridSearchService:
    """Test cases for HybridSearchService"""
    
    @pytest.fixture
    def service(self):
        """Create HybridSearchService instance"""
        return HybridSearchService(
            config=HybridSearchConfig(
                vector_weight=0.6,
                keyword_weight=0.4
            ),
            vector_strategy=None,
            keyword_strategy=None
        )
    
    @pytest.fixture
    def mock_vector_results(self):
        """Mock vector search results"""
        results = []
        for i in range(3):
            result = MagicMock()
            result.id = f"vec_doc{i}"
            result.similarity = 0.9 - i * 0.1
            result.content = f"Vector document {i}"
            result.file_path = f"vec_file{i}.py"
            result.metadata = {}
            results.append(result)
        return results
    
    @pytest.fixture
    def mock_keyword_results(self):
        """Mock keyword search results"""
        results = []
        for i in range(3):
            result = MagicMock()
            result.id = f"key_doc{i}"
            result.similarity = 0.85 - i * 0.1
            result.content = f"Keyword document {i}"
            result.file_path = f"key_file{i}.py"
            result.metadata = {"keyword_score": 0.8 - i * 0.1}
            results.append(result)
        return results
    
    def test_fuse_results_basic(self, service, mock_vector_results, mock_keyword_results):
        """基础结果融合"""
        results = service._fuse_results(
            mock_vector_results,
            mock_keyword_results,
            top_k=5
        )
        
        assert len(results) <= 6  # 3 + 3
        assert all(isinstance(r, FusionResult) for r in results)
    
    def test_fuse_results_empty(self, service):
        """空结果融合"""
        results = service._fuse_results([], [], top_k=5)
        assert results == []
    
    def test_fuse_results_only_vector(self, service, mock_vector_results):
        """仅向量结果"""
        results = service._fuse_results(
            mock_vector_results,
            [],
            top_k=5
        )
        
        assert len(results) == 3
    
    def test_fuse_results_only_keyword(self, service, mock_keyword_results):
        """仅关键词结果"""
        results = service._fuse_results(
            [],
            mock_keyword_results,
            top_k=5
        )
        
        assert len(results) == 2  # 去重后保留 2 个
    
    def test_build_rank_map(self, service, mock_vector_results):
        """排名映射构建"""
        rank_map = service._build_rank_map(mock_vector_results)
        
        assert rank_map["vec_doc0"] == 1
        assert rank_map["vec_doc1"] == 2
        assert rank_map["vec_doc2"] == 3
    
    def test_calculate_fused_score(self, service):
        """融合分数计算"""
        score = service._calculate_fused_score(
            vector_score=0.9,
            rrf_score=0.1,
            vector_rank=1,
            keyword_rank=5
        )
        
        assert 0 < score <= 1


class TestHybridSearchPipeline:
    """Test cases for HybridSearchPipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline"""
        service = HybridSearchService()
        return HybridSearchPipeline(hybrid_service=service)
    
    @pytest.mark.asyncio
    async def test_search_empty_results(self, pipeline):
        """空结果搜索"""
        service = HybridSearchService()
        pipeline = HybridSearchPipeline(hybrid_service=service)
        
        result = await pipeline.search(
            query="test",
            query_embedding=[0.1] * 384,
            workspace_id="test_workspace",
            top_k=5
        )
        
        assert "results" in result
        assert "stats" in result
        assert "metadata" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
