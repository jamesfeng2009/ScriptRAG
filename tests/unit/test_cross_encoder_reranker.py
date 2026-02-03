"""Unit Tests for Cross-Encoder Reranker"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.cross_encoder_reranker import (
    CrossEncoderReranker,
    FallbackReranker,
    RerankingPipeline,
    MMMReranker,
    RerankConfig,
    RerankResult
)


class TestRerankConfig:
    """Test cases for RerankConfig"""
    
    def test_default_config(self):
        """默认配置"""
        config = RerankConfig()
        
        assert config.model_name == "BAAI/bge-reranker-base"
        assert config.batch_size == 32
        assert config.top_k == 10
        assert config.fusion_weight_vector == 0.6
        assert config.fusion_weight_cross == 0.4
        assert config.min_score_threshold == 0.3
    
    def test_custom_config(self):
        """自定义配置"""
        config = RerankConfig(
            model_name="custom-model",
            batch_size=64,
            top_k=20,
            fusion_weight_vector=0.7,
            fusion_weight_cross=0.3
        )
        
        assert config.model_name == "custom-model"
        assert config.batch_size == 64
        assert config.top_k == 20


class TestRerankResult:
    """Test cases for RerankResult"""
    
    def test_result_creation(self):
        """结果创建"""
        result = RerankResult(
            id="doc1",
            original_index=0,
            cross_score=0.85,
            fused_score=0.75,
            content="Test content",
            file_path="test.py"
        )
        
        assert result.id == "doc1"
        assert result.cross_score == 0.85
        assert result.fused_score == 0.75
    
    def test_to_dict(self):
        """转换为字典"""
        result = RerankResult(
            id="doc1",
            original_index=0,
            cross_score=0.85,
            fused_score=0.75,
            content="A" * 300,  # 超过200字符
            file_path="test.py"
        )
        
        d = result.to_dict()
        
        assert d["id"] == "doc1"
        assert "..." in d["content"]  # 应该被截断
        assert len(d["content"]) < 250


class TestFallbackReranker:
    """Test cases for FallbackReranker"""
    
    @pytest.fixture
    def reranker(self):
        """Create FallbackReranker instance"""
        return FallbackReranker(
            keyword_weight=0.3,
            length_penalty=0.1
        )
    
    @pytest.fixture
    def mock_results(self):
        """Create mock retrieval results"""
        results = []
        for i in range(5):
            result = MagicMock()
            result.id = f"doc{i}"
            result.similarity = 0.9 - i * 0.1
            result.content = f"Document {i} content " + "word " * 50
            result.file_path = f"file{i}.py"
            results.append(result)
        return results
    
    @pytest.mark.asyncio
    async def test_rerank_basic(self, reranker, mock_results):
        """基础重排序测试"""
        results = await reranker.rerank(
            "test query",
            mock_results,
            top_k=3
        )
        
        assert len(results) == 3
        # 应该按融合分数排序
        assert results[0].fused_score >= results[1].fused_score
    
    @pytest.mark.asyncio
    async def test_rerank_empty(self, reranker):
        """空结果测试"""
        results = await reranker.rerank("query", [], top_k=5)
        assert results == []
    
    @pytest.mark.asyncio
    async def test_calculate_keyword_match(self, reranker):
        """关键词匹配分数计算"""
        query_words = {"test", "query"}
        
        # 包含所有关键词
        score1 = reranker._calculate_keyword_match(query_words, "test query content")
        assert score1 == 1.0
        
        # 包含部分关键词
        score2 = reranker._calculate_keyword_match(query_words, "test content")
        assert score2 == 0.5
        
        # 不包含关键词
        score3 = reranker._calculate_keyword_match(query_words, "other content")
        assert score3 == 0.0
    
    def test_calculate_length_score(self, reranker):
        """长度惩罚分数计算"""
        # 短内容
        score1 = reranker._calculate_length_score("short")
        assert score1 == 0.3
        
        # 中等内容
        score2 = reranker._calculate_length_score("word " * 50)
        assert score2 == 0.7
        
        # 长内容
        score3 = reranker._calculate_length_score("word " * 500)
        assert score3 == 1.0
        
        # 过长内容
        score4 = reranker._calculate_length_score("word " * 1000)
        assert score4 < 1.0


class TestMMMReranker:
    """Test cases for MMMReranker"""
    
    @pytest.fixture
    def mmr_reranker(self):
        """Create MMMReranker instance"""
        return MMMReranker(
            lambda_param=0.5,
            similarity_threshold=0.85
        )
    
    @pytest.fixture
    def mock_results(self):
        """Create mock retrieval results"""
        results = []
        for i in range(5):
            result = MagicMock()
            result.id = f"doc{i}"
            result.similarity = 0.95 - i * 0.05
            result.content = f"Document {i} with unique content"
            result.file_path = f"file{i}.py"
            results.append(result)
        return results
    
    def test_rerank_basic(self, mmr_reranker, mock_results):
        """基础 MMR 重排序"""
        results = mmr_reranker.rerank(
            "test query",
            mock_results,
            top_k=3
        )
        
        assert len(results) == 3
        # 结果应该多样化
        file_paths = [r.file_path for r in results]
        assert len(set(file_paths)) == len(file_paths)
    
    def test_rerank_same_file(self, mmr_reranker):
        """同一文件的多块重排序"""
        results = []
        for i in range(5):
            result = MagicMock()
            result.id = f"chunk{i}"
            result.similarity = 0.95 - i * 0.05
            result.content = f"Content {i}"
            result.file_path = "same_file.py"  # 同一文件
            results.append(result)
        
        reranked = mmr_reranker.rerank("query", results, top_k=3)
        
        # 同一文件应该有惩罚
        assert len(reranked) == 3
    
    def test_rerank_empty(self, mmr_reranker):
        """空结果"""
        results = mmr_reranker.rerank("query", [], top_k=5)
        assert results == []
    
    def test_rerank_less_than_top_k(self, mmr_reranker, mock_results):
        """结果少于 top_k"""
        results = mmr_reranker.rerank("query", mock_results, top_k=10)
        assert len(results) == 5  # 原始结果数量
    
    def test_content_similarity(self, mmr_reranker):
        """内容相似度计算"""
        # 相似内容
        sim1 = mmr_reranker._content_similarity(
            "async def main function await",
            "async function with await"
        )
        assert sim1 > 0.3
        
        # 不相似内容
        sim2 = mmr_reranker._content_similarity(
            "database connection query",
            "user interface button click"
        )
        assert sim2 < 0.3


class TestRerankingPipeline:
    """Test cases for RerankingPipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline with fallback reranker"""
        fallback = FallbackReranker()
        return RerankingPipeline(
            primary_reranker=None,
            fallback_reranker=fallback
        )
    
    @pytest.fixture
    def mock_results(self):
        """Create mock results"""
        results = []
        for i in range(5):
            result = MagicMock()
            result.id = f"doc{i}"
            result.similarity = 0.9 - i * 0.1
            result.content = f"Document {i} content"
            result.file_path = f"file{i}.py"
            results.append(result)
        return results
    
    @pytest.mark.asyncio
    async def test_rerank_with_fallback(self, pipeline, mock_results):
        """使用 fallback 重排序"""
        results = await pipeline.rerank("test query", mock_results, top_k=3)
        
        assert len(results) == 3
        assert all(isinstance(r, RerankResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_rerank_empty(self, pipeline):
        """空结果"""
        results = await pipeline.rerank("query", [], top_k=5)
        assert results == []
    
    def test_create_metadata(self, pipeline):
        """创建元数据"""
        results = [
            RerankResult(
                id="doc1",
                original_index=0,
                cross_score=0.8,
                fused_score=0.7,
                content="Test",
                file_path="test.py"
            ),
            RerankResult(
                id="doc2",
                original_index=1,
                cross_score=0.6,
                fused_score=0.5,
                content="Test 2",
                file_path="test2.py"
            )
        ]
        
        metadata = pipeline.create_metadata(results)
        
        assert metadata['reranked_count'] == 2
        assert metadata['avg_cross_score'] == 0.7
        assert metadata['avg_fused_score'] == 0.6
        assert metadata['top_result']['id'] == 'doc1'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
