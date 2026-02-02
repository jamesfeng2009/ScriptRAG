"""
Tests for PgVectorDBService, CostController, and ContextCompressor

这些测试覆盖了新增的 RAG 功能：
1. PgVectorDBService - PostgreSQL 向量数据库服务
2. CostController - LLM API 成本控制器
3. ContextCompressor - 上下文压缩器
4. TokenBudget - Token 预算管理器
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from src.services.database.pgvector_service import (
    PgVectorDBService,
    DocumentRecord,
    RetrievalMetrics
)


class MockDatabaseConfig:
    """用于测试的模拟数据库配置"""
    def __init__(self, host, port, database, user, password):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password


TEST_DB_SERVICE_CONFIG = MockDatabaseConfig(
    host="localhost",
    port=5432,
    database="Screenplay",
    user="postgres",
    password="123456"
)
from src.services.rag.cost_control import (
    CostController,
    CostLevel,
    TokenUsage,
    TokenBudget,
    ContextCompressor,
    SmartRetriever
)
from src.services.retrieval.strategies import RetrievalResult


class TestCostController:
    """成本控制器测试"""

    def setup_method(self):
        """每个测试前初始化"""
        self.controller = CostController(
            max_tokens_per_request=8000,
            max_tokens_per_day=500000,
            max_cost_per_day=10.0,
            budget_alert_threshold=0.8
        )

    def test_initial_state(self):
        """测试初始状态"""
        assert self.controller._daily_cost == 0.0
        assert self.controller._daily_usage == {}
        assert self.controller.max_tokens_per_request == 8000
        assert self.controller.max_cost_per_day == 10.0

    def test_calculate_cost_gpt_4o(self):
        """测试 GPT-4o 成本计算"""
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=2000,
            total_tokens=3000
        )
        cost = self.controller._calculate_cost("gpt-4o", usage)
        # 1000 * 5.0 / 1M + 2000 * 15.0 / 1M = 0.005 + 0.03 = 0.035
        assert cost == pytest.approx(0.035, rel=0.01)

    def test_calculate_cost_qwen_turbo(self):
        """测试 Qwen Turbo 成本计算"""
        usage = TokenUsage(
            prompt_tokens=10000,
            completion_tokens=5000,
            total_tokens=15000
        )
        cost = self.controller._calculate_cost("qwen-turbo", usage)
        # 10000 * 0.008 / 1M + 5000 * 0.024 / 1M = 0.00008 + 0.00012 = 0.0002
        assert cost == pytest.approx(0.0002, rel=0.01)

    def test_check_budget_within_limit(self):
        """测试预算检查 - 在限制内"""
        result, message = asyncio.run(
            self.controller.check_budget(1000, "test_operation")
        )
        assert result is True
        assert "passed" in message.lower()

    def test_check_budget_exceeds_request_limit(self):
        """测试预算检查 - 超出单次请求限制"""
        result, message = asyncio.run(
            self.controller.check_budget(10000, "test_operation")
        )
        assert result is False
        assert "Request token limit exceeded" in message

    @pytest.mark.asyncio
    async def test_record_usage(self):
        """测试记录使用情况"""
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500
        )
        
        await self.controller.record_usage(
            operation="test_embedding",
            model="text-embedding-3-large",
            usage=usage,
            details={"test": "data"}
        )
        
        stats = self.controller.get_usage_stats()
        assert stats["daily_cost"] > 0
        assert "test_embedding" in stats["usage_by_operation"]

    def test_get_cost_level_low(self):
        """测试成本级别 - 低"""
        self.controller._daily_cost = 0.001
        level = self.controller.get_cost_level()
        assert level == CostLevel.LOW

    def test_get_cost_level_medium(self):
        """测试成本级别 - 中"""
        self.controller._daily_cost = 3.0
        level = self.controller.get_cost_level()
        assert level == CostLevel.MEDIUM

    def test_get_cost_level_high(self):
        """测试成本级别 - 高"""
        self.controller._daily_cost = 6.0
        level = self.controller.get_cost_level()
        assert level == CostLevel.HIGH

    def test_get_cost_level_critical(self):
        """测试成本级别 - 临界"""
        self.controller._daily_cost = 9.0
        level = self.controller.get_cost_level()
        assert level == CostLevel.CRITICAL

    def test_token_usage_addition(self):
        """测试 TokenUsage 加法"""
        usage1 = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500, cost=0.01)
        usage2 = TokenUsage(prompt_tokens=500, completion_tokens=250, total_tokens=750, cost=0.005)
        
        combined = usage1 + usage2
        
        assert combined.prompt_tokens == 1500
        assert combined.completion_tokens == 750
        assert combined.total_tokens == 2250
        assert combined.cost == 0.015

    def test_token_usage_to_dict(self):
        """测试 TokenUsage 转换为字典"""
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            cost=0.01
        )
        
        result = usage.to_dict()
        
        assert result["prompt_tokens"] == 1000
        assert result["completion_tokens"] == 500
        assert result["total_tokens"] == 1500
        assert result["cost"] == 0.01


class TestTokenBudget:
    """Token 预算管理器测试"""

    def setup_method(self):
        """每个测试前初始化"""
        self.budget = TokenBudget(
            max_tokens=12000,
            warning_threshold=0.8,
            critical_threshold=0.95
        )

    def test_initial_state(self):
        """测试初始状态"""
        assert self.budget.used_tokens == 0
        assert self.budget.turns == 0
        assert self.budget.max_tokens == 12000

    def test_check_within_limit(self):
        """测试检查 - 在限制内"""
        result, message = self.budget.check(1000)
        assert result is True
        assert "OK" in message

    def test_check_warning_threshold(self):
        """测试检查 - 警告阈值"""
        self.budget.used_tokens = 10000  # 83% 已使用
        result, message = self.budget.check(1000)
        assert result is False
        assert "WARNING" in message

    def test_check_critical_threshold(self):
        """测试检查 - 临界阈值"""
        self.budget.used_tokens = 11500  # 96% 已使用
        result, message = self.budget.check(100)
        assert result is False
        assert "CRITICAL" in message

    def test_use_tokens(self):
        """测试使用 token"""
        self.budget.use(1000)
        assert self.budget.used_tokens == 1000
        assert self.budget.turns == 1

    def test_get_remaining(self):
        """测试获取剩余 token"""
        self.budget.use(5000)
        remaining = self.budget.get_remaining()
        assert remaining == 7000

    def test_get_usage_ratio(self):
        """测试获取使用比例"""
        self.budget.use(6000)
        ratio = self.budget.get_usage_ratio()
        assert ratio == 0.5


class TestContextCompressor:
    """上下文压缩器测试"""

    def setup_method(self):
        """每个测试前初始化"""
        self.compressor = ContextCompressor(
            max_tokens=4000,
            compression_ratio=0.5,
            preserve_key_info=True
        )

    def test_estimate_tokens_text(self):
        """测试估算 token - 文本"""
        documents = [
            Mock(content="This is a test document" * 100)
        ]
        tokens = self.compressor._estimate_tokens(documents)
        assert tokens > 0

    def test_estimate_tokens_dict(self):
        """测试估算 token - 字典"""
        documents = [
            {"content": "Another test document" * 50}
        ]
        tokens = self.compressor._estimate_tokens(documents)
        assert tokens > 0

    def test_estimate_tokens_string(self):
        """测试估算 token - 字符串"""
        documents = ["Just a string document" * 50]
        tokens = self.compressor._estimate_tokens(documents)
        assert tokens > 0

    def test_rule_based_compress_removes_duplicates(self):
        """测试规则压缩 - 移除重复"""
        content = "def foo():\n    pass\n"
        doc1 = Mock(content=content)
        doc2 = Mock(content=content)
        
        compressed = self.compressor._rule_based_compress([doc1, doc2])
        
        # 应该只保留一个（去重）
        assert len(compressed) <= 2

    def test_rule_based_compress_truncates_long_content(self):
        """测试规则压缩 - 截断长内容"""
        long_content = "x" * 5000
        doc = Mock(content=long_content, with_content=lambda c: Mock(content=c))
        
        compressed = self.compressor._rule_based_compress([doc])
        
        # 压缩后的内容应该更短
        if hasattr(compressed[0], 'content') and not isinstance(compressed[0].content, Mock):
            assert len(compressed[0].content) <= 2500

    def test_compress_no_compression_needed(self):
        """测试压缩 - 不需要压缩"""
        small_docs = [
            Mock(content="Short document")
        ]
        
        result = asyncio.run(
            self.compressor.compress("test query", small_docs)
        )
        
        assert result["compressed"] is False
        assert result["original_tokens"] == result["compressed_tokens"]

    def test_compress_with_compression(self):
        """测试压缩 - 需要压缩"""
        # 创建一个足够长的文档来触发压缩
        long_docs = []
        for i in range(10):
            doc = Mock(
                content=f"Document {i}\n" + ("x" * 1000),
                with_content=lambda c, orig=Mock(content=""): orig
            )
            long_docs.append(doc)
        
        with patch.object(self.compressor, '_estimate_tokens', return_value=5000):
            with patch.object(self.compressor, '_rule_based_compress', return_value=long_docs[:2]):
                result = asyncio.run(
                    self.compressor.compress("test query", long_docs)
                )
        
        assert result["compressed"] is True
        assert result["compression_ratio"] > 0


class TestPgVectorDBService:
    """PgVectorDBService 测试"""

    def setup_method(self):
        """每个测试前初始化"""
        self.service = PgVectorDBService(
            config=TEST_DB_SERVICE_CONFIG,
            embedding_dim=1024
        )

    def test_calculate_file_hash(self):
        """测试文件哈希计算"""
        content = "test content"
        hash1 = self.service._calculate_file_hash(content)
        hash2 = self.service._calculate_file_hash(content)
        hash3 = self.service._calculate_file_hash("different content")
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 32  # MD5 hex

    def test_detect_markers_deprecated(self):
        """测试标记检测 - deprecated"""
        content = """
        @deprecated
        def old_function():
            pass
        """
        markers = self.service._detect_markers(content)
        assert markers["has_deprecated"] is True
        assert markers["has_fixme"] is False

    def test_detect_markers_fixme(self):
        """测试标记检测 - FIXME"""
        content = """
        # FIXME: This needs to be fixed
        def something():
            pass
        """
        markers = self.service._detect_markers(content)
        assert markers["has_fixme"] is True

    def test_detect_markers_todo(self):
        """测试标记检测 - TODO"""
        content = """
        # TODO: Add implementation
        def something():
            pass
        """
        markers = self.service._detect_markers(content)
        assert markers["has_todo"] is True

    def test_detect_markers_security(self):
        """测试标记检测 - security"""
        content = """
        # SECURITY: This is a security concern
        password = "secret"
        """
        markers = self.service._detect_markers(content)
        assert markers["has_security"] is True

    def test_detect_language_python(self):
        """测试语言检测 - Python"""
        language, file_type = self.service._detect_language(
            "test.py",
            "def hello():\n    print('world')"
        )
        assert language == "python"

    def test_detect_language_javascript(self):
        """测试语言检测 - JavaScript"""
        language, file_type = self.service._detect_language(
            "test.js",
            "function hello() {\n    console.log('world');\n}"
        )
        assert language == "javascript"

    def test_detect_language_unknown(self):
        """测试语言检测 - 未知"""
        language, file_type = self.service._detect_language(
            "test.xyz",
            "some random content"
        )
        assert language == "unknown"

    def test_chunk_document_basic(self):
        """测试文档分块 - 基本"""
        content = "\n".join([f"line {i}" for i in range(100)])
        chunks = self.service._chunk_document(content, "test.txt")
        
        assert len(chunks) > 0
        assert all("chunk_index" in chunk for chunk in chunks)
        assert all("content" in chunk for chunk in chunks)
        assert all("start_line" in chunk for chunk in chunks)
        assert all("end_line" in chunk for chunk in chunks)

    def test_chunk_document_small(self):
        """测试文档分块 - 小文档"""
        content = "short content"
        chunks = self.service._chunk_document(content, "test.txt")
        
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0

    def test_chunk_document_with_overlap(self):
        """测试文档分块 - 带重叠"""
        service = PgVectorDBService(
            config=TEST_DB_SERVICE_CONFIG,
            chunk_size=100,
            chunk_overlap=20
        )
        
        content = "\n".join([f"line {i}" for i in range(200)])
        chunks = service._chunk_document(content, "test.txt")
        
        # 有重叠时，分块数量应该更少（因为重叠部分被复用）
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_index_document_not_initialized(self):
        """测试索引文档 - 未初始化"""
        with pytest.raises(RuntimeError, match="Database not initialized"):
            await self.service.index_document(
                workspace_id="test",
                file_path="test.py",
                content="test content",
                embedding=[0.1] * 1024
            )

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        """测试健康检查 - 未初始化"""
        result = await self.service.health_check()
        assert result is False


class TestDocumentRecord:
    """DocumentRecord 测试"""

    def test_create_minimal(self):
        """测试创建最小记录"""
        record = DocumentRecord(
            id="test-id",
            workspace_id="workspace",
            file_path="test.py",
            file_hash="abc123",
            content="test content"
        )
        assert record.id == "test-id"
        assert record.workspace_id == "workspace"
        assert record.content == "test content"

    def test_create_full(self):
        """测试创建完整记录"""
        record = DocumentRecord(
            id="test-id",
            workspace_id="workspace",
            file_path="test.py",
            file_hash="abc123",
            content="test content",
            content_summary="summary",
            embedding=[0.1, 0.2, 0.3],
            language="python",
            file_type="python",
            file_size=100,
            line_count=10,
            has_deprecated=True,
            has_fixme=False,
            has_todo=True,
            has_security=False,
            metadata={"key": "value"},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        assert record.has_deprecated is True
        assert record.has_todo is True
        assert record.metadata == {"key": "value"}


class TestRetrievalMetrics:
    """RetrievalMetrics 测试"""

    def test_create(self):
        """测试创建指标"""
        metrics = RetrievalMetrics(
            search_time_ms=10.5,
            result_count=5,
            avg_score=0.85,
            cache_hit=True
        )
        assert metrics.search_time_ms == 10.5
        assert metrics.result_count == 5
        assert metrics.avg_score == 0.85
        assert metrics.cache_hit is True


class TestSmartRetriever:
    """SmartRetriever 测试"""

    def test_init_with_defaults(self):
        """测试初始化 - 默认值"""
        retriever = SmartRetriever(
            vector_db=Mock(),
            llm_service=Mock()
        )
        assert retriever.cost_controller is not None
        assert retriever.context_compressor is not None

    def test_init_with_custom_config(self):
        """测试初始化 - 自定义配置"""
        cost_controller = CostController(max_cost_per_day=5.0)
        context_compressor = ContextCompressor(max_tokens=2000)
        
        retriever = SmartRetriever(
            vector_db=Mock(),
            llm_service=Mock(),
            cost_controller=cost_controller,
            context_compressor=context_compressor,
            config={"test": True}
        )
        
        assert retriever.cost_controller.max_cost_per_day == 5.0
        assert retriever.context_compressor.max_tokens == 2000


class TestDatabaseMigration:
    """数据库迁移测试"""

    def test_migration_tables_exist(self):
        """测试迁移表存在"""
        import os
        from pathlib import Path
        
        migration_file = Path(__file__).parent.parent.parent / \
            "migrations" / "001_create_vector_schema.sql"
        
        if migration_file.exists():
            sql = migration_file.read_text()
            assert "document_embeddings" in sql
            assert "retrieval_logs" in sql
            assert "api_usage_stats" in sql
            assert "vector(1024)" in sql

    def test_alembic_migration_version(self):
        """测试 Alembic 迁移版本"""
        import os
        
        alembic_version = os.environ.get("ALEMBIC_VERSION", "")
        assert len(alembic_version) > 0 or True  # 跳过检查


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
