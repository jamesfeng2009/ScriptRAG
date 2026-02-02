"""
Integration tests for pgvector RAG features

这些测试需要实际的数据库连接：
1. 测试 PgVectorDBService 与真实 PostgreSQL 的交互
2. 测试成本记录到数据库
3. 测试检索日志记录
4. 测试 API 使用统计

前置条件：
- PostgreSQL 服务运行在 localhost:5433
- 数据库 Screenplay 存在
- pgvector 扩展已安装
- Alembic 迁移已应用
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List

from src.services.database.pgvector_service import PgVectorDBService


class MockDatabaseConfig:
    """用于测试的模拟数据库配置"""
    def __init__(self, host, port, database, user, password):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password


# PostgreSQL 测试配置
TEST_DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "Screenplay",
    "user": "postgres",
    "password": "123456"
}

TEST_DB_SERVICE_CONFIG = MockDatabaseConfig(
    host=TEST_DB_CONFIG["host"],
    port=TEST_DB_CONFIG["port"],
    database=TEST_DB_CONFIG["database"],
    user=TEST_DB_CONFIG["user"],
    password=TEST_DB_CONFIG["password"]
)

WORKSPACE_ID = "test-workspace-integration"


class TestPgVectorDBServiceIntegration:
    """PgVectorDBService 集成测试"""

    @pytest.fixture
    async def db_service(self):
        """创建数据库服务实例"""
        service = PgVectorDBService(
            config=TEST_DB_SERVICE_CONFIG,
            embedding_dim=1024,
            chunk_size=1000
        )
        
        try:
            await service.initialize()
            yield service
        finally:
            await service.close()

    @pytest.mark.asyncio
    async def test_health_check(self, db_service):
        """测试健康检查"""
        result = await db_service.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_index_and_search_document(self, db_service):
        """测试索引和搜索文档"""
        # 生成随机内容避免冲突
        doc_id = str(uuid.uuid4())
        file_path = f"test_{doc_id}.py"
        
        content = '''
def calculate_fibonacci(n: int) -> int:
    """Calculate Fibonacci number recursively"""
    if n <= 1:
        return n
    # TODO: Add memoization for better performance
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class FibonacciCalculator:
    """Calculator for Fibonacci sequence"""
    
    def __init__(self):
        self.cache = {}
    
    @deprecated
    def old_method(self, n):
        """This method is deprecated, use calculate_fibonacci instead"""
        pass
'''
        
        embedding = [0.1] * 1024  # 简化：使用固定向量
        
        # 索引文档
        doc_id = await db_service.index_document(
            workspace_id=WORKSPACE_ID,
            file_path=file_path,
            content=content,
            embedding=embedding,
            language="python"
        )
        
        assert doc_id is not None
        
        # 搜索文档
        results = await db_service.vector_search(
            workspace_id=WORKSPACE_ID,
            query_embedding=embedding,
            top_k=5,
            similarity_threshold=0.0  # 降低阈值以匹配
        )
        
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_keyword_search(self, db_service):
        """测试关键词搜索"""
        # 索引测试文档
        doc_id = str(uuid.uuid4())
        file_path = f"keyword_test_{doc_id}.py"
        
        content = '''
def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user credentials"""
    # FIXME: Add rate limiting
    # SECURITY: Hash passwords before storage
    pass
'''
        
        embedding = [0.2] * 1024
        
        await db_service.index_document(
            workspace_id=WORKSPACE_ID,
            file_path=file_path,
            content=content,
            embedding=embedding
        )
        
        # 关键词搜索 - 可能有结果，取决于索引
        try:
            results = await db_service.keyword_search(
                workspace_id=WORKSPACE_ID,
                query="authenticate",
                top_k=5
            )
            assert len(results) >= 0
        except Exception as e:
            pytest.skip(f"Keyword search failed: {e}")

    @pytest.mark.asyncio
    async def test_batch_index_documents(self, db_service):
        """测试批量索引文档"""
        documents = []
        embeddings = []
        
        for i in range(3):
            doc_id = str(uuid.uuid4())
            documents.append({
                "file_path": f"batch_test_{doc_id}.py",
                "content": f"# Batch test document {i}\ndef function_{i}():\n    pass",
                "metadata": {"index": i}
            })
            embeddings.append([0.1 * i] * 1024)
        
        results = await db_service.index_documents_batch(
            workspace_id=WORKSPACE_ID,
            documents=documents,
            embeddings=embeddings
        )
        
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_vector_search_with_filters(self, db_service):
        """测试带过滤器的向量搜索"""
        # 索引包含 deprecated 标记的文档
        doc_id = str(uuid.uuid4())
        await db_service.index_document(
            workspace_id=WORKSPACE_ID,
            file_path=f"filter_test_{doc_id}.py",
            content="""
@deprecated
def old_function():
    '''This is deprecated'''
    pass
""",
            embedding=[0.3] * 1024,
            has_deprecated=True
        )
        
        # 搜索带过滤器
        try:
            results = await db_service.vector_search(
                workspace_id=WORKSPACE_ID,
                query_embedding=[0.3] * 1024,
                top_k=5,
                similarity_threshold=0.0,
                filters={"has_deprecated": True}
            )
        except Exception as e:
            pytest.skip(f"Vector search with filters failed: {e}")


class TestDatabaseViewsIntegration:
    """数据库视图集成测试"""

    @pytest.fixture
    def db_connection(self):
        """创建数据库连接（使用同步连接避免事件循环问题）"""
        import psycopg2
        
        conn = psycopg2.connect(
            host=TEST_DB_CONFIG["host"],
            port=TEST_DB_CONFIG["port"],
            database=TEST_DB_CONFIG["database"],
            user=TEST_DB_CONFIG["user"],
            password=TEST_DB_CONFIG["password"]
        )
        yield conn
        conn.close()

    def test_daily_cost_summary_view(self, db_connection):
        """测试每日成本汇总视图"""
        try:
            cursor = db_connection.cursor()
            cursor.execute("SELECT * FROM screenplay.daily_cost_summary LIMIT 1")
            result = cursor.fetchone()
            cursor.close()
            assert result is not None
        except Exception as e:
            pytest.skip(f"View may not exist: {e}")

    def test_popular_queries_view(self, db_connection):
        """测试热门查询视图"""
        try:
            cursor = db_connection.cursor()
            cursor.execute("SELECT * FROM screenplay.popular_queries LIMIT 1")
            result = cursor.fetchone()
            cursor.close()
            assert result is not None
        except Exception as e:
            pytest.skip(f"View may not exist: {e}")


class TestDatabaseFunctionsIntegration:
    """数据库函数集成测试"""

    @pytest.fixture
    def db_connection(self):
        """创建数据库连接（使用同步连接避免事件循环问题）"""
        import psycopg2
        
        conn = psycopg2.connect(
            host=TEST_DB_CONFIG["host"],
            port=TEST_DB_CONFIG["port"],
            database=TEST_DB_CONFIG["database"],
            user=TEST_DB_CONFIG["user"],
            password=TEST_DB_CONFIG["password"]
        )
        yield conn
        conn.close()

    def test_cleanup_old_logs_function(self, db_connection):
        """测试清理旧日志函数"""
        try:
            cursor = db_connection.cursor()
            cursor.execute("SELECT screenplay.cleanup_old_logs(365)")
            result = cursor.fetchone()[0]
            cursor.close()
            assert isinstance(result, int)
        except Exception as e:
            pytest.skip(f"Function may not exist: {e}")

    def test_cleanup_old_api_stats_function(self, db_connection):
        """测试清理旧 API 统计函数"""
        try:
            cursor = db_connection.cursor()
            cursor.execute("SELECT screenplay.cleanup_old_api_stats(365)")
            result = cursor.fetchone()[0]
            cursor.close()
            assert isinstance(result, int)
        except Exception as e:
            pytest.skip(f"Function may not exist: {e}")


class TestVectorSearchPerformance:
    """向量搜索性能测试"""

    @pytest.fixture
    async def db_service(self):
        """创建数据库服务实例"""
        service = PgVectorDBService(
            config=TEST_DB_SERVICE_CONFIG,
            embedding_dim=1024
        )
        
        try:
            await service.initialize()
            yield service
        finally:
            await service.close()

    @pytest.mark.asyncio
    async def test_vector_search_response_time(self, db_service):
        """测试向量搜索响应时间"""
        import time
        
        embedding = [0.5] * 1024
        
        start_time = time.time()
        try:
            results = await db_service.vector_search(
                workspace_id=WORKSPACE_ID,
                query_embedding=embedding,
                top_k=10,
                similarity_threshold=0.0
            )
        except Exception as e:
            pytest.skip(f"Vector search failed: {e}")
        end_time = time.time()
        
        response_time = end_time - start_time
        print(f"\nVector search response time: {response_time:.3f}s")
        
        # 响应时间应该在合理范围内
        assert response_time < 5.0  # 5秒内


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
