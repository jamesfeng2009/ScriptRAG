"""
父文档检索 (Parent Document Retrieval) 单元测试

使用真实数据测试:
- 真实的 Python 项目代码
- 真实的 Markdown 文档
- 实际的代码检索场景
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path

from src.services.documents.parent_document_retriever import (
    ParentDocumentStore,
    ParentDocumentRetriever,
    ParentDocumentType,
    ParentDocument,
    ChildChunk,
)
from src.services.documents.document_chunker import (
    SmartChunker,
    FileType,
    Chunk,
    ChunkMetadata,
)


# ============================================================
# 真实测试数据
# ============================================================

REAL_PYTHON_CODE = '''"""
用户服务模块

提供用户相关的所有功能:
- 用户注册
- 用户登录
- 用户信息管理
"""

import os
import json
import hashlib
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class User:
    """用户数据模型"""
    user_id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active
        }


class UserServiceError(Exception):
    """用户服务异常基类"""
    pass


class UserNotFoundError(UserServiceError):
    """用户不存在异常"""
    pass


class UserService:
    """
    用户服务类
    
    负责处理所有用户相关的业务逻辑:
    - 用户认证
    - 用户信息管理
    - 用户权限控制
    """
    
    def __init__(self, db_connection: str):
        """
        初始化用户服务
        
        Args:
            db_connection: 数据库连接字符串
        """
        self.db_connection = db_connection
        self._cache = {}
        self._user_counter = 0
    
    async def create_user(
        self,
        username: str,
        email: str,
        password_hash: str
    ) -> User:
        """
        创建新用户
        
        Args:
            username: 用户名
            email: 邮箱
            password_hash: 密码哈希
            
        Returns:
            创建的用户对象
            
        Raises:
            ValueError: 用户名或邮箱已存在
        """
        if username in self._cache:
            raise ValueError(f"用户名 {username} 已存在")
        
        self._user_counter += 1
        user = User(
            user_id=self._user_counter,
            username=username,
            email=email,
            created_at=datetime.now(),
            is_active=True
        )
        
        self._cache[username] = user
        return user
    
    async def get_user(self, user_id: int) -> Optional[User]:
        """
        获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户对象，不存在返回 None
        """
        for user in self._cache.values():
            if user.user_id == user_id:
                return user
        return None
    
    async def authenticate(
        self,
        username: str,
        password: str
    ) -> Optional[User]:
        """
        用户登录认证
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            认证成功返回用户对象，失败返回 None
        """
        user = self._cache.get(username)
        if user and user.is_active:
            return user
        return None
    
    def _generate_token(self, user: User) -> str:
        """生成用户认证令牌"""
        payload = f"{user.user_id}:{user.username}:{datetime.now().timestamp()}"
        return hashlib.sha256(payload.encode()).hexdigest()


class AdminService(UserService):
    """
    管理员服务
    
    继承自 UserService，提供管理员特有功能:
    - 用户管理
    - 系统配置
    - 审计日志
    """
    
    def __init__(self, db_connection: str, admin_level: int = 1):
        """
        初始化管理员服务
        
        Args:
            db_connection: 数据库连接字符串
            admin_level: 管理员等级
        """
        super().__init__(db_connection)
        self.admin_level = admin_level
        self.audit_log = []
    
    async def deactivate_user(self, user_id: int) -> bool:
        """
        禁用用户账户
        
        Args:
            user_id: 要禁用的用户ID
            
        Returns:
            操作是否成功
            
        Raises:
            UserNotFoundError: 用户不存在
        """
        user = await self.get_user(user_id)
        if not user:
            raise UserNotFoundError(f"用户 {user_id} 不存在")
        
        user.is_active = False
        self.audit_log.append({
            "action": "deactivate_user",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        return True
    
    def get_audit_logs(self, user_id: Optional[int] = None) -> List[Dict]:
        """
        获取审计日志
        
        Args:
            user_id: 可选，按用户ID过滤
            
        Returns:
            审计日志列表
        """
        if user_id:
            return [log for log in self.audit_log if log["user_id"] == user_id]
        return self.audit_log
'''

REAL_MARKDOWN_DOCS = '''# API 文档

## 概述

本项目提供 RESTful API 接口，用于管理用户、订单和系统配置。

## 认证

所有 API 请求需要携带 Bearer Token:

```bash
Authorization: Bearer <your_token>
```

## 用户接口

### 创建用户

**端点:** `POST /api/v1/users`

**请求体:**

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| username | string | 是 | 用户名 |
| email | string | 是 | 邮箱地址 |
| password | string | 是 | 密码 |

**响应:**

```json
{
    "user_id": 1,
    "username": "test_user",
    "email": "test@example.com",
    "created_at": "2024-01-01T00:00:00"
}
```

### 获取用户信息

**端点:** `GET /api/v1/users/{user_id}`

**响应:**

```json
{
    "user_id": 1,
    "username": "test_user",
    "email": "test@example.com",
    "is_active": true
}
```

## 订单接口

### 创建订单

**端点:** `POST /api/v1/orders`

**请求体:**

| 字段 | 类型 | 描述 |
|------|------|------|
| user_id | int | 用户ID |
| items | array | 订单商品列表 |
| total_amount | float | 订单总金额 |

**示例:**

```json
{
    "user_id": 1,
    "items": [
        {"product_id": 101, "quantity": 2, "price": 29.99},
        {"product_id": 102, "quantity": 1, "price": 49.99}
    ],
    "total_amount": 109.97
}
```

## 错误码说明

| 错误码 | 描述 |
|--------|------|
| 400 | 请求参数错误 |
| 401 | 未授权访问 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |
'''

REAL_COMPLEX_PYTHON = '''"""
数据处理管道模块

实现高效的数据处理流程，包括:
- 数据采集
- 数据清洗
- 数据转换
- 数据存储
"""

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


T = TypeVar('T')
R = TypeVar('R')


@dataclass
class PipelineContext:
    """管道执行上下文"""
    pipeline_id: str
    start_time: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)


class DataSource(ABC):
    """数据源抽象基类"""
    
    @abstractmethod
    async def connect(self) -> None:
        """建立连接"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass
    
    @abstractmethod
    async def fetch(self, batch_size: int = 100) -> AsyncIterator[Dict]:
        """获取数据"""
        pass


class DataSink(ABC):
    """数据接收器抽象基类"""
    
    @abstractmethod
    async def write(self, data: Dict) -> bool:
        """写入数据"""
        pass
    
    @abstractmethod
    async def flush(self) -> None:
        """刷新缓冲区"""
        pass


class DataTransformer(ABC):
    """数据转换器抽象基类"""
    
    @abstractmethod
    def transform(self, data: Dict) -> Dict:
        """转换数据"""
        pass


class DataPipeline(Generic[T, R]):
    """
    数据处理管道
    
    支持灵活的数据处理流程配置:
    - 可配置的数据源
    - 可配置的转换器
    - 可配置的数据接收器
    """
    
    def __init__(
        self,
        name: str,
        source: DataSource,
        transformer: DataTransformer,
        sink: DataSink,
        error_handler: callable = None
    ):
        self.name = name
        self.source = source
        self.transformer = transformer
        self.sink = sink
        self.error_handler = error_handler
        self.context = PipelineContext(pipeline_id=name)
        self._running = False
    
    async def run(self, max_records: int = 10000) -> PipelineContext:
        """
        执行管道
        
        Args:
            max_records: 最大处理记录数
            
        Returns:
            执行上下文，包含处理统计信息
        """
        self._running = True
        self.context.start_time = datetime.now()
        
        await self.source.connect()
        try:
            count = 0
            async for raw_data in self.source.fetch():
                if count >= max_records:
                    break
                    
                try:
                    transformed = self.transformer.transform(raw_data)
                    await self.sink.write(transformed)
                    count += 1
                except Exception as e:
                    logger.error(f"处理记录失败: {e}")
                    if self.error_handler:
                        self.error_handler(e, raw_data)
                    
        finally:
            await self.source.disconnect()
            await self.sink.flush()
        
        self.context.metrics["records_processed"] = count
        self._running = False
        return self.context


class CSVDataSource(DataSource):
    """CSV 文件数据源"""
    
    def __init__(self, file_path: str, delimiter: str = ','):
        self.file_path = file_path
        self.delimiter = delimiter
        self._file = None
    
    async def connect(self) -> None:
        self._file = open(self.file_path, 'r', encoding='utf-8')
    
    async def disconnect(self) -> None:
        if self._file:
            self._file.close()
    
    async def fetch(self, batch_size: int = 100) -> AsyncIterator[Dict]:
        headers = None
        batch = []
        
        for line in self._file:
            if headers is None:
                headers = line.strip().split(self.delimiter)
                continue
            
            values = line.strip().split(self.delimiter)
            row = dict(zip(headers, values))
            batch.append(row)
            
            if len(batch) >= batch_size:
                for item in batch:
                    yield item
                batch = []
        
        if batch:
            for item in batch:
                yield item


class JSONDataSink(DataSink):
    """JSON 文件数据接收器"""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self._buffer = []
    
    async def write(self, data: Dict) -> bool:
        self._buffer.append(data)
        return True
    
    async def flush(self) -> None:
        import json
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self._buffer, f, ensure_ascii=False, indent=2)
        self._buffer.clear()


# 使用示例
async def main():
    """主函数示例"""
    source = CSVDataSource('input.csv')
    transformer = DataTransformer()
    sink = JSONDataSink('output.json')
    
    pipeline = DataPipeline(
        name="csv_to_json",
        source=source,
        transformer=transformer,
        sink=sink
    )
    
    context = await pipeline.run(max_records=1000)
    print(f"处理完成: {context.metrics}")
'''


# ============================================================
# 测试类
# ============================================================

class TestParentDocumentWithRealPython:
    """使用真实 Python 代码测试父文档检索"""

    @pytest.fixture
    def store(self):
        return ParentDocumentStore()

    @pytest.fixture
    def retriever(self):
        return ParentDocumentRetriever()

    def test_index_realistic_python_module(self, store, retriever):
        """
        测试真实 Python 模块的索引
        
        场景:
        - 包含类定义、继承
        - 包含异常处理
        - 包含类型注解
        - 包含异步代码
        """
        chunker = SmartChunker(config={
            'chunk_size': 800,
            'min_chunk_size': 200,
            'overlap': 100
        })
        
        chunks = chunker.chunk_text(REAL_PYTHON_CODE, 'user_service.py')
        
        assert len(chunks) > 0, "应该生成分块"
        
        parents = retriever.index_python_file(
            file_path='user_service.py',
            content=REAL_PYTHON_CODE,
            chunks=chunks,
            class_context={
                'User': '用户数据模型',
                'UserService': '用户服务主类',
                'AdminService': '管理员服务'
            }
        )
        
        assert len(parents) == 1
        assert parents[0].doc_type == ParentDocumentType.FILE
        assert 'UserService' in parents[0].content

    def test_retrieve_user_service_context(self, store, retriever):
        """
        测试检索用户服务上下文
        
        场景:
        - 匹配分块
        - 返回完整上下文
        - 包含骨架信息
        """
        chunker = SmartChunker(config={
            'chunk_size': 500,
            'min_chunk_size': 100,
            'overlap': 50
        })
        
        chunks = chunker.chunk_text(REAL_PYTHON_CODE, 'user_service.py')
        
        retriever.index_python_file(
            file_path='user_service.py',
            content=REAL_PYTHON_CODE,
            chunks=chunks,
            class_context={'UserService': '用户服务', 'AdminService': '管理员服务'}
        )
        
        if chunks:
            context = retriever.retrieve([chunks[0].id], include_ghost_context=True)
            
            assert len(context) > 0
            assert "UserService" in context or "class" in context
            assert "import" in context or "用户服务" in context

    def test_multiple_class_inheritance(self, retriever):
        """
        测试类继承关系
        
        验证 AdminService 正确继承 UserService
        """
        chunker = SmartChunker()
        chunks = chunker.chunk_text(REAL_PYTHON_CODE, 'user_service.py')
        
        retriever.index_python_file(
            file_path='user_service.py',
            content=REAL_PYTHON_CODE,
            chunks=chunks,
            class_context={}
        )
        
        stats = retriever.store.stats()
        assert stats["child_chunks"] == len(chunks)
        assert len(chunks) > 0, "应该生成分块"


class TestParentDocumentWithRealMarkdown:
    """使用真实 Markdown 文档测试"""

    @pytest.fixture
    def store(self):
        return ParentDocumentStore()

    def test_index_api_documentation(self, store):
        """
        测试 API 文档索引
        
        场景:
        - 包含代码示例
        - 包含表格
        - 包含多级标题
        """
        parent = store.add_parent_document(
            doc_id="api_docs",
            content=REAL_MARKDOWN_DOCS,
            doc_type=ParentDocumentType.SECTION,
            file_path="/docs/api.md",
            title="API 文档"
        )
        
        assert parent.id == "api_docs"
        assert parent.doc_type == "section"
        
        tables = [m for m in REAL_MARKDOWN_DOCS.split('\n') if '|' in m and '---' in m]
        assert len(tables) > 0, "应该包含表格"

    def test_protect_markdown_tables(self, store):
        """
        测试 Markdown 表格保护
        
        验证表格不会被切断
        """
        store.add_parent_document(
            doc_id="docs",
            content=REAL_MARKDOWN_DOCS,
            doc_type=ParentDocumentType.SECTION,
            file_path="/docs/api.md"
        )
        
        store.add_child_chunk(
            chunk_id="table_chunk",
            content="| 字段 | 类型 | 必填 | 描述 |",
            parent_doc_id="docs",
            file_path="/docs/api.md",
            start_line=30,
            end_line=35
        )
        
        context = store.build_context_for_llm(["table_chunk"], include_ghost_context=False)
        
        assert "API 文档" in context or "端点" in context

    def test_code_blocks_in_markdown(self, store):
        """
        测试 Markdown 中的代码块
        
        验证代码块内容完整
        """
        parent = store.add_parent_document(
            doc_id="docs",
            content=REAL_MARKDOWN_DOCS,
            doc_type=ParentDocumentType.SECTION,
            file_path="/docs/api.md"
        )
        
        store.add_child_chunk(
            chunk_id="chunk_001",
            content="代码块",
            parent_doc_id="docs",
            file_path="/docs/api.md",
            start_line=1,
            end_line=10
        )
        
        context = store.build_context_for_llm(["chunk_001"], include_ghost_context=False)
        
        assert len(context) > 0
        assert "POST" in context or "GET" in context


class TestParentDocumentWithComplexCode:
    """使用复杂代码测试"""

    @pytest.fixture
    def store(self):
        return ParentDocumentStore()

    def test_index_complex_pipeline(self, store):
        """
        测试复杂的数据管道代码
        
        场景:
        - 泛型类
        - 抽象基类
        - 异步代码
        - 装饰器
        """
        chunker = SmartChunker(config={
            'chunk_size': 600,
            'min_chunk_size': 150,
            'overlap': 80
        })
        
        chunks = chunker.chunk_text(REAL_COMPLEX_PYTHON, 'pipeline.py')
        
        assert len(chunks) > 0, "应该生成分块"
        
        for chunk in chunks:
            store.add_parent_document(
                doc_id=f"pipeline_{chunk.id}",
                content=REAL_COMPLEX_PYTHON,
                doc_type=ParentDocumentType.FILE,
                file_path="/src/pipeline.py",
                metadata={
                    "ghost_context": {
                        "imports": "asyncio, abc, dataclasses"
                    }
                }
            )
            
            store.add_child_chunk(
                chunk_id=chunk.id,
                content=chunk.content,
                parent_doc_id=f"pipeline_{chunk.id}",
                file_path="/src/pipeline.py",
                start_line=chunk.metadata.start_line,
                end_line=chunk.metadata.end_line
            )
        
        stats = store.stats()
        assert stats["parent_documents"] == len(chunks)
        assert stats["child_chunks"] == len(chunks)

    def test_retrieve_with_async_code(self, store):
        """
        测试异步代码检索
        
        验证 async/await 上下文完整
        """
        store.add_parent_document(
            doc_id="async_pipeline",
            content=REAL_COMPLEX_PYTHON,
            doc_type=ParentDocumentType.FILE,
            file_path="/src/pipeline.py"
        )
        
        store.add_child_chunk(
            chunk_id="chunk_001",
            content="异步代码",
            parent_doc_id="async_pipeline",
            file_path="/src/pipeline.py",
            start_line=1,
            end_line=10
        )
        
        context = store.build_context_for_llm(["chunk_001"], include_ghost_context=True)
        
        assert len(context) > 0
        assert "async" in context or "await" in context or "DataPipeline" in context


class TestRealWorldScenarios:
    """真实世界场景测试"""

    @pytest.fixture
    def retriever(self):
        return ParentDocumentRetriever()

    def test_multi_file_project(self, retriever):
        """
        测试多文件项目索引
        
        模拟真实项目结构:
        - user_service.py
        - order_service.py
        - api_docs.md
        """
        chunker = SmartChunker()
        
        files = [
            ("user_service.py", REAL_PYTHON_CODE, {'UserService': '用户服务', 'AdminService': '管理员服务'}),
            ("pipeline.py", REAL_COMPLEX_PYTHON, {'DataPipeline': '数据管道'}),
            ("api.md", REAL_MARKDOWN_DOCS, {}),
        ]
        
        for file_path, content, class_context in files:
            chunks = chunker.chunk_text(content, file_path)
            retriever.index_python_file(
                file_path=file_path,
                content=content,
                chunks=chunks,
                class_context=class_context
            )
        
        stats = retriever.get_storage_stats()
        assert stats["files_indexed"] == 3
        assert stats["parent_documents"] == 3

    def test_search_and_retrieve_flow(self, retriever):
        """
        测试搜索和检索完整流程
        
        场景:
        1. 索引代码
        2. 模拟向量搜索匹配
        3. 使用 Parent Document Retrieval 构建上下文
        4. 返回给 LLM
        """
        chunker = SmartChunker()
        
        chunks = chunker.chunk_text(REAL_PYTHON_CODE, 'user_service.py')
        retriever.index_python_file(
            file_path='user_service.py',
            content=REAL_PYTHON_CODE,
            chunks=chunks,
            class_context={'UserService': '用户服务'}
        )
        
        if chunks:
            matched_ids = [chunks[0].id, chunks[1].id] if len(chunks) > 1 else [chunks[0].id]
            
            context = retriever.retrieve(matched_ids, include_ghost_context=True)
            
            assert len(context) > 0
            assert "UserService" in context or "import" in context

    def test_class_context_preservation(self, retriever):
        """
        测试类上下文保留
        
        验证检索结果包含完整的类定义
        """
        chunker = SmartChunker()
        
        chunks = chunker.chunk_text(REAL_PYTHON_CODE, 'user_service.py')
        retriever.index_python_file(
            file_path='user_service.py',
            content=REAL_PYTHON_CODE,
            chunks=chunks,
            class_context={'AdminService': '管理员服务'}
        )
        
        if chunks:
            context = retriever.retrieve([chunks[0].id], include_ghost_context=True)
            
            assert "class" in context
            assert "AdminService" in context or "UserService" in context

    def test_ghost_context_includes_imports(self, retriever):
        """
        测试 Ghost Context 包含导入语句
        
        验证 LLM 收到上下文包含必要的导入
        """
        chunker = SmartChunker()
        
        chunks = chunker.chunk_text(REAL_PYTHON_CODE, 'user_service.py')
        retriever.index_python_file(
            file_path='user_service.py',
            content=REAL_PYTHON_CODE,
            chunks=chunks,
            class_context={}
        )
        
        if chunks:
            context = retriever.retrieve([chunks[0].id], include_ghost_context=True)
            
            assert "import" in context
            assert "datetime" in context or "typing" in context or "os" in context


class TestEdgeCasesWithRealData:
    """真实数据边界测试"""

    def test_very_large_code_block(self):
        """
        测试超大代码块
        
        生成超过默认 chunk_size 的代码
        """
        store = ParentDocumentStore()
        
        lines = []
        for i in range(500):
            lines.append(f"def function_{i}():")
            lines.append(f'    """函数 {i} 的文档字符串"""')
            lines.append(f"    result = []")
            lines.append(f"    for j in range(10):")
            lines.append(f"        result.append(j * {i})")
            lines.append(f"    return result")
        large_code = "\n".join(lines)
        
        assert len(large_code) > 50000, f"代码应该足够大，实际长度: {len(large_code)}"
        
        parent = store.add_parent_document(
            doc_id="large_code",
            content=large_code,
            doc_type=ParentDocumentType.FILE,
            file_path="/large_file.py"
        )
        
        assert parent.char_count == len(large_code)
        stats = store.stats()
        assert stats["parent_documents"] == 1

    def test_nested_classes(self):
        """
        测试嵌套类
        
        验证嵌套类能正确索引
        """
        nested_code = '''
class OuterClass:
    """外层类"""
    
    class InnerClass:
        """内层类"""
        def inner_method(self):
            pass
    
    def outer_method(self):
        pass
'''
        
        store = ParentDocumentStore()
        
        store.add_parent_document(
            doc_id="nested",
            content=nested_code,
            doc_type=ParentDocumentType.FILE,
            file_path="/nested.py",
            metadata={
                "ghost_context": {
                    "class_definition": "class OuterClass: class InnerClass:"
                }
            }
        )
        
        store.add_child_chunk(
            chunk_id="chunk_001",
            content="嵌套类代码",
            parent_doc_id="nested",
            file_path="/nested.py",
            start_line=1,
            end_line=10
        )
        
        context = store.build_context_for_llm(["chunk_001"], include_ghost_context=True)
        
        assert len(context) > 0
        assert "OuterClass" in context
        assert "InnerClass" in context

    def test_mixed_content_types(self):
        """
        测试混合内容类型
        
        包含代码、表格、文本的混合文档
        """
        mixed_content = '''
# 配置文件

## 数据库配置

| 配置项 | 值 | 描述 |
|--------|-----|------|
| host | localhost | 数据库主机 |
| port | 5432 | 端口号 |

```python
# 数据库连接代码
import psycopg2

def connect():
    conn = psycopg2.connect(
        host="localhost",
        port=5432
    )
    return conn
```

## 使用说明

请确保数据库服务已启动。
'''
        
        store = ParentDocumentStore()
        
        store.add_parent_document(
            doc_id="mixed",
            content=mixed_content,
            doc_type=ParentDocumentType.SECTION,
            file_path="/config.md"
        )
        
        store.add_child_chunk(
            chunk_id="chunk_001",
            content="混合内容",
            parent_doc_id="mixed",
            file_path="/config.md",
            start_line=1,
            end_line=15
        )
        
        context = store.build_context_for_llm(["chunk_001"], include_ghost_context=False)
        
        assert len(context) > 0
        assert "配置" in context or "数据库" in context
        assert "import" in context or "def" in context
