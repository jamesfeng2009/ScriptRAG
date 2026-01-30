# 服务层 (Service Layer)

服务层提供了系统的核心基础设施服务，包括 LLM 集成、数据库访问和代码解析。

## 目录结构

```
services/
├── llm/                    # LLM 服务
│   ├── adapter.py         # LLM 适配器抽象基类
│   ├── openai_adapter.py  # OpenAI 兼容适配器
│   ├── qwen_adapter.py    # 通义千问适配器
│   ├── minimax_adapter.py # MiniMax 适配器
│   ├── glm_adapter.py     # 智谱 GLM 适配器
│   └── service.py         # 统一 LLM 服务
├── database/              # 数据库服务
│   ├── postgres.py        # PostgreSQL 服务
│   └── vector_db.py       # 向量数据库服务
└── parser/                # 代码解析服务
    └── tree_sitter_parser.py  # Tree-sitter 解析器
```

## LLM 服务

### 功能特性

- **多提供商支持**: OpenAI、通义千问、MiniMax、智谱 GLM
- **统一接口**: 所有提供商使用相同的 API 接口
- **自动回退**: 主提供商失败时自动切换到备用提供商
- **任务类型映射**: 根据任务类型自动选择合适的模型
- **日志记录**: 记录所有 LLM 调用的详细信息

### 使用示例

```python
from src.services.llm.service import LLMService

# 从配置文件加载
llm_service = LLMService.from_yaml("config/llm_providers.yaml")

# 聊天补全（高性能模型）
response = await llm_service.chat_completion(
    messages=[
        {"role": "user", "content": "解释什么是 RAG"}
    ],
    task_type="high_performance",
    temperature=0.7
)

# 聊天补全（轻量级模型）
response = await llm_service.chat_completion(
    messages=[
        {"role": "user", "content": "生成一段代码注释"}
    ],
    task_type="lightweight",
    temperature=0.7
)

# 生成嵌入向量
embeddings = await llm_service.embedding(
    texts=["这是一段文本", "这是另一段文本"]
)
```

### 配置文件

配置文件位于 `config/llm_providers.yaml`，包含以下内容：

- `providers`: 提供商配置（API 密钥、base_url 等）
- `model_mappings`: 模型映射（high_performance、lightweight、embedding）
- `active_provider`: 当前激活的提供商
- `fallback_providers`: 回退提供商列表

参考 `config/llm_providers.example.yaml` 创建配置文件。

## 向量数据库服务

### 功能特性

- **PostgreSQL + pgvector**: 使用 PostgreSQL 17 和 pgvector 扩展
- **向量搜索**: 基于余弦相似度的语义搜索
- **混合搜索**: 结合向量搜索和关键词过滤
- **连接池**: 异步连接池管理
- **标记检测**: 自动检测 @deprecated、FIXME、TODO、Security 标记

### 使用示例

```python
from src.services.database.vector_db import PostgresVectorDBService

# 初始化服务
vector_db = PostgresVectorDBService(
    host="localhost",
    port=5432,
    database="screenplay_db",
    user="screenplay_user",
    password="password"
)

await vector_db.initialize()

# 索引文档
doc_id = await vector_db.index_document(
    workspace_id="workspace-123",
    file_path="src/main.py",
    content="def main(): pass",
    embedding=[0.1, 0.2, ...],  # 1536 维向量
    language="python",
    has_deprecated=False,
    has_fixme=True
)

# 向量搜索
results = await vector_db.vector_search(
    workspace_id="workspace-123",
    query_embedding=[0.1, 0.2, ...],
    top_k=5,
    similarity_threshold=0.7
)

# 混合搜索
results = await vector_db.hybrid_search(
    workspace_id="workspace-123",
    query_embedding=[0.1, 0.2, ...],
    keyword_filters={"has_deprecated": True},
    top_k=5
)

# 关闭连接
await vector_db.close()
```

## 代码解析服务

### 功能特性

- **多语言支持**: Python、JavaScript、TypeScript、Java、C++、Go 等
- **结构提取**: 提取函数、类、方法、注释
- **标记检测**: 检测 @deprecated、FIXME、TODO、Security 标记
- **回退机制**: Tree-sitter 解析失败时自动回退到纯文本解析

### 使用示例

```python
from src.services.parser.tree_sitter_parser import TreeSitterParser

# 初始化解析器
parser = TreeSitterParser()

# 解析代码
parsed_code = parser.parse(
    file_path="src/main.py",
    content="""
    # @deprecated Use new_function instead
    def old_function():
        # TODO: Remove this function
        pass
    """
)

# 检查标记
print(parsed_code.has_deprecated)  # True
print(parsed_code.has_todo)        # True

# 提取函数
functions = parser.extract_functions(parsed_code)

# 提取注释
comments = parser.extract_comments(parsed_code)

# 检测标记
markers = parser.detect_markers(content)
print(markers)  # {'has_deprecated': True, 'has_fixme': False, ...}
```

## 架构设计原则

### 1. 接口抽象

所有服务都定义了抽象接口（ABC），便于：
- 单元测试（使用 mock）
- 切换实现（如从 PostgreSQL 迁移到 Milvus）
- 扩展功能（添加新的提供商）

### 2. 依赖注入

服务通过构造函数接收配置，支持：
- 灵活配置
- 测试隔离
- 环境切换

### 3. 错误处理

所有服务都实现了完善的错误处理：
- 记录详细错误日志
- 提供有意义的错误信息
- 支持优雅降级

### 4. 异步支持

所有 I/O 操作都使用异步接口：
- 提高并发性能
- 避免阻塞
- 支持大规模并发

## 测试

运行服务层测试：

```bash
# 运行所有服务层测试
pytest tests/unit/test_llm_service.py -v
pytest tests/unit/test_vector_db_service.py -v
pytest tests/unit/test_parser_service.py -v

# 运行所有单元测试
pytest tests/unit/ -v
```

## 依赖项

服务层依赖以下第三方库：

- `openai`: OpenAI SDK（用于所有 OpenAI 兼容提供商）
- `asyncpg`: PostgreSQL 异步客户端
- `pydantic`: 数据验证和配置管理
- `pyyaml`: YAML 配置文件解析
- `tree-sitter`: 代码解析（可选）

安装依赖：

```bash
pip install -r requirements.txt
```

## 下一步

服务层已经实现了核心接口，后续任务将：

1. 实现数据库表结构（任务 5）
2. 实现导航器智能体，集成 RAG 检索（任务 6）
3. 实现其他智能体（任务 7-15）
4. 实现 LangGraph 状态机（任务 16）

## 相关文档

- [设计文档](../../.kiro/specs/rag-screenplay-multi-agent/design.md)
- [需求文档](../../.kiro/specs/rag-screenplay-multi-agent/requirements.md)
- [任务列表](../../.kiro/specs/rag-screenplay-multi-agent/tasks.md)
