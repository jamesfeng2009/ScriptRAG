# 数据库迁移指南

## 概述

本目录包含从 PostgreSQL + pgvector 迁移到 Milvus 的完整工具集。迁移采用渐进式策略，确保零停机时间和数据一致性。

## 迁移触发条件

当以下任一指标达到阈值时，建议开始迁移：

| 指标 | 阈值 | 说明 |
|------|------|------|
| 向量数量 | 100 万 | 向量数据量超过 PostgreSQL 最佳性能范围 |
| 搜索 QPS | 100 | 查询压力超过 PostgreSQL 处理能力 |
| P99 延迟 | 500ms | 搜索延迟影响用户体验 |
| 存储大小 | 100GB | 存储成本和性能考虑 |

## 迁移流程

### 阶段 1: 监控（Monitoring）

**目标**: 持续监控系统指标，判断是否需要迁移

**操作**:
```bash
python migration_monitor.py
```

**输出**:
- 实时指标收集
- 阈值检查
- 迁移建议生成
- 告警触发

### 阶段 2: 准备（Preparation）

**目标**: 准备迁移环境和工具

**操作**:
1. 安装 Milvus
```bash
# 使用 Docker Compose
docker-compose -f milvus-docker-compose.yml up -d
```

2. 验证连接
```bash
python -c "from pymilvus import connections; connections.connect(host='localhost', port=19530); print('Connected!')"
```

3. 检查当前指标
```bash
python migration_orchestrator.py --check-status
```

### 阶段 3: 全量迁移（Full Migration）

**目标**: 将所有现有数据从 PostgreSQL 迁移到 Milvus

**操作**:
```bash
python postgres_to_milvus.py
```

**特性**:
- 批量迁移（默认 1000 条/批）
- 进度条显示
- 自动重试
- 数据验证

**验证**:
```python
# 验证迁移结果
verification = await migrator.verify_migration(sample_size=100)
print(f"Count match: {verification['count_match']}")
print(f"Sample match rate: {verification['sample_match_rate']:.2%}")
```

### 阶段 4: 双写（Dual Write）

**目标**: 新数据同时写入 PostgreSQL 和 Milvus

**操作**:
```python
from dual_write_manager import DualWriteManager

# 初始化双写管理器
dual_write = DualWriteManager(
    pg_pool=pg_pool,
    milvus_collection=milvus_collection,
    enable_milvus=True  # 启用 Milvus 写入
)

# 插入文档（自动双写）
doc_id = await dual_write.insert_document(
    workspace_id="workspace-123",
    file_path="src/main.py",
    content="...",
    embedding=[0.1, 0.2, ...],
    language="python"
)
```

**特性**:
- 自动双写到两个数据库
- PostgreSQL 失败则整体失败
- Milvus 失败不影响主流程（通过增量同步补齐）
- 灰度开关控制

### 阶段 5: 灰度切流（Gradual Cutover）

**目标**: 逐步将读流量从 PostgreSQL 切换到 Milvus

**操作**:
```python
from gradual_cutover import GradualCutoverManager, CutoverScheduler

# 初始化切流管理器
cutover = GradualCutoverManager(pg_pool, milvus_collection)

# 设置切流策略
cutover.set_cutover_strategy("workspace")  # 按工作空间切流

# 手动控制流量百分比
cutover.set_traffic_percentage(10)  # 10% 流量到 Milvus

# 或使用自动调度器
scheduler = CutoverScheduler(cutover)
await scheduler.execute_gradual_cutover()
```

**切流计划**:
| 阶段 | 流量百分比 | 持续时间 | 说明 |
|------|-----------|---------|------|
| 1 | 1% | 24 小时 | 小流量验证 |
| 2 | 5% | 24 小时 | 扩大验证范围 |
| 3 | 10% | 24 小时 | 持续观察 |
| 4 | 25% | 48 小时 | 四分之一流量 |
| 5 | 50% | 48 小时 | 一半流量 |
| 6 | 75% | 24 小时 | 大部分流量 |
| 7 | 100% | - | 完全切换 |

**切流策略**:
- `random`: 随机切流（适合均匀分布）
- `workspace`: 按工作空间切流（适合隔离测试）
- `user`: 按用户切流（适合用户分组）

### 阶段 6: 完成（Completion）

**目标**: 验证迁移成功，PostgreSQL 降级为只读备份

**操作**:
1. 最终验证
```python
# 对比搜索结果
comparison = await cutover.compare_results(
    workspace_id="test-workspace",
    query_embedding=test_embedding,
    top_k=10
)
print(f"Overlap rate: {comparison['overlap_rate']:.2%}")
```

2. 禁用 PostgreSQL 写入
```python
dual_write.disable_milvus_write()  # 只写 Milvus
```

3. 配置 PostgreSQL 为只读
```sql
ALTER DATABASE screenplay_db SET default_transaction_read_only = on;
```

## 使用编排器（推荐）

编排器自动化整个迁移流程：

```python
from migration_orchestrator import MigrationOrchestrator

# 创建编排器
orchestrator = MigrationOrchestrator(
    pg_config={...},
    milvus_config={...}
)

# 初始化
await orchestrator.initialize()

# 启动监控
await orchestrator.start_monitoring()

# 获取状态
status = await orchestrator.get_status()
print(status)

# 开始迁移（当建议迁移时）
await orchestrator.start_migration()
```

## 监控指标

### 实时指标

```python
from migration_monitor import MigrationMonitor

monitor = MigrationMonitor(pg_config)
await monitor.connect()

# 收集指标
metrics = await monitor.collect_metrics()
print(f"Vector count: {metrics['vector_count']}")
print(f"Search QPS: {metrics['search_qps']:.2f}")
print(f"P99 latency: {metrics['p99_latency_ms']:.2f}ms")
print(f"Storage size: {metrics['storage_size_gb']:.2f}GB")
```

### 告警配置

```python
# 自定义告警回调
async def send_email_alert(alert_data):
    # 发送邮件
    pass

async def send_slack_alert(alert_data):
    # 发送 Slack 消息
    pass

monitor = MigrationMonitor(
    pg_config=pg_config,
    alert_callbacks=[send_email_alert, send_slack_alert]
)
```

## 回滚策略

如果迁移过程中出现问题，可以快速回滚：

### 1. 双写阶段回滚
```python
# 禁用 Milvus 写入
dual_write.disable_milvus_write()

# 继续使用 PostgreSQL
```

### 2. 灰度切流阶段回滚
```python
# 将流量切回 PostgreSQL
cutover.set_traffic_percentage(0)

# 或立即切回
cutover.set_traffic_percentage(0)
```

### 3. 数据回滚
```python
# 从 PostgreSQL 恢复数据（如果需要）
# PostgreSQL 始终保持完整数据
```

## 性能优化

### PostgreSQL 优化
```sql
-- 增加共享内存
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';

-- 优化向量索引
ALTER INDEX idx_code_documents_embedding SET (m = 16, ef_construction = 256);
```

### Milvus 优化
```python
# 调整索引参数
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {
        "M": 16,              # 增加连接数以提高召回率
        "efConstruction": 256  # 增加构建深度
    }
}

# 调整搜索参数
search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 128}  # 增加搜索深度
}
```

## 故障排查

### 问题 1: 迁移速度慢
**原因**: 批量大小不合适或网络延迟
**解决**:
```python
migrator = PostgresToMilvusMigration(
    pg_config=pg_config,
    milvus_config=milvus_config,
    batch_size=2000  # 增加批量大小
)
```

### 问题 2: 搜索结果不一致
**原因**: 向量精度或索引参数差异
**解决**:
```python
# 对比搜索结果
comparison = await cutover.compare_results(...)
print(f"Overlap rate: {comparison['overlap_rate']:.2%}")
print(f"Avg similarity diff: {comparison['avg_similarity_diff']:.4f}")

# 如果差异过大，调整索引参数
```

### 问题 3: Milvus 连接失败
**原因**: Milvus 服务未启动或配置错误
**解决**:
```bash
# 检查 Milvus 状态
docker ps | grep milvus

# 查看日志
docker logs milvus-standalone

# 重启 Milvus
docker-compose restart milvus-standalone
```

## 最佳实践

1. **提前规划**: 在达到阈值前 1-2 个月开始准备
2. **充分测试**: 在测试环境完整演练迁移流程
3. **监控告警**: 配置完善的监控和告警系统
4. **灰度切流**: 严格按照灰度计划执行，不要跳过阶段
5. **保留备份**: PostgreSQL 数据至少保留 3 个月
6. **文档记录**: 记录每个阶段的操作和结果
7. **团队协作**: 确保团队成员了解迁移计划和回滚策略

## 依赖安装

```bash
pip install pymilvus asyncpg psycopg2-binary tqdm
```

## 配置文件示例

```yaml
# migration_config.yaml
postgresql:
  host: localhost
  port: 5432
  database: screenplay_db
  user: postgres
  password: password

milvus:
  host: localhost
  port: 19530

migration:
  batch_size: 1000
  verify_sample_size: 100
  
monitoring:
  interval_seconds: 60
  alert_email: admin@example.com
  alert_slack_webhook: https://hooks.slack.com/...

cutover:
  strategy: workspace
  schedule:
    - percentage: 1
      duration_hours: 24
    - percentage: 5
      duration_hours: 24
    - percentage: 10
      duration_hours: 24
    - percentage: 25
      duration_hours: 48
    - percentage: 50
      duration_hours: 48
    - percentage: 75
      duration_hours: 24
    - percentage: 100
      duration_hours: 0
```

## 联系支持

如有问题，请联系：
- 技术支持: support@example.com
- 文档: https://docs.example.com/migration
- Issue 跟踪: https://github.com/example/screenplay/issues
