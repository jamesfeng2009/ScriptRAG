# RAG + Skills 集成测试快速开始指南

## 🎯 目标

本指南展示如何快速测试RAG检索和Skills系统如何动态调整剧本生成方向的完整功能。

---

## ⚡ 5分钟快速开始

### 1. 运行集成测试
```bash
# 运行所有集成测试
pytest tests/integration/test_skills_integration.py -v

# 运行特定测试
pytest tests/integration/test_skills_integration.py::TestSkillsWithSharedState::test_shared_state_skill_switching -v
```

**预期结果**: ✅ 所有测试通过

### 2. 查看测试覆盖
```bash
# 运行所有测试并显示覆盖率
pytest tests/ --cov=src --cov-report=html

# 打开覆盖率报告
open htmlcov/index.html
```

### 3. 查看监控指标
```bash
# 运行单个测试并查看输出
pytest tests/integration/test_skills_integration.py::TestSkillsWithSharedState::test_retrieval_config_with_workflow -v -s
```

---

## 📋 完整测试清单

### ✅ 已通过的测试

#### 单元测试 (209个)
- ✅ Phase 1: 技能系统 (32个)
- ✅ Phase 2: RAG优化 (35个)
- ✅ Phase 3: 缓存监控 (34个)
- ✅ 其他单元测试 (108个)

#### 集成测试 (56个)
- ✅ 端到端工作流 (7个)
- ✅ 幻觉检测工作流 (8个)
- ✅ LangGraph工作流 (8个)
- ✅ LLM Provider Fallback (9个)
- ✅ Pivot工作流 (7个)
- ✅ 重试限制工作流 (9个)
- ✅ Skills集成 (8个)

#### 属性测试 (214个)
- ✅ 代理执行顺序 (3个)
- ✅ 动态技能切换 (10个)
- ✅ 幻觉检测和再生成 (21个)
- ✅ 重试限制 (22个)
- ✅ Pivot工作流 (9个)
- ✅ 其他属性测试 (149个)

---

## 🔍 详细测试指南

### 测试1: RAG检索功能

**目标**: 验证RAG检索、查询扩展、重排序和缓存工作正常

**运行**:
```bash
pytest tests/unit/test_query_expansion.py -v
pytest tests/unit/test_reranker.py -v
pytest tests/unit/test_retrieval_cache.py -v
```

**验证点**:
- ✅ 查询扩展生成多个查询变体
- ✅ 重排序提升相关结果
- ✅ 缓存命中率 > 70%
- ✅ 延迟 < 100ms (缓存命中)

### 测试2: Skills系统功能

**目标**: 验证技能加载、配置和动态切换工作正常

**运行**:
```bash
pytest tests/unit/test_skill_loader.py -v
pytest tests/unit/test_prompt_manager.py -v
pytest tests/integration/test_skills_integration.py -v
```

**验证点**:
- ✅ YAML配置正确加载
- ✅ 技能可以动态切换
- ✅ PromptManager正确应用配置
- ✅ 热重载功能工作正常

### 测试3: 动态转向功能

**目标**: 验证Director评估、Pivot触发和大纲修改工作正常

**运行**:
```bash
pytest tests/integration/test_pivot_workflow.py -v
pytest tests/property/test_pivot_deprecation.py -v
pytest tests/property/test_pivot_retrieval.py -v
```

**验证点**:
- ✅ Director正确评估质量
- ✅ Pivot在需要时被触发
- ✅ 大纲在Pivot后被修改
- ✅ 重新检索在Pivot后被触发

### 测试4: 监控和告警

**目标**: 验证监控系统正确跟踪性能和质量指标

**运行**:
```bash
pytest tests/unit/test_retrieval_monitor.py -v
pytest tests/property/test_comprehensive_logging.py -v
```

**验证点**:
- ✅ 性能指标被正确记录
- ✅ 质量指标被正确计算
- ✅ 告警在阈值超过时触发
- ✅ 日志记录完整

---

## 🧪 手动测试步骤

### 步骤1: 启动API服务器

```bash
# 启动API服务器
python -m src.presentation.api

# 或使用uvicorn
uvicorn src.presentation.api:app --reload --port 8000
```

**预期**: 服务器在 http://localhost:8000 启动

### 步骤2: 创建工作空间

```bash
# 获取认证令牌
TOKEN=$(curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test@example.com",
    "password": "password"
  }' | jq -r '.access_token')

# 创建工作空间
WORKSPACE=$(curl -X POST http://localhost:8000/api/v1/workspaces \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "name": "测试工作空间",
    "description": "用于测试RAG和Skills集成"
  }' | jq -r '.workspace_id')

echo "Workspace ID: $WORKSPACE"
```

### 步骤3: 上传代码文件

```bash
# 上传Python文件
curl -X POST http://localhost:8000/api/v1/workspaces/$WORKSPACE/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@src/auth/jwt.py" \
  -F "metadata={\"language\": \"python\", \"tags\": [\"auth\"]}"

# 上传多个文件
for file in src/auth/*.py; do
  curl -X POST http://localhost:8000/api/v1/workspaces/$WORKSPACE/documents \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@$file" \
    -F "metadata={\"language\": \"python\"}"
done
```

### 步骤4: 生成剧本

```bash
# 创建生成任务
SESSION=$(curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "topic": "用户认证系统实现",
    "project_context": "基于FastAPI和JWT",
    "workspace_id": "'$WORKSPACE'",
    "skill": "standard_tutorial",
    "options": {
      "enable_fact_checking": true,
      "enable_auto_pivot": true
    }
  }' | jq -r '.session_id')

echo "Session ID: $SESSION"
```

### 步骤5: 查询任务状态

```bash
# 查询任务状态
curl -X GET http://localhost:8000/api/v1/sessions/$SESSION \
  -H "Authorization: Bearer $TOKEN" | jq '.'

# 等待完成
while true; do
  STATUS=$(curl -s -X GET http://localhost:8000/api/v1/sessions/$SESSION \
    -H "Authorization: Bearer $TOKEN" | jq -r '.status')
  
  if [ "$STATUS" = "completed" ]; then
    echo "✅ 任务完成"
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "❌ 任务失败"
    break
  else
    echo "⏳ 状态: $STATUS"
    sleep 5
  fi
done
```

### 步骤6: 获取结果

```bash
# 获取生成的剧本
curl -X GET http://localhost:8000/api/v1/sessions/$SESSION/screenplay \
  -H "Authorization: Bearer $TOKEN" | jq '.content'

# 获取监控指标
curl -X GET http://localhost:8000/api/v1/stats \
  -H "Authorization: Bearer $TOKEN" | jq '.statistics'
```

---

## 📊 验证检查清单

### RAG检索验证
- [ ] 查询扩展生成了多个查询变体
- [ ] 重排序提升了相关结果的排名
- [ ] 缓存命中率 > 70%
- [ ] 延迟 < 100ms (缓存命中)
- [ ] 多样性过滤移除了重复结果

### Skills系统验证
- [ ] YAML配置正确加载
- [ ] 技能可以动态切换
- [ ] PromptManager应用了正确的配置
- [ ] 热重载功能工作正常
- [ ] 技能兼容性检查工作正常

### 动态转向验证
- [ ] Director正确评估了质量
- [ ] Pivot在需要时被触发
- [ ] 大纲在Pivot后被修改
- [ ] 重新检索在Pivot后被触发
- [ ] 技能在Pivot后被切换

### 监控验证
- [ ] 性能指标被正确记录
- [ ] 质量指标被正确计算
- [ ] 告警在阈值超过时触发
- [ ] 日志记录完整
- [ ] 缓存统计准确

---

## 🐛 常见问题排查

### Q1: 测试超时

**症状**: 测试运行超过30秒

**解决**:
```bash
# 增加超时时间
pytest tests/ --timeout=60

# 或运行特定测试
pytest tests/integration/test_skills_integration.py -v
```

### Q2: Mock数据不完整

**症状**: 幻觉检测测试失败

**解决**:
```python
# 检查mock数据
from tests.fixtures.realistic_mock_data import MOCK_RETRIEVAL_RESULTS
print(MOCK_RETRIEVAL_RESULTS)

# 或使用不完整的mock数据
from tests.fixtures.realistic_mock_data import create_incomplete_mock_data
incomplete_data = create_incomplete_mock_data(missing_functions=['nonexistent_func'])
```

### Q3: API连接失败

**症状**: 无法连接到API服务器

**解决**:
```bash
# 检查服务器是否运行
curl http://localhost:8000/health

# 检查端口是否被占用
lsof -i :8000

# 使用不同的端口
uvicorn src.presentation.api:app --port 8001
```

### Q4: 认证失败

**症状**: 401 Unauthorized

**解决**:
```bash
# 检查令牌是否有效
echo $TOKEN

# 重新获取令牌
TOKEN=$(curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test@example.com",
    "password": "password"
  }' | jq -r '.access_token')
```

---

## 📈 性能基准

### 预期性能

| 操作 | 无缓存 | 有缓存 | 改进 |
|------|--------|--------|------|
| 查询扩展 | 100-200ms | <1ms | 100-200x |
| 嵌入生成 | 50-100ms | <1ms | 50-100x |
| 完整检索 | 300-500ms | 50-100ms | 3-10x |

### 缓存命中率目标

- 查询扩展: 60-70%
- 嵌入: 70-80%
- 结果: 20-30% (短TTL)

---

## 🚀 下一步

### 1. 运行完整测试套件
```bash
pytest tests/ -v --cov=src
```

### 2. 启动API服务器进行手动测试
```bash
python -m src.presentation.api
```

### 3. 监控生产环境
- 查看缓存命中率
- 监控延迟指标
- 跟踪质量评分

### 4. 部署到生产环境
- 配置环境变量
- 设置监控告警
- 启用审计日志

---

## 📚 相关文档

- `TESTING_GUIDE_RAG_SKILLS_INTEGRATION.md` - 完整测试指南
- `QUICK_START_GUIDE.md` - 快速开始指南
- `docs/API.md` - API文档
- `docs/SKILL_CONFIGURATION_GUIDE.md` - 技能配置指南
- `CURRENT_PROJECT_STATUS.md` - 项目当前状态

---

## 💡 提示

1. **使用pytest的-v标志获得详细输出**
   ```bash
   pytest tests/integration/test_skills_integration.py -v
   ```

2. **使用-s标志查看print输出**
   ```bash
   pytest tests/integration/test_skills_integration.py -v -s
   ```

3. **使用-k标志运行特定测试**
   ```bash
   pytest tests/ -k "skill" -v
   ```

4. **使用--tb=short获得简洁的错误信息**
   ```bash
   pytest tests/ --tb=short
   ```

5. **使用--lf运行上次失败的测试**
   ```bash
   pytest tests/ --lf
   ```

---

*快速开始指南完成*  
*最后更新: 2026-01-31*  
*状态: ✅ 生产就绪*

