# 测试指南 - RAG与Skills动态调整剧本方向

## 概述

本指南展示如何测试RAG检索和Skills系统如何动态调整剧本生成方向的完整功能。

---

## 系统架构

### 核心流程

```
用户请求
    ↓
Planner (规划大纲)
    ↓
Navigator (RAG检索相关代码)
    ↓
Director (评估质量，决定是否转向)
    ↓
Writer (使用Skills生成内容)
    ↓
FactChecker (验证事实)
    ↓
Compiler (编译最终剧本)
```

### 关键组件

1. **RAG检索服务** (`src/services/retrieval_service.py`)
   - 查询扩展
   - 多因素重排序
   - 多样性过滤
   - 缓存优化

2. **Skills系统** (`src/domain/skills.py`)
   - 动态技能加载
   - 配置驱动
   - 热重载支持

3. **多智能体编排** (`src/application/orchestrator.py`)
   - LangGraph状态机
   - 智能体协调
   - 动态转向

---

## 测试方法

### 方法1: 直接Python测试 (推荐用于开发)

#### 1.1 基础集成测试

```python
import asyncio
from src.services.retrieval_service import RetrievalService, RetrievalConfig
from src.services.cache.retrieval_cache import RetrievalCache
from src.services.monitoring.retrieval_monitor import RetrievalMonitor
from src.domain.skill_loader import SkillConfigLoader
from src.domain.skills import SkillManager
from src.services.llm.service import LLMService

async def test_rag_skills_integration():
    """测试RAG和Skills的集成"""
    
    # 1. 初始化LLM服务
    llm_service = LLMService(config={
        "provider": "openai",
        "model": "gpt-4",
        "api_key": "your_api_key"
    })
    
    # 2. 初始化向量数据库
    vector_db = VectorDBService(config={
        "type": "milvus",
        "host": "localhost",
        "port": 19530
    })
    
    # 3. 初始化RAG检索服务 (带缓存和监控)
    retrieval_config = RetrievalConfig(
        enable_query_expansion=True,
        enable_reranking=True,
        enable_diversity=True,
        enable_quality_monitoring=True
    )
    
    cache = RetrievalCache()
    monitor = RetrievalMonitor()
    
    retrieval_service = RetrievalService(
        vector_db_service=vector_db,
        llm_service=llm_service,
        config=retrieval_config,
        cache=cache,
        monitor=monitor
    )
    
    # 4. 初始化Skills系统
    skill_loader = SkillConfigLoader()
    skills_config = skill_loader.load_from_file("config/skills.yaml")
    
    skill_manager = SkillManager()
    skill_manager.load_from_config(skills_config)
    
    # 5. 测试查询
    workspace_id = "test_workspace"
    query = "如何实现用户认证系统"
    
    # 执行RAG检索
    print("=== 执行RAG检索 ===")
    results = await retrieval_service.hybrid_retrieve(
        workspace_id=workspace_id,
        query=query,
        top_k=5
    )
    
    print(f"检索到 {len(results)} 个结果")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.file_path} (相似度: {result.similarity:.3f})")
    
    # 6. 获取监控指标
    print("\n=== 监控指标 ===")
    metrics = monitor.get_metrics(time_window=300)
    print(f"P95延迟: {metrics['performance']['latency']['p95']:.0f}ms")
    print(f"缓存命中率: {metrics['performance']['cache']['embedding']['hit_rate']:.1%}")
    print(f"平均相似度: {metrics['quality']['avg_similarity']:.3f}")
    
    # 7. 测试Skills应用
    print("\n=== 应用Skills生成内容 ===")
    
    # 选择合适的Skill
    available_skills = skill_manager.list_skills()
    print(f"可用Skills: {available_skills}")
    
    # 使用检索结果和Skill生成内容
    skill_name = "standard_tutorial"
    skill = skill_manager.get_skill(skill_name)
    
    # 构建提示词
    context = "\n".join([r.content[:200] for r in results])
    prompt = f"""
    基于以下代码片段，使用{skill_name}风格生成教程：
    
    {context}
    
    请生成详细的教程内容。
    """
    
    # 生成内容
    response = await llm_service.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=skill.temperature,
        max_tokens=skill.max_tokens
    )
    
    print(f"生成的内容:\n{response}")
    
    # 8. 测试缓存效果
    print("\n=== 测试缓存效果 ===")
    
    # 第二次查询相同内容 (应该命中缓存)
    print("执行相同查询 (应该命中缓存)...")
    results2 = await retrieval_service.hybrid_retrieve(
        workspace_id=workspace_id,
        query=query,
        top_k=5
    )
    
    cache_stats = cache.get_stats()
    print(f"缓存统计:")
    print(f"  查询扩展命中率: {cache_stats['query_expansion']['hit_rate']:.1%}")
    print(f"  嵌入命中率: {cache_stats['embedding']['hit_rate']:.1%}")
    print(f"  结果命中率: {cache_stats['result']['hit_rate']:.1%}")

# 运行测试
asyncio.run(test_rag_skills_integration())
```

#### 1.2 动态转向测试

```python
async def test_dynamic_pivot():
    """测试基于RAG结果的动态转向"""
    
    from src.domain.agents.director import Director
    from src.domain.models import SharedState
    
    # 初始化Director智能体
    director = Director(llm_service)
    
    # 创建共享状态
    state = SharedState(
        topic="用户认证系统",
        current_step=3,
        outline=["概述", "实现", "测试"],
        fragments=["第1部分内容", "第2部分内容"],
        rag_results=[
            {"file": "auth.py", "similarity": 0.92},
            {"file": "jwt.py", "similarity": 0.88}
        ],
        quality_score=0.75,
        should_pivot=False
    )
    
    # 执行Director评估
    print("=== Director评估 ===")
    result = await director.evaluate(state)
    
    print(f"质量评分: {result['quality_score']:.3f}")
    print(f"建议转向: {result['should_pivot']}")
    print(f"转向原因: {result['pivot_reason']}")
    
    if result['should_pivot']:
        print(f"新方向: {result['new_direction']}")
        print(f"新Skill: {result['suggested_skill']}")

asyncio.run(test_dynamic_pivot())
```

#### 1.3 Skills热重载测试

```python
async def test_skills_hot_reload():
    """测试Skills的热重载功能"""
    
    from src.domain.skill_loader import SkillConfigLoader
    from src.domain.skills import SkillManager
    import time
    
    # 初始化Skills管理器
    skill_manager = SkillManager()
    skill_loader = SkillConfigLoader()
    
    # 加载初始配置
    print("=== 加载初始Skills ===")
    skills_config = skill_loader.load_from_file("config/skills.yaml")
    skill_manager.load_from_config(skills_config)
    
    initial_skills = skill_manager.list_skills()
    print(f"初始Skills: {initial_skills}")
    
    # 修改config/skills.yaml文件
    print("\n=== 修改Skills配置 ===")
    # 手动编辑config/skills.yaml，添加新的Skill或修改现有Skill
    
    # 等待文件变化被检测
    print("等待文件变化...")
    time.sleep(2)
    
    # 重新加载
    print("重新加载Skills...")
    skills_config = skill_loader.load_from_file("config/skills.yaml")
    skill_manager.load_from_config(skills_config)
    
    updated_skills = skill_manager.list_skills()
    print(f"更新后的Skills: {updated_skills}")
    
    # 验证新Skills可用
    for skill_name in updated_skills:
        skill = skill_manager.get_skill(skill_name)
        print(f"  {skill_name}: temperature={skill.temperature}, max_tokens={skill.max_tokens}")

asyncio.run(test_skills_hot_reload())
```

---

### 方法2: REST API测试 (推荐用于集成测试)

#### 2.1 启动API服务器

```bash
# 启动API服务器
python -m src.presentation.api

# 或使用uvicorn
uvicorn src.presentation.api:app --reload --port 8000
```

#### 2.2 使用cURL测试

```bash
# 1. 登录获取令牌
TOKEN=$(curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test@example.com",
    "password": "password"
  }' | jq -r '.access_token')

echo "Token: $TOKEN"

# 2. 创建工作空间
WORKSPACE=$(curl -X POST http://localhost:8000/api/v1/workspaces \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "name": "测试工作空间",
    "description": "用于测试RAG和Skills集成"
  }' | jq -r '.workspace_id')

echo "Workspace: $WORKSPACE"

# 3. 上传代码文件
curl -X POST http://localhost:8000/api/v1/workspaces/$WORKSPACE/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@src/auth/jwt.py" \
  -F "metadata={\"language\": \"python\", \"tags\": [\"auth\"]}"

# 4. 创建剧本生成任务
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

echo "Session: $SESSION"

# 5. 查询任务状态
curl -X GET http://localhost:8000/api/v1/sessions/$SESSION \
  -H "Authorization: Bearer $TOKEN" | jq '.'

# 6. 获取生成结果
curl -X GET http://localhost:8000/api/v1/sessions/$SESSION/screenplay \
  -H "Authorization: Bearer $TOKEN" | jq '.content'

# 7. 获取监控指标
curl -X GET http://localhost:8000/api/v1/stats \
  -H "Authorization: Bearer $TOKEN" | jq '.statistics'
```

#### 2.3 使用Python requests测试

```python
import requests
import time
import json

BASE_URL = "http://localhost:8000/api/v1"

class ScreenplayTester:
    def __init__(self, username, password):
        self.token = None
        self.login(username, password)
    
    def login(self, username, password):
        """登录获取令牌"""
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"username": username, "password": password}
        )
        self.token = response.json()["access_token"]
        print(f"✓ 登录成功")
    
    def create_workspace(self, name):
        """创建工作空间"""
        response = requests.post(
            f"{BASE_URL}/workspaces",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"name": name}
        )
        workspace_id = response.json()["workspace_id"]
        print(f"✓ 创建工作空间: {workspace_id}")
        return workspace_id
    
    def upload_document(self, workspace_id, file_path):
        """上传代码文件"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'metadata': json.dumps({"language": "python"})}
            response = requests.post(
                f"{BASE_URL}/workspaces/{workspace_id}/documents",
                headers={"Authorization": f"Bearer {self.token}"},
                files=files,
                data=data
            )
        print(f"✓ 上传文件: {file_path}")
    
    def generate_screenplay(self, workspace_id, topic, skill="standard_tutorial"):
        """创建剧本生成任务"""
        response = requests.post(
            f"{BASE_URL}/generate",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "topic": topic,
                "workspace_id": workspace_id,
                "skill": skill,
                "options": {
                    "enable_fact_checking": True,
                    "enable_auto_pivot": True
                }
            }
        )
        session_id = response.json()["session_id"]
        print(f"✓ 创建任务: {session_id}")
        return session_id
    
    def wait_for_completion(self, session_id, timeout=300):
        """等待任务完成"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(
                f"{BASE_URL}/sessions/{session_id}",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            data = response.json()
            status = data["status"]
            progress = data.get("progress", {})
            
            print(f"  状态: {status} ({progress.get('percentage', 0)}%)")
            
            if status == "completed":
                print(f"✓ 任务完成")
                return True
            elif status == "failed":
                print(f"✗ 任务失败")
                return False
            
            time.sleep(5)
        
        print(f"✗ 任务超时")
        return False
    
    def get_screenplay(self, session_id):
        """获取生成的剧本"""
        response = requests.get(
            f"{BASE_URL}/sessions/{session_id}/screenplay",
            headers={"Authorization": f"Bearer {self.token}"}
        )
        return response.json()["content"]
    
    def get_stats(self):
        """获取统计信息"""
        response = requests.get(
            f"{BASE_URL}/stats",
            headers={"Authorization": f"Bearer {self.token}"}
        )
        return response.json()["statistics"]

# 使用示例
def main():
    # 初始化测试器
    tester = ScreenplayTester("test@example.com", "password")
    
    # 创建工作空间
    workspace_id = tester.create_workspace("测试工作空间")
    
    # 上传代码文件
    tester.upload_document(workspace_id, "src/auth/jwt.py")
    tester.upload_document(workspace_id, "src/auth/models.py")
    
    # 生成剧本
    print("\n=== 生成剧本 ===")
    session_id = tester.generate_screenplay(
        workspace_id,
        "用户认证系统实现",
        skill="standard_tutorial"
    )
    
    # 等待完成
    print("\n=== 等待完成 ===")
    if tester.wait_for_completion(session_id):
        # 获取结果
        print("\n=== 获取结果 ===")
        screenplay = tester.get_screenplay(session_id)
        print(f"生成的剧本:\n{screenplay[:500]}...")
        
        # 获取统计信息
        print("\n=== 统计信息 ===")
        stats = tester.get_stats()
        print(f"总会话数: {stats['total_sessions']}")
        print(f"已完成: {stats['completed_sessions']}")
        print(f"令牌使用: {stats['total_tokens_used']}")

if __name__ == "__main__":
    main()
```

---

### 方法3: 集成测试 (推荐用于CI/CD)

```python
# tests/integration/test_rag_skills_dynamic_direction.py

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from src.services.retrieval_service import RetrievalService, RetrievalConfig
from src.domain.skills import SkillManager
from src.domain.agents.director import Director
from src.domain.models import SharedState


@pytest.fixture
def mock_llm_service():
    """Mock LLM服务"""
    service = MagicMock()
    service.embedding = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    service.chat_completion = AsyncMock(return_value="生成的内容")
    return service


@pytest.fixture
def mock_vector_db():
    """Mock向量数据库"""
    db = MagicMock()
    db.vector_search = AsyncMock(return_value=[
        MagicMock(
            id="1",
            file_path="auth.py",
            content="认证代码",
            similarity=0.92,
            has_security=True
        )
    ])
    return db


@pytest.mark.asyncio
async def test_rag_retrieval_with_expansion(mock_llm_service, mock_vector_db):
    """测试RAG检索与查询扩展"""
    
    config = RetrievalConfig(
        enable_query_expansion=True,
        enable_reranking=True
    )
    
    retrieval_service = RetrievalService(
        vector_db_service=mock_vector_db,
        llm_service=mock_llm_service,
        config=config
    )
    
    # 执行检索
    results = await retrieval_service.hybrid_retrieve(
        workspace_id="test",
        query="认证系统",
        top_k=5
    )
    
    # 验证
    assert len(results) > 0
    assert results[0].similarity > 0.8


@pytest.mark.asyncio
async def test_skills_dynamic_selection(mock_llm_service):
    """测试Skills的动态选择"""
    
    skill_manager = SkillManager()
    
    # 加载Skills
    skills_config = {
        "skills": [
            {
                "name": "standard_tutorial",
                "enabled": True,
                "temperature": 0.7,
                "max_tokens": 2000
            },
            {
                "name": "advanced_guide",
                "enabled": True,
                "temperature": 0.5,
                "max_tokens": 3000
            }
        ]
    }
    
    skill_manager.load_from_config(skills_config)
    
    # 验证Skills可用
    available_skills = skill_manager.list_skills()
    assert "standard_tutorial" in available_skills
    assert "advanced_guide" in available_skills


@pytest.mark.asyncio
async def test_director_pivot_decision(mock_llm_service):
    """测试Director的转向决策"""
    
    director = Director(mock_llm_service)
    
    # 创建低质量状态
    state = SharedState(
        topic="认证系统",
        current_step=3,
        outline=["概述", "实现", "测试"],
        fragments=["内容1", "内容2"],
        quality_score=0.5,  # 低质量
        should_pivot=False
    )
    
    # 执行评估
    result = await director.evaluate(state)
    
    # 验证转向决策
    assert "should_pivot" in result
    assert "quality_score" in result


@pytest.mark.asyncio
async def test_end_to_end_rag_skills_workflow(
    mock_llm_service,
    mock_vector_db
):
    """端到端测试RAG和Skills工作流"""
    
    # 1. 初始化服务
    retrieval_service = RetrievalService(
        vector_db_service=mock_vector_db,
        llm_service=mock_llm_service
    )
    
    skill_manager = SkillManager()
    skill_manager.load_from_config({
        "skills": [{"name": "standard_tutorial", "enabled": True}]
    })
    
    # 2. 执行RAG检索
    results = await retrieval_service.hybrid_retrieve(
        workspace_id="test",
        query="认证系统",
        top_k=5
    )
    
    assert len(results) > 0
    
    # 3. 选择合适的Skill
    skill = skill_manager.get_skill("standard_tutorial")
    assert skill is not None
    
    # 4. 生成内容
    response = await mock_llm_service.chat_completion(
        messages=[{"role": "user", "content": "生成教程"}],
        temperature=skill.temperature,
        max_tokens=skill.max_tokens
    )
    
    assert response is not None
```

---

## 监控和调试

### 查看缓存效果

```python
# 获取缓存统计
cache_stats = retrieval_service.cache.get_stats()

print("缓存统计:")
print(f"  查询扩展: {cache_stats['query_expansion']['hit_rate']:.1%}")
print(f"  嵌入: {cache_stats['embedding']['hit_rate']:.1%}")
print(f"  结果: {cache_stats['result']['hit_rate']:.1%}")
```

### 查看性能指标

```python
# 获取监控指标
metrics = retrieval_service.monitor.get_metrics(time_window=300)

print("性能指标 (最近5分钟):")
print(f"  P50延迟: {metrics['performance']['latency']['p50']:.0f}ms")
print(f"  P95延迟: {metrics['performance']['latency']['p95']:.0f}ms")
print(f"  P99延迟: {metrics['performance']['latency']['p99']:.0f}ms")
print(f"  吞吐量: {metrics['performance']['throughput']:.2f} qps")
print(f"  错误率: {metrics['performance']['errors']['error_rate']:.2%}")
```

### 查看质量指标

```python
# 获取质量指标
quality = metrics['quality']

print("质量指标:")
print(f"  平均相似度: {quality['avg_similarity']:.3f}")
print(f"  多样性: {quality['avg_diversity']:.3f}")
print(f"  安全标记: {quality['marker_distribution'].get('security', 0)}")
print(f"  已弃用标记: {quality['marker_distribution'].get('deprecated', 0)}")
```

---

## 常见问题

### Q1: 如何验证RAG检索是否工作正常？

**A**: 检查以下指标：
1. 检索结果的相似度 (应该 > 0.7)
2. 缓存命中率 (应该 > 70% 预热后)
3. 延迟 (应该 < 200ms 缓存命中)

### Q2: 如何测试Skills的动态切换？

**A**: 
1. 修改 `config/skills.yaml`
2. 观察系统是否自动重新加载
3. 验证新Skill被应用到生成过程

### Q3: 如何调试转向决策？

**A**:
1. 启用Director的详细日志
2. 检查质量评分
3. 查看转向原因和新方向

### Q4: 缓存没有命中怎么办？

**A**:
1. 检查缓存是否启用
2. 验证TTL设置
3. 检查缓存大小限制
4. 查看缓存统计信息

---

## 性能基准

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

## 下一步

1. **运行完整测试套件**
   ```bash
   pytest tests/integration/test_rag_skills_dynamic_direction.py -v
   ```

2. **启动API服务器进行手动测试**
   ```bash
   python -m src.presentation.api
   ```

3. **监控生产环境**
   - 查看缓存命中率
   - 监控延迟指标
   - 跟踪质量评分

---

*测试指南完成*  
*最后更新: 2026-01-31*
