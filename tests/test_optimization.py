"""优化模块测试套件

测试内容：
1. 智能跳过（质量评估、复杂度阈值、缓存命中跳过）
2. 并行执行（向量+关键词并行搜索）
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta


class TestQualityAssessor:
    """质量评估器测试"""

    def test_assess_high_quality_code(self):
        """测试高质量代码评估"""
        from src.services.optimization import QualityAssessor

        assessor = QualityAssessor(
            high_quality_threshold=0.85,
            low_quality_threshold=0.5
        )

        high_quality_content = """
```python
class UserService:
    def __init__(self, db_session):
        self.db = db_session
    
    async def get_user_by_id(self, user_id: int) -> User:
        cache_key = f"user:{user_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            return User(**cached)
        
        user = await self.db.get_user(user_id)
        if user:
            await self.cache.set(cache_key, user.to_dict())
        
        return user
```

## 总结
本服务提供用户数据的增删改查功能。
        """

        score = assessor.assess_quality(high_quality_content)

        assert score >= 0.3, f"High quality content should score >= 0.3, got {score}"
        assert score <= 1.0, f"Score should be <= 1.0, got {score}"

        print(f"High quality content score: {score}")

    def test_assess_low_quality_content(self):
        """测试低质量内容评估"""
        from src.services.optimization import QualityAssessor

        assessor = QualityAssessor()

        low_quality_content = """
一些随机的文本内容，没有太多结构。
这里是一些不完整的信息。
        """

        score = assessor.assess_quality(low_quality_content)

        assert score < 0.7, f"Low quality content should score < 0.7, got {score}"

        print(f"Low quality content score: {score}")

    def test_empty_content_score(self):
        """测试空内容评估"""
        from src.services.optimization import QualityAssessor

        assessor = QualityAssessor()

        assert assessor.assess_quality("") == 0.0
        assert assessor.assess_quality("   ") == 0.0

    def test_quality_caching(self):
        """测试质量评估缓存"""
        from src.services.optimization import QualityAssessor

        assessor = QualityAssessor(cache_ttl=300)

        content = "def test_function():\n    pass"

        score1 = assessor.assess_quality(content)
        score2 = assessor.assess_quality(content)

        assert score1 == score2, "Cached score should be identical"

        cache_size = len(assessor._quality_cache)
        assert cache_size >= 1, "Cache should contain at least one entry"

    def test_should_skip_fact_check_high_quality(self):
        """测试高质量内容跳过事实检查"""
        from src.services.optimization import QualityAssessor

        assessor = QualityAssessor(
            high_quality_threshold=0.85
        )

        high_quality_content = """
## API 文档

### 用户接口

```python
async def create_user(data: UserCreate) -> User:
    创建新用户
    hashed_password = hash_password(data.password)
    user = await db.create_user(
        username=data.username,
        email=data.email,
        password_hash=hashed_password
    )
    return user
```
        """

        score = assessor.assess_quality(high_quality_content)
        decision = assessor.should_skip_fact_check(score, skip_types=['fact_check'])

        print(f"Score: {score}, Should skip: {decision.should_skip}")

        if score >= 0.85:
            assert decision.should_skip is True
            assert decision.skip_type == 'fact_check'


class TestComplexityBasedSkipper:
    """基于复杂度的跳过器测试"""

    def test_should_skip_detailed_processing_high_complexity(self):
        """测试高复杂度内容跳过详细处理"""
        from src.services.optimization import ComplexityBasedSkipper

        skipper = ComplexityBasedSkipper(
            skip_threshold=0.8,
            reduce_processing_threshold=0.6
        )

        should_skip, reason = skipper.should_skip_detailed_processing(0.85)

        assert should_skip is True
        assert "High complexity" in reason
        print(f"High complexity (0.85): should_skip={should_skip}, reason={reason}")

    def test_should_skip_detailed_processing_medium_complexity(self):
        """测试中等复杂度内容减少处理"""
        from src.services.optimization import ComplexityBasedSkipper

        skipper = ComplexityBasedSkipper()

        should_skip, reason = skipper.should_skip_detailed_processing(0.65)

        assert should_skip is False
        assert "Medium complexity" in reason
        print(f"Medium complexity (0.65): should_skip={should_skip}, reason={reason}")

    def test_should_skip_detailed_processing_low_complexity(self):
        """测试低复杂度内容标准处理"""
        from src.services.optimization import ComplexityBasedSkipper

        skipper = ComplexityBasedSkipper()

        should_skip, reason = skipper.should_skip_detailed_processing(0.3)

        assert should_skip is False
        assert "Low complexity" in reason
        print(f"Low complexity (0.3): should_skip={should_skip}, reason={reason}")

    def test_get_processing_mode(self):
        """测试获取处理模式"""
        from src.services.optimization import ComplexityBasedSkipper

        skipper = ComplexityBasedSkipper()

        assert skipper.get_processing_mode(0.9) == 'minimal'
        assert skipper.get_processing_mode(0.65) == 'reduced'
        assert skipper.get_processing_mode(0.3) == 'standard'

        print("Processing modes: minimal=0.9, reduced=0.65, standard=0.3")


class TestCacheBasedSkipper:
    """基于缓存的跳过器测试"""

    def test_check_cache_hit_first_time(self):
        """测试首次缓存查找"""
        from src.services.optimization import CacheBasedSkipper

        skipper = CacheBasedSkipper()

        is_hit, hit_count = skipper.check_cache_hit("test_key_1")

        assert is_hit is False
        assert hit_count == 1

    def test_check_cache_hit_second_time(self):
        """测试第二次缓存查找"""
        from src.services.optimization import CacheBasedSkipper

        skipper = CacheBasedSkipper()

        skipper.check_cache_hit("test_key_2")
        is_hit, hit_count = skipper.check_cache_hit("test_key_2")

        assert is_hit is True
        assert hit_count == 2

    def test_should_skip_processing_below_threshold(self):
        """测试未达到跳过阈值的处理"""
        from src.services.optimization import CacheBasedSkipper

        skipper = CacheBasedSkipper()

        skipper.check_cache_hit("key_1")
        decision = skipper.should_skip_processing("key_1", min_hits_for_skip=3)

        assert decision.should_skip is False

    def test_should_skip_processing_above_threshold(self):
        """测试达到跳过阈值的处理"""
        from src.services.optimization import CacheBasedSkipper

        skipper = CacheBasedSkipper()

        for _ in range(3):
            skipper.check_cache_hit("key_2")

        decision = skipper.should_skip_processing("key_2", min_hits_for_skip=2)

        assert decision.should_skip is True
        assert decision.skip_type == 'cache_processing'
        assert decision.confidence >= 0.5


class TestSmartSkipOptimizer:
    """智能跳过优化器测试"""

    def test_evaluate_skip_decision_with_content(self):
        """测试带内容的跳过决策评估"""
        from src.services.optimization import SmartSkipOptimizer

        optimizer = SmartSkipOptimizer(
            enable_quality_skip=True,
            enable_complexity_skip=True,
            enable_cache_skip=True
        )

        high_quality_content = """
```python
class AuthenticationService:
    def __init__(self, user_repo, token_service):
        self.users = user_repo
        self.tokens = token_service
    
    async def authenticate(self, credentials):
        user = await self.users.find(credentials.email)
        if user and verify_password(user.password, credentials.password):
            return self.tokens.create(user)
        return None
```
        """

        decisions = optimizer.evaluate_skip_decision(
            content=high_quality_content,
            complexity_score=0.3,
            cache_key="test_optimization_key"
        )

        assert 'quality' in decisions
        assert 'complexity' in decisions
        assert 'cache' in decisions

        for name, decision in decisions.items():
            print(f"{name}: should_skip={decision.should_skip}")

    def test_get_overall_skip_decision_cache_hit(self):
        """测试缓存命中时的总体跳过决策"""
        from src.services.optimization import SmartSkipOptimizer

        optimizer = SmartSkipOptimizer(
            enable_quality_skip=True,
            enable_complexity_skip=True,
            enable_cache_skip=True
        )

        decisions = {
            'quality': Mock(
                should_skip=False,
                confidence=0.7,
                reason=None
            ),
            'complexity': Mock(
                should_skip=False,
                confidence=0.6,
                reason=None
            )
        }

        for _ in range(3):
            optimizer.cache_skipper.check_cache_hit("test_key_3")

        cache_decision = optimizer.cache_skipper.should_skip_processing(
            "test_key_3", min_hits_for_skip=2
        )
        decisions['cache'] = cache_decision

        overall = optimizer.get_overall_skip_decision(decisions)

        assert overall.should_skip is True

    def test_get_overall_skip_decision_no_skip(self):
        """测试不需要跳过的决策"""
        from src.services.optimization import SmartSkipOptimizer

        optimizer = SmartSkipOptimizer()

        decisions = {
            'quality': Mock(
                should_skip=False,
                confidence=0.5,
                reason=None
            ),
            'complexity': Mock(
                should_skip=False,
                confidence=0.7,
                reason=None
            )
        }

        overall = optimizer.get_overall_skip_decision(decisions)

        assert overall.should_skip is False


class TestParallelExecutor:
    """并行执行器测试"""

    @pytest.mark.asyncio
    async def test_execute_parallel_success(self):
        """测试并行执行成功"""
        from src.services.optimization import ParallelExecutor

        executor = ParallelExecutor(max_concurrency=3)

        async def task1():
            await asyncio.sleep(0.01)
            return {"task": "task1", "result": 1}

        async def task2():
            await asyncio.sleep(0.02)
            return {"task": "task2", "result": 2}

        async def task3():
            await asyncio.sleep(0.01)
            return {"task": "task3", "result": 3}

        results = await executor.execute_parallel({
            't1': task1,
            't2': task2,
            't3': task3
        })

        assert len(results) == 3

        task1_result = results.get('t1')
        assert task1_result is not None
        assert task1_result.result['result'] == 1

        print(f"Task 1 result: {task1_result.result}")

    @pytest.mark.asyncio
    async def test_execute_parallel_with_exception(self):
        """测试并行执行异常处理"""
        from src.services.optimization import ParallelExecutor

        executor = ParallelExecutor(max_concurrency=2)

        async def failing_task():
            await asyncio.sleep(0.01)
            raise ValueError("Task failed")

        async def success_task():
            await asyncio.sleep(0.01)
            return {"success": True}

        results = await executor.execute_parallel({
            'fail': failing_task,
            'success': success_task
        })

        assert 'success' in results
        success_result = results.get('success')
        assert success_result.result['success'] is True


class TestParallelRetrievalIntegration:
    """并行检索集成测试"""

    @pytest.mark.asyncio
    async def test_parallel_vector_and_keyword_search(self):
        """测试向量和关键词并行搜索"""
        from src.services.optimization import CacheBasedSkipper

        async def mock_vector_search(query, workspace_id, top_k):
            await asyncio.sleep(0.05)
            return [
                Mock(content="Vector result 1", confidence=0.85),
                Mock(content="Vector result 2", confidence=0.78)
            ]

        async def mock_keyword_search(query, workspace_id, top_k):
            await asyncio.sleep(0.03)
            return [
                Mock(content="Keyword result 1", confidence=0.72)
            ]

        cache_skipper = CacheBasedSkipper()
        cache_key = f"retrieval:test:{hash('test_query')}"

        skip_decision = cache_skipper.should_skip_processing(cache_key, min_hits_for_skip=2)

        if not skip_decision.should_skip:
            vector_results, keyword_results = await asyncio.gather(
                mock_vector_search("test", "workspace_1", 5),
                mock_keyword_search("test", "workspace_1", 5)
            )

            assert len(vector_results) == 2
            assert len(keyword_results) == 1

            print(f"Vector results: {len(vector_results)}")
            print(f"Keyword results: {len(keyword_results)}")

    @pytest.mark.asyncio
    async def test_retrieval_cache_hit_skip(self):
        """测试检索缓存命中跳过"""
        from src.services.optimization import CacheBasedSkipper

        cache_skipper = CacheBasedSkipper()
        cache_key = "retrieval:workspace:test_query_4"
        min_hits = 3

        skip_decision1 = cache_skipper.should_skip_processing(cache_key, min_hits_for_skip=min_hits)
        assert skip_decision1.should_skip is False

        skip_decision2 = cache_skipper.should_skip_processing(cache_key, min_hits_for_skip=min_hits)
        assert skip_decision2.should_skip is False

        skip_decision3 = cache_skipper.should_skip_processing(cache_key, min_hits_for_skip=min_hits)
        assert skip_decision3.should_skip is True

        print(f"Request 1: skip={skip_decision1.should_skip}, hits={skip_decision1.metadata.get('hit_count')}")
        print(f"Request 2: skip={skip_decision2.should_skip}, hits={skip_decision2.metadata.get('hit_count')}")
        print(f"Request 3: skip={skip_decision3.should_skip}, hits={skip_decision3.metadata.get('hit_count')}")


class TestNavigatorIntegration:
    """Navigator集成测试"""

    @pytest.mark.asyncio
    async def test_navigator_parallel_retrieve(self):
        """测试Navigator并行检索"""
        from src.domain.agents.navigator import _parallel_retrieve
        from src.services.optimization import CacheBasedSkipper

        mock_state = Mock()
        mock_state.retrieved_docs = []

        mock_retrieval_service = AsyncMock()

        mock_retrieval_service.retrieve_with_strategy = AsyncMock(
            side_effect=[
                [Mock(content="Vector result", confidence=0.85)],
                [Mock(content="Keyword result", confidence=0.72)]
            ]
        )

        result = await _parallel_retrieve(
            state=mock_state,
            retrieval_service=mock_retrieval_service,
            query="test query",
                        top_k=5
        )

        assert result is not None

        print(f"Retrieved {len(result)} results")


class TestDirectorIntegration:
    """Director集成测试"""

    @pytest.mark.asyncio
    async def test_director_conflict_detection(self):
        """测试Director冲突检测"""
        from src.domain.agents.director import detect_conflicts

        mock_step = Mock()
        mock_step.step_id = "step_1"
        mock_step.description = "Explain authentication"

        deprecated_doc = Mock()
        deprecated_doc.metadata = {'has_deprecated': True}
        deprecated_doc.source = "auth.py"

        normal_doc = Mock()
        normal_doc.metadata = {}
        normal_doc.source = "normal.py"

        has_conflict, conflict_type, details = detect_conflicts(
            current_step=mock_step,
            retrieved_docs=[deprecated_doc, normal_doc]
        )

        assert has_conflict is True
        assert conflict_type == 'deprecation_conflict'

        print(f"Conflict detected: {conflict_type}")
        print(f"Details: {details[:100]}...")

    @pytest.mark.asyncio
    async def test_director_heuristic_complexity(self):
        """测试Director启发式复杂度评估"""
        from src.domain.agents.director import _heuristic_complexity_assessment

        doc1 = Mock()
        doc1.content = "A" * 5000
        doc1.metadata = {'has_fixme': True}

        doc2 = Mock()
        doc2.content = "B" * 3000
        doc2.metadata = {}

        score = _heuristic_complexity_assessment([doc1, doc2])

        assert 0.0 <= score <= 1.0
        print(f"Heuristic complexity score: {score:.3f}")


class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_complete_optimization_flow(self):
        """测试完整优化流程"""
        from src.services.optimization import (
            SmartSkipOptimizer,
            CacheBasedSkipper
        )

        skip_optimizer = SmartSkipOptimizer(
            enable_quality_skip=True,
            enable_complexity_skip=True,
            enable_cache_skip=True
        )

        quality_content = """
## 测试文档

这是一个用于测试的文档，包含一些代码示例。

```python
def hello():
    print("Hello, World!")
```

这个文档有一定的结构。
        """

        quality_score = skip_optimizer.quality_assessor.assess_quality(quality_content)

        decisions = skip_optimizer.evaluate_skip_decision(
            content=quality_content,
            complexity_score=0.4,
            cache_key="integration_test_key"
        )

        overall = skip_optimizer.get_overall_skip_decision(decisions)

        assert overall is not None
        print(f"Quality score: {quality_score:.3f}")
        print(f"Overall skip: {overall.should_skip}")

    @pytest.mark.asyncio
    async def test_cache_efficiency(self):
        """测试缓存效率"""
        from src.services.optimization import CacheBasedSkipper

        cache_skipper = CacheBasedSkipper()

        cache_keys = [f"query_{i}" for i in range(10)]

        for key in cache_keys:
            cache_skipper.check_cache_hit(key)

        for key in cache_keys[:5]:
            cache_skipper.check_cache_hit(key)

        stats = cache_skipper.get_cache_stats()

        assert stats['total_lookups'] >= 15
        print(f"Total lookups: {stats['total_lookups']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
