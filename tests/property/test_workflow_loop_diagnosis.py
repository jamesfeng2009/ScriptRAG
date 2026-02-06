"""
诊断脚本：追踪工作流循环位置

目的：找出工作流在哪个节点/条件上循环
"""
import asyncio
import sys
import os
import logging
from datetime import datetime
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.domain.state_types import GlobalState
from src.domain.models import RetrievedDocument, OutlineStep
from src.domain.task_stack import TaskContext
from src.infrastructure.langgraph_error_handler import ErrorCategory
from src.application.orchestrator import WorkflowOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_llm():
    """创建 Mock LLM 服务"""
    mock = Mock()
    mock.generate = AsyncMock(return_value={
        "content": "这是一个测试内容",
        "confidence": 0.9,
        "metadata": {"model": "test"}
    })
    mock.agenerate = AsyncMock(return_value={
        "content": "这是一个测试内容",
        "confidence": 0.9,
        "metadata": {"model": "test"}
    })
    mock.structured_generate = AsyncMock(return_value={
        "primary_intent": "explanation",
        "keywords": ["测试"],
        "search_sources": ["web"],
        "confidence": 0.9,
        "intent_type": "lightweight"
    })
    return mock

def create_mock_retrieval():
    """创建 Mock 检索服务 - 故意返回低质量结果"""
    mock = Mock()
    
    async def mock_search(query: str, top_k: int = 5, filters: Dict = None) -> List[Dict]:
        # 故意返回低质量结果（短文本）来触发重试
        return [
            {
                "content": "短内容",
                "score": 0.3,  # 低分数
                "metadata": {"source": "test", "file_path": "/test/doc1.txt"}
            }
        ]
    
    async def mock_vector_search(query: str, top_k: int = 5, filters: Dict = None) -> List[Dict]:
        return mock_search(query, top_k, filters)
    
    async def mock_keyword_search(query: str, top_k: int = 5, filters: Dict = None) -> List[Dict]:
        return mock_search(query, top_k, filters)
    
    async def mock_retrieve_with_strategy(query: str, strategy: str = "hybrid", top_k: int = 5) -> List[Dict]:
        return mock_search(query, top_k, filters)
    
    mock.search = mock_search
    mock.vector_search = mock_vector_search
    mock.keyword_search = mock_keyword_search
    mock.retrieve_with_strategy = mock_retrieve_with_strategy
    
    return mock

def create_mock_parser():
    """创建 Mock 解析服务"""
    mock = Mock()
    mock.parse_outline = Mock(return_value=[
        {"step_id": 0, "title": "步骤 1", "description": "第一步描述"},
        {"step_id": 1, "title": "步骤 2", "description": "第二步描述"},
    ])
    mock.parse_query = Mock(return_value={
        "primary_intent": "informational",
        "keywords": ["测试"],
        "search_sources": ["web"],
        "confidence": 0.9,
        "intent_type": "lightweight"
    })
    mock.validate_format = Mock(return_value={"valid": True, "errors": []})
    return mock

def create_mock_summarization():
    """创建 Mock 总结服务"""
    mock = Mock()
    mock.check_size = Mock(return_value=False)  # 总是返回 False，表示内容在限制内
    mock.summarize = AsyncMock(return_value={"summary": "测试总结", "confidence": 0.9})
    return mock

def create_test_state() -> GlobalState:
    """创建测试状态"""
    return {
        "user_topic": "测试主题",
        "project_context": "测试上下文",
        "outline": [
            {
                "step_id": 0,
                "title": "步骤 1: 介绍",
                "description": "介绍测试主题的背景信息",
                "status": "pending"
            },
            {
                "step_id": 1,
                "title": "步骤 2: 详细说明",
                "description": "详细说明测试主题的各个方面",
                "status": "pending"
            },
            {
                "step_id": 2,
                "title": "步骤 3: 总结",
                "description": "总结测试主题的关键点",
                "status": "pending"
            }
        ],
        "current_step_index": 0,
        "fragments": [],
        "retrieved_docs": [],
        "director_feedback": None,
        "execution_log": [],
        "session_id": "test_session",
        "user_id": "test_user"
    }

class ExecutionTracer:
    """执行追踪器 - 记录每次节点调用"""
    
    def __init__(self):
        self.call_history = []
        self.step_visits = {}  # 记录每个步骤被访问的次数
        self.node_visits = {}  # 记录每个节点被访问的次数
    
    def record(self, node_name: str, state: Dict[str, Any], decision: str = None):
        step_index = state.get("current_step_index", -1)
        timestamp = datetime.now().isoformat()
        
        # 记录节点访问
        if node_name not in self.node_visits:
            self.node_visits[node_name] = 0
        self.node_visits[node_name] += 1
        
        # 记录步骤访问
        if step_index not in self.step_visits:
            self.step_visits[step_index] = 0
        self.step_visits[step_index] += 1
        
        # 记录详细历史
        feedback = state.get("director_feedback") or {}
        entry = {
            "timestamp": timestamp,
            "node": node_name,
            "step_index": step_index,
            "decision": decision,
            "retry_count": state.get("retrieval_retry_count", 0),
            "quality_score": feedback.get("metadata", {}).get("quality_score", None) if feedback else None,
            "has_retrieved_docs": len(state.get("retrieved_docs", [])) > 0
        }
        self.call_history.append(entry)
        
        # 打印追踪信息
        print(f"\n{'='*60}")
        print(f"[{timestamp}] 节点: {node_name}")
        print(f"  步骤索引: {step_index}")
        print(f"  重试次数: {entry['retry_count']}")
        print(f"  质量分数: {entry['quality_score']}")
        print(f"  有检索结果: {entry['has_retrieved_docs']}")
        if decision:
            print(f"  决策: {decision}")
        print(f"  节点访问次数: {self.node_visits[node_name]}")
        print(f"  步骤 {step_index} 访问次数: {self.step_visits.get(step_index, 0)}")
        print(f"{'='*60}")
        
        # 检测循环
        if self.step_visits.get(step_index, 0) > 3:
            print(f"\n⚠️  警告: 步骤 {step_index} 已被访问超过 3 次！")
            print(f"    可能存在循环！")
        
        if self.node_visits.get(node_name, 0) > 10:
            print(f"\n⚠️  警告: 节点 {node_name} 已被访问超过 10 次！")
            print(f"    可能存在死循环！")
    
    def summary(self):
        print("\n" + "="*60)
        print("执行追踪摘要")
        print("="*60)
        print(f"总节点调用次数: {len(self.call_history)}")
        print("\n节点访问统计:")
        for node, count in sorted(self.node_visits.items(), key=lambda x: -x[1]):
            print(f"  {node}: {count} 次")
        print("\n步骤访问统计:")
        for step, count in sorted(self.step_visits.items(), key=lambda x: -x[0]):
            status = "🔴 循环" if count > 3 else "✓"
            print(f"  步骤 {step}: {count} 次 {status}")
        print("="*60)

async def trace_workflow_execution():
    """追踪工作流执行"""
    print("\n" + "="*60)
    print("开始工作流执行追踪")
    print("="*60)
    
    # 创建 Mock 服务
    mock_llm = create_mock_llm()
    mock_retrieval = create_mock_retrieval()
    mock_parser = create_mock_parser()
    mock_summarization = create_mock_summarization()
    
    # 创建追踪器
    tracer = ExecutionTracer()
    
    # 创建 Orchestrator
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        enable_agentic_rag=True,  # 启用 Agentic RAG
        enable_dynamic_adjustment=False,
        max_retrieval_retries=3  # 允许更多重试以便观察
    )
    
    # 初始状态
    initial_state = create_test_state()
    tracer.record("INITIAL", initial_state)
    
    # 设置执行超时
    import signal
    
    def timeout_handler(signum, frame):
        print("\n" + "="*60)
        print("⏰ 执行超时！")
        print("="*60)
        tracer.summary()
        raise asyncio.TimeoutError("执行超时")
    
    # 设置 30 秒超时
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        # 执行工作流
        result = await orchestrator.execute(initial_state)
        
        print("\n" + "="*60)
        print("✓ 工作流执行完成")
        print("="*60)
        
        tracer.summary()
        
        return result
        
    except asyncio.TimeoutError:
        print("\n⚠️  工作流执行超时！")
        tracer.summary()
        raise
        
    except Exception as e:
        print(f"\n✗ 工作流执行出错: {e}")
        tracer.summary()
        raise
        
    finally:
        signal.alarm(0)  # 取消超时

if __name__ == "__main__":
    asyncio.run(trace_workflow_execution())
