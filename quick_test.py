"""Quick test to diagnose workflow execution issues"""

import asyncio
from unittest.mock import Mock
from src.domain.models import SharedState, OutlineStep
from src.application.orchestrator import WorkflowOrchestrator
from tests.fixtures.realistic_mock_data import (
    create_mock_llm_service,
    create_mock_retrieval_service,
    create_mock_parser_service
)

def create_mock_summarization_service():
    from unittest.mock import Mock
    mock_service = Mock()
    mock_service.check_size = Mock(return_value=False)
    return mock_service

async def quick_test():
    print("Creating mock services...")
    mock_llm = create_mock_llm_service()
    mock_retrieval = create_mock_retrieval_service()
    mock_parser = create_mock_parser_service()
    mock_summarization = create_mock_summarization_service()

    outline = [
        OutlineStep(
            step_id=0,
            title="第一步",
            description="测试主题的第一步内容",
            status="pending",
            retry_count=0
        ),
        OutlineStep(
            step_id=1,
            title="第二步",
            description="测试主题的第二步内容",
            status="pending",
            retry_count=0
        )
    ]

    initial_state = SharedState(
        user_topic="测试主题",
        project_context="测试上下文",
        outline=outline,
        current_step_index=0,
        retrieved_docs=[],
        fragments=[],
        current_skill="standard_tutorial",
        global_tone="professional",
        pivot_triggered=False,
        pivot_reason=None,
        max_retries=3,
        awaiting_user_input=False,
        user_input_prompt=None,
        execution_log=[],
        fact_check_passed=True
    )

    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        workspace_id="test-workspace"
    )

    print("Starting workflow execution (limit=50)...")
    try:
        result = await asyncio.wait_for(
            orchestrator.execute(initial_state, recursion_limit=50),
            timeout=30.0
        )
        print(f"Workflow completed!")
        print(f"Success: {result.get('success')}")
        print(f"Final step index: {result['state'].get('current_step_index')}")
        print(f"Fragments count: {len(result['state'].get('fragments', []))}")
        print(f"Pivot triggered: {result['state'].get('pivot_triggered')}")
    except asyncio.TimeoutError:
        print("TIMEOUT: Workflow execution took too long!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(quick_test())
