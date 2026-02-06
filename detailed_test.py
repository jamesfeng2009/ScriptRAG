"""Detailed test to trace workflow execution"""

import asyncio
import logging
from unittest.mock import Mock
from src.domain.models import SharedState, OutlineStep
from src.application.orchestrator import WorkflowOrchestrator
from tests.fixtures.realistic_mock_data import (
    create_mock_llm_service,
    create_mock_retrieval_service,
    create_mock_parser_service
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_summarization_service():
    from unittest.mock import Mock
    mock_service = Mock()
    mock_service.check_size = Mock(return_value=False)
    return mock_service

async def detailed_test():
    print("=== Detailed Workflow Test ===\n")

    mock_llm = create_mock_llm_service()
    mock_retrieval = create_mock_retrieval_service()
    mock_parser = create_mock_parser_service()
    mock_summarization = create_mock_summarization_service()

    outline = [
        OutlineStep(
            step_id=0,
            title="第一步",
            description="Python 异步编程基础",
            status="pending",
            retry_count=0
        ),
        OutlineStep(
            step_id=1,
            title="第二步",
            description="Python 异步编程进阶",
            status="pending",
            retry_count=0
        )
    ]

    initial_state = SharedState(
        user_topic="Python异步编程",
        project_context="技术教程",
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

    print("Step 1: Testing mock retrieval directly...")
    docs = await mock_retrieval.hybrid_retrieve("test-workspace", "Python 异步编程基础", 3)
    print(f"  Retrieved {len(docs)} docs")

    print("\nStep 2: Running workflow (recursion_limit=30)...")
    try:
        result = await asyncio.wait_for(
            orchestrator.execute(initial_state, recursion_limit=30),
            timeout=20.0
        )
        print(f"\nWorkflow result: success={result.get('success')}")
        print(f"Current step: {result['state'].get('current_step_index')}")
        print(f"Fragments: {len(result['state'].get('fragments', []))}")
    except asyncio.TimeoutError:
        print("\n!!! TIMEOUT !!!")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(detailed_test())
