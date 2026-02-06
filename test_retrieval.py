"""Quick test to diagnose retrieval service"""

import asyncio
from tests.fixtures.realistic_mock_data import create_realistic_retrieval_results, create_mock_retrieval_service

async def test_retrieval():
    print("Testing create_realistic_retrieval_results...")
    results = create_realistic_retrieval_results("Python async", 3)
    print(f"Direct call returned {len(results)} results")
    for i, r in enumerate(results):
        print(f"  Result {i}: id={r.id}, file={r.file_path}")

    print("\nTesting mock retrieval service...")
    mock_retrieval = create_mock_retrieval_service()
    results2 = await mock_retrieval.hybrid_retrieve("test-workspace", "Python async", 3)
    print(f"hybrid_retrieve returned {len(results2)} results")
    for i, r in enumerate(results2):
        print(f"  Result {i}: id={r.id}, file={r.file_path}")

if __name__ == "__main__":
    asyncio.run(test_retrieval())
