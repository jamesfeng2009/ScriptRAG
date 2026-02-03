#!/usr/bin/env python3
"""
Test script for all database tables via API

This script tests:
1. document_embeddings - via existing /documents API
2. document_chunks - auto-created when indexing long documents
3. knowledge_nodes - via /knowledge/nodes API
4. knowledge_relations - via /knowledge/relations API
5. retrieval_logs - via /retrieval-logs API
6. api_usage_stats - via /api-usage API

Usage:
    python test_db_tables.py [--base-url http://localhost:8000]
"""

import argparse
import asyncio
import httpx
import json
import sys
from datetime import datetime
from typing import Optional


BASE_URL = "http://localhost:8000"
WORKSPACE_ID = "test-workspace"
KEEP_DATA = False


class APITester:
    def __init__(self, base_url: str, keep_data: bool = False):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.keep_data = keep_data
    
    async def close(self):
        await self.client.aclose()
    
    async def health_check(self) -> bool:
        """Check if API is running"""
        try:
            r = await self.client.get(f"{self.base_url}/health")
            return r.status_code == 200
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    async def test_documents_table(self) -> bool:
        """Test document_embeddings and document_chunks tables"""
        print("\n📄 Testing document_embeddings table...")
        
        success = True
        
        try:
            content = '''
def calculate_fibonacci(n: int) -> int:
    """Calculate Fibonacci number recursively with memoization"""
    if n <= 1:
        return n
    
    # Use memoization to avoid redundant calculations
    cache = {}
    
    def fib_with_cache(k: int) -> int:
        if k in cache:
            return cache[k]
        if k <= 1:
            result = k
        else:
            result = fib_with_cache(k-1) + fib_with_cache(k-2)
        cache[k] = result
        return result
    
    return fib_with_cache(n)

@deprecated
def old_calculate_fibonacci(n: int) -> int:
    """Old implementation without memoization"""
    if n <= 1:
        return n
    return old_calculate_fibonacci(n-1) + old_calculate_fibonacci(n-2)
'''
            
            r = await self.client.post(
                f"{self.base_url}/documents",
                data={
                    "content": content,
                    "file_name": f"test_fibonacci_{datetime.now().timestamp()}.py",
                    "title": "Test Fibonacci Code"
                }
            )
            
            if r.status_code == 200:
                data = r.json()
                print(f"  ✅ Document indexed: {data.get('doc_id', 'unknown')[:8]}...")
                
                search_r = await self.client.get(
                    f"{self.base_url}/documents/search",
                    params={"query": "fibonacci function", "top_k": 5}
                )
                
                if search_r.status_code == 200:
                    search_data = search_r.json()
                    print(f"  ✅ Document search: found {search_data['total_results']} results")
                else:
                    print(f"  ⚠️  Document search failed: {search_r.status_code}")
            else:
                print(f"  ❌ Document indexing failed: {r.status_code}")
                print(f"     {r.text[:200]}")
                success = False
                
        except Exception as e:
            print(f"  ❌ Document test error: {e}")
            success = False
        
        return success
    
    async def test_document_chunks_table(self) -> bool:
        """Test document_chunks table"""
        print("\n📦 Testing document_chunks table...")
        
        success = True
        created_chunk_ids = []
        doc_id = None
        
        try:
            print("  📄 First creating a document for chunking...")
            doc_content = '''def calculate_fibonacci(n: int) -> int:
    """Calculate Fibonacci number recursively with memoization"""
    if n <= 1:
        return n
    
    cache = {}
    
    def fib_with_cache(k: int) -> int:
        if k in cache:
            return cache[k]
        if k <= 1:
            result = k
        else:
            result = fib_with_cache(k-1) + fib_with_cache(k-2)
        cache[k] = result
        return result
    
    return fib_with_cache(n)'''
            
            r = await self.client.post(
                f"{self.base_url}/documents",
                files={"file": ("test.py", doc_content, "text/plain")},
                data={"title": "fibonacci_test.py", "file_name": "fibonacci_test.py"}
            )
            
            if r.status_code == 200:
                doc_data = r.json()
                doc_id = doc_data.get("doc_id")
                print(f"  ✅ Document created: {doc_id[:8]}...")
            else:
                print(f"  ⚠️  Document creation failed: {r.status_code}, trying JSON...")
                r = await self.client.post(
                    f"{self.base_url}/documents",
                    json={
                        "title": "fibonacci_test.py",
                        "file_name": "fibonacci_test.py",
                        "content": doc_content
                    }
                )
                if r.status_code == 200:
                    doc_data = r.json()
                    doc_id = doc_data.get("doc_id")
                    print(f"  ✅ Document created: {doc_id[:8]}...")
                else:
                    print(f"  ❌ Document creation failed: {r.status_code}")
                    doc_id = "00000000-0000-0000-0000-000000000001"
            
            chunks = [
                {
                    "document_id": doc_id,
                    "chunk_index": 0,
                    "content": "def calculate_fibonacci(n: int) -> int:\n    \"\"\"Calculate Fibonacci number recursively with memoization\"\"\"\n    if n <= 1:\n        return n",
                    "start_line": 1,
                    "end_line": 5,
                    "embedding": [0.1] * 1024
                },
                {
                    "document_id": doc_id,
                    "chunk_index": 1,
                    "content": "    cache = {}\n    \n    def fib_with_cache(k: int) -> int:\n        if k in cache:\n            return cache[k]",
                    "start_line": 6,
                    "end_line": 10,
                    "embedding": [0.2] * 1024
                },
                {
                    "document_id": doc_id,
                    "chunk_index": 2,
                    "content": "        if k <= 1:\n            result = k\n        else:\n            result = fib_with_cache(k-1) + fib_with_cache(k-2)\n        cache[k] = result\n        return result\n    \n    return fib_with_cache(n)",
                    "start_line": 11,
                    "end_line": 18,
                    "embedding": [0.3] * 1024
                }
            ]
            
            for chunk_data in chunks:
                r = await self.client.post(
                    f"{self.base_url}/document-chunks",
                    json=chunk_data
                )
                
                if r.status_code == 200:
                    data = r.json()
                    chunk_id = data.get("chunk_id")
                    created_chunk_ids.append(chunk_id)
                    print(f"  ✅ Chunk created: index={chunk_data['chunk_index']} ({chunk_id[:8]}...)")
                else:
                    print(f"  ❌ Chunk creation failed: {r.status_code}")
                    print(f"     {r.text[:200]}")
                    success = False
            
            if created_chunk_ids:
                chunk_id = created_chunk_ids[0]
                r = await self.client.get(f"{self.base_url}/document-chunks/{chunk_id}")
                if r.status_code == 200:
                    print(f"  ✅ Chunk retrieved: index=0")
                else:
                    print(f"  ⚠️  Chunk retrieval failed: {r.status_code}")
            
            r = await self.client.get(f"{self.base_url}/document-chunks?limit=10")
            if r.status_code == 200:
                data = r.json()
                print(f"  ✅ Chunk list: {data['total_count']} chunks")
            else:
                print(f"  ⚠️  Chunk list failed: {r.status_code}")
            
            r = await self.client.get(f"{self.base_url}/document-chunks/stats")
            if r.status_code == 200:
                stats = r.json()
                print(f"  ✅ Chunk stats: {stats['total_chunks']} total chunks")
            else:
                print(f"  ⚠️  Chunk stats failed: {r.status_code}")
            
            for chunk_id in created_chunk_ids:
                if not self.keep_data:
                    r = await self.client.delete(f"{self.base_url}/document-chunks/{chunk_id}")
                    if r.status_code == 200:
                        print(f"  ✅ Chunk deleted: {chunk_id[:8]}...")
                else:
                    print(f"  ✅ Chunk kept: {chunk_id[:8]}...")
        
        except Exception as e:
            print(f"  ❌ Document chunks test error: {e}")
            success = False
        
        return success
    
    async def test_knowledge_nodes_table(self) -> bool:
        """Test knowledge_nodes table"""
        print("\n🧠 Testing knowledge_nodes table...")
        
        success = True
        created_node_ids = []
        
        try:
            nodes = [
                {
                    "workspace_id": WORKSPACE_ID,
                    "node_type": "function",
                    "name": "calculate_fibonacci",
                    "content": "Main Fibonacci calculation function with memoization",
                    "properties": {"complexity": "O(n)", "language": "python"},
                    "source_file": "fibonacci.py",
                    "line_number": 1
                },
                {
                    "workspace_id": WORKSPACE_ID,
                    "node_type": "function",
                    "name": "old_calculate_fibonacci",
                    "content": "Old Fibonacci implementation without optimization",
                    "properties": {"complexity": "O(2^n)", "deprecated": True},
                    "source_file": "fibonacci.py",
                    "line_number": 25
                },
                {
                    "workspace_id": WORKSPACE_ID,
                    "node_type": "concept",
                    "name": "Memoization",
                    "content": "Optimization technique to cache results of expensive function calls",
                    "properties": {"category": "algorithm"}
                }
            ]
            
            for node_data in nodes:
                r = await self.client.post(
                    f"{self.base_url}/knowledge/nodes",
                    json=node_data
                )
                
                if r.status_code == 200:
                    data = r.json()
                    node_id = data.get("node_id")
                    created_node_ids.append(node_id)
                    print(f"  ✅ Node created: {node_data['name']} ({node_id[:8]}...)")
                else:
                    print(f"  ❌ Node creation failed: {r.status_code}")
                    print(f"     {r.text[:200]}")
                    success = False
            
            if created_node_ids:
                node_id = created_node_ids[0]
                r = await self.client.get(f"{self.base_url}/knowledge/nodes/{node_id}")
                if r.status_code == 200:
                    print(f"  ✅ Node retrieved: {r.json()['name']}")
                else:
                    print(f"  ⚠️  Node retrieval failed: {r.status_code}")
            
            r = await self.client.get(
                f"{self.base_url}/knowledge/nodes",
                params={"workspace_id": WORKSPACE_ID, "limit": 10}
            )
            if r.status_code == 200:
                nodes_list = r.json()
                print(f"  ✅ Node list: {len(nodes_list)} nodes")
            else:
                print(f"  ⚠️  Node list failed: {r.status_code}")
            
        except Exception as e:
            print(f"  ❌ Knowledge nodes test error: {e}")
            success = False
        
        return success, created_node_ids
    
    async def test_knowledge_relations_table(self, node_ids: list) -> bool:
        """Test knowledge_relations table"""
        print("\n🔗 Testing knowledge_relations table...")
        
        success = True
        
        try:
            if len(node_ids) < 2:
                print("  ⚠️  Not enough nodes to create relations")
                return True
            
            relations = [
                {
                    "workspace_id": WORKSPACE_ID,
                    "source_node_id": node_ids[0],
                    "target_node_id": node_ids[2],
                    "relation_type": "depends_on",
                    "strength": 0.9,
                    "properties": {"description": "Uses memoization concept"}
                },
                {
                    "workspace_id": WORKSPACE_ID,
                    "source_node_id": node_ids[1],
                    "target_node_id": node_ids[0],
                    "relation_type": "replaced_by",
                    "strength": 1.0,
                    "properties": {"replaced_in_version": "2.0"}
                }
            ]
            
            created_relation_ids = []
            for rel_data in relations:
                r = await self.client.post(
                    f"{self.base_url}/knowledge/relations",
                    json=rel_data
                )
                
                if r.status_code == 200:
                    data = r.json()
                    relation_id = data.get("relation_id")
                    created_relation_ids.append(relation_id)
                    print(f"  ✅ Relation created: {rel_data['relation_type']} ({relation_id[:8]}...)")
                else:
                    print(f"  ❌ Relation creation failed: {r.status_code}")
                    print(f"     {r.text[:200]}")
                    success = False
            
            r = await self.client.get(
                f"{self.base_url}/knowledge/relations",
                params={"workspace_id": WORKSPACE_ID, "limit": 10}
            )
            if r.status_code == 200:
                relations_list = r.json()
                print(f"  ✅ Relation list: {len(relations_list)} relations")
            else:
                print(f"  ⚠️  Relation list failed: {r.status_code}")
            
            r = await self.client.get(
                f"{self.base_url}/knowledge/stats",
                params={"workspace_id": WORKSPACE_ID}
            )
            if r.status_code == 200:
                stats = r.json()
                print(f"  ✅ Graph stats: {stats['total_nodes']} nodes, {stats['total_relations']} relations")
            else:
                print(f"  ⚠️  Graph stats failed: {r.status_code}")
            
            for rel_id in created_relation_ids:
                if not self.keep_data:
                    r = await self.client.delete(f"{self.base_url}/knowledge/relations/{rel_id}")
                    if r.status_code == 200:
                        print(f"  ✅ Relation deleted: {rel_id[:8]}...")
                else:
                    print(f"  ✅ Relation kept: {rel_id[:8]}...")
            
        except Exception as e:
            print(f"  ❌ Knowledge relations test error: {e}")
            success = False
        
        return success
    
    async def test_retrieval_logs_table(self) -> bool:
        """Test retrieval_logs table"""
        print("\n📊 Testing retrieval_logs table...")
        
        success = True
        
        try:
            logs = [
                {
                    "workspace_id": WORKSPACE_ID,
                    "query": "How to calculate fibonacci efficiently?",
                    "top_k": 5,
                    "result_count": 3,
                    "total_score": 2.8,
                    "search_strategy": "hybrid",
                    "duration_ms": 150,
                    "cost_usd": 0.0015,
                    "user_id": "test_user_1"
                },
                {
                    "workspace_id": WORKSPACE_ID,
                    "query": "What is memoization in Python?",
                    "top_k": 3,
                    "result_count": 5,
                    "total_score": 4.2,
                    "search_strategy": "vector",
                    "duration_ms": 80,
                    "cost_usd": 0.0008,
                    "user_id": "test_user_2"
                }
            ]
            
            created_log_ids = []
            for log_data in logs:
                r = await self.client.post(f"{self.base_url}/retrieval-logs", params=log_data)
                
                if r.status_code == 200:
                    data = r.json()
                    log_id = data.get("log_id")
                    created_log_ids.append(log_id)
                    print(f"  ✅ Log created: '{log_data['query'][:30]}...' ({log_id[:8]}...)")
                else:
                    print(f"  ❌ Log creation failed: {r.status_code}")
                    print(f"     {r.text[:200]}")
                    success = False
            
            r = await self.client.get(
                f"{self.base_url}/retrieval-logs",
                params={"workspace_id": WORKSPACE_ID, "days": 7, "limit": 10}
            )
            if r.status_code == 200:
                logs_list = r.json()
                print(f"  ✅ Log list: {len(logs_list)} logs")
            else:
                print(f"  ⚠️  Log list failed: {r.status_code}")
            
            r = await self.client.get(
                f"{self.base_url}/retrieval-logs/popular",
                params={"workspace_id": WORKSPACE_ID, "limit": 5}
            )
            if r.status_code == 200:
                popular = r.json()
                print(f"  ✅ Popular queries: {len(popular)} queries")
            else:
                print(f"  ⚠️  Popular queries failed: {r.status_code}")
            
            r = await self.client.get(
                f"{self.base_url}/retrieval-logs/stats",
                params={"workspace_id": WORKSPACE_ID, "days": 7}
            )
            if r.status_code == 200:
                stats = r.json()
                print(f"  ✅ Retrieval stats: {stats['total_retrievals']} total retrievals")
            else:
                print(f"  ⚠️  Retrieval stats failed: {r.status_code}")
            
        except Exception as e:
            print(f"  ❌ Retrieval logs test error: {e}")
            success = False
        
        return success
    
    async def test_api_usage_stats_table(self) -> bool:
        """Test api_usage_stats table"""
        print("\n💰 Testing api_usage_stats table...")
        
        success = True
        
        try:
            usage_records = [
                {
                    "provider": "openai",
                    "model": "gpt-4",
                    "operation": "chat_completion",
                    "prompt_tokens": 1500,
                    "completion_tokens": 500,
                    "cost_usd": 0.06,
                    "workspace_id": WORKSPACE_ID
                },
                {
                    "provider": "glm",
                    "model": "glm-4",
                    "operation": "chat_completion",
                    "prompt_tokens": 800,
                    "completion_tokens": 300,
                    "cost_usd": 0.015,
                    "workspace_id": WORKSPACE_ID
                },
                {
                    "provider": "qwen",
                    "model": "qwen-turbo",
                    "operation": "embedding",
                    "prompt_tokens": 100,
                    "completion_tokens": 0,
                    "cost_usd": 0.001,
                    "workspace_id": WORKSPACE_ID
                }
            ]
            
            for usage in usage_records:
                r = await self.client.post(f"{self.base_url}/api-usage", params=usage)
                
                if r.status_code == 200:
                    data = r.json()
                    print(f"  ✅ Usage recorded: {usage['provider']}/{usage['model']} ({data.get('record_id', 'unknown')[:8]}...)")
                else:
                    print(f"  ❌ Usage recording failed: {r.status_code}")
                    print(f"     {r.text[:200]}")
                    success = False
            
            r = await self.client.get(
                f"{self.base_url}/api-usage/stats",
                params={"workspace_id": WORKSPACE_ID, "days": 30}
            )
            if r.status_code == 200:
                stats = r.json()
                print(f"  ✅ API usage stats: {stats['total_tokens']} tokens, ${stats['total_cost_usd']:.4f}")
            else:
                print(f"  ⚠️  API usage stats failed: {r.status_code}")
            
            r = await self.client.get(
                f"{self.base_url}/api-usage/daily-cost",
                params={"workspace_id": WORKSPACE_ID, "days": 7}
            )
            if r.status_code == 200:
                daily = r.json()
                print(f"  ✅ Daily cost trend: {len(daily)} days of data")
            else:
                print(f"  ⚠️  Daily cost trend failed: {r.status_code}")
            
        except Exception as e:
            print(f"  ❌ API usage stats test error: {e}")
            success = False
        
        return success
    
    async def cleanup_test_data(self, node_ids: list):
        """Clean up test data"""
        print("\n🧹 Cleaning up test data...")
        
        try:
            for node_id in node_ids:
                r = await self.client.delete(f"{self.base_url}/knowledge/nodes/{node_id}")
                if r.status_code == 200:
                    print(f"  ✅ Node deleted: {node_id[:8]}...")
        
            r = await self.client.post(
                f"{self.base_url}/retrieval-logs/cleanup",
                params={"retention_days": 0}
            )
            if r.status_code == 200:
                data = r.json()
                print(f"  ✅ Cleanup: {data['deleted_count']} logs deleted")
        
        except Exception as e:
            print(f"  ⚠️  Cleanup error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Test database tables via API")
    parser.add_argument("--base-url", default=BASE_URL, help="Base URL of the API server")
    parser.add_argument("--keep-data", action="store_true", help="Keep test data after running tests")
    args = parser.parse_args()
    
    tester = APITester(args.base_url, keep_data=args.keep_data)
    
    print("=" * 60)
    print("🧪 Database Tables API Test")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")
    print(f"Workspace ID: {WORKSPACE_ID}")
    print(f"Keep test data: {args.keep_data}")
    print("=" * 60)
    
    if not await tester.health_check():
        print("\n❌ API is not available. Please start the server first.")
        print(f"   Run: uvicorn src.presentation.api:app --host 0.0.0.0 --port 8000")
        await tester.close()
        sys.exit(1)
    
    print("✅ API is available")
    
    results = {
        "documents": False,
        "document_chunks": False,
        "knowledge_nodes": False,
        "knowledge_relations": False,
        "retrieval_logs": False,
        "api_usage_stats": False
    }
    
    node_ids = []
    
    try:
        results["documents"] = await tester.test_documents_table()
        
        results["document_chunks"] = await tester.test_document_chunks_table()
        
        nodes_success, node_ids = await tester.test_knowledge_nodes_table()
        results["knowledge_nodes"] = nodes_success
        
        results["knowledge_relations"] = await tester.test_knowledge_relations_table(node_ids)
        
        results["retrieval_logs"] = await tester.test_retrieval_logs_table()
        
        results["api_usage_stats"] = await tester.test_api_usage_stats_table()
        
        if not args.keep_data:
            await tester.cleanup_test_data(node_ids)
        else:
            print("\n📊 Test data kept for inspection:")
            print(f"   - Knowledge nodes: {len(node_ids)} nodes")
            print(f"   - API endpoints available at: {args.base_url}/docs")
            
            print("\n📋 Data in tables:")
            nodes_r = await tester.client.get(f"{args.base_url}/knowledge/nodes", params={"workspace_id": WORKSPACE_ID, "limit": 10})
            if nodes_r.status_code == 200:
                nodes = nodes_r.json()
                print(f"   - knowledge_nodes: {len(nodes)} records")
            
            rels_r = await tester.client.get(f"{args.base_url}/knowledge/relations", params={"workspace_id": WORKSPACE_ID, "limit": 10})
            if rels_r.status_code == 200:
                rels = rels_r.json()
                print(f"   - knowledge_relations: {len(rels)} records")
            
            logs_r = await tester.client.get(f"{args.base_url}/retrieval-logs", params={"workspace_id": WORKSPACE_ID, "days": 7, "limit": 10})
            if logs_r.status_code == 200:
                logs = logs_r.json()
                print(f"   - retrieval_logs: {len(logs)} records")
            
            chunks_r = await tester.client.get(f"{args.base_url}/document-chunks/stats")
            if chunks_r.status_code == 200:
                chunks_stats = chunks_r.json()
                print(f"   - document_chunks: {chunks_stats.get('total_chunks', 0)} records")
            
            stats_r = await tester.client.get(f"{args.base_url}/api-usage/stats", params={"days": 30})
            if stats_r.status_code == 200:
                stats = stats_r.json()
                print(f"   - api_usage_stats: {stats.get('total_requests', 0)} requests")
            
            print("\n✅ All data is preserved in the database!")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await tester.close()
    
    print("\n" + "=" * 60)
    print("📋 Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for table, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {table}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
