"""Mock Services - 测试用的 Mock 服务实现

提供所有核心接口的 Mock 实现，用于单元测试和集成测试。
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .interfaces import (
    IService,
    IServiceStatus,
    IRetrievalService,
    IRetrievalStrategy,
    IQueryResult,
    IDocument,
    ILLMService,
    IRAGService,
    ICacheService,
    IStorageService,
    IMonitoringService,
)


@dataclass
class MockDocument(IDocument):
    """Mock 文档实现"""
    def __init__(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None
    ):
        super().__init__(id, content, metadata, score)


@dataclass
class MockQueryResult(IQueryResult):
    """Mock 查询结果实现"""
    def __init__(
        self,
        documents: List[IDocument],
        query: str,
        total_count: int = 0,
        execution_time_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(documents, query, total_count, execution_time_ms, metadata)


class MockRetrievalService(IRetrievalService):
    """Mock 检索服务 - 用于测试"""

    def __init__(
        self,
        documents: Optional[List[Dict[str, Any]]] = None,
        fail_on_retrieve: bool = False
    ):
        self._documents = documents or []
        self.fail_on_retrieve = fail_on_retrieve
        self.call_count = 0
        self.last_query = None

    @property
    def status(self) -> IServiceStatus:
        return IServiceStatus.HEALTHY

    async def health_check(self) -> bool:
        return True

    async def retrieve(
        self,
        query: str,
        strategy: IRetrievalStrategy = IRetrievalStrategy.AUTO,
        top_k: int = 10,
        **kwargs
    ) -> IQueryResult:
        self.call_count += 1
        self.last_query = query

        if self.fail_on_retrieve:
            from .errors import RetrievalServiceError
            raise RetrievalServiceError(
                message="Mock retrieval failure",
                error_code="MOCK_RETRIEVAL_ERROR"
            )

        documents = []
        for doc_data in self._documents[:top_k]:
            doc = MockDocument(
                id=doc_data.get("id", f"doc_{len(documents)}"),
                content=doc_data.get("content", ""),
                metadata=doc_data.get("metadata", {}),
                score=doc_data.get("score", 1.0)
            )
            documents.append(doc)

        return MockQueryResult(
            documents=documents,
            query=query,
            total_count=len(documents)
        )

    async def retrieve_with_filter(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 10,
        **kwargs
    ) -> IQueryResult:
        filtered_docs = [
            doc for doc in self._documents
            if all(doc.get(k) == v for k, v in filters.items())
        ]
        return await self.retrieve(query, top_k=top_k, **kwargs)

    async def optimize_query(self, query: str) -> str:
        """优化查询"""
        return query

    async def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        **kwargs
    ) -> List[IQueryResult]:
        return [await self.retrieve(q, top_k=top_k, **kwargs) for q in queries]


class MockLLMService(ILLMService):
    """Mock LLM 服务 - 用于测试"""

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        fail_on_generate: bool = False
    ):
        self._responses = responses or {}
        self.fail_on_generate = fail_on_generate
        self.call_count = 0
        self.last_prompt = None

    @property
    def status(self) -> IServiceStatus:
        return IServiceStatus.HEALTHY

    async def health_check(self) -> bool:
        return True

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        self.call_count += 1
        self.last_prompt = prompt

        if self.fail_on_generate:
            from .errors import LLMServiceError
            raise LLMServiceError(
                message="Mock LLM generation failure",
                error_code="MOCK_LLM_ERROR"
            )

        return self._responses.get(prompt, f"Mock response to: {prompt[:50]}...")

    async def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        self.call_count += 1
        content = messages[-1].get("content", "") if messages else ""
        return f"Mock response to messages: {content[:50]}..."

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4

    async def get_embedding(self, text: str) -> List[float]:
        self.call_count += 1
        return [0.1] * 10

    async def generate_stream(self, prompt: str, **kwargs):
        response = await self.generate(prompt, **kwargs)
        for chunk in response:
            yield chunk


class MockRAGService(IRAGService):
    """Mock RAG 服务 - 用于测试"""

    def __init__(self, retrieval_service: Optional[IRetrievalService] = None):
        self._retrieval_service = retrieval_service or MockRetrievalService()
        self.call_count = 0

    @property
    def status(self) -> IServiceStatus:
        return IServiceStatus.HEALTHY

    async def health_check(self) -> bool:
        return True

    async def rag_retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> IQueryResult:
        self.call_count += 1
        return await self._retrieval_service.retrieve(query, top_k=top_k, **kwargs)

    async def rag_generate(
        self,
        query: str,
        context: str,
        **kwargs
    ) -> str:
        return f"Generated response for '{query}' based on context: {context[:100]}..."

    async def generate_with_retrieval(
        self,
        query: str,
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        self.call_count += 1
        retrieval_result = await self._retrieval_service.retrieve(query, top_k=5)
        rag_context = context or "\n".join([doc.content for doc in retrieval_result.documents])
        generated = await self.rag_generate(query, rag_context)
        return {
            "response": generated,
            "context_used": rag_context,
            "retrieved_docs": len(retrieval_result.documents)
        }

    async def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> IQueryResult:
        self.call_count += 1
        return await self._retrieval_service.retrieve(query, top_k=top_k, **kwargs)

    async def get_relevant_context(
        self,
        query: str,
        max_length: int = 4000,
        **kwargs
    ) -> str:
        self.call_count += 1
        result = await self._retrieval_service.retrieve(query, top_k=5)
        context = "\n".join([doc.content for doc in result.documents])
        return context[:max_length]


class MockCacheService(ICacheService):
    """Mock 缓存服务 - 用于测试"""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self.call_count = 0

    @property
    def status(self) -> IServiceStatus:
        return IServiceStatus.HEALTHY

    async def health_check(self) -> bool:
        return True

    async def get(self, key: str) -> Optional[Any]:
        self.call_count += 1
        return self._cache.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self.call_count += 1
        self._cache[key] = value
        return True

    async def delete(self, key: str) -> bool:
        self.call_count += 1
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    async def clear(self) -> bool:
        self.call_count += 1
        self._cache.clear()
        return True

    async def exists(self, key: str) -> bool:
        self.call_count += 1
        return key in self._cache


class MockStorageService(IStorageService):
    """Mock 存储服务 - 用于测试"""

    def __init__(self):
        self._files: Dict[str, bytes] = {}
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._collections: set = set(["default"])
        self.call_count = 0

    @property
    def status(self) -> IServiceStatus:
        return IServiceStatus.HEALTHY

    async def health_check(self) -> bool:
        return True

    async def save(
        self,
        path: str,
        content: bytes,
        **kwargs
    ) -> bool:
        self.call_count += 1
        self._files[path] = content
        return True

    async def load(self, path: str) -> Optional[bytes]:
        self.call_count += 1
        return self._files.get(path)

    async def delete(self, path: str) -> bool:
        self.call_count += 1
        if path in self._files:
            del self._files[path]
            return True
        return False

    async def exists(self, path: str) -> bool:
        self.call_count += 1
        return path in self._files

    async def store_document(
        self,
        document: IDocument,
        collection: str = "default",
        **kwargs
    ) -> bool:
        self.call_count += 1
        self._documents[document.id] = {
            "id": document.id,
            "content": document.content,
            "metadata": document.metadata,
            "collection": collection
        }
        self._collections.add(collection)
        return True

    async def store_documents(
        self,
        documents: List[IDocument],
        collection: str = "default",
        **kwargs
    ) -> int:
        self.call_count += 1
        count = 0
        for doc in documents:
            if await self.store_document(doc, collection, **kwargs):
                count += 1
        return count

    async def delete_document(
        self,
        doc_id: str,
        collection: str = "default",
        **kwargs
    ) -> bool:
        self.call_count += 1
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False

    async def list_collections(self) -> List[str]:
        self.call_count += 1
        return list(self._collections)

    async def get_document_count(
        self,
        collection: str = "default"
    ) -> int:
        self.call_count += 1
        return sum(1 for doc in self._documents.values() if doc.get("collection") == collection)


class MockMonitoringService(IMonitoringService):
    """Mock 监控服务 - 用于测试"""

    def __init__(self):
        self._metrics: Dict[str, Any] = {}
        self._logs: List[Dict[str, Any]] = []
        self._error_count: int = 0
        self.call_count = 0

    @property
    def status(self) -> IServiceStatus:
        return IServiceStatus.HEALTHY

    async def health_check(self) -> bool:
        return True

    async def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        self.call_count += 1
        self._metrics[name] = {
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.now().isoformat()
        }

    async def get_metric(self, name: str) -> Optional[float]:
        self.call_count += 1
        if name in self._metrics:
            return self._metrics[name].get("value")
        return None

    async def get_metrics(
        self,
        name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        self.call_count += 1
        metrics = []
        for metric_name, metric_data in self._metrics.items():
            if name and metric_name != name:
                continue
            metrics.append({
                "name": metric_name,
                **metric_data
            })
        return metrics[:limit]

    async def log_event(self, event_type: str, data: Dict[str, Any], **kwargs):
        self.call_count += 1
        self._logs.append({
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

    async def record_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        self.call_count += 1
        self._error_count += 1
        self._logs.append({
            "type": "error",
            "error": str(error),
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        })

    async def get_error_rate(
        self,
        time_window: int = 3600
    ) -> float:
        self.call_count += 1
        return self._error_count


def create_mock_services() -> Dict[str, IService]:
    """创建一套完整的 Mock 服务"""
    return {
        "retrieval": MockRetrievalService(),
        "llm": MockLLMService(),
        "rag": MockRAGService(),
        "cache": MockCacheService(),
        "storage": MockStorageService(),
        "monitoring": MockMonitoringService(),
    }
