"""RAG Service - 用户问答流水线

功能：
- 查询预处理
- 向量检索
- Rerank 重排序
- LLM 生成回答
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.config import get_database_config
from src.services.database.postgres import PostgresService
from src.services.llm.service import LLMService
from src.services.reranker import MultiFactorReranker
from src.services.retrieval.strategies import RetrievalResult


logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """问答结果"""
    answer: str
    sources: List[Dict[str, Any]]


class RAGService:
    """RAG 服务 - 用户问答流水线

    流程：
    1. 查询预处理
    2. 向量检索
    3. Rerank 重排序
    4. LLM 生成回答
    """

    def __init__(
        self,
        llm_service: LLMService = None,
        vector_store: PostgresService = None,
        reranker: MultiFactorReranker = None,
        workspace_id: str = "default",
        top_k: int = 10,
        rerank_top_k: int = 5
    ):
        """
        初始化 RAG 服务

        Args:
            llm_service: LLM 服务
            vector_store: 向量数据库服务
            reranker: 重排序器
            workspace_id: 工作空间 ID
            top_k: 检索返回数量
            rerank_top_k: Rerank 后返回数量
        """
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.reranker = reranker or MultiFactorReranker()
        self.workspace_id = workspace_id
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

        if self.llm_service is None:
            self.llm_service = LLMService()

        if self.vector_store is None:
            db_config = get_database_config()
            self.vector_store = PostgresService({
                'host': db_config.host,
                'port': db_config.port,
                'database': db_config.database,
                'user': db_config.user,
                'password': db_config.password
            })

        logger.info("RAGService initialized")

    async def initialize(self) -> None:
        """初始化服务"""
        await self.vector_store.connect()

    async def close(self) -> None:
        """关闭服务"""
        await self.vector_store.disconnect()

    async def query(
        self,
        question: str,
        history: List[Dict[str, str]] = None
    ) -> QueryResult:
        """
        问答查询

        Args:
            question: 用户问题
            history: 对话历史（可选）

        Returns:
            问答结果
        """
        try:
            logger.info(f"处理问题: {question[:50]}...")

            processed_query = self._preprocess_query(question, history)

            query_emb = await self.llm_service.embed([processed_query])
            query_emb = query_emb[0]

            results = await self._vector_search(query_emb)

            if not results:
                return QueryResult(
                    answer="未找到相关信息",
                    sources=[]
                )

            reranked = self._rerank(processed_query, results)

            context = self._build_context(reranked)

            answer = await self._generate(processed_query, context)

            sources = [
                {
                    "text": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "score": round(r.metadata.get('rerank_score', r.similarity), 4),
                    "file_path": r.file_path
                }
                for r in reranked[:self.rerank_top_k]
            ]

            logger.info(f"问答完成，来源数: {len(sources)}")
            return QueryResult(answer=answer, sources=sources)

        except Exception as e:
            logger.error(f"问答查询失败: {e}")
            return QueryResult(
                answer=f"处理问题时发生错误: {str(e)}",
                sources=[]
            )

    def _preprocess_query(
        self,
        question: str,
        history: List[Dict[str, str]] = None
    ) -> str:
        """查询预处理"""
        query = " ".join(question.split())

        if history and len(history) > 0:
            context_parts = []
            for h in history[-2:]:
                if 'question' in h:
                    context_parts.append(h['question'])
            if context_parts:
                query = " | ".join(context_parts) + " | " + query

        return query

    async def _vector_search(
        self,
        query_embedding: List[float],
        limit: int = None
    ) -> List[RetrievalResult]:
        """向量检索"""
        limit = limit or self.top_k

        try:
            rows = await self.vector_store.fetch(
                """
                SELECT * FROM search_similar_documents($1, $2, $3, $4)
                """,
                self.workspace_id,
                query_embedding,
                limit * 2,
                0.0
            )

            results = []
            for row in rows:
                results.append(RetrievalResult(
                    id=str(row['id']),
                    source="vector",
                    file_path=row['file_path'],
                    content=row['content'],
                    workspace_id=self.workspace_id,
                    similarity=row['similarity'],
                    confidence=row['similarity'],
                    metadata={}
                ))

            logger.info(f"向量检索返回 {len(results)} 结果")
            return results

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def _rerank(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank 重排序"""
        if not results:
            return results

        reranked = self.reranker.rerank(
            query=query,
            results=results,
            top_k=len(results)
        )

        logger.info(f"Rerank 完成，Top1 分数: {reranked[0].confidence:.4f}")
        return reranked

    def _build_context(self, results: List[RetrievalResult]) -> str:
        """构建上下文"""
        parts = []
        for i, r in enumerate(results):
            parts.append(f"[来源{i+1}] {r.content}")

        return "\n\n".join(parts)

    async def _generate(self, question: str, context: str) -> str:
        """LLM 生成回答"""
        prompt = f"""你是一个专业的信息检索助手。请严格遵循以下规则：

## 回答规则
1. 仅基于参考信息回答，不要添加外部知识
2. 如果参考信息不足以回答问题，明确说"未找到相关信息"
3. 回答要简洁准确，不要冗长

## 参考信息
{context}

## 用户问题
{question}

## 回答
"""

        try:
            response = await self.llm_service.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )

            return response

        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            return f"生成回答时发生错误: {str(e)}"

    async def search_similar(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """仅检索（不生成回答）"""
        query_emb = await self.llm_service.embed([query])
        query_emb = query_emb[0]

        results = await self._vector_search(query_emb, limit)

        return [
            {
                "content": r.content,
                "score": r.similarity,
                "file_path": r.file_path
            }
            for r in results
        ]


async def create_rag_service(workspace_id: str = "default") -> RAGService:
    """工厂方法：创建 RAG 服务"""
    db_config = get_database_config()

    pg_service = PostgresService({
        'host': db_config.host,
        'port': db_config.port,
        'database': db_config.database,
        'user': db_config.user,
        'password': db_config.password
    })
    await pg_service.connect()

    return RAGService(
        vector_store=pg_service,
        workspace_id=workspace_id
    )
