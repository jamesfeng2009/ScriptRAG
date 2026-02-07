"""
真实数据检索服务

从圣斗士星矢知识库中检索相关内容，而不是使用 Mock 数据。
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.services.interfaces import IDocument, IQueryResult
from src.services.retrieval_service import RetrievalService as BaseRetrievalService


@dataclass
class KnowledgeDocument(IDocument):
    """知识库文档"""
    id: str
    content: str
    metadata: Dict[str, Any] = None
    score: float = None


@dataclass
class KnowledgeQueryResult(IQueryResult):
    """知识库查询结果"""
    documents: List[IDocument]
    query: str
    total_count: int = 0
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = None


class KnowledgeBaseRetrievalService:
    """
    从知识库文件检索内容的服务
    
    从 data/knowledge 目录加载 YAML/JSON 文件，支持基于关键词和内容的检索。
    """

    def __init__(self, knowledge_dir: str = None):
        """
        初始化知识库检索服务
        
        Args:
            knowledge_dir: 知识库目录路径，默认使用项目的 data/knowledge
        """
        if knowledge_dir is None:
            # 默认使用项目根目录下的 data/knowledge
            knowledge_dir = Path(__file__).parent.parent.parent / "data" / "knowledge"
        else:
            knowledge_dir = Path(knowledge_dir)
        
        self.knowledge_dir = Path(knowledge_dir)
        self.documents: Dict[str, KnowledgeDocument] = {}
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """加载知识库文件"""
        if not self.knowledge_dir.exists():
            print(f"⚠️ 知识库目录不存在: {self.knowledge_dir}")
            return
        
        print(f"📚 正在加载知识库: {self.knowledge_dir}")
        
        # 加载 JSON 文件
        json_file = self.knowledge_dir / "saint_seiya_knowledge.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                docs = json.load(f)
                for doc in docs:
                    self._add_document(doc)
        
        # 加载 YAML 文件
        for yaml_file in self.knowledge_dir.glob("*.yaml"):
            if yaml_file.name == "saint_seiya_knowledge.yaml":
                continue
            with open(yaml_file, 'r', encoding='utf-8') as f:
                doc = yaml.safe_load(f)
                if doc:
                    self._add_document(doc)
        
        print(f"✅ 已加载 {len(self.documents)} 条知识库文档")
    
    def _add_document(self, doc: Dict[str, Any]):
        """添加文档到知识库"""
        doc_id = doc.get('id', doc.get('title', 'unknown'))
        
        document = KnowledgeDocument(
            id=doc_id,
            content=doc.get('content', ''),
            metadata=doc.get('metadata', {}),
            score=1.0
        )
        
        self.documents[doc_id] = document
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ) -> KnowledgeQueryResult:
        """
        检索知识库
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件
        
        Returns:
            查询结果
        """
        import time
        start_time = time.time()
        
        # 基于关键词匹配
        query_keywords = query.lower().split()
        
        # 如果没有空格（连续中文），尝试按字符匹配
        if len(query_keywords) == 1 and len(query_keywords[0]) > 4:
            # 长中文查询，尝试按字符和词组匹配
            query_keywords = [
                query.lower(),
                query.lower()[:len(query)//2],
                query.lower()[len(query)//2:]
            ]
        
        scored_docs = []
        
        for doc in self.documents.values():
            # 计算匹配分数
            score = 0.0
            content_lower = doc.content.lower()
            metadata = doc.metadata or {}
            
            # 标题匹配
            title = doc.id.lower()
            for keyword in query_keywords:
                if keyword in title:
                    score += 3.0
            
            # 内容关键词匹配
            for keyword in query_keywords:
                if keyword in content_lower:
                    score += 1.0
            
            # 元数据匹配
            for keyword in query_keywords:
                if keyword in str(metadata).lower():
                    score += 2.0
            
            # 标签匹配
            tags = metadata.get('tags', [])
            for keyword in query_keywords:
                for tag in tags:
                    if keyword in tag.lower():
                        score += 1.5
            
            # 如果有内容匹配，增加基础分
            if any(keyword in content_lower for keyword in query_keywords):
                score += 0.5
            
            if score > 0:
                # 创建副本，设置分数
                doc_copy = KnowledgeDocument(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=score
                )
                scored_docs.append((doc_copy, score))
        
        # 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 取 top_k
        top_docs = [doc for doc, score in scored_docs[:top_k]]
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return KnowledgeQueryResult(
            documents=top_docs,
            query=query,
            total_count=len(top_docs),
            execution_time_ms=execution_time_ms,
            metadata={"filters": filters}
        )
    
    async def async_retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ) -> KnowledgeQueryResult:
        """
        异步检索知识库
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件
        
        Returns:
            查询结果
        """
        return self.retrieve(query, top_k, filters)
    
    def retrieve_with_strategy(
        self,
        workspace_id: str,
        query: str,
        strategy_name: str,
        top_k: int = 5
    ) -> KnowledgeQueryResult:
        """
        使用指定策略检索
        
        Args:
            workspace_id: 工作区ID（用于多租户支持）
            query: 查询文本
            strategy_name: 策略名称（vector_search, keyword_search, hybrid）
            top_k: 返回结果数量
        
        Returns:
            查询结果
        """
        return self.retrieve(query=query, top_k=top_k)
    
    def get_document(self, doc_id: str) -> Optional[KnowledgeDocument]:
        """
        获取指定文档
        
        Args:
            doc_id: 文档ID
        
        Returns:
            文档对象，不存在返回 None
        """
        return self.documents.get(doc_id)
    
    def list_documents(self, category: str = None) -> List[KnowledgeDocument]:
        """
        列出知识库文档
        
        Args:
            category: 可选，按分类过滤
        
        Returns:
            文档列表
        """
        docs = list(self.documents.values())
        
        if category:
            docs = [
                doc for doc in docs
                if doc.metadata and doc.metadata.get('category') == category
            ]
        
        return docs


def create_knowledge_retrieval_service() -> KnowledgeBaseRetrievalService:
    """
    创建知识库检索服务
    
    Returns:
        知识库检索服务实例
    """
    return KnowledgeBaseRetrievalService()


if __name__ == "__main__":
    # 测试知识库检索服务
    service = create_knowledge_retrieval_service()
    
    print("\n📚 知识库检索测试")
    print("=" * 60)
    
    # 测试查询
    queries = [
        "星矢 狮子宫 艾欧里亚",
        "热血战斗 友情",
        "天马流星拳"
    ]
    
    for query in queries:
        print(f"\n🔍 查询: {query}")
        result = service.retrieve(query, top_k=3)
        print(f"   找到 {result.total_count} 条结果")
        for doc in result.documents:
            print(f"   - {doc.id}: {doc.metadata.get('tags', [])[:3]}")
    
    print("\n" + "=" * 60)
    print(f"📊 知识库文档总数: {len(service.documents)}")
