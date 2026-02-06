"""
服务接口定义 - 抽象基类 ABC

定义核心服务的接口契约，支持依赖注入和可测试性。

使用示例：
    from src.services.interfaces import IRetrievalService, ILLMService

    class MyRetrievalService(IRetrievalService):
        async def retrieve(self, query: str, **kwargs) -> IDocument:
            ...
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar
from dataclasses import dataclass
from enum import Enum


T = TypeVar("T")


class IDocument:
    """文档接口"""
    
    def __init__(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None
    ):
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.score = score
    
    def __repr__(self):
        return f"IDocument(id={self.id}, score={self.score})"


class IQueryResult:
    """查询结果接口"""
    
    def __init__(
        self,
        documents: List[IDocument],
        query: str,
        total_count: int = 0,
        execution_time_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.documents = documents
        self.query = query
        self.total_count = total_count
        self.execution_time_ms = execution_time_ms
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"IQueryResult(documents={len(self.documents)}, query={self.query})"


class IRetrievalStrategy(Enum):
    """检索策略"""
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    GRAPH = "graph"
    AUTO = "auto"


class IServiceStatus(Enum):
    """服务状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class IServiceConfig:
    """服务配置基类"""
    pass


class IService(ABC):
    """所有服务的基类接口"""
    
    @property
    @abstractmethod
    def status(self) -> IServiceStatus:
        """获取服务状态"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass


class IRetrievalService(IService):
    """
    检索服务接口
    
    定义检索操作的标准接口，支持多种检索策略。
    """
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        strategy: IRetrievalStrategy = IRetrievalStrategy.AUTO,
        top_k: int = 10,
        **kwargs
    ) -> IQueryResult:
        """
        执行检索
        
        Args:
            query: 查询字符串
            strategy: 检索策略
            top_k: 返回结果数量
            **kwargs: 其他参数
            
        Returns:
            IQueryResult: 查询结果
        """
        pass
    
    @abstractmethod
    async def retrieve_with_filter(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 10,
        **kwargs
    ) -> IQueryResult:
        """
        带过滤条件的检索
        
        Args:
            query: 查询字符串
            filters: 过滤条件
            top_k: 返回结果数量
            **kwargs: 其他参数
            
        Returns:
            IQueryResult: 查询结果
        """
        pass
    
    @abstractmethod
    async def optimize_query(self, query: str) -> str:
        """
        优化查询
        
        Args:
            query: 原始查询
            
        Returns:
            str: 优化后的查询
        """
        pass
    
    @abstractmethod
    async def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        **kwargs
    ) -> List[IQueryResult]:
        """
        批量检索
        
        Args:
            queries: 查询列表
            top_k: 返回结果数量
            **kwargs: 其他参数
            
        Returns:
            List[IQueryResult]: 结果列表
        """
        pass


class ILLMService(IService):
    """
    LLM 服务接口
    
    定义与大语言模型交互的标准接口。
    """
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 提示词
            max_tokens: 最大 token 数
            temperature: 温度参数
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        pass
    
    @abstractmethod
    async def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        使用消息格式生成文本
        
        Args:
            messages: 消息列表
            max_tokens: 最大 token 数
            temperature: 温度参数
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        计算 token 数量
        
        Args:
            text: 输入文本
            
        Returns:
            int: token 数量
        """
        pass
    
    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """
        获取文本嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 嵌入向量
        """
        pass


class IRAGService(IService):
    """
    RAG 服务接口
    
    定义检索增强生成的标准接口。
    """
    
    @abstractmethod
    async def generate_with_retrieval(
        self,
        query: str,
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        基于检索生成
        
        Args:
            query: 查询
            context: 额外上下文
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 生成结果和元数据
        """
        pass
    
    @abstractmethod
    async def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> IQueryResult:
        """
        检索并重排序
        
        Args:
            query: 查询
            top_k: 返回数量
            **kwargs: 其他参数
            
        Returns:
            IQueryResult: 重排序后的结果
        """
        pass
    
    @abstractmethod
    async def get_relevant_context(
        self,
        query: str,
        max_length: int = 4000,
        **kwargs
    ) -> str:
        """
        获取相关上下文
        
        Args:
            query: 查询
            max_length: 最大长度
            **kwargs: 其他参数
            
        Returns:
            str: 上下文文本
        """
        pass


class ICacheService(IService):
    """
    缓存服务接口
    
    定义缓存操作的标准接口。
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存值
        """
        pass
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
            
        Returns:
            bool: 是否成功
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否成功
        """
        pass
    
    @abstractmethod
    async def clear(self, pattern: Optional[str] = None) -> int:
        """
        清除缓存
        
        Args:
            pattern: 匹配模式
            
        Returns:
            int: 删除的键数量
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        检查键是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否存在
        """


class IStorageService(IService):
    """
    存储服务接口
    
    定义文档存储操作的标准接口。
    """
    
    @abstractmethod
    async def store_document(
        self,
        document: IDocument,
        collection: str = "default",
        **kwargs
    ) -> bool:
        """
        存储文档
        
        Args:
            document: 文档
            collection: 集合名称
            **kwargs: 其他参数
            
        Returns:
            bool: 是否成功
        """
        pass
    
    @abstractmethod
    async def store_documents(
        self,
        documents: List[IDocument],
        collection: str = "default",
        **kwargs
    ) -> int:
        """
        批量存储文档
        
        Args:
            documents: 文档列表
            collection: 集合名称
            **kwargs: 其他参数
            
        Returns:
            int: 成功存储的数量
        """
        pass
    
    @abstractmethod
    async def delete_document(
        self,
        doc_id: str,
        collection: str = "default",
        **kwargs
    ) -> bool:
        """
        删除文档
        
        Args:
            doc_id: 文档 ID
            collection: 集合名称
            **kwargs: 其他参数
            
        Returns:
            bool: 是否成功
        """
        pass
    
    @abstractmethod
    async def list_collections(self) -> List[str]:
        """
        列出集合
        
        Returns:
            List[str]: 集合名称列表
        """
        pass
    
    @abstractmethod
    async def get_document_count(
        self,
        collection: str = "default"
    ) -> int:
        """
        获取文档数量
        
        Args:
            collection: 集合名称
            
        Returns:
            int: 文档数量
        """
        pass


class IMonitoringService(IService):
    """
    监控服务接口
    
    定义监控和指标操作的标准接口。
    """
    
    @abstractmethod
    async def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        记录指标
        
        Args:
            name: 指标名称
            value: 指标值
            tags: 标签
        """
        pass
    
    @abstractmethod
    async def get_metrics(
        self,
        name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取指标
        
        Args:
            name: 指标名称
            tags: 标签
            limit: 限制数量
            
        Returns:
            List[Dict[str, Any]]: 指标列表
        """
        pass
    
    @abstractmethod
    async def record_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        记录错误
        
        Args:
            error: 异常
            context: 上下文
        """
        pass
    
    @abstractmethod
    async def get_error_rate(
        self,
        time_window: int = 3600
    ) -> float:
        """
        获取错误率
        
        Args:
            time_window: 时间窗口（秒）
            
        Returns:
            float: 错误率
        """
        pass


__all__ = [
    # 基本类型
    "IDocument",
    "IQueryResult",
    "IRetrievalStrategy",
    "IServiceStatus",
    "IServiceConfig",
    
    # 接口
    "IService",
    "IRetrievalService",
    "ILLMService",
    "IRAGService",
    "ICacheService",
    "IStorageService",
    "IMonitoringService",
]
