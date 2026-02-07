"""
通用知识库检索服务

支持多主题动态切换，根据主题自动加载对应的知识库。

设计原则：
1. 主题无关的核心检索逻辑
2. 根据运行时上下文动态加载知识库
3. 支持知识库目录结构和配置文件
4. 保持与现有 RetrievalService 接口兼容
"""

import json
import yaml
import glob
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from src.services.interfaces import IDocument, IQueryResult
from src.services.retrieval_service import RetrievalService as BaseRetrievalService

logger = logging.getLogger(__name__)


class MatchingStrategy(Enum):
    """匹配策略"""
    KEYWORD = "keyword"  # 关键词匹配
    SEMANTIC = "semantic"  # 语义匹配
    HYBRID = "hybrid"  # 混合匹配


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


@dataclass
class ThemeConfig:
    """主题配置"""
    theme_id: str
    name: str
    description: str
    knowledge_dir: str
    keywords: List[str] = field(default_factory=list)
    default_skill: str = "default"


class UniversalKnowledgeRetrievalService:
    """
    通用知识库检索服务

    功能：
    - 支持多主题动态切换
    - 根据主题自动加载对应知识库
    - 多种检索策略（关键词、语义、混合）
    - 主题无关的核心检索逻辑

    使用方式：
    1. 单主题模式：直接加载指定目录
    2. 多主题模式：根据主题名称动态切换
    3. 自动检测模式：根据查询内容自动识别主题
    """

    def __init__(
        self,
        base_knowledge_dir: str = None,
        default_theme: str = "general",
        enable_theme_detection: bool = True
    ):
        """
        初始化通用知识库检索服务

        Args:
            base_knowledge_dir: 知识库基础目录
            default_theme: 默认主题
            enable_theme_detection: 是否自动检测主题
        """
        if base_knowledge_dir is None:
            # 默认使用项目根目录下的 data/knowledge
            base_knowledge_dir = Path(__file__).parent.parent.parent / "data" / "knowledge"
            # 检查目录是否存在，如果不存在则使用备选路径
            if not base_knowledge_dir.exists():
                base_knowledge_dir = Path.cwd() / "data" / "knowledge"
        
        self.base_knowledge_dir = Path(base_knowledge_dir)
        self.default_theme = default_theme
        self.enable_theme_detection = enable_theme_detection

        # 已加载的知识库缓存 {theme_id: {doc_id: KnowledgeDocument}}
        self._knowledge_cache: Dict[str, Dict[str, KnowledgeDocument]] = {}

        # 主题配置缓存 {theme_id: ThemeConfig}
        self._theme_configs: Dict[str, ThemeConfig] = {}

        # 当前激活的主题
        self._current_theme: Optional[str] = None

        logger.info(f"Knowledge base directory: {self.base_knowledge_dir}")
        
        # 加载主题配置
        self._load_theme_configs()
        
        # 预加载默认主题
        self._ensure_theme_loaded(default_theme)

    def _load_theme_configs(self):
        """加载所有主题配置"""
        config_file = self.base_knowledge_dir / "themes.json"
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                configs = json.load(f)
                for config_data in configs:
                    theme = ThemeConfig(
                        theme_id=config_data['theme_id'],
                        name=config_data['name'],
                        description=config_data.get('description', ''),
                        knowledge_dir=config_data.get('knowledge_dir', config_data['theme_id']),
                        keywords=config_data.get('keywords', []),
                        default_skill=config_data.get('default_skill', 'default')
                    )
                    self._theme_configs[theme.theme_id] = theme
                    logger.info(f"Loaded theme config: {theme.name} ({theme.theme_id})")

    def set_theme(self, theme_id: str) -> bool:
        """
        设置当前主题

        Args:
            theme_id: 主题ID

        Returns:
            是否设置成功
        """
        if theme_id in self._theme_configs or self._ensure_theme_loaded(theme_id):
            self._current_theme = theme_id
            logger.info(f"Switched to theme: {theme_id}")
            return True
        return False

    def get_current_theme(self) -> Optional[str]:
        """获取当前主题"""
        return self._current_theme

    def get_available_themes(self) -> List[ThemeConfig]:
        """获取所有可用主题"""
        return list(self._theme_configs.values())

    def _ensure_theme_loaded(self, theme_id: str) -> bool:
        """
        确保指定主题的知识库已加载

        Args:
            theme_id: 主题ID

        Returns:
            是否加载成功
        """
        if theme_id in self._knowledge_cache:
            return True

        theme_dir = self.base_knowledge_dir / theme_id
        if not theme_dir.exists():
            # 尝试加载通用知识库
            if theme_id != self.default_theme:
                return self._ensure_theme_loaded(self.default_theme)
            return False

        # 加载主题知识库
        docs = self._load_knowledge_from_dir(theme_dir)
        if docs:
            self._knowledge_cache[theme_id] = docs
            logger.info(f"Loaded {len(docs)} documents for theme: {theme_id}")
            return True

        return False

    def _load_knowledge_from_dir(self, dir_path: Path) -> Dict[str, KnowledgeDocument]:
        """
        从目录加载知识库文档

        Args:
            dir_path: 目录路径

        Returns:
            文档字典 {doc_id: KnowledgeDocument}
        """
        docs = {}

        # 加载 JSON 文件
        for json_file in dir_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for doc in data:
                            self._add_knowledge_document(docs, doc)
                    else:
                        self._add_knowledge_document(docs, data)
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")

        # 加载 YAML 文件
        for yaml_file in dir_path.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, list):
                        for doc in data:
                            self._add_knowledge_document(docs, doc)
                    else:
                        self._add_knowledge_document(docs, data)
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")

        return docs

    def _add_knowledge_document(
        self,
        docs: Dict[str, KnowledgeDocument],
        doc_data: Dict[str, Any]
    ):
        """添加知识文档到字典"""
        doc_id = doc_data.get('id', doc_data.get('title', 'unknown'))
        
        document = KnowledgeDocument(
            id=doc_id,
            content=doc_data.get('content', ''),
            metadata=doc_data.get('metadata', {}),
            score=1.0
        )
        docs[doc_id] = document

    def detect_theme(self, query: str) -> str:
        """
        根据查询自动检测主题

        Args:
            query: 查询文本

        Returns:
            检测到的主题ID
        """
        query_lower = query.lower()

        if not self.enable_theme_detection or not self._theme_configs:
            return self.default_theme

        best_match = None
        best_score = 0

        for theme_id, config in self._theme_configs.items():
            score = 0
            keywords = config.keywords

            if isinstance(keywords, list):
                for kw_group in keywords:
                    if isinstance(kw_group, dict):
                        words = kw_group.get('words', [])
                        priority = kw_group.get('priority', 1)
                    else:
                        words = [kw_group]
                        priority = 1
                    for keyword in words:
                        if keyword.lower() in query_lower:
                            score += (1 + (10 - priority) * 0.1)
            else:
                for keyword in keywords:
                    if keyword.lower() in query_lower:
                        score += 1

            if score > best_score:
                best_score = score
                best_match = theme_id

        return best_match or self.default_theme

    def add_theme_knowledge(
        self,
        theme_id: str,
        documents: List[Dict[str, Any]]
    ):
        """
        动态添加知识到指定主题

        Args:
            theme_id: 主题ID
            documents: 文档列表
        """
        if theme_id not in self._knowledge_cache:
            self._knowledge_cache[theme_id] = {}

        for doc in documents:
            self._add_knowledge_document(self._knowledge_cache[theme_id], doc)

        logger.info(f"Added {len(documents)} documents to theme: {theme_id}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None,
        theme: str = None,
        strategy: MatchingStrategy = MatchingStrategy.KEYWORD
    ) -> KnowledgeQueryResult:
        """
        检索知识库

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件
            theme: 指定主题，默认使用当前主题或自动检测
            strategy: 匹配策略

        Returns:
            查询结果
        """
        import time
        start_time = time.time()

        # 确定使用的主题
        if theme is None:
            if self._current_theme:
                theme = self._current_theme
            else:
                theme = self.detect_theme(query)

        # 确保主题已加载
        self._ensure_theme_loaded(theme)

        # 获取知识库
        knowledge_base = self._knowledge_cache.get(theme, {})

        if not knowledge_base:
            logger.warning(f"No knowledge found for theme: {theme}")
            return KnowledgeQueryResult(
                documents=[],
                query=query,
                total_count=0,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"theme": theme, "filters": filters}
            )

        # 执行检索
        scored_docs = self._search_knowledge(
            knowledge_base=knowledge_base,
            query=query,
            top_k=top_k,
            strategy=strategy,
            filters=filters
        )

        # 取 top_k
        top_docs = [doc for doc, score in scored_docs[:top_k]]

        return KnowledgeQueryResult(
            documents=top_docs,
            query=query,
            total_count=len(top_docs),
            execution_time_ms=(time.time() - start_time) * 1000,
            metadata={"theme": theme, "filters": filters, "strategy": strategy.value}
        )

    def _search_knowledge(
        self,
        knowledge_base: Dict[str, KnowledgeDocument],
        query: str,
        top_k: int,
        strategy: MatchingStrategy,
        filters: Dict[str, Any] = None
    ) -> List[tuple]:
        """
        执行知识检索

        Args:
            knowledge_base: 知识库
            query: 查询
            top_k: 返回数量
            strategy: 匹配策略
            filters: 过滤条件

        Returns:
            排序后的文档列表 [(doc, score), ...]
        """
        # 解析查询关键词
        query_keywords = self._parse_query_keywords(query)

        scored_docs = []

        for doc in knowledge_base.values():
            # 应用过滤器
            if filters and not self._apply_filters(doc, filters):
                continue

            # 计算匹配分数
            score = self._calculate_match_score(
                doc=doc,
                query=query,
                query_keywords=query_keywords,
                strategy=strategy
            )

            if score > 0:
                scored_docs.append((doc, score))

        # 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs

    def _parse_query_keywords(self, query: str) -> List[str]:
        """解析查询关键词"""
        keywords = query.lower().split()

        # 处理中文查询
        if len(keywords) == 1 and len(keywords[0]) > 4:
            keywords = [
                query.lower(),
                query.lower()[:len(query) // 2],
                query.lower()[len(query) // 2:]
            ]

        return keywords

    def _calculate_match_score(
        self,
        doc: KnowledgeDocument,
        query: str,
        query_keywords: List[str],
        strategy: MatchingStrategy
    ) -> float:
        """
        计算文档与查询的匹配分数

        Args:
            doc: 文档
            query: 原始查询
            query_keywords: 解析后的关键词
            strategy: 匹配策略

        Returns:
            匹配分数
        """
        score = 0.0
        content_lower = doc.content.lower()
        metadata = doc.metadata or {}

        if strategy == MatchingStrategy.KEYWORD:
            # 关键词匹配策略
            for keyword in query_keywords:
                if keyword in doc.id.lower():
                    score += 3.0
                if keyword in content_lower:
                    score += 1.0
                if keyword in str(metadata).lower():
                    score += 2.0

                tags = metadata.get('tags', [])
                for tag in tags:
                    if keyword in tag.lower():
                        score += 1.5

        elif strategy == MatchingStrategy.SEMANTIC:
            # 简化语义匹配（实际应该用向量模型）
            # 这里用关键词近似
            for keyword in query_keywords:
                if keyword in content_lower[:500]:
                    score += 2.0

        else:  # HYBRID
            # 混合匹配
            keyword_score = 0.0
            for keyword in query_keywords:
                if keyword in doc.id.lower():
                    keyword_score += 3.0
                if keyword in content_lower:
                    keyword_score += 1.0
                if keyword in str(metadata).lower():
                    keyword_score += 2.0

                tags = metadata.get('tags', [])
                for tag in tags:
                    if keyword in tag.lower():
                        keyword_score += 1.5

            score = keyword_score * 0.7 + (len(query_keywords) / 10) * 0.3

        return score

    def _apply_filters(
        self,
        doc: KnowledgeDocument,
        filters: Dict[str, Any]
    ) -> bool:
        """应用过滤器"""
        metadata = doc.metadata or {}

        for key, value in filters.items():
            if key == 'category':
                if metadata.get('category') != value:
                    return False
            elif key == 'tags':
                if not all(tag in metadata.get('tags', []) for tag in value):
                    return False
            elif key == 'source':
                if metadata.get('source') != value:
                    return False

        return True

    def async_retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None,
        theme: str = None
    ) -> KnowledgeQueryResult:
        """
        异步检索接口

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件
            theme: 指定主题

        Returns:
            查询结果
        """
        return self.retrieve(query, top_k, filters, theme)

    def retrieve_with_strategy(
        self,
        workspace_id: str,
        query: str,
        strategy_name: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ) -> KnowledgeQueryResult:
        """
        使用指定策略检索

        Args:
            workspace_id: 工作区ID
            query: 查询文本
            strategy_name: 策略名称
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            查询结果
        """
        strategy = MatchingStrategy(strategy_name)
        return self.retrieve(query, top_k, filters, strategy=strategy)

    def get_document(self, doc_id: str, theme: str = None) -> Optional[KnowledgeDocument]:
        """
        获取指定文档

        Args:
            doc_id: 文档ID
            theme: 主题

        Returns:
            文档对象
        """
        if theme is None:
            theme = self._current_theme or self.default_theme

        knowledge_base = self._knowledge_cache.get(theme, {})
        return knowledge_base.get(doc_id)

    def list_documents(
        self,
        theme: str = None,
        category: str = None
    ) -> List[KnowledgeDocument]:
        """
        列出文档

        Args:
            theme: 主题
            category: 分类过滤

        Returns:
            文档列表
        """
        if theme is None:
            theme = self._current_theme or self.default_theme

        knowledge_base = self._knowledge_cache.get(theme, {})
        docs = list(knowledge_base.values())

        if category:
            docs = [
                doc for doc in docs
                if doc.metadata and doc.metadata.get('category') == category
            ]

        return docs

    def clear_cache(self, theme: str = None):
        """
        清空知识库缓存

        Args:
            theme: 指定主题，为None则清空所有
        """
        if theme:
            if theme in self._knowledge_cache:
                del self._knowledge_cache[theme]
                logger.info(f"Cleared cache for theme: {theme}")
        else:
            self._knowledge_cache.clear()
            logger.info("Cleared all knowledge cache")

    async def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ) -> List[KnowledgeDocument]:
        """
        混合检索接口

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            检索结果列表
        """
        result = self.retrieve(
            query=query,
            top_k=top_k,
            filters=filters
        )
        return result.documents


class RetrievalService(BaseRetrievalService):
    """
    兼容旧接口的检索服务

    包装 UniversalKnowledgeRetrievalService，保持与现有接口兼容
    """

    def __init__(
        self,
        knowledge_dir: str = None,
        default_theme: str = "general"
    ):
        """
        初始化检索服务

        Args:
            knowledge_dir: 知识库目录
            default_theme: 默认主题
        """
        self._service = UniversalKnowledgeRetrievalService(
            base_knowledge_dir=knowledge_dir,
            default_theme=default_theme
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ):
        """检索接口"""
        return self._service.retrieve(query, top_k, filters)

    async def async_retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ):
        """异步检索接口"""
        return self._service.async_retrieve(query, top_k, filters)

    def retrieve_with_strategy(
        self,
        workspace_id: str,
        query: str,
        strategy_name: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ):
        """策略检索接口"""
        return self._service.retrieve_with_strategy(
            workspace_id, query, strategy_name, top_k, filters
        )

    async def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ):
        """
        混合检索接口

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            检索结果列表
        """
        result = self._service.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
            strategy=MatchingStrategy.HYBRID
        )
        return result.documents
