"""Hybrid Search with RRF - 混合检索与 Reciprocal Rank Fusion

功能：
1. 混合检索：结合向量检索和 BM25 关键词检索
2. RRF 融合：使用 Reciprocal Rank Fusion 算法融合多个检索结果
3. 加权融合：支持自定义向量和关键词检索的权重

解决的问题：
- 向量检索擅长语义（"异步" ≈ "非阻塞"）
- 关键词检索擅长精确匹配（"asyncio" 必须匹配 "asyncio"）
- 两者融合可获得最佳检索效果
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import math

from .retrieval.strategies import RetrievalResult, VectorSearchStrategy, KeywordSearchStrategy

try:
    from .bm25 import BM25KeywordSearch
except ImportError:
    BM25KeywordSearch = None
from .retrieval.mergers import FusionMerger, ReciprocalRankMerger

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """混合检索配置"""
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    top_k_vector: int = 20
    top_k_keyword: int = 20
    rrf_k: int = 60
    enable_fusion: bool = True
    min_score_threshold: float = 0.3


@dataclass
class FusionResult:
    """融合结果"""
    id: str
    file_path: str
    content: str
    vector_score: float
    keyword_score: float
    rrf_score: float
    fused_score: float
    rank_vector: int
    rank_keyword: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
            "rrf_score": self.rrf_score,
            "fused_score": self.fused_score,
            "rank_vector": self.rank_vector,
            "rank_keyword": self.rank_keyword
        }


class HybridSearchService:
    """
    混合检索服务
    
    整合向量检索和关键词检索，使用 RRF 算法融合结果。
    
    优势：
    - 向量检索：捕捉语义相似性
    - 关键词检索：精确匹配专业术语
    - RRF 融合：平衡两种检索方式的结果
    """
    
    def __init__(
        self,
        config: Optional[HybridSearchConfig] = None,
        vector_strategy: Optional[VectorSearchStrategy] = None,
        keyword_strategy: Optional[KeywordSearchStrategy] = None,
        bm25_searcher: Optional['BM25KeywordSearch'] = None
    ):
        self.config = config or HybridSearchConfig()
        self.vector_strategy = vector_strategy
        self.keyword_strategy = keyword_strategy
        self.bm25_searcher = bm25_searcher
    
    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        workspace_id: str,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[FusionResult]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            query_embedding: 查询嵌入向量
            workspace_id: 工作空间 ID
            top_k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            融合后的检索结果
        """
        logger.info(
            f"Hybrid search: query={query[:50]}..., "
            f"vector_weight={self.config.vector_weight}, "
            f"keyword_weight={self.config.keyword_weight}"
        )
        
        vector_results = []
        keyword_results = []
        
        # 并行执行两种检索
        if self.vector_strategy:
            vector_results = await self._vector_search(
                query, query_embedding, workspace_id, filters
            )
        
        if self.keyword_strategy:
            keyword_results = await self._keyword_search(
                query, workspace_id, filters
            )
        
        # 融合结果
        fused_results = self._fuse_results(
            vector_results,
            keyword_results,
            top_k
        )
        
        logger.info(
            f"Hybrid search complete: {len(vector_results)} vector, "
            f"{len(keyword_results)} keyword → {len(fused_results)} fused"
        )
        
        return fused_results
    
    async def _vector_search(
        self,
        query: str,
        query_embedding: List[float],
        workspace_id: str,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """向量检索"""
        if not self.vector_strategy:
            return []
        
        try:
            results = await self.vector_strategy.search(
                query=query,
                query_embedding=query_embedding,
                workspace_id=workspace_id,
                top_k=self.config.top_k_vector,
                filters=filters
            )
            
            logger.debug(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []
    
    async def _keyword_search(
        self,
        query: str,
        workspace_id: str,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """关键词检索"""
        if not self.keyword_strategy and not self.bm25_searcher:
            return []
        
        try:
            # 优先使用 KeywordSearchStrategy
            if self.keyword_strategy:
                results = await self.keyword_strategy.search(
                    query=query,
                    query_embedding=None,
                    workspace_id=workspace_id,
                    top_k=self.config.top_k_keyword,
                    filters=filters
                )
                logger.debug(f"Keyword search returned {len(results)} results")
                return results
            
            # 回退到 BM25 搜索
            if self.bm25_searcher and self.bm25_searcher._doc_count > 0:
                results = self.bm25_searcher.search(query, top_k=self.config.top_k_keyword)
                logger.debug(f"BM25 search returned {len(results)} results")
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []
    
    def _fuse_results(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        top_k: int
    ) -> List[FusionResult]:
        """
        融合检索结果
        
        使用 RRF (Reciprocal Rank Fusion) 算法：
        RRF(d) = Σ (1 / (k + rank_i(d)))
        
        其中：
        - rank_i(d): 文档 d 在第 i 个检索系统中的排名
        - k: 平滑因子（通常60）
        """
        if not vector_results and not keyword_results:
            return []
        
        # 获取所有文档 ID
        all_doc_ids: Set[str] = set()
        for r in vector_results:
            all_doc_ids.add(r.id)
        for r in keyword_results:
            all_doc_ids.add(r.id)
        
        # 构建排名映射
        vector_ranks = self._build_rank_map(vector_results)
        keyword_ranks = self._build_rank_map(keyword_results)
        
        # 计算 RRF 分数
        k = self.config.rrf_k
        rrf_scores = {}
        
        for doc_id in all_doc_ids:
            vector_rank = vector_ranks.get(doc_id, len(vector_results) + 1)
            keyword_rank = keyword_ranks.get(doc_id, len(keyword_results) + 1)
            
            rrf_vector = 1.0 / (k + vector_rank)
            rrf_keyword = 1.0 / (k + keyword_rank)
            
            rrf_scores[doc_id] = {
                'rrf': self.config.vector_weight * rrf_vector + 
                        self.config.keyword_weight * rrf_keyword,
                'vector_rank': vector_rank,
                'keyword_rank': keyword_rank
            }
        
        # 按 RRF 分数排序
        sorted_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x]['rrf'],
            reverse=True
        )
        
        # 构建融合结果
        result_map = {r.id: r for r in vector_results}
        result_map.update({r.id: r for r in keyword_results})
        
        fused_results = []
        
        for rank, doc_id in enumerate(sorted_doc_ids[:top_k], 1):
            original = result_map.get(doc_id)
            if not original:
                continue
            
            scores = rrf_scores[doc_id]
            
            # 计算最终融合分数
            fused_score = self._calculate_fused_score(
                original.similarity,
                scores['rrf'],
                scores['vector_rank'],
                scores['keyword_rank']
            )
            
            # 过滤低分结果
            if fused_score >= self.config.min_score_threshold:
                fusion_result = FusionResult(
                    id=doc_id,
                    file_path=original.file_path,
                    content=original.content,
                    vector_score=original.similarity,
                    keyword_score=original.metadata.get('keyword_score', 0.0),
                    rrf_score=scores['rrf'],
                    fused_score=fused_score,
                    rank_vector=scores['vector_rank'],
                    rank_keyword=scores['keyword_rank'],
                    metadata={
                        'strategy': 'hybrid',
                        'rrf_k': k,
                        **original.metadata
                    }
                )
                fused_results.append(fusion_result)
        
        logger.debug(f"Fusion complete: {len(fused_results)} results after filtering")
        
        return fused_results
    
    def _build_rank_map(
        self,
        results: List[RetrievalResult]
    ) -> Dict[str, int]:
        """构建文档 ID 到排名的映射"""
        rank_map = {}
        
        for rank, result in enumerate(results, 1):
            rank_map[result.id] = rank
        
        return rank_map
    
    def _calculate_fused_score(
        self,
        vector_score: float,
        rrf_score: float,
        vector_rank: int,
        keyword_rank: int
    ) -> float:
        """
        计算最终融合分数
        
        综合考虑：
        1. 向量相似度分数
        2. RRF 融合分数
        3. 排名位置
        """
        # 排名惩罚
        rank_penalty = 1.0 / (1 + math.log(vector_rank + keyword_rank))
        
        # 综合分数
        fused = (
            0.3 * vector_score +
            0.5 * rrf_score +
            0.2 * rank_penalty
        )
        
        return fused


class RRFEngine:
    """
    RRF 引擎
    
    提供纯 RRF 融合功能，可与其他检索系统配合使用
    """
    
    def __init__(self, k: int = 60):
        """
        初始化 RRF 引擎
        
        Args:
            k: RRF 平滑因子，通常 60
        """
        self.k = k
    
    def fuse(
        self,
        *ranked_lists: List[RetrievalResult],
        weights: Optional[List[float]] = None
    ) -> List[Tuple[str, float]]:
        """
        融合多个排名列表
        
        Args:
            ranked_lists: 多个排名列表（每个列表按相关度降序）
            weights: 每个列表的权重
            
        Returns:
            [(doc_id, rrf_score), ...] 按 RRF 分数降序
        """
        if not ranked_lists:
            return []
        
        weights = weights or [1.0] * len(ranked_lists)
        
        # 收集所有文档
        all_doc_ids: Set[str] = set()
        for ranking in ranked_lists:
            for result in ranking:
                all_doc_ids.add(result.id)
        
        # 计算每个文档的 RRF 分数
        doc_scores: Dict[str, float] = defaultdict(float)
        
        for ranking, weight in zip(ranked_lists, weights):
            for rank, result in enumerate(ranking, 1):
                rrf_score = weight * (1.0 / (self.k + rank))
                doc_scores[result.id] += rrf_score
        
        # 排序
        sorted_results = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results
    
    def fuse_with_scores(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[FusionResult]:
        """
        融合向量和关键词检索结果
        
        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            vector_weight: 向量权重
            keyword_weight: 关键词权重
            
        Returns:
            融合结果列表
        """
        # 使用权重进行融合
        weights = [vector_weight, keyword_weight]
        ranked_lists = [vector_results, keyword_results]
        
        sorted_doc_ids = self.fuse(*ranked_lists, weights=weights)
        
        # 构建结果
        result_map = {r.id: r for r in vector_results}
        result_map.update({r.id: r for r in keyword_results})
        
        results = []
        
        for rank, (doc_id, rrf_score) in enumerate(sorted_doc_ids, 1):
            original = result_map.get(doc_id)
            if not original:
                continue
            
            # 确定来源
            is_vector = any(r.id == doc_id for r in vector_results)
            is_keyword = any(r.id == doc_id for r in keyword_results)
            
            vector_rank = next(
                (i for i, r in enumerate(vector_results, 1) if r.id == doc_id),
                len(vector_results) + 1
            )
            
            keyword_rank = next(
                (i for i, r in enumerate(keyword_results, 1) if r.id == doc_id),
                len(keyword_results) + 1
            )
            
            fusion_result = FusionResult(
                id=doc_id,
                file_path=original.file_path,
                content=original.content,
                vector_score=original.similarity if is_vector else 0.0,
                keyword_score=original.metadata.get('keyword_score', 0.0) if is_keyword else 0.0,
                rrf_score=rrf_score,
                fused_score=rrf_score,
                rank_vector=vector_rank,
                rank_keyword=keyword_rank,
                metadata={
                    'type': 'hybrid',
                    'is_vector': is_vector,
                    'is_keyword': is_keyword
                }
            )
            
            results.append(fusion_result)
        
        return results


class BM25KeywordSearch:
    """
    BM25 关键词检索实现
    
    基于 BM25 算法的关键词检索，适合与向量检索混合使用
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        min_length: int = 3
    ):
        """
        初始化 BM25 检索器
        
        Args:
            k1: 词频饱和参数（通常 1.2-2.0）
            b: 文档长度归一化参数（通常 0.75）
            min_length: 最小匹配词数
        """
        self.k1 = k1
        self.b = b
        self.min_length = min_length
        
        self._doc_store: Dict[str, Dict] = {}
        self._doc_lengths: Dict[str, int] = {}
        self._avg_doc_length: float = 0
        self._term_doc_freqs: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._doc_count: int = 0
    
    def index_documents(
        self,
        documents: List[Dict[str, str]]
    ):
        """
        索引文档
        
        Args:
            documents: [{"id": "doc1", "content": "...", "file_path": "..."}]
        """
        self._doc_store.clear()
        self._doc_lengths.clear()
        self._term_doc_freqs.clear()
        
        total_length = 0
        
        for doc in documents:
            doc_id = doc['id']
            content = doc['content']
            file_path = doc.get('file_path', '')
            
            words = self._tokenize(content)
            word_count = len(words)
            
            self._doc_store[doc_id] = {
                'content': content,
                'file_path': file_path,
                'words': set(words)
            }
            self._doc_lengths[doc_id] = word_count
            total_length += word_count
            
            # 更新词频
            word_freqs = defaultdict(int)
            for word in words:
                word_freqs[word] += 1
            
            for word, freq in word_freqs.items():
                self._term_doc_freqs[word][doc_id] = freq
        
        self._doc_count = len(documents)
        self._avg_doc_length = total_length / self._doc_count if self._doc_count > 0 else 0
        
        logger.info(f"Indexed {self._doc_count} documents, avg length: {self._avg_doc_length:.1f}")
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        BM25 检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            按 BM25 分数排序的检索结果
        """
        if not self._doc_store:
            logger.warning("No documents indexed")
            return []
        
        query_words = self._tokenize(query)
        
        if not query_words:
            return []
        
        # 计算每个文档的 BM25 分数
        doc_scores: Dict[str, float] = {}
        
        for doc_id, doc_info in self._doc_store.items():
            score = self._calculate_bm25(
                doc_id,
                query_words,
                doc_info['words']
            )
            
            if score > 0:
                doc_scores[doc_id] = score
        
        # 排序并返回 top_k
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        results = []
        
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            doc_info = self._doc_store[doc_id]
            
            # 归一化分数到 [0, 1]
            max_score = max(doc_scores.values()) if doc_scores else 1
            normalized_score = score / max_score if max_score > 0 else 0
            
            results.append(RetrievalResult(
                id=doc_id,
                file_path=doc_info['file_path'],
                content=doc_info['content'],
                similarity=normalized_score,
                confidence=normalized_score,
                strategy_name='bm25',
                metadata={
                    'bm25_score': score,
                    'keyword_rank': rank,
                    'keyword_score': normalized_score
                }
            ))
        
        return results
    
    def _calculate_bm25(
        self,
        doc_id: str,
        query_words: List[str],
        doc_words: Set[str]
    ) -> float:
        """计算 BM25 分数"""
        score = 0.0
        doc_length = self._doc_lengths[doc_id]
        
        for word in query_words:
            if word not in self._term_doc_freqs:
                continue
            
            # 词频
            tf = self._term_doc_freqs[word].get(doc_id, 0)
            
            # 文档频率
            df = len(self._term_doc_freqs[word])
            
            # IDF
            idf = math.log(
                (self._doc_count - df + 0.5) / (df + 0.5) + 1
            )
            
            # TF 饱和
            tf_saturation = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_length / self._avg_doc_length)
            )
            
            score += idf * tf_saturation
        
        return score
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        import re
        text = text.lower()
        words = re.findall(r'\b[a-z_]+\b', text)
        return [w for w in words if len(w) >= self.min_length]


class HybridSearchPipeline:
    """
    混合检索流水线
    
    整合混合检索、RRF 融合、重排序
    """
    
    def __init__(
        self,
        hybrid_service: Optional[HybridSearchService] = None,
        reranker: Optional[Any] = None
    ):
        self.hybrid_service = hybrid_service or HybridSearchService()
        self.reranker = reranker
    
    async def search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        enable_rerank: bool = True
    ) -> Dict[str, Any]:
        """
        执行完整检索流程
        
        Returns:
            {
                "results": [...],
                "stats": {...},
                "metadata": {...}
            }
        """
        workspace_id = "default"
        
        # 混合检索
        results = await self.hybrid_service.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            workspace_id=workspace_id,
            top_k=top_k * 2  # 预留重排序空间
        )
        
        # 可选：重排序
        if enable_rerank and self.reranker and len(results) > top_k:
            reranked = await self.reranker.rerank(
                query,
                [self._to_retrieval_result(r) for r in results],
                top_k
            )
            results = [self._from_rerank_result(r) for r in reranked]
        else:
            results = results[:top_k]
        
        # 构建响应
        return {
            "results": [r.to_dict() for r in results],
            "stats": {
                "total_results": len(results),
                "vector_weight": self.hybrid_service.config.vector_weight,
                "keyword_weight": self.hybrid_service.config.keyword_weight,
                "avg_fused_score": (
                    sum(r.fused_score for r in results) / len(results)
                    if results else 0
                )
            },
            "metadata": {
                "query": query,
                "workspace_id": workspace_id,
                "top_k": top_k
            }
        }
    
    def _to_retrieval_result(self, fusion: FusionResult) -> RetrievalResult:
        """FusionResult 转换为 RetrievalResult"""
        return RetrievalResult(
            id=fusion.id,
            file_path=fusion.file_path,
            content=fusion.content,
            similarity=fusion.fused_score,
            confidence=fusion.fused_score,
            strategy_name='hybrid',
            metadata={
                'vector_score': fusion.vector_score,
                'keyword_score': fusion.keyword_score,
                'rrf_score': fusion.rrf_score
            }
        )
    
    def _from_rerank_result(self, rerank) -> FusionResult:
        """从重排序结果转换"""
        return FusionResult(
            id=rerank.id,
            file_path=rerank.file_path,
            content=rerank.content,
            vector_score=rerank.metadata.get('vector_score', 0),
            keyword_score=rerank.metadata.get('keyword_score', 0),
            rrf_score=rerank.metadata.get('rrf_score', 0),
            fused_score=rerank.fused_score,
            rank_vector=0,
            rank_keyword=0,
            metadata={'type': 'hybrid_reranked'}
        )
