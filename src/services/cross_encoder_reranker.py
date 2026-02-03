"""Cross-Encoder Reranker - Cross-Encoder 精排模块

功能：
1. Cross-Encoder 重排序：使用交叉编码器进行细粒度排序
2. 批量处理：支持大批量结果的批量打分
3. 分数融合：将 Cross-Encoder 分数与原始分数融合

优势：
- 逐字对比 Query 和 Document，精度远超向量相似度
- 能捕捉细微语义差异
- 适合作为检索后的精排步骤
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """重排序结果"""
    id: str
    original_index: int
    cross_score: float
    fused_score: float
    content: str
    file_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "original_index": self.original_index,
            "cross_score": self.cross_score,
            "fused_score": self.fused_score,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "file_path": self.file_path
        }


@dataclass
class RerankConfig:
    """重排序配置"""
    model_name: str = "BAAI/bge-reranker-base"
    batch_size: int = 32
    top_k: int = 10
    fusion_weight_vector: float = 0.6
    fusion_weight_cross: float = 0.4
    min_score_threshold: float = 0.3
    device: str = "auto"


class BaseReranker(ABC):
    """重排序器基类"""
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: List["RetrievalResult"],
        top_k: int = None
    ) -> List[RerankResult]:
        pass
    
    @abstractmethod
    async def score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        pass


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder 重排序器
    
    使用 Cross-Encoder 模型对检索结果进行精排。
    
    优势：
    - 逐字对比 Query 和 Document
    - 精度远超向量相似度
    - 能捕捉细微语义差异
    
    使用场景：
    - 检索出 Top-50 结果后精排到 Top-10
    - 对精度要求高的关键查询
    """
    
    def __init__(
        self,
        config: Optional[RerankConfig] = None,
        model: Optional[Any] = None
    ):
        self.config = config or RerankConfig()
        self.model = model
        self._model_loaded = False
    
    async def rerank(
        self,
        query: str,
        results: List["RetrievalResult"],
        top_k: int = None
    ) -> List[RerankResult]:
        """
        重排序检索结果
        
        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回结果数量
            
        Returns:
            重排序后的结果
        """
        if not results:
            return []
        
        top_k = top_k or self.config.top_k
        
        logger.info(f"Reranking {len(results)} results with Cross-Encoder...")
        
        # 初始化模型
        await self._ensure_model_loaded()
        
        # 准备文档
        documents = [r.content for r in results]
        
        # 批量计算分数
        cross_scores = await self.score(query, documents)
        
        # 融合分数
        reranked_results = []
        for i, (result, cross_score) in enumerate(zip(results, cross_scores)):
            # 融合分数
            fused = self._fuse_scores(
                result.similarity,
                cross_score,
                self.config.fusion_weight_vector,
                self.config.fusion_weight_cross
            )
            
            # 过滤低分结果
            if cross_score >= self.config.min_score_threshold:
                reranked_results.append(RerankResult(
                    id=result.id,
                    original_index=i,
                    cross_score=cross_score,
                    fused_score=fused,
                    content=result.content,
                    file_path=result.file_path
                ))
        
        # 按融合分数排序
        reranked_results.sort(key=lambda x: x.fused_score, reverse=True)
        
        logger.info(
            f"Cross-Encoder rerank complete: {len(reranked_results)} "
            f"results passed threshold"
        )
        
        return reranked_results[:top_k]
    
    async def score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        计算 Query-Document 对的 Cross-Encoder 分数
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            Cross-Encoder 分数列表
        """
        await self._ensure_model_loaded()
        
        if not self.model:
            logger.warning("Model not loaded, returning default scores")
            return [0.5] * len(documents)
        
        # 批量处理
        batch_size = self.config.batch_size
        all_scores = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            try:
                # 构造 Query-Document 对
                pairs = [[query, doc] for doc in batch_docs]
                
                # 计算分数
                scores = self.model.predict(pairs)
                
                # 归一化到 [0, 1]
                normalized = self._normalize_scores(scores)
                all_scores.extend(normalized)
                
            except Exception as e:
                logger.error(f"Batch scoring failed: {str(e)}")
                all_scores.extend([0.5] * len(batch_docs))
        
        return all_scores
    
    async def _ensure_model_loaded(self):
        """确保模型已加载"""
        if self._model_loaded:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                self.config.model_name,
                device=self.config.device
            )
            self._model_loaded = True
            logger.info(f"Cross-Encoder model loaded: {self.config.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self._model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model: {str(e)}")
            self._model_loaded = True
    
    def _fuse_scores(
        self,
        vector_score: float,
        cross_score: float,
        weight_vector: float,
        weight_cross: float
    ) -> float:
        """
        融合向量分数和 Cross-Encoder 分数
        
        fusion_formula = w1 * vector_score + w2 * cross_score
        """
        return weight_vector * vector_score + weight_cross * cross_score
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """将分数归一化到 [0, 1]"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]


class FallbackReranker(BaseReranker):
    """
    降级重排序器
    
    当 Cross-Encoder 不可用时，提供基于规则的重排序
    """
    
    def __init__(
        self,
        keyword_weight: float = 0.3,
        length_penalty: float = 0.1
    ):
        self.keyword_weight = keyword_weight
        self.length_penalty = length_penalty
    
    async def rerank(
        self,
        query: str,
        results: List["RetrievalResult"],
        top_k: int = 10
    ) -> List[RerankResult]:
        """基于规则的重排序"""
        if not results:
            return []
        
        query_words = set(query.lower().split())
        
        reranked = []
        for i, result in enumerate(results):
            # 计算关键词匹配分数
            keyword_score = self._calculate_keyword_match(
                query_words,
                result.content
            )
            
            # 计算长度惩罚
            length_score = self._calculate_length_score(result.content)
            
            # 综合分数
            fused = (
                0.6 * result.similarity +
                0.3 * keyword_score +
                0.1 * length_score
            )
            
            reranked.append(RerankResult(
                id=result.id,
                original_index=i,
                cross_score=keyword_score,
                fused_score=fused,
                content=result.content,
                file_path=result.file_path
            ))
        
        reranked.sort(key=lambda x: x.fused_score, reverse=True)
        return reranked[:top_k]
    
    async def score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """计算关键词匹配分数"""
        query_words = set(query.lower().split())
        return [
            self._calculate_keyword_match(query_words, doc)
            for doc in documents
        ]
    
    def _calculate_keyword_match(
        self,
        query_words: set,
        content: str
    ) -> float:
        """计算关键词匹配分数"""
        content_words = set(content.lower().split())
        if not query_words:
            return 0.5
        
        matches = len(query_words & content_words)
        return min(1.0, matches / len(query_words))
    
    def _calculate_length_score(self, content: str) -> float:
        """计算长度惩罚分数"""
        word_count = len(content.split())
        
        if word_count < 10:
            return 0.3
        elif word_count < 100:
            return 0.7
        elif word_count < 500:
            return 1.0
        else:
            # 过长内容略微惩罚
            return max(0.5, 1.0 - (word_count - 500) / 1000)


class RerankingPipeline:
    """
    重排序流水线
    
    整合多种重排序策略，提供优雅降级
    """
    
    def __init__(
        self,
        primary_reranker: Optional[BaseReranker] = None,
        fallback_reranker: Optional[BaseReranker] = None
    ):
        self.primary = primary_reranker or CrossEncoderReranker()
        self.fallback = fallback_reranker or FallbackReranker()
    
    async def rerank(
        self,
        query: str,
        results: List["RetrievalResult"],
        top_k: int = 10,
        use_primary: bool = True
    ) -> List[RerankResult]:
        """
        执行重排序
        
        Args:
            query: 查询文本
            results: 检索结果
            top_k: 返回数量
            use_primary: 是否优先使用主重排序器
            
        Returns:
            重排序结果
        """
        if not results:
            return []
        
        if use_primary:
            try:
                return await self.primary.rerank(query, results, top_k)
            except Exception as e:
                logger.warning(
                    f"Primary reranker failed: {str(e)}. "
                    f"Falling back to rule-based reranker."
                )
        
        return await self.fallback.rerank(query, results, top_k)
    
    async def score(
        self,
        query: str,
        documents: List[str],
        use_primary: bool = True
    ) -> List[float]:
        """计算文档分数"""
        if use_primary:
            try:
                return await self.primary.score(query, documents)
            except Exception as e:
                logger.warning(f"Primary scoring failed: {str(e)}")
        
        return await self.fallback.score(query, documents)
    
    def create_metadata(self, results: List[RerankResult]) -> Dict[str, Any]:
        """创建重排序元数据"""
        return {
            "reranked_count": len(results),
            "avg_cross_score": (
                sum(r.cross_score for r in results) / len(results)
                if results else 0
            ),
            "avg_fused_score": (
                sum(r.fused_score for r in results) / len(results)
                if results else 0
            ),
            "top_result": (
                results[0].to_dict() if results else None
            )
        }


class MMMReranker:
    """
    MMR（最大边际相关性）重排序器
    
    在相关性和多样性之间取得平衡
    
    MMR公式：
    MMR = λ * sim(Q, D) - (1-λ) * max(sim(D_i, D_j))
    
    其中：
    - sim(Q, D): 查询与文档的相似度
    - max(sim(D_i, D_j)): 文档与已选文档的最大相似度
    - λ: 平衡参数（通常0.5）
    """

    def __init__(
        self,
        lambda_param: float = 0.5,
        similarity_threshold: float = 0.85
    ):
        """
        初始化 MMR 重排序器
        
        Args:
            lambda_param: 平衡参数 (0-1)
                - 接近1: 更注重相关性
                - 接近0: 更注重多样性
            similarity_threshold: 相似度阈值
                - 超过此阈值认为内容重复
        """
        self.lambda_param = lambda_param
        self.similarity_threshold = similarity_threshold
    
    def rerank(
        self,
        query: str,
        results: List["RetrievalResult"],
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        MMR 重排序
        
        Args:
            query: 查询文本
            results: 检索结果列表（按相似度降序）
            top_k: 返回结果数量
            
        Returns:
            MMR 排序后的结果
        """
        if not results:
            return []
        
        if len(results) <= top_k:
            # 结果数量少于 top_k，无需重排序
            return [
                RerankResult(
                    id=r.id,
                    original_index=i,
                    cross_score=r.similarity,
                    fused_score=r.similarity,
                    content=r.content,
                    file_path=r.file_path
                )
                for i, r in enumerate(results)
            ]
        
        selected = []
        candidates = results.copy()
        
        while len(selected) < top_k and candidates:
            mmr_scores = []
            
            for doc in candidates:
                # 相关性分数
                relevance = doc.similarity
                
                # 多样性分数（与已选文档的最大相似度）
                diversity = self._calculate_max_similarity(doc, selected)
                
                # MMR 分数
                mmr_score = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * diversity
                )
                
                mmr_scores.append((doc, mmr_score))
            
            # 选择 MMR 分数最高的文档
            best_doc, best_score = max(
                mmr_scores,
                key=lambda x: x[1]
            )
            
            selected.append(best_doc)
            candidates.remove(best_doc)
            
            logger.debug(
                f"MMR selected: {best_doc.id[:8]}... "
                f"(mmr={best_score:.3f}, relevance={best_doc.similarity:.3f})"
            )
        
        # 转换为 RerankResult
        return [
            RerankResult(
                id=r.id,
                original_index=results.index(r),
                cross_score=r.similarity,
                fused_score=r.similarity,
                content=r.content,
                file_path=r.file_path
            )
            for r in selected
        ]
    
    def _calculate_max_similarity(
        self,
        doc: "RetrievalResult",
        selected: List["RetrievalResult"]
    ) -> float:
        """计算与已选文档的最大相似度"""
        if not selected:
            return 0.0
        
        max_sim = 0.0
        
        for selected_doc in selected:
            # 使用文件路径和内容计算相似度
            if doc.file_path == selected_doc.file_path:
                sim = 0.9  # 同一文件，相似度较高
            else:
                # 简化的相似度计算
                sim = self._content_similarity(
                    doc.content,
                    selected_doc.content
                )
            
            max_sim = max(max_sim, sim)
        
        return max_sim
    
    def _content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度（简化版）"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
