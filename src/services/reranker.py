"""重排序器 - 重排序检索结果以提高精确度

本模块实现各种重排序策略，以提高 top-K 检索结果的质量。
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval_service import RetrievalResult


logger = logging.getLogger(__name__)


class MultiFactorReranker:
    """
    多因素重排序器
    
    综合多个因素对检索结果重排序：
    - 向量相似度
    - 关键词匹配
    - 文档新鲜度
    - 文档热度
    - 敏感标记
    """
    
    def __init__(
        self,
        similarity_weight: float = 0.4,
        keyword_weight: float = 0.3,
        recency_weight: float = 0.2,
        popularity_weight: float = 0.1
    ):
        """
        初始化重排序器
        
        Args:
            similarity_weight: 相似度权重
            keyword_weight: 关键词匹配权重
            recency_weight: 新鲜度权重
            popularity_weight: 热度权重
        """
        self.similarity_weight = similarity_weight
        self.keyword_weight = keyword_weight
        self.recency_weight = recency_weight
        self.popularity_weight = popularity_weight
        
        # 确保权重和为1
        total = sum([
            similarity_weight,
            keyword_weight,
            recency_weight,
            popularity_weight
        ])
        
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing...")
            self.similarity_weight /= total
            self.keyword_weight /= total
            self.recency_weight /= total
            self.popularity_weight /= total
    
    def rerank(
        self,
        query: str,
        results: List["RetrievalResult"],
        top_k: int = None,
        query_keywords: List[str] = None
    ) -> List["RetrievalResult"]:
        """
        重排序检索结果
        
        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回结果数量
            query_keywords: 查询关键词（可选）
            
        Returns:
            重排序后的结果列表
        """
        if not results:
            return results
        
        logger.info(f"Reranking {len(results)} results...")
        
        # 提取查询关键词
        if query_keywords is None:
            query_keywords = self._extract_keywords(query)
        
        # 计算每个结果的综合分数
        for result in results:
            # 1. 相似度分数（已归一化）
            similarity_score = result.similarity
            
            # 2. 关键词匹配分数
            keyword_score = self._calculate_keyword_match(
                query_keywords,
                result
            )
            
            # 3. 新鲜度分数
            recency_score = self._calculate_recency_score(result)
            
            # 4. 热度分数
            popularity_score = self._calculate_popularity_score(result)
            
            # 综合分数
            final_score = (
                self.similarity_weight * similarity_score +
                self.keyword_weight * keyword_score +
                self.recency_weight * recency_score +
                self.popularity_weight * popularity_score
            )
            
            # 敏感标记加权
            if result.has_deprecated or result.has_security:
                final_score *= 1.2  # 提升20%
            elif result.has_fixme:
                final_score *= 1.1  # 提升10%
            
            # 保存分数和详细信息
            result.metadata['rerank_score'] = final_score
            result.metadata['score_breakdown'] = {
                'similarity': similarity_score,
                'keyword': keyword_score,
                'recency': recency_score,
                'popularity': popularity_score
            }
            
            # 更新confidence为最终分数
            result.confidence = final_score
        
        # 按最终分数排序
        reranked = sorted(
            results,
            key=lambda x: x.confidence,
            reverse=True
        )
        
        logger.info(f"Reranking complete. Top score: {reranked[0].confidence:.3f}")
        
        return reranked
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取查询关键词"""
        import re
        
        # 简单分词
        words = re.findall(r'\w+', query.lower())
        
        # 过滤停用词和短词
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            '的', '了', '在', '是', '我', '有', '和', '就', '不'
        }
        
        keywords = [
            word for word in words
            if word not in stopwords and len(word) > 2
        ]
        
        return keywords
    
    def _calculate_keyword_match(
        self,
        query_keywords: List[str],
        result: "RetrievalResult"
    ) -> float:
        """
        计算关键词匹配分数
        
        Returns:
            0.0-1.0之间的分数
        """
        if not query_keywords:
            return 0.5  # 默认中等分数
        
        # 提取文档关键词
        doc_text = (result.content + ' ' + result.file_path).lower()
        
        # 计算匹配的关键词数量
        matches = sum(1 for kw in query_keywords if kw in doc_text)
        
        # 归一化到0-1
        score = matches / len(query_keywords)
        
        return min(score, 1.0)
    
    def _calculate_recency_score(self, result: "RetrievalResult") -> float:
        """
        计算新鲜度分数
        
        Returns:
            0.0-1.0之间的分数
        """
        # 如果有时间戳元数据
        if 'timestamp' in result.metadata:
            try:
                timestamp = result.metadata['timestamp']
                if isinstance(timestamp, str):
                    doc_time = datetime.fromisoformat(timestamp)
                else:
                    doc_time = timestamp
                
                # 计算天数差
                days_old = (datetime.now() - doc_time).days
                
                # 使用指数衰减
                # 30天内的文档得分较高
                score = np.exp(-days_old / 30.0)
                
                return min(score, 1.0)
                
            except Exception as e:
                logger.debug(f"Failed to parse timestamp: {e}")
        
        # 默认中等分数（假设文档不太旧也不太新）
        return 0.5
    
    def _calculate_popularity_score(self, result: "RetrievalResult") -> float:
        """
        计算热度分数
        
        Returns:
            0.0-1.0之间的分数
        """
        # 如果有访问次数元数据
        if 'access_count' in result.metadata:
            try:
                count = int(result.metadata['access_count'])
                
                # 使用对数缩放
                # 假设100次访问为高热度
                score = np.log1p(count) / np.log1p(100)
                
                return min(score, 1.0)
                
            except Exception as e:
                logger.debug(f"Failed to parse access_count: {e}")
        
        # 默认中等分数
        return 0.5


class DiversityFilter:
    """
    多样性过滤器
    
    确保检索结果的多样性，避免返回过于相似的文档
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        初始化多样性过滤器
        
        Args:
            similarity_threshold: 相似度阈值，超过此值认为文档过于相似
        """
        self.similarity_threshold = similarity_threshold
    
    def filter(
        self,
        results: List["RetrievalResult"],
        threshold: float = 0.85,
        top_k: int = None
    ) -> List["RetrievalResult"]:
        """
        过滤结果以确保多样性
        
        Args:
            results: 检索结果列表（已排序）
            top_k: 需要返回的结果数量
            
        Returns:
            多样性过滤后的结果列表
        """
        if len(results) <= top_k:
            return results
        
        logger.info(f"Applying diversity filter to {len(results)} results...")
        
        diverse_results = []
        
        for result in results:
            # 检查与已选结果的相似度
            is_diverse = True
            
            for selected in diverse_results:
                similarity = self._calculate_content_similarity(
                    result.content,
                    selected.content
                )
                
                if similarity > self.similarity_threshold:
                    is_diverse = False
                    logger.debug(
                        f"Filtered out similar document: {result.file_path} "
                        f"(similarity: {similarity:.2f})"
                    )
                    break
            
            if is_diverse:
                diverse_results.append(result)
            
            # 达到目标数量
            if len(diverse_results) >= top_k:
                break
        
        logger.info(f"Diversity filter returned {len(diverse_results)} results")
        
        return diverse_results
    
    def _calculate_content_similarity(
        self,
        content1: str,
        content2: str
    ) -> float:
        """
        计算两个文档内容的相似度
        
        使用简单的Jaccard相似度
        
        Returns:
            0.0-1.0之间的相似度分数
        """
        # 分词
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        # Jaccard相似度
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0.0
        
        return intersection / union


class RetrievalQualityMonitor:
    """
    检索质量监控器
    
    评估和监控检索结果的质量
    """
    
    def calculate_metrics(
        self,
        query: str,
        results: List["RetrievalResult"]
    ) -> Dict[str, Any]:
        """
        评估检索质量
        
        Args:
            query: 查询文本
            results: 检索结果列表
            
        Returns:
            质量指标字典
        """
        if not results:
            return {
                'num_results': 0,
                'avg_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0,
                'diversity': 0.0,
                'has_deprecated': False,
                'has_security': False
            }
        
        # 计算指标
        similarities = [r.similarity for r in results]
        
        metrics = {
            'num_results': len(results),
            'avg_similarity': np.mean(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'std_similarity': np.std(similarities),
            'diversity': self._calculate_diversity(results),
            'has_deprecated': any(r.has_deprecated for r in results),
            'has_security': any(r.has_security for r in results),
            'has_fixme': any(r.has_fixme for r in results),
            'source_distribution': self._calculate_source_distribution(results)
        }
        
        logger.info(f"Retrieval quality metrics: {metrics}")
        
        return metrics
    
    def _calculate_diversity(self, results: List["RetrievalResult"]) -> float:
        """
        计算结果多样性
        
        Returns:
            0.0-1.0之间的多样性分数
        """
        if len(results) < 2:
            return 1.0
        
        # 计算文件路径的多样性
        unique_paths = len(set(r.file_path for r in results))
        diversity = unique_paths / len(results)
        
        return diversity
    
    def _calculate_source_distribution(
        self,
        results: List["RetrievalResult"]
    ) -> Dict[str, int]:
        """
        计算来源分布
        
        Returns:
            来源类型到数量的映射
        """
        distribution = {}
        
        for result in results:
            source = result.source
            distribution[source] = distribution.get(source, 0) + 1
        
        return distribution
