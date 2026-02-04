"""检索质量评估器

功能：
1. 检索结果覆盖度评估
2. 结果一致性分析
3. 信息新鲜度检测
4. 完整性评估与补充检索建议
5. 负样本过滤

解决的问题：
- 检索结果质量难以量化
- 难以判断是否需要补充检索
- 明显不相关的结果未被过滤
"""

import logging
import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter

from .retrieval.strategies import RetrievalResult
from .llm.service import LLMService

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """质量维度"""
    COVERAGE = "coverage"       # 覆盖度
    CONSISTENCY = "consistency" # 一致性
    FRESHNESS = "freshness"    # 新鲜度
    COMPLETENESS = "completeness" # 完整性
    RELEVANCE = "relevance"    # 相关性


@dataclass
class QualityAssessment:
    """质量评估结果"""
    coverage_score: float = 0.0
    consistency_score: float = 0.0
    freshness_score: float = 0.0
    completeness_score: float = 0.0
    relevance_score: float = 0.0
    overall_score: float = 0.0
    needs_supplemental_retrieval: bool = False
    suggested_supplemental_queries: List[str] = field(default_factory=list)
    identified_gaps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    assessment_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalIssue:
    """检索问题"""
    issue_type: str
    description: str
    severity: str  # low, medium, high
    affected_results: List[str]
    suggested_fix: str


class CoverageAnalyzer:
    """
    覆盖度分析器
    
    分析检索结果是否覆盖查询的所有方面
    """
    
    def __init__(self):
        self.aspect_keywords = {
            'terminology': ['定义', '概念', '什么是', 'explain', 'definition'],
            'usage': ['使用', '如何', 'how to', '用法', '教程'],
            'implementation': ['实现', '代码', '示例', 'example', 'code'],
            'configuration': ['配置', '设置', 'config', 'setup'],
            'troubleshooting': ['问题', '错误', '调试', 'debug', 'error', '故障'],
            'architecture': ['架构', '结构', '设计', 'architecture', 'design'],
        }
    
    def analyze_coverage(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> Tuple[float, List[str]]:
        """
        分析覆盖度
        
        Args:
            query: 查询文本
            results: 检索结果列表
            
        Returns:
            (覆盖度分数, 识别的缺失方面)
        """
        query_lower = query.lower()
        identified_aspects = set()
        covered_aspects = set()
        
        for aspect, keywords in self.aspect_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    identified_aspects.add(aspect)
                    break
        
        for result in results:
            content_lower = result.content.lower()
            for aspect, keywords in self.aspect_keywords.items():
                for keyword in keywords:
                    if keyword in content_lower:
                        covered_aspects.add(aspect)
                        break
        
        if not identified_aspects:
            coverage_score = 0.8 if len(results) >= 3 else 0.5
            return coverage_score, []
        
        if not covered_aspects:
            missing_aspects = list(identified_aspects)
            coverage_score = 0.3
            return coverage_score, missing_aspects
        
        overlap = identified_aspects & covered_aspects
        coverage_score = len(overlap) / len(identified_aspects) if identified_aspects else 0.5
        
        missing_aspects = list(identified_aspects - covered_aspects)
        
        if len(results) < 3:
            coverage_score *= 0.8
        
        return min(coverage_score, 1.0), missing_aspects


class ConsistencyAnalyzer:
    """
    一致性分析器
    
    分析检索结果之间是否有矛盾
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service
        self.key_value_patterns = [
            r'(\w+)\s*[:=]\s*([^,\n]+)',
            r'(\w+)\s+is\s+([^,\n]+)',
        ]
    
    def extract_key_values(self, text: str) -> Dict[str, str]:
        """提取键值对"""
        values = {}
        for pattern in self.key_value_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for key, value in matches:
                key = key.strip().lower()
                value = value.strip()
                if len(value) < 100:
                    values[key] = value
        return values
    
    def analyze_consistency(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> Tuple[float, List[str]]:
        """
        分析一致性
        
        Args:
            query: 查询文本
            results: 检索结果列表
            
        Returns:
            (一致性分数, 发现的不一致列表)
        """
        if len(results) < 2:
            return 0.9, []
        
        inconsistencies = []
        
        extracted_info: List[Dict[str, str]] = []
        for result in results:
            info = self.extract_key_values(result.content)
            extracted_info.append((result.id, info))
        
        all_keys: Set[str] = set()
        for _, info in extracted_info:
            all_keys.update(info.keys())
        
        for key in all_keys:
            values = [info.get(key) for _, info in extracted_info if info.get(key)]
            if len(values) >= 2:
                unique_values = set(v.lower() for v in values)
                if len(unique_values) > 1:
                    inconsistencies.append(
                        f"Key '{key}' has conflicting values: {values}"
                    )
        
        similarity_scores = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                content_i = results[i].content.lower()
                content_j = results[j].content.lower()
                
                words_i = set(re.findall(r'\b\w+\b', content_i))
                words_j = set(re.findall(r'\b\w+\b', content_j))
                
                if words_i and words_j:
                    overlap = len(words_i & words_j) / len(words_i | words_j)
                    similarity_scores.append(overlap)
        
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
        else:
            avg_similarity = 0.5
        
        consistency_score = avg_similarity
        
        if len(inconsistencies) > 2:
            consistency_score *= 0.7
        elif len(inconsistencies) > 0:
            consistency_score *= 0.9
        
        return max(consistency_score, 0.3), inconsistencies


class FreshnessAnalyzer:
    """
    新鲜度分析器
    
    检测结果是否包含过时信息
    """
    
    def __init__(self):
        self.deprecated_indicators = [
            'deprecated', '已废弃', '已弃用', '过时的',
            'legacy', 'legacy code', 'old version',
            'v1.0', 'v2.0', 'v0.x',
            '不再支持', '不再推荐', 'will be removed'
        ]
        self.version_patterns = [
            r'v?(\d+\.\d+)',
            r'version\s*(\d+\.\d+)',
        ]
        self.date_patterns = [
            r'(\d{4})-(\d{2})-(\d{2})',
            r'(\d{2})/(\d{2})/(\d{4})',
        ]
    
    def analyze_freshness(
        self,
        results: List[RetrievalResult]
    ) -> Tuple[float, List[str]]:
        """
        分析新鲜度
        
        Args:
            results: 检索结果列表
            
        Returns:
            (新鲜度分数, 过时信息列表)
        """
        deprecated_count = 0
        outdated_warnings = []
        
        for result in results:
            content_lower = result.content.lower()
            
            for indicator in self.deprecated_indicators:
                if indicator in content_lower:
                    deprecated_count += 1
                    outdated_warnings.append(
                        f"Result {result.id}: contains '{indicator}'"
                    )
                    break
        
        if len(results) == 0:
            return 0.5, []
        
        freshness_score = 1.0 - (deprecated_count / len(results) * 0.5)
        
        if deprecated_count > 0:
            logger.warning(f"Found {deprecated_count} deprecated results")
        
        return max(freshness_score, 0.5), outdated_warnings


class CompletenessAnalyzer:
    """
    完整性分析器
    
    评估是否需要补充检索
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service
    
    def analyze_completeness(
        self,
        query: str,
        results: List[RetrievalResult],
        coverage_score: float,
        identified_gaps: List[str]
    ) -> Tuple[float, bool, List[str]]:
        """
        分析完整性
        
        Args:
            query: 查询文本
            results: 检索结果列表
            coverage_score: 覆盖度分数
            identified_gaps: 识别的缺失方面
            
        Returns:
            (完整性分数, 是否需要补充检索, 补充查询建议)
        """
        supplemental_queries = []
        
        query_lower = query.lower()
        
        if len(results) < 3:
            supplemental_queries.append(
                f"扩展检索: {query} 官方文档"
            )
        
        if coverage_score < 0.5:
            for gap in identified_gaps[:2]:
                supplemental_queries.append(
                    f"{gap} 相关内容 {query}"
                )
        
        multi_indicators = ['和', '与', '以及', ' or ', ' and ', '区别']
        if any(ind in query_lower for ind in multi_indicators):
            parts = re.split(r'\s+(?:和|与|以及|and|or|区别)\s+', query, maxsplit=1)
            if len(parts) > 1:
                supplemental_queries.append(f"扩展检索: {parts[1].strip()}")
        
        if any(kw in query_lower for kw in ['配置', 'config', '设置']):
            supplemental_queries.append(f"{query} 配置最佳实践")
        
        if any(kw in query_lower for kw in ['错误', 'error', '问题', 'debug']):
            supplemental_queries.append(f"{query} 常见问题和解决方案")
        
        completeness_score = 0.5
        
        if len(results) >= 3:
            completeness_score += 0.2
        if coverage_score >= 0.7:
            completeness_score += 0.2
        if not identified_gaps:
            completeness_score += 0.1
        
        completeness_score = min(completeness_score, 1.0)
        
        needs_supplemental = (
            len(results) < 3 or
            coverage_score < 0.5 or
            len(identified_gaps) > 0
        )
        
        return completeness_score, needs_supplemental, supplemental_queries


class RetrievalQualityAssessor:
    """
    检索质量评估器 v1.0
    
    整合多种分析器，提供检索结果的整体质量评估
    
    使用示例:
    ```python
    assessor = RetrievalQualityAssessor(llm_service=llm_service)
    assessment = await assessor.assess_quality(
        query="如何在 FastAPI 中实现认证",
        results=retrieval_results
    )
    
    if assessment.needs_supplemental_retrieval:
        for query in assessment.suggested_supplemental_queries:
            await supplemental_retrieval(query)
    ```
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化质量评估器
        
        Args:
            llm_service: LLM 服务（可选，用于高级分析）
            config: 配置选项
        """
        self.llm_service = llm_service
        self.config = config or {}
        
        self.coverage_analyzer = CoverageAnalyzer()
        self.consistency_analyzer = ConsistencyAnalyzer(llm_service)
        self.freshness_analyzer = FreshnessAnalyzer()
        self.completeness_analyzer = CompletenessAnalyzer(llm_service)
        
        self.min_results_threshold = self.config.get('min_results_threshold', 3)
        self.min_coverage_threshold = self.config.get('min_coverage_threshold', 0.6)
        self.min_consistency_threshold = self.config.get('min_consistency_threshold', 0.7)
    
    async def assess_quality(
        self,
        query: str,
        results: List[RetrievalResult],
        context: Optional[Dict[str, Any]] = None
    ) -> QualityAssessment:
        """
        评估检索结果的整体质量
        
        Args:
            query: 查询文本
            results: 检索结果列表
            context: 额外上下文信息
            
        Returns:
            质量评估结果
        """
        logger.info(f"Assessing retrieval quality for query: {query[:50]}...")
        start_time = time.time()
        
        if not results:
            return QualityAssessment(
                coverage_score=0.0,
                consistency_score=0.0,
                freshness_score=0.0,
                completeness_score=0.0,
                relevance_score=0.0,
                overall_score=0.0,
                needs_supplemental_retrieval=True,
                suggested_supplemental_queries=[query],
                identified_gaps=["无检索结果"],
                warnings=["没有找到任何相关结果"]
            )
        
        relevance_score = self._calculate_relevance_score(query, results)
        
        coverage_score, identified_gaps = self.coverage_analyzer.analyze_coverage(
            query, results
        )
        
        consistency_score, inconsistencies = self.consistency_analyzer.analyze_consistency(
            query, results
        )
        
        freshness_score, outdated_warnings = self.freshness_analyzer.analyze_freshness(
            results
        )
        
        completeness_score, needs_supplemental, suggested_queries = (
            self.completeness_analyzer.analyze_completeness(
                query, results, coverage_score, identified_gaps
            )
        )
        
        weights = self.config.get('dimension_weights', {
            'coverage': 0.25,
            'consistency': 0.2,
            'freshness': 0.15,
            'completeness': 0.2,
            'relevance': 0.2
        })
        
        overall_score = (
            coverage_score * weights['coverage'] +
            consistency_score * weights['consistency'] +
            freshness_score * weights['freshness'] +
            completeness_score * weights['completeness'] +
            relevance_score * weights['relevance']
        )
        
        warnings = []
        if coverage_score < self.min_coverage_threshold:
            warnings.append(f"覆盖度不足 ({coverage_score:.2f} < {self.min_coverage_threshold})")
        if consistency_score < self.min_consistency_threshold:
            warnings.append(f"一致性较低 ({consistency_score:.2f} < {self.min_consistency_threshold})")
        if inconsistencies:
            warnings.append(f"发现 {len(inconsistencies)} 个不一致问题")
        if outdated_warnings:
            warnings.append(f"发现 {len(outdated_warnings)} 个可能过时的结果")
        
        assessment = QualityAssessment(
            coverage_score=coverage_score,
            consistency_score=consistency_score,
            freshness_score=freshness_score,
            completeness_score=completeness_score,
            relevance_score=relevance_score,
            overall_score=overall_score,
            needs_supplemental_retrieval=needs_supplemental,
            suggested_supplemental_queries=suggested_queries,
            identified_gaps=identified_gaps,
            warnings=warnings,
            assessment_details={
                'num_results': len(results),
                'num_inconsistencies': len(inconsistencies),
                'num_outdated': len(outdated_warnings),
                'query_length': len(query),
                'context_provided': context is not None
            }
        )
        
        elapsed = time.time() - start_time
        logger.info(
            f"Quality assessment completed in {elapsed:.2f}s: "
            f"overall={overall_score:.2f}, "
            f"needs_supplemental={needs_supplemental}"
        )
        
        return assessment
    
    def _calculate_relevance_score(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> float:
        """计算平均相关性分数"""
        if not results:
            return 0.0
        
        scores = [r.similarity for r in results if r.similarity > 0]
        
        if not scores:
            return 0.0
        
        avg_score = sum(scores) / len(scores)
        
        score_distribution = Counter(r.similarity > 0.5 for r in results)
        high_quality_ratio = score_distribution.get(True, 0) / len(results)
        
        combined_score = avg_score * 0.6 + high_quality_ratio * 0.4
        
        return min(combined_score, 1.0)
    
    def batch_assess(
        self,
        queries: List[str],
        results_map: Dict[str, List[RetrievalResult]]
    ) -> Dict[str, QualityAssessment]:
        """
        批量评估多个查询的检索质量
        
        Args:
            queries: 查询列表
            results_map: 查询到结果的映射
            
        Returns:
            查询到评估结果的映射
        """
        assessments = {}
        for query in queries:
            results = results_map.get(query, [])
            assessment = None
            
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            assessment = loop.run_until_complete(
                self.assess_quality(query, results)
            )
            
            assessments[query] = assessment
        
        return assessments
    
    def get_quality_report(self, assessment: QualityAssessment) -> str:
        """
        生成质量报告
        
        Args:
            assessment: 质量评估结果
            
        Returns:
            格式化的质量报告
        """
        lines = [
            "=" * 50,
            "检索质量评估报告",
            "=" * 50,
            f"整体评分: {assessment.overall_score:.2f}/1.00",
            "",
            "各维度评分:",
            f"  - 覆盖度: {assessment.coverage_score:.2f}",
            f"  - 一致性: {assessment.consistency_score:.2f}",
            f"  - 新鲜度: {assessment.freshness_score:.2f}",
            f"  - 完整性: {assessment.completeness_score:.2f}",
            f"  - 相关性: {assessment.relevance_score:.2f}",
            "",
        ]
        
        if assessment.needs_supplemental_retrieval:
            lines.append("⚠️ 需要补充检索")
            if assessment.suggested_supplemental_queries:
                lines.append("建议的补充查询:")
                for i, query in enumerate(assessment.suggested_supplemental_queries, 1):
                    lines.append(f"  {i}. {query}")
        else:
            lines.append("✓ 检索结果质量良好")
        
        if assessment.identified_gaps:
            lines.append("")
            lines.append("识别的缺失方面:")
            for gap in assessment.identified_gaps:
                lines.append(f"  - {gap}")
        
        if assessment.warnings:
            lines.append("")
            lines.append("警告:")
            for warning in assessment.warnings:
                lines.append(f"  ⚠️ {warning}")
        
        lines.append("=" * 50)
        
        return '\n'.join(lines)


class NegativeSampleFilter:
    """
    负样本过滤器
    
    过滤明显不相关的检索结果
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.3,
        keyword_match_threshold: float = 0.2,
        enable_semantic_filter: bool = True,
        llm_service: Optional[LLMService] = None
    ):
        """
        初始化负样本过滤器
        
        Args:
            similarity_threshold: 相似度阈值（低于此值被过滤）
            keyword_match_threshold: 关键词匹配阈值
            enable_semantic_filter: 是否启用语义过滤
            llm_service: LLM 服务（用于高级语义分析）
        """
        self.similarity_threshold = similarity_threshold
        self.keyword_match_threshold = keyword_match_threshold
        self.enable_semantic_filter = enable_semantic_filter
        self.llm_service = llm_service
        
        self.irrelevant_patterns = [
            r'^import\s+\w+$',
            r'^\s*#include\s*<',
            r'^\s*(?:def|func|function)\s+\w+\s*\(\s*\)\s*$',
        ]
    
    def filter_negative_samples(
        self,
        query: str,
        results: List[RetrievalResult],
        threshold: Optional[float] = None
    ) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        """
        过滤负样本
        
        Args:
            query: 查询文本
            results: 检索结果列表
            threshold: 自定义阈值
            
        Returns:
            (过滤后的结果, 被过滤的结果)
        """
        effective_threshold = threshold if threshold is not None else self.similarity_threshold
        
        filtered_results = []
        removed_results = []
        
        query_keywords = self._extract_keywords(query)
        
        for result in results:
            removal_reason = None
            
            if result.similarity < effective_threshold:
                removal_reason = f"similarity_below_threshold ({result.similarity:.2f} < {effective_threshold})"
            elif self._is_irrelevant_content(result.content):
                removal_reason = "irrelevant_content_pattern"
            elif query_keywords and not self._has_keyword_match(result.content, query_keywords):
                if result.similarity < effective_threshold + 0.1:
                    removal_reason = "no_keyword_match"
            
            if removal_reason:
                logger.debug(f"Filtered out {result.id}: {removal_reason}")
                removed_results.append({
                    'id': result.id,
                    'reason': removal_reason,
                    'similarity': result.similarity,
                    'file_path': result.file_path
                })
            else:
                filtered_results.append(result)
        
        return filtered_results, removed_results
    
    def _extract_keywords(self, query: str) -> Set[str]:
        """提取查询关键词"""
        keywords = set()
        
        query_cleaned = re.sub(r'[^\w\s]', ' ', query.lower())
        words = query_cleaned.split()
        
        stopwords = {'的', '是', '在', '如何', '怎么', '什么', 'how', 'what', 'the', 'a', 'an'}
        
        for word in words:
            if len(word) > 2 and word not in stopwords:
                keywords.add(word)
        
        return keywords
    
    def _is_irrelevant_content(self, content: str) -> bool:
        """检查内容是否明显不相关"""
        content_stripped = content.strip()
        
        for pattern in self.irrelevant_patterns:
            if re.match(pattern, content_stripped):
                return True
        
        if len(content_stripped) < 10:
            return True
        
        code_indicators = content_stripped.count('\n') / max(len(content_stripped), 1)
        if code_indicators > 0.5 and len(content_stripped) < 50:
            return True
        
        return False
    
    def _has_keyword_match(
        self,
        content: str,
        keywords: Set[str]
    ) -> bool:
        """检查内容是否包含关键词"""
        if not keywords:
            return True
        
        content_lower = content.lower()
        
        matched_count = 0
        for keyword in keywords:
            if keyword in content_lower:
                matched_count += 1
        
        match_ratio = matched_count / len(keywords) if keywords else 0
        
        return match_ratio >= self.keyword_match_threshold
    
    def filter_with_scores(
        self,
        query: str,
        results: List[RetrievalResult],
        score_type: str = "similarity"
    ) -> List[RetrievalResult]:
        """
        根据指定分数过滤结果
        
        Args:
            query: 查询文本
            results: 检索结果列表
            score_type: 使用的分数类型 ('similarity', 'confidence', 'combined')
            
        Returns:
            过滤后的结果列表
        """
        threshold = self.similarity_threshold
        
        filtered = []
        for result in results:
            if score_type == "similarity":
                score = result.similarity
            elif score_type == "confidence":
                score = result.confidence
            else:
                score = (result.similarity + result.confidence) / 2
            
            if score >= threshold:
                filtered.append(result)
        
        logger.info(
            f"Filtered {len(results) - len(filtered)} results "
            f"(score_type={score_type}, threshold={threshold})"
        )
        
        return filtered
    
    def get_filtering_report(
        self,
        query: str,
        original_results: List[RetrievalResult],
        filtered_results: List[RetrievalResult],
        removed_details: List[Dict[str, Any]]
    ) -> str:
        """生成过滤报告"""
        lines = [
            "=" * 50,
            "负样本过滤报告",
            "=" * 50,
            f"查询: {query[:100]}",
            f"原始结果数: {len(original_results)}",
            f"过滤后结果数: {len(filtered_results)}",
            f"过滤掉的结果数: {len(removed_details)}",
            "",
            "过滤详情:",
        ]
        
        for detail in removed_details:
            lines.append(
                f"  - {detail['id']}: {detail['reason']} "
                f"(similarity={detail['similarity']:.2f})"
            )
        
        lines.append("=" * 50)
        
        return '\n'.join(lines)
