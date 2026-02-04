"""Query Rewriter - 查询改写与分解

功能：
1. 查询意图改写：将不完整/模糊查询改写为更适合检索的形式
2. 复杂查询分解：将复杂查询分解为多个子查询
3. 上下文增强：基于项目上下文补充隐含信息
4. 智能缓存：避免重复查询的重复计算
5. 混合策略：规则 + LLM 混合改写

解决的问题：
- 用户查询不完整导致召回率低
- 隐含的技术栈/框架信息丢失
- 复杂查询难以精确匹配
- 重复查询重复计算
- LLM 调用成本高
"""

import hashlib
import json
import logging
import re
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .llm.service import LLMService

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型"""
    TERMINOLOGY = "terminology"       # 术语查询（什么是X）
    USAGE = "usage"                   # 用法查询（如何使用X）
    IMPLEMENTATION = "implementation" # 实现查询（X是如何实现的）
    COMPARISON = "comparison"         # 比较查询（X和Y的区别）
    DEBUGGING = "debugging"           # 调试查询（X出错了）
    ARCHITECTURE = "architecture"     # 架构查询（X的架构是怎样的）
    GENERAL = "general"               # 一般查询


@dataclass
class RewriteResult:
    """改写结果"""
    original_query: str
    rewritten_query: str
    query_type: QueryType
    confidence: float
    added_context: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)


@dataclass
class QueryContext:
    """查询上下文"""
    project_type: str = "general"           # 项目类型
    detected_language: str = ""             # 检测到的编程语言
    detected_framework: str = ""            # 检测到的框架
    file_path: str = ""                     # 当前文件路径
    recent_queries: List[str] = field(default_factory=list)
    workspace_id: str = ""


class IntentClassifier:
    """
    查询意图分类器
    
    功能：
    - 基于规则的快速意图分类
    - 置信度评估
    - 分类结果验证
    """
    
    def __init__(self):
        self.type_indicators = {
            QueryType.TERMINOLOGY: ['什么是', 'what is', 'explain', '解释', '定义', '概念'],
            QueryType.USAGE: ['如何使用', 'how to', 'how do i', '用法', '使用', '怎么用'],
            QueryType.IMPLEMENTATION: ['如何实现', '实现', 'implementation', '怎么写', '编写', '开发'],
            QueryType.COMPARISON: ['区别', 'difference', 'vs', '比较', '对比', '不同'],
            QueryType.DEBUGGING: ['错误', 'error', 'bug', '问题', '异常', '失败', '报错'],
            QueryType.ARCHITECTURE: ['架构', 'architecture', '设计', '结构', '组成'],
        }
        
        self.simple_query_patterns = [
            r'^什么是\w+$',
            r'^如何\w+\w*$',
            r'^\w+是什么$',
            r'^explain\s+\w+$',
            r'^how to \w+$',
        ]
    
    def classify(self, query: str) -> Tuple[QueryType, float]:
        """
        分类查询意图
        
        Args:
            query: 查询文本
            
        Returns:
            (查询类型, 置信度)
        """
        query_lower = query.lower().strip()
        
        type_scores: Dict[QueryType, float] = {}
        
        for qtype, indicators in self.type_indicators.items():
            matches = sum(1 for ind in indicators if ind in query_lower)
            if matches > 0:
                type_scores[qtype] = min(matches / len(indicators) + 0.3, 1.0)
        
        if not type_scores:
            return QueryType.GENERAL, 0.5
        
        best_type = max(type_scores.items(), key=lambda x: x[1])
        return best_type[0], best_type[1]
    
    def is_simple_query(self, query: str) -> bool:
        """
        判断是否为简单查询
        
        简单查询特征：
        - 长度短（< 20字符）
        - 关键词明确
        - 不包含复杂逻辑
        """
        query = query.strip()
        
        if len(query) < 5 or len(query) > 50:
            return False
        
        for pattern in self.simple_query_patterns:
            if re.match(pattern, query, re.IGNORECASE):
                return True
        
        simple_keywords = ['什么是', 'how to', 'explain', '如何使用']
        if any(kw in query.lower() for kw in simple_keywords):
            word_count = len(query.split())
            if word_count <= 5:
                return True
        
        return False
    
    def validate_classification(
        self,
        query: str,
        query_type: QueryType,
        confidence: float
    ) -> bool:
        """
        验证分类结果的有效性
        
        Args:
            query: 查询文本
            query_type: 分类结果
            confidence: 置信度
            
        Returns:
            是否有效
        """
        if confidence < 0.3:
            return False
        
        if query_type == QueryType.TERMINOLOGY:
            return bool(re.search(r'什么|what|explain|定义', query.lower()))
        
        if query_type == QueryType.USAGE:
            return bool(re.search(r'使用|how|用法', query.lower()))
        
        if query_type == QueryType.DEBUGGING:
            return bool(re.search(r'错误|error|bug|问题|异常|失败', query.lower()))
        
        return True


@dataclass
class RewriteCacheEntry:
    """缓存条目"""
    rewritten_query: str
    query_type: QueryType
    confidence: float
    added_context: List[str]
    sub_queries: List[str]
    timestamp: float


class QueryRewriter:
    """
    查询改写器 v2.0
    
    增强功能：
    - 智能缓存：避免重复查询的重复计算
    - 混合策略：规则 + LLM 混合改写
    - 独立意图分类器
    - 简单查询快速处理
    - 复杂查询 LLM 深度改写
    
    功能：
    - 意图消歧：确定查询的具体意图
    - 信息补充：添加隐含的技术栈/框架信息
    - 规范化：统一术语表述
    - 分解：将复杂查询分解为多个子查询
    """
    
    SIMPLE_QUERY_THRESHOLD = 0.7
    
    def __init__(
        self,
        llm_service: LLMService,
        project_context: Optional[Dict] = None,
        cache_ttl: int = 3600,
        enable_hybrid_strategy: bool = True
    ):
        self.llm_service = llm_service
        self.project_context = project_context or {}
        self.cache_ttl = cache_ttl
        self.enable_hybrid_strategy = enable_hybrid_strategy
        
        self._cache: Dict[str, RewriteCacheEntry] = {}
        
        self.intent_classifier = IntentClassifier()
        
        self.term_normalization = {
            '异步': ['async', 'asyncio', 'await', '非阻塞'],
            '并发': ['concurrency', 'parallel', '多线程'],
            'api': ['api', '接口', 'endpoint', '接口'],
            '请求': ['request', 'http请求', '网络请求'],
        }
        
        self._simple_query_cache: Dict[str, str] = {}
    
    def _get_cache_key(
        self,
        query: str,
        context: Optional[QueryContext] = None
    ) -> str:
        """
        生成缓存键
        
        Args:
            query: 查询文本
            context: 查询上下文
            
        Returns:
            缓存键字符串
        """
        cache_data = {
            'query': query,
            'project_type': context.project_type if context else 'general',
            'language': context.detected_language if context else '',
            'framework': context.detected_framework if context else '',
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _is_cache_valid(self, entry: RewriteCacheEntry) -> bool:
        """
        检查缓存是否有效
        
        Args:
            entry: 缓存条目
            
        Returns:
            是否有效
        """
        current_time = time.time()
        return (current_time - entry.timestamp) < self.cache_ttl
    
    def _rule_based_rewrite(
        self,
        query: str,
        query_type: QueryType
    ) -> str:
        """
        基于规则的查询改写
        
        Args:
            query: 原始查询
            query_type: 查询类型
            
        Returns:
            改写后的查询
        """
        rewritten = query
        
        if query_type == QueryType.TERMINOLOGY:
            if not query.lower().startswith(('what is', '什么是')):
                rewritten = f"什么是 {query.replace('什么是', '').strip()}"
        
        elif query_type == QueryType.USAGE:
            if not any(kw in query.lower() for kw in ['如何使用', 'how to']):
                rewritten = f"如何使用{query}"
        
        elif query_type == QueryType.IMPLEMENTATION:
            if not any(kw in query.lower() for kw in ['如何实现', '怎么写']):
                rewritten = f"如何实现{query}"
        
        return rewritten
    
    async def _llm_rewrite_with_retry(
        self,
        query: str,
        query_type: QueryType,
        added_context: List[str],
        context: Optional[QueryContext] = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        带重试的 LLM 改写
        
        Args:
            query: 原始查询
            query_type: 查询类型
            added_context: 添加的上下文
            context: 查询上下文
            max_retries: 最大重试次数
            
        Returns:
            改写结果字典
        """
        context_str = '\n'.join(added_context) if added_context else "无额外上下文"
        project_info = ""
        if context:
            if context.project_type != 'general':
                project_info += f"项目类型: {context.project_type}\n"
            if context.detected_language:
                project_info += f"检测语言: {context.detected_language}\n"
            if context.detected_framework:
                project_info += f"检测框架: {context.detected_framework}\n"
        
        prompt = f"""请将以下查询改写为更适合代码/文档检索的形式。

## 查询信息
原始查询类型: {query_type.value}
项目背景:
{project_info if project_info else context_str}

原始查询: {query}

## 改写要求
1. 补充隐含的技术栈、框架或语言信息
2. 规范化专业术语
3. 保持查询的核心意图不变
4. 如果查询简短且是特定术语，直接返回原查询
5. 输出 JSON 格式: {{"rewritten": "改写后的查询", "confidence": 0.85}}

改写结果:"""

        for attempt in range(max_retries):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "你是一个搜索查询优化专家，擅长将用户的模糊查询改写为精确的检索查询。输出必须是有效的 JSON 格式。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                
                response = await self.llm_service.chat_completion(
                    messages=messages,
                    task_type="lightweight",
                    temperature=0.3,
                    max_tokens=200
                )
                
                import json as json_lib
                try:
                    result = json_lib.loads(response)
                    if 'rewritten' in result:
                        return {
                            'rewritten': result['rewritten'],
                            'confidence': result.get('confidence', 0.8)
                        }
                except:
                    pass
                
                rewritten = response.strip()
                if len(rewritten) < len(query) * 0.3:
                    return {'rewritten': query, 'confidence': 0.5}
                
                return {'rewritten': rewritten, 'confidence': 0.75}
                
            except Exception as e:
                logger.warning(f"LLM rewrite attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return {'rewritten': query, 'confidence': 0.5}
        
        return {'rewritten': query, 'confidence': 0.5}
    
    async def rewrite(
        self,
        query: str,
        context: Optional[QueryContext] = None,
        bypass_cache: bool = False
    ) -> RewriteResult:
        """
        改写查询 v2.0
        
        混合策略流程：
        1. 检查缓存
        2. 简单查询使用规则改写
        3. 复杂查询使用 LLM 改写
        4. 缓存结果
        
        Args:
            query: 原始查询
            context: 查询上下文
            bypass_cache: 强制绕过缓存
            
        Returns:
            改写结果
        """
        logger.info(f"Rewriting query: {query[:100]}...")
        
        cache_key = self._get_cache_key(query, context)
        
        if not bypass_cache and cache_key in self._cache:
            entry = self._cache[cache_key]
            if self._is_cache_valid(entry):
                logger.info(f"Cache hit for query: {query[:50]}...")
                return RewriteResult(
                    original_query=query,
                    rewritten_query=entry.rewritten_query,
                    query_type=entry.query_type,
                    confidence=entry.confidence * 0.95,
                    added_context=entry.added_context,
                    sub_queries=entry.sub_queries
                )
        
        query_type, confidence = self.intent_classifier.classify(query)
        
        added_context = await self._extract_context(query, context)
        
        if self.enable_hybrid_strategy and self.intent_classifier.is_simple_query(query):
            logger.info(f"Using rule-based rewrite for simple query: {query[:50]}...")
            rewritten = self._rule_based_rewrite(query, query_type)
            
            sub_queries = []
            if self._is_complex_query(query):
                sub_queries = await self._decompose_query(query, context)
            
            result = RewriteResult(
                original_query=query,
                rewritten_query=rewritten,
                query_type=query_type,
                confidence=confidence * 0.9,
                added_context=added_context,
                sub_queries=sub_queries
            )
        else:
            logger.info(f"Using LLM rewrite for complex query: {query[:50]}...")
            llm_result = await self._llm_rewrite_with_retry(
                query, query_type, added_context, context
            )
            
            rewritten = llm_result['rewritten']
            llm_confidence = llm_result['confidence']
            
            sub_queries = []
            if self._is_complex_query(query):
                sub_queries = await self._decompose_query(query, context)
            
            combined_confidence = (confidence + llm_confidence) / 2
            
            result = RewriteResult(
                original_query=query,
                rewritten_query=rewritten,
                query_type=query_type,
                confidence=combined_confidence,
                added_context=added_context,
                sub_queries=sub_queries
            )
        
        self._cache[cache_key] = RewriteCacheEntry(
            rewritten_query=result.rewritten_query,
            query_type=result.query_type,
            confidence=result.confidence,
            added_context=result.added_context,
            sub_queries=result.sub_queries,
            timestamp=time.time()
        )
        
        logger.info(
            f"Query rewritten: type={query_type.value}, "
            f"confidence={result.confidence:.2f}, "
            f"sub_queries={len(sub_queries)}"
        )
        
        return result
    
    async def rewrite_batch(
        self,
        queries: List[str],
        context: Optional[QueryContext] = None,
        parallel: bool = True
    ) -> List[RewriteResult]:
        """
        批量改写查询
        
        Args:
            queries: 查询列表
            context: 查询上下文
            parallel: 是否并行处理
            
        Returns:
            改写结果列表
        """
        if not parallel:
            results = []
            for query in queries:
                result = await self.rewrite(query, context)
                results.append(result)
            return results
        
        import asyncio
        
        tasks = [self.rewrite(query, context) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch rewrite failed for query {i}: {str(result)}")
                final_results.append(RewriteResult(
                    original_query=queries[i],
                    rewritten_query=queries[i],
                    query_type=QueryType.GENERAL,
                    confidence=0.0
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def clear_cache(self, older_than: Optional[float] = None) -> int:
        """
        清除缓存
        
        Args:
            older_than: 只清除早于此时间戳的条目（秒）
            
        Returns:
            清除的条目数量
        """
        if older_than is None:
            count = len(self._cache)
            self._cache.clear()
            return count
        
        current_time = time.time()
        keys_to_remove = [
            key for key, entry in self._cache.items()
            if (current_time - entry.timestamp) > older_than
        ]
        
        for key in keys_to_remove:
            del self._cache[key]
        
        return len(keys_to_remove)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        current_time = time.time()
        valid_entries = sum(
            1 for entry in self._cache.values()
            if self._is_cache_valid(entry)
        )
        
        return {
            'total_entries': len(self._cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self._cache) - valid_entries,
            'cache_ttl_seconds': self.cache_ttl
        }
    
    async def _detect_query_type(self, query: str) -> QueryType:
        """检测查询类型"""
        query_lower = query.lower()
        
        type_indicators = {
            QueryType.TERMINOLOGY: ['什么是', 'what is', 'explain', '解释'],
            QueryType.USAGE: ['如何使用', 'how to', 'how do i', '用法', '使用'],
            QueryType.IMPLEMENTATION: ['如何实现', '实现', 'implementation', '怎么写'],
            QueryType.COMPARISON: ['区别', 'difference', 'vs', '比较', '对比'],
            QueryType.DEBUGGING: ['错误', 'error', 'bug', '问题', '异常', '失败'],
            QueryType.ARCHITECTURE: ['架构', 'architecture', '设计', '结构'],
        }
        
        max_matches = 0
        detected_type = QueryType.GENERAL
        
        for qtype, indicators in type_indicators.items():
            matches = sum(1 for ind in indicators if ind in query_lower)
            if matches > max_matches:
                max_matches = matches
                detected_type = qtype
        
        return detected_type
    
    async def _detect_language_framework(
        self,
        query: str,
        context: Optional[QueryContext] = None
    ) -> Dict[str, str]:
        """检测编程语言和框架"""
        detected = {
            'language': '',
            'framework': ''
        }
        
        query_lower = query.lower()
        
        # 编程语言检测
        languages = {
            'python': ['python', 'py', 'python的', 'python中'],
            'javascript': ['javascript', 'js', 'nodejs', 'node.js', 'js的'],
            'typescript': ['typescript', 'ts', 'typescript的'],
            'java': ['java', 'java的'],
            'go': ['go ', 'golang', 'go语言的'],
            'rust': ['rust', 'rust的'],
        }
        
        for lang, indicators in languages.items():
            if any(ind in query_lower for ind in indicators):
                detected['language'] = lang
                break
        
        # 框架检测
        frameworks = {
            'fastapi': ['fastapi', 'fast api'],
            'django': ['django'],
            'flask': ['flask'],
            'react': ['react', 'react的'],
            'vue': ['vue', 'vue的'],
            'spring': ['spring', 'spring boot'],
            'express': ['express', 'expressjs'],
        }
        
        for framework, indicators in frameworks.items():
            if any(ind in query_lower for ind in indicators):
                detected['framework'] = framework
                break
        
        # 从上下文补充
        if context:
            if not detected['language'] and context.detected_language:
                detected['language'] = context.detected_language
            if not detected['framework'] and context.detected_framework:
                detected['framework'] = context.detected_framework
        
        return detected
    
    async def _extract_context(
        self,
        query: str,
        context: Optional[QueryContext] = None
    ) -> List[str]:
        """提取上下文信息"""
        added_context = []
        
        if not context:
            return added_context
        
        # 项目类型上下文
        if context.project_type != 'general':
            added_context.append(f"项目类型: {context.project_type}")
        
        # 文件路径上下文
        if context.file_path:
            added_context.append(f"当前文件: {context.file_path}")
        
        return added_context
    
    async def _execute_rewrite(
        self,
        query: str,
        query_type: QueryType,
        detected: Dict[str, str],
        added_context: List[str],
        context: Optional[QueryContext] = None
    ) -> str:
        """执行改写"""
        # 策略1：基于检测结果改写
        if detected['language'] or detected['framework']:
            rewrite_parts = [query]
            
            if detected['language']:
                rewrite_parts.append(f"[语言: {detected['language']}]")
            if detected['framework']:
                rewrite_parts.append(f"[框架: {detected['framework']}]")
            
            # 根据查询类型调整
            if query_type == QueryType.USAGE:
                rewrite_parts.insert(1, "在")
                rewrite_parts.insert(2, detected['framework'] or detected['language'])
                rewrite_parts.insert(3, "中")
            elif query_type == QueryType.IMPLEMENTATION:
                rewrite_parts.insert(1, f"在{detected['language']}中")
            
            return ' '.join(rewrite_parts)
        
        # 策略2：使用LLM进行智能改写（当检测不到语言/框架时）
        return await self._llm_rewrite(query, query_type, added_context)
    
    async def _llm_rewrite(
        self,
        query: str,
        query_type: QueryType,
        added_context: List[str]
    ) -> str:
        """使用LLM进行智能改写"""
        context_str = '\n'.join(added_context) if added_context else "无额外上下文"
        
        prompt = f"""请将以下查询改写为更适合代码/文档检索的形式。

原始查询类型: {query_type.value}
背景上下文:
{context_str}

原始查询: {query}

要求：
1. 补充隐含的技术栈、框架或语言信息（如果可能）
2. 规范化专业术语
3. 保持查询的核心意图不变
4. 如果查询简短且是特定术语，直接返回原查询
5. 输出改写后的查询，不要输出其他内容

改写后的查询:"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个搜索查询优化专家，擅长将用户的模糊查询改写为精确的检索查询。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await self.llm_service.chat_completion(
                messages=messages,
                task_type="lightweight",
                temperature=0.3,
                max_tokens=150
            )
            
            rewritten = response.strip()
            
            # 如果改写结果与原查询差异太大，使用原查询
            if len(rewritten) < len(query) * 0.5:
                return query
            
            return rewritten
            
        except Exception as e:
            logger.error(f"LLM rewrite failed: {str(e)}")
            return query
    
    def _is_complex_query(self, query: str) -> bool:
        """判断是否为复杂查询"""
        # 复杂查询指标
        complexity_indicators = [
            '和', '与', '以及', ' and ', ' or ',
            '首先', '然后', '接着', '最后',
            '流程', '步骤', '步骤'
        ]
        
        query_lower = query.lower()
        word_count = len(query.split())
        
        # 多词查询 + 包含复杂度指标
        if word_count >= 5:
            for indicator in complexity_indicators:
                if indicator in query_lower:
                    return True
        
        # 特别长的问题
        if word_count >= 10:
            return True
        
        return False
    
    async def _decompose_query(
        self,
        query: str,
        context: Optional[QueryContext] = None
    ) -> List[str]:
        """分解复杂查询为多个子查询"""
        prompt = f"""请将以下复杂查询分解为多个独立的、可检索的子查询。

原始查询: {query}

要求：
1. 分解后的子查询应该能够独立检索
2. 保持每个子查询的核心意图
3. 每个子查询一行，不要编号
4. 输出3-5个子查询

子查询:"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个查询分解专家，擅长将复杂问题分解为多个可独立检索的子问题。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await self.llm_service.chat_completion(
                messages=messages,
                task_type="lightweight",
                temperature=0.5,
                max_tokens=300
            )
            
            # 解析子查询
            sub_queries = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                line = re.sub(r'^[\d\.\-\*]+\s*', '', line)
                if line and len(line) > 3:
                    sub_queries.append(line)
            
            return sub_queries[:5]
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {str(e)}")
            return []


class QueryNormalizer:
    """
    查询标准化器
    
    功能：
    - 术语规范化
    - 同义词扩展
    - 拼写纠错
    """
    
    def __init__(self):
        self.synonym_map = {
            'async': ['async', 'asynchronous', '异步'],
            'await': ['await', '等待', 'awaiting'],
            'function': ['function', '函数', '方法', 'method'],
            'class': ['class', '类'],
            'import': ['import', '导入', '引入'],
            'api': ['api', '接口', 'endpoint', '接口', 'api接口'],
        }
    
    def normalize(self, query: str) -> str:
        """标准化查询"""
        query_lower = query.lower()
        
        # 移除多余空格
        query = re.sub(r'\s+', ' ', query).strip()
        
        # 规范化常见缩写
        replacements = {
            'http request': 'HTTP请求',
            'http response': 'HTTP响应',
            'rest api': 'REST API',
            'json': 'JSON',
            'sql': 'SQL',
        }
        
        for pattern, replacement in replacements.items():
            query_lower = query_lower.replace(pattern, replacement)
        
        return query
    
    def expand_synonyms(self, query: str) -> List[str]:
        """同义词扩展"""
        expanded = [query]
        
        for term, synonyms in self.synonym_map.items():
            if term in query.lower():
                for synonym in synonyms:
                    if synonym not in query and synonym not in expanded:
                        expanded.append(synonym)
        
        return expanded
