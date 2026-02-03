"""Query Rewriter - 查询改写与分解

功能：
1. 查询意图改写：将不完整/模糊查询改写为更适合检索的形式
2. 复杂查询分解：将复杂查询分解为多个子查询
3. 上下文增强：基于项目上下文补充隐含信息

解决的问题：
- 用户查询不完整导致召回率低
- 隐含的技术栈/框架信息丢失
- 复杂查询难以精确匹配
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
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


class QueryRewriter:
    """
    查询改写器
    
    功能：
    - 意图消歧：确定查询的具体意图
    - 信息补充：添加隐含的技术栈/框架信息
    - 规范化：统一术语表述
    - 分解：将复杂查询分解为多个子查询
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        project_context: Optional[Dict] = None
    ):
        self.llm_service = llm_service
        self.project_context = project_context or {}
        
        # 术语规范化映射
        self.term_normalization = {
            '异步': ['async', 'asyncio', 'await', '非阻塞'],
            '并发': ['concurrency', 'parallel', '多线程'],
            'api': ['api', '接口', 'endpoint', '接口'],
            '请求': ['request', 'http请求', '网络请求'],
        }
    
    async def rewrite(
        self,
        query: str,
        context: Optional[QueryContext] = None
    ) -> RewriteResult:
        """
        改写查询
        
        Args:
            query: 原始查询
            context: 查询上下文
            
        Returns:
            改写结果
        """
        logger.info(f"Rewriting query: {query[:100]}...")
        
        # 1. 检测查询类型
        query_type = await self._detect_query_type(query)
        
        # 2. 检测编程语言和框架
        detected = await self._detect_language_framework(query, context)
        
        # 3. 提取上下文信息
        added_context = await self._extract_context(query, context)
        
        # 4. 执行改写
        rewritten = await self._execute_rewrite(
            query, query_type, detected, added_context, context
        )
        
        # 5. 如果是复杂查询，分解为子查询
        sub_queries = []
        if self._is_complex_query(query):
            sub_queries = await self._decompose_query(query, context)
        
        result = RewriteResult(
            original_query=query,
            rewritten_query=rewritten,
            query_type=query_type,
            confidence=0.85,
            added_context=added_context,
            sub_queries=sub_queries
        )
        
        logger.info(
            f"Query rewritten: type={query_type.value}, "
            f"sub_queries={len(sub_queries)}"
        )
        
        return result
    
    async def rewrite_batch(
        self,
        queries: List[str],
        context: Optional[QueryContext] = None
    ) -> List[RewriteResult]:
        """批量改写查询"""
        results = []
        for query in queries:
            result = await self.rewrite(query, context)
            results.append(result)
        return results
    
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
