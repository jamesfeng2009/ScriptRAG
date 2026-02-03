"""查询扩展 - 扩展查询以提高召回率

本模块实现查询扩展策略，通过生成相关查询和提取关键术语来提高检索召回率。
"""

import logging
import re
from typing import List, Optional
from ..services.llm.service import LLMService


logger = logging.getLogger(__name__)


class QueryExpansion:
    """
    查询扩展器
    
    功能：
    - 使用LLM生成相关查询
    - 提取关键词
    - 同义词扩展
    - 查询重写
    """
    
    def __init__(self, llm_service: LLMService, max_expansions: int = 2):
        """
        初始化查询扩展器
        
        Args:
            llm_service: LLM服务实例
            max_expansions: 最大扩展查询数量
        """
        self.llm_service = llm_service
        self.max_expansions = max_expansions
    
    async def expand_query(self, query: str) -> List[str]:
        """
        扩展查询
        
        Args:
            query: 原始查询
            
        Returns:
            扩展后的查询列表（包含原始查询）
        """
        logger.info(f"Expanding query: {query[:100]}...")
        
        # 原始查询始终包含
        queries = [query]
        
        try:
            # 生成相关查询
            related_queries = await self._generate_related_queries(query)
            
            # 添加扩展查询（最多max_expansions个）
            queries.extend(related_queries[:self.max_expansions])
            
            logger.info(f"Query expanded to {len(queries)} queries")
            return queries
            
        except Exception as e:
            logger.error(f"Query expansion failed: {str(e)}")
            # 失败时返回原始查询
            return queries
    
    async def _generate_related_queries(self, query: str) -> List[str]:
        """
        使用LLM生成相关查询
        
        Args:
            query: 原始查询
            
        Returns:
            相关查询列表
        """
        prompt = f"""请为以下查询生成2个相关的搜索查询，用于提高检索召回率。

原始查询: {query}

要求：
1. 生成的查询应该与原始查询语义相关
2. 使用不同的表达方式或同义词
3. 保持查询的核心意图
4. 每行一个查询，不要编号

示例：
原始查询: FastAPI的异步特性
相关查询1: FastAPI异步编程实现
相关查询2: FastAPI async/await用法

请生成相关查询："""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个搜索查询优化专家，擅长生成相关的搜索查询。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await self.llm_service.chat_completion(
                messages=messages,
                task_type="lightweight",
                temperature=0.7,
                max_tokens=200
            )
            
            # 解析响应
            related_queries = self._parse_related_queries(response)
            
            logger.info(f"Generated {len(related_queries)} related queries")
            return related_queries
            
        except Exception as e:
            logger.error(f"Failed to generate related queries: {str(e)}")
            return []
    
    def _parse_related_queries(self, response: str) -> List[str]:
        """
        解析LLM响应中的相关查询
        
        Args:
            response: LLM响应文本
            
        Returns:
            解析出的查询列表
        """
        queries = []
        
        # 按行分割
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 跳过空行和标题行
            if not line or '原始查询' in line or '相关查询' in line:
                continue
            
            # 移除编号和标记
            line = re.sub(r'^[\d\.\-\*]+\s*', '', line)
            line = re.sub(r'^相关查询\d*[:：]\s*', '', line)
            
            if line and len(line) > 3:  # 至少3个字符
                queries.append(line)
        
        return queries
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        提取查询中的关键词
        
        Args:
            query: 查询文本
            
        Returns:
            关键词列表
        """
        # 简单的关键词提取（可以使用更复杂的NLP方法）
        # 移除常见停用词
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这'
        }
        
        # 分词（简单按空格和标点分割）
        words = re.findall(r'\w+', query)
        
        # 过滤停用词和短词
        keywords = [
            word for word in words
            if word not in stopwords and len(word) > 1
        ]
        
        return keywords


class QueryOptimizer:
    """
    查询优化器
    
    功能：
    - 清理查询文本
    - 规范化表达
    - 提取核心意图
    """
    
    def optimize_query(self, query: str) -> str:
        """
        优化查询文本
        
        Args:
            query: 原始查询
            
        Returns:
            优化后的查询
        """
        # 1. 清理空白字符
        optimized = ' '.join(query.split())
        
        # 2. 移除特殊字符（保留中英文、数字、常用标点）
        optimized = re.sub(r'[^\w\s\u4e00-\u9fff\-_/.]', ' ', optimized)
        
        # 3. 规范化空格
        optimized = ' '.join(optimized.split())
        
        # 4. 转换为小写（英文部分）
        # 保留中文不变
        
        logger.debug(f"Query optimized: '{query}' -> '{optimized}'")
        
        return optimized.strip()
    
    def extract_intent(self, query: str) -> dict:
        """
        提取查询意图
        
        Args:
            query: 查询文本
            
        Returns:
            意图信息字典
        """
        intent = {
            'type': 'general',  # general, how_to, what_is, troubleshooting
            'keywords': [],
            'entities': []
        }
        
        # 检测查询类型
        if any(word in query.lower() for word in ['how', '如何', '怎么', '怎样']):
            intent['type'] = 'how_to'
        elif any(word in query.lower() for word in ['what', '什么', '是什么']):
            intent['type'] = 'what_is'
        elif any(word in query.lower() for word in ['error', '错误', '问题', 'bug']):
            intent['type'] = 'troubleshooting'
        
        # 提取关键词（简化版）
        words = re.findall(r'\w+', query)
        intent['keywords'] = [w for w in words if len(w) > 2]
        
        return intent
