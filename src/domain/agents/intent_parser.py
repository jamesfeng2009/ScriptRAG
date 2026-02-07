"""Intent Parser Agent - 意图解析智能体

本模块实现意图解析智能体，负责：
1. 分析用户查询意图
2. 提取检索关键词
3. 推荐数据源
4. 生成备选意图
5. 从配置文件动态检测主题

用于 Agentic RAG 系统的第一步，让检索更智能。
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any

from ...services.llm.service import LLMService
from ...infrastructure.logging import get_agent_logger
from ...domain.models import IntentAnalysis
from ...domain.skill_loader import SkillConfigLoader


logger = logging.getLogger(__name__)
agent_logger = get_agent_logger(__name__)


class IntentParserAgent:
    """意图解析智能体
    
    使用 LLM 分析用户查询，提取：
    1. 主要意图 - 用户真正想查询什么
    2. 关键词 - 用于向量检索的关键词
    3. 建议数据源 - rag/mysql/es/neo4j/web
    4. 置信度 - 分析的可信程度
    5. 主题检测 - 从独立的主题配置文件动态检测
    
    示例:
        query = "Python 异步编程怎么实现"
        result = await agent.parse_intent(query)
        # result.primary_intent = "了解 Python 异步编程的实现方式"
        # result.keywords = ["async", "await", "asyncio", "异步编程"]
        # result.search_sources = ["rag"]
    """
    
    def __init__(self, llm_service: LLMService, config_path: str = "config/skills.yaml"):
        """
        初始化意图解析智能体
        
        Args:
            llm_service: LLM 服务实例
            config_path: 技能配置文件路径
        """
        self.llm_service = llm_service
        self.config_path = config_path
        self._theme_loader = SkillConfigLoader(config_path)
    
    async def parse_intent(self, query: str) -> IntentAnalysis:
        """
        解析查询意图
        
        Args:
            query: 用户原始查询
            
        Returns:
            IntentAnalysis: 意图分析结果
        """
        logger.info(f"Parsing intent for query: {query[:100]}...")
        
        messages = self._build_prompt(query)
        
        try:
            response = await self.llm_service.chat_completion(
                messages=messages,
                task_type="lightweight",
                temperature=0.3,
                max_tokens=1000
            )
            
            intent_data = self._parse_response(response)
            
            agent_logger.log_agent_transition(
                from_agent="user",
                to_agent="intent_parser",
                reason="parse_intent"
            )
            
            logger.info(
                f"Intent parsed: primary_intent={intent_data.primary_intent[:50]}..., "
                f"keywords={intent_data.keywords}, "
                f"sources={intent_data.search_sources}, "
                f"confidence={intent_data.confidence:.2f}"
            )
            
            return intent_data
            
        except Exception as e:
            logger.error(f"Intent parsing failed: {str(e)}")
            return self._fallback_intent(query)
    
    def detect_theme(self, query: str) -> Optional[str]:
        """
        从配置文件中检测主题
        
        Args:
            query: 用户查询
            
        Returns:
            检测到的主题名称，未检测到返回 None
        """
        try:
            theme = self._theme_loader.detect_theme(query)
            if theme:
                logger.info(f"Detected theme from config: {theme}")
                return theme
            return None
        except Exception as e:
            logger.warning(f"Theme detection failed: {str(e)}")
            return None
    
    def get_theme_skills(self, theme: str) -> Optional[Dict[str, Any]]:
        """
        获取主题的可用技能列表
        
        Args:
            theme: 主题名称
            
        Returns:
            技能配置字典，未找到返回 None
        """
        try:
            skills = self._theme_loader.get_theme_skills(theme)
            if skills:
                logger.info(f"Loaded {len(skills)} skills for theme: {theme}")
                return skills
            return None
        except Exception as e:
            logger.warning(f"Failed to load theme skills: {str(e)}")
            return None
    
    def get_theme_skill_options(self, theme: str) -> List[Dict[str, Any]]:
        """
        获取主题的技能选项（给前端展示）
        
        Args:
            theme: 主题名称
            
        Returns:
            技能选项列表
        """
        try:
            return self._theme_loader.get_theme_skill_options(theme)
        except Exception as e:
            logger.warning(f"Failed to load skill options: {str(e)}")
            return []
    
    def get_theme_default_skill(self, theme: str) -> str:
        """
        获取主题的默认技能
        
        Args:
            theme: 主题名称
            
        Returns:
            默认技能名称
        """
        try:
            return self._theme_loader.get_theme_default_skill(theme)
        except Exception as e:
            logger.warning(f"Failed to load default skill: {str(e)}")
            return ""
    
    def list_available_themes(self) -> List[str]:
        """
        列出可用的主题列表
        
        Returns:
            主题名称列表
        """
        try:
            return self._theme_loader.list_available_themes()
        except Exception as e:
            logger.warning(f"Failed to list themes: {str(e)}")
            return []
    
    def _build_prompt(self, query: str) -> List[Dict[str, str]]:
        """
        构建意图解析的 prompt
        
        Args:
            query: 用户查询
            
        Returns:
            Messages 列表
        """
        return [
            {
                "role": "system",
                "content": """你是一个查询意图解析专家。你的任务是分析用户查询，提取用于检索的关键信息。

请分析用户查询，返回以下信息（JSON 格式）：

1. **primary_intent**: 用户真正想查询什么（用简洁的中文描述）
2. **keywords**: 用于检索的关键词列表（5-10个，包含技术术语和核心概念）
3. **search_sources**: 建议的数据源（从以下选项中选择）：
   - "rag": 向量数据库检索（用于语义搜索）
   - "mysql": MySQL 检索（用于结构化数据查询）
   - "es": Elasticsearch 检索（用于全文搜索）
   - "neo4j": Neo4j 检索（用于关系和图查询）
   - "web": Web 搜索（用于获取最新信息）
4. **confidence**: 分析置信度（0.0-1.0）
5. **alternative_intents**: 可能的备选意图列表（每个包含 intent 和 keywords）
6. **intent_type**: 意图类型
   - "informational": 用户想了解某个概念或知识
   - "navigational": 用户想找到特定的资源或页面
   - "transactional": 用户想执行某个操作
   - "computational": 用户想获取计算结果
7. **language**: 查询语言（zh/en/mixed）

返回格式示例：
```json
{
    "primary_intent": "了解 Python 异步编程的 async/await 语法",
    "keywords": ["async", "await", "asyncio", "异步编程", "Python", "coroutine", "异步函数"],
    "search_sources": ["rag"],
    "confidence": 0.95,
    "alternative_intents": [
        {
            "intent": "了解 asyncio 库的使用方法",
            "keywords": ["asyncio", "Python", "异步库", "事件循环"]
        }
    ],
    "intent_type": "informational",
    "language": "zh"
}
```

请只返回 JSON，不要有其他内容。"""
            },
            {
                "role": "user",
                "content": f"请分析以下查询的意图：\n\n{query}"
            }
        ]
    
    def _parse_response(self, response: str) -> IntentAnalysis:
        """
        解析 LLM 返回的 JSON 响应
        
        Args:
            response: LLM 原始响应
            
        Returns:
            IntentAnalysis: 解析后的意图分析结果
        """
        try:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
            
            return IntentAnalysis(
                primary_intent=data.get("primary_intent", ""),
                keywords=data.get("keywords", []),
                search_sources=data.get("search_sources", ["rag"]),
                confidence=data.get("confidence", 0.8),
                alternative_intents=data.get("alternative_intents", []),
                intent_type=data.get("intent_type", "informational"),
                language=data.get("language", "zh")
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse intent JSON: {e}, response: {response[:200]}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except KeyError as e:
            logger.warning(f"Missing required field in intent: {e}")
            raise ValueError(f"Missing required field in intent: {e}")
    
    def _fallback_intent(self, query: str) -> IntentAnalysis:
        """
        意图解析失败时的回退方法
        
        使用简单的规则提取关键词，而不是调用 LLM
        
        Args:
            query: 用户查询
            
        Returns:
            IntentAnalysis: 基本意图分析结果
        """
        logger.warning(f"Using fallback intent parsing for: {query[:50]}")
        
        words = re.findall(r'\b\w+\b', query.lower())
        
        stop_words = {
            '的', '是', '在', '和', '与', '或', '如何', '怎么', '什么',
            'how', 'what', 'the', 'is', 'and', 'to', 'of', 'in'
        }
        
        keywords = [w for w in words if w not in stop_words and len(w) > 1]
        
        return IntentAnalysis(
            primary_intent=query,
            keywords=keywords[:10],
            search_sources=["rag"],
            confidence=0.5,
            alternative_intents=[],
            intent_type="informational",
            language="zh" if any('\u4e00' <= c <= '\u9fff' for c in query) else "en"
        )
    
    async def parse_intent_with_context(
        self,
        query: str,
        context: Optional[str] = None
    ) -> IntentAnalysis:
        """
        带上下文的意图解析
        
        当有额外上下文信息时，使用更精确的意图解析
        
        Args:
            query: 用户查询
            context: 额外上下文（如项目背景、历史对话等）
            
        Returns:
            IntentAnalysis: 意图分析结果
        """
        logger.info(f"Parsing intent with context for query: {query[:100]}...")
        
        messages = self._build_prompt_with_context(query, context)
        
        try:
            response = await self.llm_service.chat_completion(
                messages=messages,
                task_type="high_performance",
                temperature=0.2,
                max_tokens=1200
            )
            
            intent_data = self._parse_response(response)
            
            logger.info(
                f"Intent parsed with context: primary_intent={intent_data.primary_intent[:50]}..., "
                f"confidence={intent_data.confidence:.2f}"
            )
            
            return intent_data
            
        except Exception as e:
            logger.error(f"Intent parsing with context failed: {str(e)}")
            return await self.parse_intent(query)
    
    def _build_prompt_with_context(
        self,
        query: str,
        context: Optional[str]
    ) -> List[Dict[str, str]]:
        """
        构建带上下文的 prompt
        
        Args:
            query: 用户查询
            context: 上下文信息
            
        Returns:
            Messages 列表
        """
        context_section = ""
        if context:
            context_section = f"""
额外上下文信息：
{context}

请结合上下文信息分析用户查询的意图。
"""
        
        return [
            {
                "role": "system",
                "content": f"""你是一个查询意图解析专家，负责分析技术查询的意图。

任务目标：
理解用户真正想查询什么，提取用于检索的关键词和建议数据源。

分析维度：
1. primary_intent: 用户的核心需求（1-2句话描述）
2. keywords: 关键词列表（8-12个，包含技术术语）
3. search_sources: 建议数据源
4. confidence: 置信度
5. alternative_intents: 可能的备选意图
6. intent_type: 意图类型
7. language: 查询语言

注意事项：
- 关键词应该包含技术术语、框架名、API 名等
- 数据源选择要考虑查询类型（代码、文档、最新信息等）
- 如果是代码相关查询，优先选择 "rag"
- 如果是最新信息查询，添加 "web" 数据源

{context_section}请返回 JSON 格式的分析结果。"""
            },
            {
                "role": "user",
                "content": f"查询：{query}"
            }
        ]


async def parse_intent(
    query: str,
    llm_service: LLMService
) -> IntentAnalysis:
    """
    便捷的意图解析函数
    
    Args:
        query: 用户查询
        llm_service: LLM 服务
        
    Returns:
        IntentAnalysis: 意图分析结果
    """
    agent = IntentParserAgent(llm_service)
    return await agent.parse_intent(query)


async def parse_intent_with_context(
    query: str,
    llm_service: LLMService,
    context: Optional[str] = None
) -> IntentAnalysis:
    """
    带上下文的意图解析函数
    
    Args:
        query: 用户查询
        llm_service: LLM 服务
        context: 上下文信息
        
    Returns:
        IntentAnalysis: 意图分析结果
    """
    agent = IntentParserAgent(llm_service)
    return await agent.parse_intent_with_context(query, context)
