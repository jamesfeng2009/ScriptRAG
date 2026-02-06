"""
Unit tests for Intent Parser Agent

测试意图解析智能体的各项功能，包括：
- IntentAnalysis 数据模型验证
- 意图解析功能
- 回退机制
- Prompt 构建
- JSON 响应解析
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.domain.agents.intent_parser import (
    IntentParserAgent,
    IntentAnalysis,
    parse_intent,
    parse_intent_with_context
)


class TestIntentAnalysisModel:
    """测试 IntentAnalysis 数据模型"""
    
    def test_valid_intent_analysis_creation(self):
        """测试创建有效的 IntentAnalysis"""
        intent = IntentAnalysis(
            primary_intent="了解 Python 异步编程的实现",
            keywords=["async", "await", "asyncio", "异步编程"],
            search_sources=["rag"],
            confidence=0.95,
            intent_type="informational",
            language="zh"
        )
        
        assert intent.primary_intent == "了解 Python 异步编程的实现"
        assert len(intent.keywords) == 4
        assert intent.confidence == 0.95
        assert intent.intent_type == "informational"
        assert intent.language == "zh"
    
    def test_intent_analysis_default_values(self):
        """测试默认值的设置"""
        intent = IntentAnalysis(
            primary_intent="测试意图",
            keywords=["test"]
        )
        
        assert intent.search_sources == []
        assert intent.confidence == 0.8
        assert intent.alternative_intents == []
        assert intent.intent_type == "informational"
        assert intent.language == "zh"
    
    def test_intent_analysis_confidence_validation(self):
        """测试置信度边界值"""
        intent_min = IntentAnalysis(
            primary_intent="测试",
            keywords=[],
            confidence=0.0
        )
        assert intent_min.confidence == 0.0
        
        intent_max = IntentAnalysis(
            primary_intent="测试",
            keywords=[],
            confidence=1.0
        )
        assert intent_max.confidence == 1.0
    
    def test_intent_analysis_confidence_out_of_range(self):
        """测试置信度超出范围"""
        with pytest.raises(ValueError):
            IntentAnalysis(
                primary_intent="测试",
                keywords=[],
                confidence=1.5
            )
        
        with pytest.raises(ValueError):
            IntentAnalysis(
                primary_intent="测试",
                keywords=[],
                confidence=-0.1
            )
    
    def test_intent_analysis_empty_primary_intent(self):
        """测试空主要意图"""
        with pytest.raises(ValueError):
            IntentAnalysis(
                primary_intent="",
                keywords=[]
            )
    
    def test_intent_analysis_logging_format(self):
        """测试日志格式转换"""
        intent = IntentAnalysis(
            primary_intent="测试意图",
            keywords=["test", "demo"],
            search_sources=["rag"],
            confidence=0.9,
            intent_type="informational",
            language="zh",
            alternative_intents=[
                {"intent": "备选意图1", "keywords": ["kw1"]}
            ]
        )
        
        log_data = intent.model_dump_for_logging()
        
        assert log_data["primary_intent"] == "测试意图"
        assert log_data["keywords"] == ["test", "demo"]
        assert log_data["confidence"] == 0.9
        assert log_data["alternative_intents_count"] == 1


class TestIntentParserAgent:
    """测试 IntentParserAgent 智能体"""
    
    @pytest.mark.asyncio
    async def test_parse_intent_success(self):
        """测试成功解析意图"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "了解 Python 异步编程的实现方式",
    "keywords": ["async", "await", "asyncio", "异步编程", "coroutine"],
    "search_sources": ["rag"],
    "confidence": 0.95,
    "alternative_intents": [
        {
            "intent": "了解 asyncio 库的使用",
            "keywords": ["asyncio", "事件循环"]
        }
    ],
    "intent_type": "informational",
    "language": "zh"
}
```""")
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent("Python 异步编程怎么实现")
        
        assert result.primary_intent == "了解 Python 异步编程的实现方式"
        assert "async" in result.keywords
        assert "await" in result.keywords
        assert "rag" in result.search_sources
        assert result.confidence == 0.95
        assert result.intent_type == "informational"
        assert len(result.alternative_intents) == 1
    
    @pytest.mark.asyncio
    async def test_parse_intent_without_json_markdown(self):
        """测试不带 JSON 标记的响应"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
{
    "primary_intent": "查询 REST API 设计原则",
    "keywords": ["REST", "API", "设计", "原则"],
    "search_sources": ["rag"],
    "confidence": 0.88,
    "intent_type": "informational",
    "language": "zh"
}
""")
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent("REST API 怎么设计")
        
        assert result.primary_intent == "查询 REST API 设计原则"
        assert "REST" in result.keywords
        assert "API" in result.keywords
    
    @pytest.mark.asyncio
    async def test_parse_intent_english_query(self):
        """测试英文查询"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "Learn how to implement async/await in Python",
    "keywords": ["async", "await", "Python", "coroutine", "asyncio"],
    "search_sources": ["rag"],
    "confidence": 0.92,
    "intent_type": "informational",
    "language": "en"
}
```""")
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent("How to use async await in Python")
        
        assert result.language == "en"
        assert result.intent_type == "informational"
    
    @pytest.mark.asyncio
    async def test_parse_intent_multiple_sources(self):
        """测试多数据源推荐"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "了解最新的 AI 发展趋势",
    "keywords": ["AI", "机器学习", "最新趋势"],
    "search_sources": ["rag", "web"],
    "confidence": 0.85,
    "intent_type": "informational",
    "language": "zh"
}
```""")
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent("AI 未来发展趋势是什么")
        
        assert "rag" in result.search_sources
        assert "web" in result.search_sources
    
    @pytest.mark.asyncio
    async def test_parse_intent_llm_failure(self):
        """测试 LLM 调用失败时的回退"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(side_effect=Exception("API Error"))
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent("Python 异步编程")
        
        assert result.primary_intent == "Python 异步编程"
        assert len(result.keywords) > 0
        assert result.confidence == 0.5
        assert result.search_sources == ["rag"]
    
    @pytest.mark.asyncio
    async def test_parse_intent_invalid_json(self):
        """测试无效 JSON 响应"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="This is not JSON")
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent("test query")
        
        assert result.primary_intent == "test query"
        assert result.confidence == 0.5
        assert len(result.keywords) > 0
    
    @pytest.mark.asyncio
    async def test_parse_intent_missing_fields(self):
        """测试缺少字段的响应"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "测试意图",
    "keywords": ["test"]
}
```""")
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent("test")
        
        assert result.primary_intent == "测试意图"
        assert result.search_sources == ["rag"]
        assert result.confidence == 0.8


class TestIntentParserFallback:
    """测试意图解析回退机制"""
    
    def test_fallback_keyword_extraction(self):
        """测试回退时的关键词提取"""
        llm_service = MagicMock()
        agent = IntentParserAgent(llm_service)
        
        result = agent._fallback_intent("Python 异步编程怎么实现")
        
        assert result.primary_intent == "Python 异步编程怎么实现"
        assert "python" in result.keywords or "异步" in result.keywords
        assert result.confidence == 0.5
    
    def test_fallback_removes_stop_words(self):
        """测试回退时移除停用词"""
        llm_service = MagicMock()
        agent = IntentParserAgent(llm_service)
        
        result = agent._fallback_intent("这个的怎么是什么")
        
        assert "这个" not in result.keywords
        assert "的" not in result.keywords
        assert "怎么" not in result.keywords
        assert "是" not in result.keywords
        assert "什么" not in result.keywords
    
    def test_fallback_language_detection_chinese(self):
        """测试中文语言检测"""
        llm_service = MagicMock()
        agent = IntentParserAgent(llm_service)
        
        result = agent._fallback_intent("Python 异步编程")
        
        assert result.language == "zh"
    
    def test_fallback_language_detection_english(self):
        """测试英文语言检测"""
        llm_service = MagicMock()
        agent = IntentParserAgent(llm_service)
        
        result = agent._fallback_intent("how to use async await")
        
        assert result.language == "en"
    
    def test_fallback_mixed_language(self):
        """测试混合语言检测"""
        llm_service = MagicMock()
        agent = IntentParserAgent(llm_service)
        
        result = agent._fallback_intent("Python 异步编程 async await")
        
        assert result.language == "zh"


class TestIntentParserWithContext:
    """测试带上下文的意图解析"""
    
    @pytest.mark.asyncio
    async def test_parse_intent_with_context_success(self):
        """测试带上下文成功解析"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "了解 FastAPI 中的异步路由实现",
    "keywords": ["FastAPI", "async", "路由", "APIRouter"],
    "search_sources": ["rag"],
    "confidence": 0.95,
    "intent_type": "informational",
    "language": "zh"
}
```""")
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent_with_context(
            "异步路由怎么写",
            context="项目使用 FastAPI 框架开发 Web 应用"
        )
        
        assert "FastAPI" in result.keywords or "路由" in result.keywords
        assert result.confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_parse_intent_with_context_failure(self):
        """测试带上下文时 LLM 失败"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(side_effect=Exception("API Error"))
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent_with_context(
            "test query",
            context="some context"
        )
        
        assert result.primary_intent == "test query"
        assert len(result.keywords) > 0
    
    @pytest.mark.asyncio
    async def test_parse_intent_with_empty_context(self):
        """测试空上下文"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "测试查询",
    "keywords": ["test"],
    "search_sources": ["rag"],
    "confidence": 0.9,
    "intent_type": "informational",
    "language": "zh"
}
```""")
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent_with_context("test query", context="")
        
        assert result.primary_intent == "测试查询"


class TestPromptBuilding:
    """测试 Prompt 构建"""
    
    def test_build_prompt_structure(self):
        """测试 prompt 包含必要信息"""
        llm_service = MagicMock()
        agent = IntentParserAgent(llm_service)
        
        messages = agent._build_prompt("test query")
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "test query" in messages[1]["content"]
    
    def test_build_prompt_with_context(self):
        """测试带上下文的 prompt 构建"""
        llm_service = MagicMock()
        agent = IntentParserAgent(llm_service)
        
        messages = agent._build_prompt_with_context(
            "怎么实现",
            context="使用 FastAPI 开发"
        )
        
        assert len(messages) == 2
        assert "FastAPI" in messages[0]["content"]
        assert "怎么实现" in messages[1]["content"]


class TestJSONParsing:
    """测试 JSON 响应解析"""
    
    def test_parse_json_with_markdown(self):
        """测试带 markdown 标记的 JSON 解析"""
        llm_service = MagicMock()
        agent = IntentParserAgent(llm_service)
        
        response = """
Some text before
```json
{
    "primary_intent": "测试",
    "keywords": ["test"],
    "confidence": 0.9
}
```
Some text after
"""
        
        result = agent._parse_response(response)
        
        assert result.primary_intent == "测试"
        assert "test" in result.keywords
    
    def test_parse_json_without_markdown(self):
        """测试不带 markdown 标记的 JSON 解析"""
        llm_service = MagicMock()
        agent = IntentParserAgent(llm_service)
        
        response = '{"primary_intent": "直接解析", "keywords": ["demo"], "confidence": 0.8}'
        
        result = agent._parse_response(response)
        
        assert result.primary_intent == "直接解析"
        assert "demo" in result.keywords
    
    def test_parse_invalid_json(self):
        """测试无效 JSON"""
        llm_service = MagicMock()
        agent = IntentParserAgent(llm_service)
        
        with pytest.raises(ValueError):
            agent._parse_response("not json at all")
    
    def test_parse_json_missing_primary_intent(self):
        """测试缺少 primary_intent 的 JSON"""
        llm_service = MagicMock()
        agent = IntentParserAgent(llm_service)
        
        with pytest.raises(ValueError):
            agent._parse_response('{"keywords": ["test"]}')


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    @pytest.mark.asyncio
    async def test_parse_intent_convenience_function(self):
        """测试 parse_intent 便捷函数"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "便捷函数测试",
    "keywords": ["test"],
    "confidence": 0.9
}
```""")
        
        result = await parse_intent("test query", llm_service)
        
        assert result.primary_intent == "便捷函数测试"
    
    @pytest.mark.asyncio
    async def test_parse_intent_with_context_function(self):
        """测试 parse_intent_with_context 便捷函数"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "带上下文测试",
    "keywords": ["context"],
    "confidence": 0.95
}
```""")
        
        result = await parse_intent_with_context(
            "query",
            llm_service,
            context="some context"
        )
        
        assert result.primary_intent == "带上下文测试"


class TestIntentTypes:
    """测试不同意图类型"""
    
    @pytest.mark.asyncio
    async def test_navigational_intent(self):
        """测试导航意图"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "找到认证模块的代码",
    "keywords": ["auth", "认证", "登录"],
    "search_sources": ["rag"],
    "confidence": 0.9,
    "intent_type": "navigational",
    "language": "zh"
}
```""")
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent("认证模块在哪里")
        
        assert result.intent_type == "navigational"
    
    @pytest.mark.asyncio
    async def test_transactional_intent(self):
        """测试事务意图"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "执行用户注册操作",
    "keywords": ["注册", "register", "用户"],
    "search_sources": ["mysql"],
    "confidence": 0.88,
    "intent_type": "transactional",
    "language": "zh"
}
```""")
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent("怎么实现用户注册")
        
        assert result.intent_type == "transactional"
    
    @pytest.mark.asyncio
    async def test_computational_intent(self):
        """测试计算意图"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "计算平均值",
    "keywords": ["平均", "avg", "计算"],
    "search_sources": ["mysql"],
    "confidence": 0.92,
    "intent_type": "computational",
    "language": "zh"
}
```""")
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent("怎么计算平均值")
        
        assert result.intent_type == "computational"


class TestAlternativeIntents:
    """测试备选意图"""
    
    @pytest.mark.asyncio
    async def test_alternative_intents_structure(self):
        """测试备选意图结构"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "主要意图",
    "keywords": ["kw1", "kw2"],
    "search_sources": ["rag"],
    "confidence": 0.9,
    "alternative_intents": [
        {"intent": "备选1", "keywords": ["alt1"]},
        {"intent": "备选2", "keywords": ["alt2", "alt3"]}
    ],
    "intent_type": "informational",
    "language": "zh"
}
```""")
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent("test")
        
        assert len(result.alternative_intents) == 2
        assert result.alternative_intents[0]["intent"] == "备选1"
        assert "alt1" in result.alternative_intents[0]["keywords"]
    
    @pytest.mark.asyncio
    async def test_no_alternative_intents(self):
        """测试无备选意图"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "唯一意图",
    "keywords": ["only"],
    "search_sources": ["rag"],
    "confidence": 0.95,
    "intent_type": "informational",
    "language": "zh"
}
```""")
        
        agent = IntentParserAgent(llm_service)
        result = await agent.parse_intent("test")
        
        assert len(result.alternative_intents) == 0
