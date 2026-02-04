"""幻觉检测完善模块

功能：
1. 细粒度幻觉检测 - 句子级别的幻觉检测
2. 幻觉类型分类 - 区分代码幻觉、事实幻觉、逻辑幻觉等
3. 幻觉预防机制 - 在生成前添加约束
4. 幻觉修复机制 - 检测到幻觉后尝试修复而非重新生成

解决的问题：
- 幻觉检测粒度太粗，只能检测整个片段
- 缺少幻觉类型分类，无法针对性处理
- 缺少预防机制，只能事后检测
- 只能重新生成，无法修复局部幻觉

使用示例:
```python
# 细粒度检测
detector = GranularHallucinationDetector(llm_service)
results = await detector.detect_sentence_level(fragment, retrieved_docs)

# 幻觉分类
classifier = HallucinationClassifier()
hallucination_type = classifier.classify_hallucination("使用了不存在的函数")

# 预防机制
prevention = HallucinationPrevention()
enhanced_prompt = prevention.enhance_prompt_with_constraints(original_prompt, docs)

# 幻觉修复
repairer = HallucinationRepairer(llm_service)
repaired = await repairer.repair_hallucination(fragment, hallucinations, docs)
```
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple
from difflib import SequenceMatcher
from urllib.parse import urlparse

from ..services.llm.service import LLMService

logger = logging.getLogger(__name__)


class HallucinationType(Enum):
    """幻觉类型"""
    CODE = "code"              # 代码幻觉：不存在的函数、类、参数
    FACT = "fact"             # 事实幻觉：错误的技术细节
    LOGIC = "logic"           # 逻辑幻觉：前后矛盾的陈述
    SOURCE = "source"          # 来源幻觉：引用不存在的文档
    TEMPORAL = "temporal"      # 时间幻觉：过时或不准确的时间信息
    VERSION = "version"        # 版本幻觉：错误版本信息


class HallucinationSeverity(Enum):
    """幻觉严重程度"""
    CRITICAL = "critical"      # 代码幻觉：严重（会误导用户）
    HIGH = "high"              # 事实幻觉：高
    MEDIUM = "medium"          # 逻辑幻觉：中等
    LOW = "low"                # 来源幻觉：轻微


@dataclass
class SentenceHallucinationResult:
    """句子级幻觉检测结果"""
    sentence: str
    sentence_index: int
    is_hallucination: bool
    confidence: float
    hallucination_type: Optional[HallucinationType] = None
    severity: Optional[HallucinationSeverity] = None
    supporting_source: Optional[str] = None
    evidence: Optional[str] = None
    suggested_correction: Optional[str] = None
    reasoning: str = ""


@dataclass
class FragmentHallucinationResult:
    """片段级幻觉检测结果"""
    fragment_id: str
    overall_valid: bool
    sentence_results: List[SentenceHallucinationResult] = field(default_factory=list)
    hallucination_summary: Dict[str, int] = field(default_factory=dict)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    needs_regeneration: bool = False
    needs_repair: bool = False


@dataclass
class CodeEntity:
    """代码实体"""
    name: str
    entity_type: str  # function, class, method, parameter, variable
    signature: Optional[str] = None
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    docstring: Optional[str] = None


@dataclass
class RepairSuggestion:
    """修复建议"""
    original_text: str
    corrected_text: str
    reason: str
    confidence: float
    source: Optional[str] = None


class PatternExtractor:
    """模式提取器"""
    
    CODE_PATTERNS = {
        'function_call': r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        'method_call': r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        'class_definition': r'\bclass\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
        'import_statement': r'\b(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
        'decorator': r'@([a-zA-Z_][a-zA-Z0-9_]*)',
        'parameter': r'(?:def|async\s+def)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*?\b([a-zA-Z_][a-zA-Z0-9_]*)\s*[:)]',
        'assignment': r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*',
    }
    
    def __init__(self):
        self.patterns = {
            key: re.compile(pattern)
            for key, pattern in self.CODE_PATTERNS.items()
        }
    
    def extract_functions(self, text: str) -> List[CodeEntity]:
        """提取函数调用"""
        functions = []
        seen = set()
        
        for match in self.patterns['function_call'].finditer(text):
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            
            if name in ['if', 'while', 'for', 'with', 'assert', 'return', 'yield', 'print', 'len', 'str', 'int', 'list', 'dict']:
                continue
            
            functions.append(CodeEntity(
                name=name,
                entity_type='function',
                signature=match.group(0).strip()
            ))
        
        for match in self.patterns['method_call'].finditer(text):
            obj_name = match.group(1)
            method_name = match.group(2)
            full_name = f"{obj_name}.{method_name}"
            
            if full_name in seen:
                continue
            seen.add(full_name)
            
            functions.append(CodeEntity(
                name=full_name,
                entity_type='method',
                signature=match.group(0).strip()
            ))
        
        return functions
    
    def extract_classes(self, text: str) -> List[CodeEntity]:
        """提取类定义"""
        classes = []
        for match in self.patterns['class_definition'].finditer(text):
            classes.append(CodeEntity(
                name=match.group(1),
                entity_type='class',
                signature=match.group(0).strip()
            ))
        return classes
    
    def extract_imports(self, text: str) -> List[CodeEntity]:
        """提取导入语句"""
        imports = []
        for match in self.patterns['import_statement'].finditer(text):
            name = match.group(1)
            imports.append(CodeEntity(
                name=name,
                entity_type='import',
                signature=match.group(0).strip()
            ))
        return imports
    
    def extract_decorators(self, text: str) -> List[CodeEntity]:
        """提取装饰器"""
        decorators = []
        for match in self.patterns['decorator'].finditer(text):
            decorators.append(CodeEntity(
                name=match.group(1),
                entity_type='decorator',
                signature=match.group(0).strip()
            ))
        return decorators
    
    def extract_all(self, text: str) -> Dict[str, List[CodeEntity]]:
        """提取所有代码实体"""
        return {
            'functions': self.extract_functions(text),
            'classes': self.extract_classes(text),
            'imports': self.extract_imports(text),
            'decorators': self.extract_decorators(text),
        }


class SourceVerifier:
    """源文档验证器"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service
        self.pattern_extractor = PatternExtractor()
    
    def build_source_index(self, docs: List[Any]) -> Dict[str, Set[str]]:
        """构建源文档索引"""
        index = {
            'functions': set(),
            'classes': set(),
            'imports': set(),
            'decorators': set(),
            'keywords': set(),
            'content_snippets': set(),
        }
        
        for doc in docs:
            content = doc.content if hasattr(doc, 'content') else str(doc)
            
            entities = self.pattern_extractor.extract_all(content)
            index['functions'].update(e.name for e in entities['functions'])
            index['classes'].update(e.name for e in entities['classes'])
            index['imports'].update(e.name for e in entities['imports'])
            index['decorators'].update(e.name for e in entities['decorators'])
            
            words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', content.lower())
            index['keywords'].update(words[:100])
            
            snippets = content.split('\n')
            for snippet in snippets[:20]:
                snippet = snippet.strip()
                if len(snippet) > 20:
                    index['content_snippets'].add(snippet.lower()[:100])
        
        return index
    
    def verify_entity_exists(
        self,
        entity: CodeEntity,
        source_index: Dict[str, Set[str]],
        docs: List[Any]
    ) -> Tuple[bool, Optional[str]]:
        """验证实体是否存在"""
        if entity.entity_type in ['function', 'method']:
            if entity.name in source_index.get('functions', set()):
                return True, f"Found in source functions"
            
            if self._fuzzy_match(entity.name, source_index.get('functions', set())):
                return True, "Found similar function name"
        
        elif entity.entity_type == 'class':
            if entity.name in source_index.get('classes', set()):
                return True, f"Found in source classes"
        
        elif entity.entity_type == 'decorator':
            if entity.name in source_index.get('decorators', set()):
                return True, f"Found in source decorators"
        
        elif entity.entity_type == 'import':
            if entity.name in source_index.get('imports', set()):
                return True, f"Found in source imports"
        
        return False, "Entity not found in source documents"
    
    def _fuzzy_match(self, name: str, candidates: Set[str], threshold: float = 0.7) -> Optional[str]:
        """模糊匹配"""
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            score = SequenceMatcher(None, name, candidate).ratio()
            if score > threshold and score > best_score:
                best_score = score
                best_match = candidate
        
        return best_match


class GranularHallucinationDetector:
    """
    细粒度幻觉检测器
    
    功能：
    - 句子级别的幻觉检测
    - 返回具体位置和证据
    - 支持多种检测策略
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        enable_llm_verification: bool = True
    ):
        """
        初始化检测器
        
        Args:
            llm_service: LLM 服务（用于深度验证）
            enable_llm_verification: 是否启用 LLM 验证
        """
        self.llm_service = llm_service
        self.enable_llm_verification = enable_llm_verification
        
        self.pattern_extractor = PatternExtractor()
        self.source_verifier = SourceVerifier(llm_service)
        
        self.Builtin_FUNCTIONS = {
            'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set',
            'tuple', 'range', 'open', 'input', 'abs', 'max', 'min', 'sum',
            'sorted', 'reversed', 'enumerate', 'zip', 'map', 'filter',
            'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr',
            'delattr', 'property', 'classmethod', 'staticmethod',
        }
        
        self.common_library_functions = {
            'requests': ['get', 'post', 'put', 'delete', 'head', 'options'],
            'pandas': ['DataFrame', 'read_csv', 'read_excel'],
            'numpy': ['array', 'linspace', 'arange', 'zeros', 'ones'],
            'fastapi': ['FastAPI', 'APIRouter', 'Depends'],
            'flask': ['Flask', 'render_template', 'request'],
        }
    
    def split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        sentences = re.split(r'(?<=[。！？!?])\s+', text)
        
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5:
                result.append(sentence)
        
        if not result and text.strip():
            result = [text.strip()]
        
        return result
    
    async def detect_sentence_level(
        self,
        content: str,
        retrieved_docs: List[Any],
        fragment_id: str = ""
    ) -> FragmentHallucinationResult:
        """
        句子级别的幻觉检测
        
        Args:
            content: 要检测的内容
            retrieved_docs: 检索到的源文档
            fragment_id: 片段 ID
            
        Returns:
            片段级幻觉检测结果
        """
        sentences = self.split_into_sentences(content)
        
        source_index = self.source_verifier.build_source_index(retrieved_docs)
        
        sentence_results = []
        
        for i, sentence in enumerate(sentences):
            result = await self._verify_sentence(
                sentence=sentence,
                sentence_index=i,
                source_index=source_index,
                retrieved_docs=retrieved_docs
            )
            sentence_results.append(result)
        
        hallucination_summary = self._summarize_hallucinations(sentence_results)
        
        critical_count = sum(1 for r in sentence_results 
                           if r.severity == HallucinationSeverity.CRITICAL)
        high_count = sum(1 for r in sentence_results 
                        if r.severity == HallucinationSeverity.HIGH)
        medium_count = sum(1 for r in sentence_results 
                          if r.severity == HallucinationSeverity.MEDIUM)
        low_count = sum(1 for r in sentence_results 
                       if r.severity == HallucinationSeverity.LOW)
        
        needs_regeneration = critical_count > 0
        needs_repair = high_count > 0 or medium_count > 0
        
        return FragmentHallucinationResult(
            fragment_id=fragment_id,
            overall_valid=critical_count == 0 and high_count == 0,
            sentence_results=sentence_results,
            hallucination_summary=hallucination_summary,
            critical_count=critical_count,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            needs_regeneration=needs_regeneration,
            needs_repair=needs_repair
        )
    
    async def _verify_sentence(
        self,
        sentence: str,
        sentence_index: int,
        source_index: Dict[str, Set[str]],
        retrieved_docs: List[Any]
    ) -> SentenceHallucinationResult:
        """验证单个句子"""
        entities = self.pattern_extractor.extract_all(sentence)
        
        hallucinations = []
        for func in entities['functions']:
            if func.name in self.Builtin_FUNCTIONS:
                continue
            
            exists, evidence = self.source_verifier.verify_entity_exists(
                func, source_index, retrieved_docs
            )
            
            if not exists:
                if self._is_common_library_function(func.name):
                    evidence = f"Common library function: {func.name}"
                else:
                    hallucinations.append({
                        'entity': func.name,
                        'type': HallucinationType.CODE,
                        'evidence': evidence
                    })
        
        for cls in entities['classes']:
            exists, evidence = self.source_verifier.verify_entity_exists(
                cls, source_index, retrieved_docs
            )
            if not exists:
                hallucinations.append({
                    'entity': cls.name,
                    'type': HallucinationType.CODE,
                    'evidence': evidence
                })
        
        if hallucinations:
            most_severe = self._get_most_severe(hallucinations)
            
            return SentenceHallucinationResult(
                sentence=sentence,
                sentence_index=sentence_index,
                is_hallucination=True,
                confidence=0.85,
                hallucination_type=most_severe['type'],
                severity=self._type_to_severity(most_severe['type']),
                evidence=most_severe.get('evidence'),
                reasoning=f"Detected hallucination: {most_severe['entity']}"
            )
        
        if self.enable_llm_verification and self.llm_service:
            llm_result = await self._llm_verify_sentence(
                sentence, retrieved_docs
            )
            if llm_result['is_hallucination'] and llm_result['confidence'] > 0.7:
                return SentenceHallucinationResult(
                    sentence=sentence,
                    sentence_index=sentence_index,
                    is_hallucination=True,
                    confidence=llm_result['confidence'],
                    hallucination_type=llm_result.get('type', HallucinationType.FACT),
                    severity=self._type_to_severity(llm_result.get('type', HallucinationType.FACT)),
                    evidence=llm_result.get('evidence'),
                    reasoning=llm_result.get('reasoning', 'LLM detected hallucination')
                )
        
        return SentenceHallucinationResult(
            sentence=sentence,
            sentence_index=sentence_index,
            is_hallucination=False,
            confidence=0.9,
            reasoning="No hallucination detected"
        )
    
    def _is_common_library_function(self, name: str) -> bool:
        """检查是否为常见库函数"""
        for lib, funcs in self.common_library_functions.items():
            if any(name.endswith(f".{f}") or name == f for f in funcs):
                return True
        return False
    
    def _get_most_severe(self, hallucinations: List[Dict]) -> Dict:
        """获取最严重的幻觉"""
        severity_order = {
            HallucinationType.CODE: 4,
            HallucinationType.VERSION: 3,
            HallucinationType.FACT: 2,
            HallucinationType.LOGIC: 2,
            HallucinationType.SOURCE: 1,
            HallucinationType.TEMPORAL: 1,
        }
        
        return max(hallucinations, 
                  key=lambda h: severity_order.get(h.get('type', HallucinationType.FACT), 0))
    
    def _type_to_severity(self, htype: HallucinationType) -> HallucinationSeverity:
        """转换幻觉类型到严重程度"""
        mapping = {
            HallucinationType.CODE: HallucinationSeverity.CRITICAL,
            HallucinationType.VERSION: HallucinationSeverity.HIGH,
            HallucinationType.FACT: HallucinationSeverity.HIGH,
            HallucinationType.LOGIC: HallucinationSeverity.MEDIUM,
            HallucinationType.SOURCE: HallucinationSeverity.LOW,
            HallucinationType.TEMPORAL: HallucinationSeverity.MEDIUM,
        }
        return mapping.get(htype, HallucinationSeverity.MEDIUM)
    
    async def _llm_verify_sentence(
        self,
        sentence: str,
        docs: List[Any]
    ) -> Dict[str, Any]:
        """LLM 验证句子"""
        sources_summary = "\n".join([
            doc.content[:500] for doc in docs[:3]
        ])
        
        prompt = f"""请验证以下句子是否与源文档一致。

源文档内容:
{sources_summary}

待验证句子:
{sentence}

请判断句子中是否存在幻觉，即与源文档不符的信息。

输出 JSON 格式:
{{
    "is_hallucination": true/false,
    "confidence": 0.0-1.0,
    "type": "code/fact/logic/source",
    "evidence": "具体证据",
    "reasoning": "判断理由"
}}

验证结果:"""

        try:
            response = await self.llm_service.chat_completion(
                messages=[
                    {"role": "system", "content": "你是幻觉检测专家，输出必须是有效的 JSON。"},
                    {"role": "user", "content": prompt}
                ],
                task_type="fact_check",
                temperature=0.1,
                max_tokens=200
            )
            
            import json
            result = json.loads(response)
            return result
            
        except Exception as e:
            logger.error(f"LLM verification failed: {str(e)}")
            return {"is_hallucination": False, "confidence": 0.5}
    
    def _summarize_hallucinations(
        self,
        results: List[SentenceHallucinationResult]
    ) -> Dict[str, int]:
        """总结幻觉类型分布"""
        summary = {}
        for result in results:
            if result.is_hallucination and result.hallucination_type:
                key = result.hallucination_type.value
                summary[key] = summary.get(key, 0) + 1
        return summary


class HallucinationClassifier:
    """
    幻觉类型分类器
    
    功能：
    - 分类幻觉类型
    - 评估严重程度
    - 提供处理建议
    """
    
    CODE_KEYWORDS = ['函数', '方法', '类', '参数', '返回值', 'import', 'def ', 'class ']
    FACT_KEYWORDS = ['版本', '特性', '支持', '提供', '基于', '使用']
    LOGIC_KEYWORDS = ['首先', '然后', '因此', '所以', '因为', '但是', '然而']
    SOURCE_KEYWORDS = ['文档', '来源', '根据', '依据', '引用']
    
    def __init__(self):
        pass
    
    def classify_hallucination(self, text: str, context: str = "") -> HallucinationType:
        """
        分类幻觉类型
        
        Args:
            text: 幻觉文本
            context: 上下文信息
            
        Returns:
            幻觉类型
        """
        text_lower = (text + " " + context).lower()
        
        if any(kw in text_lower for kw in self.CODE_KEYWORDS):
            return HallucinationType.CODE
        elif any(kw in text_lower for kw in self.SOURCE_KEYWORDS):
            return HallucinationType.SOURCE
        elif any(kw in text_lower for kw in self.LOGIC_KEYWORDS):
            return HallucinationType.LOGIC
        elif any(kw in text_lower for kw in self.FACT_KEYWORDS):
            return HallucinationType.FACT
        elif re.search(r'v\d+\.\d+', text):
            return HallucinationType.VERSION
        elif re.search(r'\d{4}年|\d+年前|最近|目前', text):
            return HallucinationType.TEMPORAL
        
        return HallucinationType.FACT
    
    def get_severity(self, hallucination_type: HallucinationType) -> HallucinationSeverity:
        """
        获取严重程度
        
        Args:
            hallucination_type: 幻觉类型
            
        Returns:
            严重程度
        """
        severity_map = {
            HallucinationType.CODE: HallucinationSeverity.CRITICAL,
            HallucinationType.VERSION: HallucinationSeverity.HIGH,
            HallucinationType.FACT: HallucinationSeverity.HIGH,
            HallucinationType.LOGIC: HallucinationSeverity.MEDIUM,
            HallucinationType.SOURCE: HallucinationSeverity.LOW,
            HallucinationType.TEMPORAL: HallucinationSeverity.MEDIUM,
        }
        return severity_map.get(hallucination_type, HallucinationSeverity.MEDIUM)
    
    def get_handling_suggestion(
        self,
        hallucination_type: HallucinationType,
        severity: HallucinationSeverity
    ) -> str:
        """获取处理建议"""
        suggestions = {
            (HallucinationType.CODE, HallucinationSeverity.CRITICAL):
                "代码幻觉会误导用户，必须修复。建议查找正确的函数/类名或重新生成。",
            (HallucinationType.VERSION, HallucinationSeverity.HIGH):
                "版本信息错误会误导用户，建议更新为正确的版本信息。",
            (HallucinationType.FACT, HallucinationSeverity.HIGH):
                "事实错误会影响内容准确性，建议修正或标注不确定性。",
            (HallucinationType.LOGIC, HallucinationSeverity.MEDIUM):
                "逻辑矛盾会影响内容连贯性，建议重新组织表述。",
            (HallucinationType.SOURCE, HallucinationSeverity.LOW):
                "来源引用错误影响可信度，建议标注具体来源或移除引用。",
            (HallucinationType.TEMPORAL, HallucinationSeverity.MEDIUM):
                "时间信息可能已过时，建议核实并更新。",
        }
        
        key = (hallucination_type, severity)
        return suggestions.get(key, "根据具体情况处理")


class HallucinationPrevention:
    """
    幻觉预防机制
    
    功能：
    - 在 prompt 中添加约束
    - 提取可用实体列表
    - 添加正确/错误示例
    """
    
    def __init__(self):
        self.pattern_extractor = PatternExtractor()
    
    def enhance_prompt_with_constraints(
        self,
        original_prompt: str,
        retrieved_docs: List[Any],
        available_functions: Optional[List[str]] = None,
        available_classes: Optional[List[str]] = None
    ) -> str:
        """
        在 prompt 中添加约束
        
        Args:
            original_prompt: 原始 prompt
            retrieved_docs: 检索到的文档
            available_functions: 可用函数列表
            available_classes: 可用类列表
            
        Returns:
            增强后的 prompt
        """
        if available_functions is None:
            available_functions = self._extract_functions(retrieved_docs)
        
        if available_classes is None:
            available_classes = self._extract_classes(retrieved_docs)
        
        func_list = ", ".join(available_functions[:30])
        class_list = ", ".join(available_classes[:20])
        
        constraint_text = f"""

## 严格约束

### 可用代码实体
**只能使用以下函数**：
{func_list if func_list else "无特定限制"}

**只能使用以下类**：
{class_list if class_list else "无特定限制"}

### 生成规则
1. 所有代码示例必须来自提供的文档
2. 如果引用第三方库，必须确认该库在导入语句中存在
3. 如果不确定某个函数/类的用法，使用"根据文档..."的表述
4. 禁止编造任何函数名、参数名或类名
5. 版本信息必须准确，标明具体版本号

### 禁止行为
- 禁止使用不存在的函数或方法
- 禁止编造 API 端点或参数
- 禁止添加源文档中没有的代码示例
"""
        
        return original_prompt + constraint_text
    
    def add_verification_examples(self, prompt: str) -> str:
        """
        添加正确和错误的示例
        
        Args:
            prompt: 原始 prompt
            
        Returns:
            添加示例后的 prompt
        """
        examples = """

## 正确 vs 错误示例

### ✅ 正确示例
- "根据文档，FastAPI 使用 `@app.get()` 装饰器定义路由"
- "文档中展示了如何使用 `Request` 对象获取查询参数"
- "根据官方示例，`Depends` 用于依赖注入"

### ❌ 错误示例（避免）
- "FastAPI 使用 `@app.magic_route()` 装饰器" （函数不存在）
- "通过 `get_magic_params()` 获取参数" （函数不存在）
- "该框架内置了 AI 生成功能" （未在文档中提及）

## 自检清单
在生成每个陈述前，请检查：
- [ ] 代码示例是否在文档中真实存在？
- [ ] 函数名、类名是否准确？
- [ ] 版本信息是否准确？
- [ ] 技术描述是否与文档一致？
"""
        
        return prompt + examples
    
    def _extract_functions(self, docs: List[Any]) -> List[str]:
        """提取可用函数"""
        functions = set()
        for doc in docs:
            content = doc.content if hasattr(doc, 'content') else str(doc)
            entities = self.pattern_extractor.extract_all(content)
            for func in entities['functions']:
                if func.name not in ['if', 'while', 'for', 'with', 'return', 'print']:
                    functions.add(func.name)
        return list(functions)
    
    def _extract_classes(self, docs: List[Any]) -> List[str]:
        """提取可用类"""
        classes = set()
        for doc in docs:
            content = doc.content if hasattr(doc, 'content') else str(doc)
            entities = self.pattern_extractor.extract_all(content)
            for cls in entities['classes']:
                classes.add(cls.name)
        return list(classes)


class HallucinationRepairer:
    """
    幻觉修复器
    
    功能：
    - 尝试修复幻觉而非重新生成
    - 支持多种修复策略
    - 提供修复建议
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        max_repair_attempts: int = 2
    ):
        """
        初始化修复器
        
        Args:
            llm_service: LLM 服务
            max_repair_attempts: 最大修复尝试次数
        """
        self.llm_service = llm_service
        self.max_repair_attempts = max_repair_attempts
        self.pattern_extractor = PatternExtractor()
    
    async def repair_hallucination(
        self,
        content: str,
        hallucinations: List[SentenceHallucinationResult],
        retrieved_docs: List[Any]
    ) -> Tuple[str, List[RepairSuggestion]]:
        """
        修复幻觉
        
        Args:
            content: 原始内容
            hallucinations: 幻觉列表
            retrieved_docs: 源文档
            
        Returns:
            (修复后的内容, 修复建议列表)
        """
        repaired_content = content
        suggestions = []
        
        source_index = self._build_source_index(retrieved_docs)
        
        for hallucination in hallucinations:
            if hallucination.hallucination_type == HallucinationType.CODE:
                suggestion = await self._repair_code_hallucination(
                    hallucination, source_index, retrieved_docs
                )
                if suggestion and suggestion.confidence > 0.6:
                    repaired_content = repaired_content.replace(
                        hallucination.sentence,
                        suggestion.corrected_text
                    )
                    suggestions.append(suggestion)
        
        if self.llm_service and any(
            h.hallucination_type in [HallucinationType.FACT, HallucinationType.LOGIC]
            for h in hallucinations
        ):
            llm_suggestions = await self._llm_repair(
                content, hallucinations, retrieved_docs
            )
            suggestions.extend(llm_suggestions)
            
            if llm_suggestions:
                best_suggestion = max(llm_suggestions, key=lambda s: s.confidence)
                if best_suggestion.confidence > 0.7:
                    repaired_content = repaired_content.replace(
                        best_suggestion.original_text,
                        best_suggestion.corrected_text
                    )
        
        return repaired_content, suggestions
    
    def _build_source_index(self, docs: List[Any]) -> Dict[str, Set[str]]:
        """构建源文档索引"""
        index = {
            'functions': set(),
            'classes': set(),
            'imports': set(),
            'content_patterns': set(),
        }
        
        for doc in docs:
            content = doc.content if hasattr(doc, 'content') else str(doc)
            
            entities = self.pattern_extractor.extract_all(content)
            index['functions'].update(e.name for e in entities['functions'])
            index['classes'].update(e.name for e in entities['classes'])
            index['imports'].update(e.name for e in entities['imports'])
        
        return index
    
    async def _repair_code_hallucination(
        self,
        hallucination: SentenceHallucinationResult,
        source_index: Dict[str, Set[str]],
        docs: List[Any]
    ) -> Optional[RepairSuggestion]:
        """修复代码幻觉"""
        entities = self.pattern_extractor.extract_all(hallucination.sentence)
        
        for func in entities['functions']:
            if func.name in source_index.get('functions', set()):
                continue
            
            similar = self._find_similar_function(func.name, source_index.get('functions', set()))
            
            if similar:
                return RepairSuggestion(
                    original_text=func.name,
                    corrected_text=similar,
                    reason=f"Found similar function: {similar}",
                    confidence=0.8,
                    source="source_index"
                )
            
            if self._is_likely_typo(func.name):
                return RepairSuggestion(
                    original_text=func.name,
                    corrected_text=func.name,
                    reason="Function appears to be correct or not in source",
                    confidence=0.5,
                    source="unknown"
                )
        
        return None
    
    def _find_similar_function(
        self,
        name: str,
        candidates: Set[str],
        threshold: float = 0.6
    ) -> Optional[str]:
        """查找相似函数"""
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            score = SequenceMatcher(None, name, candidate).ratio()
            if score > threshold and score > best_score:
                best_score = score
                best_match = candidate
        
        return best_match
    
    def _is_likely_typo(self, name: str) -> bool:
        """检查是否为拼写错误"""
        common_names = {
            'fastapi': ['FastAPI', 'fast_api'],
            'pydantic': ['Pydantic', 'pydantic'],
            'requests': ['Requests', 'requests'],
            'pandas': ['Pandas', 'pandas'],
            'numpy': ['NumPy', 'numpy'],
        }
        
        name_lower = name.lower()
        for common, variants in common_names.items():
            if name_lower in [v.lower() for v in variants]:
                return True
        
        return False
    
    async def _llm_repair(
        self,
        content: str,
        hallucinations: List[SentenceHallucinationResult],
        docs: List[Any]
    ) -> List[RepairSuggestion]:
        """LLM 辅助修复"""
        hallucination_texts = [
            f"- 句子 {i+1}: {h.sentence} (可能幻觉)"
            for i, h in enumerate(hallucinations)
        ]
        
        sources_summary = "\n".join([
            doc.content[:300] for doc in docs[:3]
        ])
        
        prompt = f"""请尝试修复以下可能存在幻觉的句子。

源文档:
{sources_summary}

原始内容:
{content}

检测到的可能幻觉:
{chr(10).join(hallucination_texts)}

请对每个幻觉句子提供修正建议。

输出 JSON 格式:
[
    {{
        "original_text": "原句子",
        "corrected_text": "修正后句子",
        "reason": "修正理由",
        "confidence": 0.0-1.0
    }}
]

修复建议:"""

        try:
            response = await self.llm_service.chat_completion(
                messages=[
                    {"role": "system", "content": "你是内容修复专家，输出必须是有效的 JSON 数组。"},
                    {"role": "user", "content": prompt}
                ],
                task_type="fact_check",
                temperature=0.3,
                max_tokens=500
            )
            
            import json
            suggestions = json.loads(response)
            
            return [
                RepairSuggestion(
                    original_text=s.get('original_text', ''),
                    corrected_text=s.get('corrected_text', ''),
                    reason=s.get('reason', ''),
                    confidence=s.get('confidence', 0.5)
                )
                for s in suggestions
            ]
            
        except Exception as e:
            logger.error(f"LLM repair failed: {str(e)}")
            return []


class UnifiedHallucinationService:
    """
    统一幻觉检测服务
    
    整合所有幻觉检测功能，提供统一接口
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化服务
        
        Args:
            llm_service: LLM 服务
            config: 配置选项
        """
        self.config = config or {}
        
        self.detector = GranularHallucinationDetector(
            llm_service=llm_service,
            enable_llm_verification=self.config.get('enable_llm_verification', True)
        )
        
        self.classifier = HallucinationClassifier()
        
        self.prevention = HallucinationPrevention()
        
        self.repairer = HallucinationRepairer(
            llm_service=llm_service,
            max_repair_attempts=self.config.get('max_repair_attempts', 2)
        )
    
    async def detect(
        self,
        content: str,
        retrieved_docs: List[Any],
        fragment_id: str = ""
    ) -> FragmentHallucinationResult:
        """
        检测幻觉
        
        Args:
            content: 要检测的内容
            retrieved_docs: 源文档
            fragment_id: 片段 ID
            
        Returns:
            检测结果
        """
        return await self.detector.detect_sentence_level(
            content=content,
            retrieved_docs=retrieved_docs,
            fragment_id=fragment_id
        )
    
    async def detect_and_repair(
        self,
        content: str,
        retrieved_docs: List[Any],
        fragment_id: str = ""
    ) -> Dict[str, Any]:
        """
        检测并尝试修复
        
        Args:
            content: 要处理的内容
            retrieved_docs: 源文档
            fragment_id: 片段 ID
            
        Returns:
            处理结果
        """
        result = await self.detect(content, retrieved_docs, fragment_id)
        
        repaired_content = content
        suggestions = []
        
        if result.needs_repair:
            repaired_content, suggestions = await self.repairer.repair_hallucination(
                content=content,
                hallucinations=result.sentence_results,
                retrieved_docs=retrieved_docs
            )
        
        return {
            'original_content': content,
            'repaired_content': repaired_content if result.needs_repair else content,
            'detection_result': result,
            'repair_suggestions': suggestions,
            'needs_regeneration': result.needs_regeneration,
            'needs_repair': result.needs_repair
        }
    
    def enhance_prompt(
        self,
        prompt: str,
        retrieved_docs: List[Any],
        add_examples: bool = True
    ) -> str:
        """
        增强 prompt 以预防幻觉
        
        Args:
            prompt: 原始 prompt
            retrieved_docs: 检索到的文档
            add_examples: 是否添加示例
            
        Returns:
            增强后的 prompt
        """
        enhanced = self.prevention.enhance_prompt_with_constraints(
            original_prompt=prompt,
            retrieved_docs=retrieved_docs
        )
        
        if add_examples:
            enhanced = self.prevention.add_verification_examples(enhanced)
        
        return enhanced
    
    def get_detection_report(
        self,
        result: FragmentHallucinationResult
    ) -> str:
        """生成检测报告"""
        lines = [
            "=" * 60,
            "幻觉检测报告",
            "=" * 60,
            f"片段 ID: {result.fragment_id}",
            f"整体有效: {'✓' if result.overall_valid else '✗'}",
            f"需要重新生成: {'是' if result.needs_regeneration else '否'}",
            f"需要修复: {'是' if result.needs_repair else '否'}",
            "",
            "幻觉统计:",
            f"  - 严重: {result.critical_count}",
            f"  - 高: {result.high_count}",
            f"  - 中: {result.medium_count}",
            f"  - 低: {result.low_count}",
            "",
            "类型分布:",
        ]
        
        for htype, count in result.hallucination_summary.items():
            lines.append(f"  - {htype}: {count}")
        
        lines.append("")
        lines.append("详细结果:")
        
        for i, sentence_result in enumerate(result.sentence_results):
            status = "✗" if sentence_result.is_hallucination else "✓"
            lines.append(f"  {i+1}. [{status}] {sentence_result.sentence[:50]}...")
            
            if sentence_result.is_hallucination:
                lines.append(f"     类型: {sentence_result.hallucination_type.value}")
                lines.append(f"     严重程度: {sentence_result.severity.value}")
                lines.append(f"     原因: {sentence_result.reasoning}")
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)
