"""Tree-sitter Parser - Code structure extraction"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from enum import Enum


logger = logging.getLogger(__name__)


class CodeElementType(str, Enum):
    """代码元素类型"""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    COMMENT = "comment"
    IMPORT = "import"


class CodeElement(BaseModel):
    """代码元素"""
    type: CodeElementType
    name: str
    content: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ParsedCode(BaseModel):
    """解析后的代码"""
    file_path: str
    language: str
    elements: List[CodeElement]
    has_deprecated: bool = False
    has_fixme: bool = False
    has_todo: bool = False
    has_security: bool = False
    raw_content: str
    metadata: Dict[str, Any] = {}


class IParserService(ABC):
    """代码解析服务接口（抽象）"""
    
    @abstractmethod
    def parse(
        self,
        file_path: str,
        content: str,
        language: Optional[str] = None
    ) -> ParsedCode:
        """
        解析代码文件
        
        Args:
            file_path: 文件路径
            content: 文件内容
            language: 编程语言（可选，自动检测）
            
        Returns:
            解析后的代码结构
        """
        pass
    
    @abstractmethod
    def extract_functions(
        self,
        parsed_code: ParsedCode
    ) -> List[CodeElement]:
        """
        提取函数
        
        Args:
            parsed_code: 解析后的代码
            
        Returns:
            函数列表
        """
        pass
    
    @abstractmethod
    def extract_classes(
        self,
        parsed_code: ParsedCode
    ) -> List[CodeElement]:
        """
        提取类
        
        Args:
            parsed_code: 解析后的代码
            
        Returns:
            类列表
        """
        pass
    
    @abstractmethod
    def extract_comments(
        self,
        parsed_code: ParsedCode
    ) -> List[CodeElement]:
        """
        提取注释
        
        Args:
            parsed_code: 解析后的代码
            
        Returns:
            注释列表
        """
        pass
    
    @abstractmethod
    def detect_markers(
        self,
        content: str
    ) -> Dict[str, bool]:
        """
        检测代码标记
        
        Args:
            content: 代码内容
            
        Returns:
            标记字典（has_deprecated, has_fixme, has_todo, has_security）
        """
        pass


class TreeSitterParser(IParserService):
    """
    Tree-sitter 代码解析器实现
    
    功能：
    - 支持多种编程语言
    - 提取函数、类、注释
    - 检测代码标记（@deprecated, FIXME, TODO, Security）
    - 解析失败时回退到纯文本
    """
    
    # 支持的语言映射
    LANGUAGE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
    }
    
    # 标记关键词
    MARKERS = {
        'deprecated': ['@deprecated', 'deprecated', 'DEPRECATED'],
        'fixme': ['FIXME', 'fixme', 'FIX ME'],
        'todo': ['TODO', 'todo', 'TO DO'],
        'security': ['Security', 'SECURITY', 'security', 'XXE', 'SQL injection', 'XSS']
    }
    
    def __init__(self):
        """初始化 Tree-sitter 解析器"""
        self.parsers = {}
        self._initialize_parsers()
    
    def _initialize_parsers(self):
        """初始化语言解析器"""
        try:
            from tree_sitter import Language, Parser
            
            # 这里需要预先编译 Tree-sitter 语言库
            # 实际使用时需要根据项目配置加载
            logger.info("Tree-sitter parsers initialized")
        except ImportError:
            logger.warning("Tree-sitter not installed, parser will use fallback mode")
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """
        根据文件扩展名检测语言
        
        Args:
            file_path: 文件路径
            
        Returns:
            语言名称
        """
        import os
        ext = os.path.splitext(file_path)[1].lower()
        return self.LANGUAGE_EXTENSIONS.get(ext)
    
    def parse(
        self,
        file_path: str,
        content: str,
        language: Optional[str] = None
    ) -> ParsedCode:
        """解析代码文件"""
        # 检测语言
        if not language:
            language = self._detect_language(file_path)
        
        if not language:
            logger.warning(f"Unknown language for file: {file_path}, using fallback")
            return self._fallback_parse(file_path, content)
        
        try:
            # 尝试使用 Tree-sitter 解析
            return self._tree_sitter_parse(file_path, content, language)
        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed for {file_path}: {str(e)}, using fallback")
            return self._fallback_parse(file_path, content, language)
    
    def _tree_sitter_parse(
        self,
        file_path: str,
        content: str,
        language: str
    ) -> ParsedCode:
        """使用 Tree-sitter 解析"""
        # TODO: 实现 Tree-sitter 解析逻辑
        # 这里需要使用 tree-sitter 库进行实际解析
        
        # 暂时使用回退解析
        return self._fallback_parse(file_path, content, language)
    
    def _fallback_parse(
        self,
        file_path: str,
        content: str,
        language: Optional[str] = None
    ) -> ParsedCode:
        """回退到纯文本解析"""
        # 检测标记
        markers = self.detect_markers(content)
        
        # 简单的行级解析
        elements = []
        lines = content.split('\n')
        
        # 提取注释
        for i, line in enumerate(lines):
            stripped = line.strip()
            # 检测注释
            if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
                elements.append(CodeElement(
                    type=CodeElementType.COMMENT,
                    name=f"comment_{i}",
                    content=stripped,
                    line_start=i + 1,
                    line_end=i + 1
                ))
        
        # 简单的函数检测（Python 和 JavaScript）
        if language in ['python', 'javascript', 'typescript']:
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Python 函数
                if language == 'python' and stripped.startswith('def '):
                    func_name = stripped.split('(')[0].replace('def ', '').strip()
                    elements.append(CodeElement(
                        type=CodeElementType.FUNCTION,
                        name=func_name,
                        content=line,
                        line_start=i + 1,
                        line_end=i + 1
                    ))
                # JavaScript/TypeScript 函数
                elif language in ['javascript', 'typescript'] and ('function ' in stripped or '=>' in stripped):
                    # 简单提取函数名
                    if 'function ' in stripped:
                        func_name = stripped.split('function ')[1].split('(')[0].strip()
                    else:
                        func_name = stripped.split('=')[0].strip() if '=' in stripped else f"func_{i}"
                    elements.append(CodeElement(
                        type=CodeElementType.FUNCTION,
                        name=func_name,
                        content=line,
                        line_start=i + 1,
                        line_end=i + 1
                    ))
        
        # 简单的类检测
        if language in ['python', 'javascript', 'typescript', 'java']:
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('class '):
                    class_name = stripped.split('class ')[1].split('(')[0].split('{')[0].split(':')[0].strip()
                    elements.append(CodeElement(
                        type=CodeElementType.CLASS,
                        name=class_name,
                        content=line,
                        line_start=i + 1,
                        line_end=i + 1
                    ))
        
        return ParsedCode(
            file_path=file_path,
            language=language or "unknown",
            elements=elements,
            has_deprecated=markers['has_deprecated'],
            has_fixme=markers['has_fixme'],
            has_todo=markers['has_todo'],
            has_security=markers['has_security'],
            raw_content=content
        )
    
    def extract_functions(
        self,
        parsed_code: ParsedCode
    ) -> List[CodeElement]:
        """提取函数"""
        return [
            elem for elem in parsed_code.elements
            if elem.type == CodeElementType.FUNCTION
        ]
    
    def extract_classes(
        self,
        parsed_code: ParsedCode
    ) -> List[CodeElement]:
        """提取类"""
        return [
            elem for elem in parsed_code.elements
            if elem.type == CodeElementType.CLASS
        ]
    
    def extract_comments(
        self,
        parsed_code: ParsedCode
    ) -> List[CodeElement]:
        """提取注释"""
        return [
            elem for elem in parsed_code.elements
            if elem.type == CodeElementType.COMMENT
        ]
    
    def detect_markers(
        self,
        content: str
    ) -> Dict[str, bool]:
        """检测代码标记"""
        content_lower = content.lower()
        
        return {
            'has_deprecated': any(marker.lower() in content_lower for marker in self.MARKERS['deprecated']),
            'has_fixme': any(marker.lower() in content_lower for marker in self.MARKERS['fixme']),
            'has_todo': any(marker.lower() in content_lower for marker in self.MARKERS['todo']),
            'has_security': any(marker.lower() in content_lower for marker in self.MARKERS['security'])
        }
