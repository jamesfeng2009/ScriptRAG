"""
智能文档分块器 (Smart Document Chunker)

生产级文档分块解决方案，支持:
- 多种文件类型 (Python, Markdown, JSON, YAML, SQL, etc.)
- 复杂嵌套结构处理 (Python 类/函数嵌套)
- 特殊格式文件检测 (minified JS, proto, 二进制)
- 大文件流式处理 (避免内存溢出)
- 多编码支持 (UTF-8, GBK, etc.)

架构:
- Strategy Pattern: 灵活的分块策略
- Chain of Responsibility: 处理链
- Template Method: 模板方法模式
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple, Union, Set
import re
import hashlib
import mmap
import logging
import asyncio
import time
import pickle
from io import StringIO, BytesIO
from collections import defaultdict
from threading import Lock


logger = logging.getLogger(__name__)


class FileType(Enum):
    """支持的文件类型"""
    PYTHON = "python"
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    SQL = "sql"
    PROTO = "proto"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    HTML = "html"
    CSS = "css"
    BINARY = "binary"
    TEXT = "text"
    UNKNOWN = "unknown"


class SemanticRelationType(Enum):
    """语义关系类型 - 解决指代消解和因果关系丢失问题"""
    CONDITIONAL_GROUP = "conditional_group"  # if-elif-else 条件组
    TRY_EXCEPT_GROUP = "try_except_group"  # try-except-finally 异常处理组
    CLASS_METHOD_GROUP = "class_method_group"  # 类及其方法的关联
    CONTROL_FLOW_GROUP = "control_flow_group"  # 循环/with/上下文管理器组
    DECORATOR_TARGET = "decorator_target"  # 装饰器与被装饰函数的关系
    PROPERTY_CHAIN = "property_chain"  # 属性访问链（如 self.config.timeout）
    CAUSAL_DEPENDENCY = "causal_dependency"  # 因果依赖（如验证后使用）
    IMPORT_DEPENDENCY = "import_dependency"  # 导入依赖关系
    TYPE_REFERENCE = "type_reference"  # 类型引用关系（如返回 User 对象）


@dataclass
class SemanticRelation:
    """语义关系 - 保留指代消解和因果逻辑"""
    relation_type: SemanticRelationType
    related_chunk_ids: List[str]
    context: Optional[Dict[str, Any]] = None
    strength: float = 1.0  # 关系强度 0-1


class ChunkingStrategy(ABC):
    """分块策略基类"""
    
    @abstractmethod
    def chunk(self, content: str, file_path: str) -> List['Chunk']:
        """执行分块"""
        pass
    
    @abstractmethod
    def can_handle(self, file_type: FileType) -> bool:
        """是否支持该文件类型"""
        pass
    
    @abstractmethod
    def is_binary_content(self, content: bytes) -> bool:
        """检测是否为二进制内容"""
        pass


@dataclass
class ChunkMetadata:
    """分块元数据 - 增强版：保留语义完整性"""
    file_path: str
    file_type: FileType
    start_line: int
    end_line: int
    char_count: int
    content_hash: str
    language_structure: Optional[str] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    parent_chunk: Optional[str] = None
    parent_document_id: Optional[str] = None
    related_chunks: List[str] = field(default_factory=list)
    encoding: Optional[str] = None
    is_truncated: bool = False
    original_size: Optional[int] = None
    ghost_context: Optional[Dict[str, str]] = None
    semantic_type: Optional[str] = None
    semantic_relations: List[SemanticRelation] = field(default_factory=list)
    control_flow_context: Optional[Dict[str, Any]] = None
    reference_context: Optional[Dict[str, List[str]]] = None
    causal_chain: Optional[List[Dict[str, Any]]] = None
    cross_file_references: Optional[Dict[str, List[str]]] = None
    is_atomic: bool = False


@dataclass
class Chunk:
    """文档分块"""
    id: str
    content: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": {
                "file_path": self.metadata.file_path,
                "file_type": self.metadata.file_type.value,
                "start_line": self.metadata.start_line,
                "end_line": self.metadata.end_line,
                "char_count": self.metadata.char_count,
                "content_hash": self.metadata.content_hash,
                "language_structure": self.metadata.language_structure,
                "function_name": self.metadata.function_name,
                "class_name": self.metadata.class_name,
                "parent_chunk": self.metadata.parent_chunk,
                "related_chunks": self.metadata.related_chunks,
                "encoding": self.metadata.encoding,
                "is_truncated": self.metadata.is_truncated,
                "semantic_type": self.metadata.semantic_type,
                "semantic_relations": [
                    {
                        "relation_type": rel.relation_type.value,
                        "related_chunk_ids": rel.related_chunk_ids,
                        "context": rel.context,
                        "strength": rel.strength
                    }
                    for rel in self.metadata.semantic_relations
                ] if self.metadata.semantic_relations else [],
                "control_flow_context": self.metadata.control_flow_context,
                "reference_context": self.metadata.reference_context,
                "causal_chain": self.metadata.causal_chain,
                "cross_file_references": self.metadata.cross_file_references,
            }
        }


class EncodingDetector:
    """编码检测器"""
    
    UTF8_BOM = b'\xef\xbb\xbf'
    UTF16_LE_BOM = b'\xff\xfe'
    UTF16_BE_BOM = b'\xfe\xff'
    UTF32_LE_BOM = b'\xff\xfe\x00\x00'
    UTF32_BE_BOM = b'\x00\x00\xfe\xff'
    
    SUPPORTED_ENCODINGS = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1', 'cp1252']
    
    @classmethod
    def detect(cls, content: bytes) -> Tuple[str, bool]:
        """
        检测文件编码
        
        Returns:
            Tuple[编码名称, 是否成功]
        """
        if len(content) == 0:
            return 'utf-8', True
        
        if content.startswith(cls.UTF8_BOM):
            return 'utf-8-sig', True
        
        if content.startswith(cls.UTF16_LE_BOM):
            return 'utf-16-le', True
        
        if content.startswith(cls.UTF16_BE_BOM):
            return 'utf-16-be', True
        
        if content.startswith(cls.UTF32_LE_BOM):
            return 'utf-32-le', True
        
        if content.startswith(cls.UTF32_BE_BOM):
            return 'utf-32-be', True
        
        for encoding in cls.SUPPORTED_ENCODINGS:
            try:
                content.decode(encoding)
                return encoding, True
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        try:
            content.decode('utf-8', errors='strict')
            return 'utf-8', True
        except UnicodeDecodeError:
            pass
        
        for encoding in ['latin-1', 'cp1252']:
            try:
                content.decode(encoding, errors='strict')
                return encoding, True
            except UnicodeDecodeError:
                continue
        
        logger.warning(f"无法检测编码，尝试使用 latin-1 (可能损坏)")
        return 'latin-1', False
    
    @classmethod
    def decode_content(cls, content: bytes) -> Tuple[str, str]:
        """
        解码内容
        
        Returns:
            Tuple[解码后内容, 编码名称]
        """
        encoding, success = cls.detect(content)
        
        try:
            decoded = content.decode(encoding, errors='replace')
            return decoded, encoding
        except Exception as e:
            logger.error(f"解码失败: {e}")
            decoded = content.decode('latin-1', errors='replace')
            return decoded, 'latin-1'


class BinaryDetector:
    """二进制文件检测器"""
    
    BINARY_SIGNATURES = {
        b'\x89PNG\r\n\x1a\n': 'png',
        b'\xff\xd8\xff': 'jpeg',
        b'GIF87a': 'gif',
        b'GIF89a': 'gif',
        b'%PDF': 'pdf',
        b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1': 'ms-office',
        b'PK\x03\x04': 'zip',
        b'\x1f\x8b': 'gzip',
        b'\x00\x00\x00\x18ftyp': 'mp4',
        b'ID3': 'mp3',
        b'\x7fELF': 'elf',
        b'\xca\xfe\xba\xbe': 'java',
        b'\x1f\x9d': 'compress',
        b'BZh': 'bzip2',
    }
    
    NULL_BYTES_THRESHOLD = 0.05
    
    @classmethod
    def is_binary(cls, content: bytes) -> bool:
        """
        检测是否为二进制内容
        
        Args:
            content: 文件字节内容
            
        Returns:
            是否为二进制文件
        """
        if len(content) == 0:
            return False
        
        null_count = content.count(b'\x00')
        null_ratio = null_count / len(content)
        
        if null_ratio > cls.NULL_BYTES_THRESHOLD:
            return True
        
        for signature in cls.BINARY_SIGNATURES:
            if content.startswith(signature):
                return True
        
        try:
            content.decode('utf-8', errors='strict')
            return False
        except (UnicodeDecodeError, UnicodeError):
            pass
        
        try:
            content.decode('ascii', errors='strict')
            return False
        except (UnicodeDecodeError, UnicodeError):
            pass
        
        non_text_ratio = sum(1 for byte in content if byte < 32 and byte not in b'\n\r\t\x0b\x0c') / len(content)
        return non_text_ratio > 0.3
    
    @classmethod
    def get_file_type(cls, content: bytes) -> Optional[str]:
        """获取具体的二进制文件类型"""
        for signature, file_type in cls.BINARY_SIGNATURES.items():
            if content.startswith(signature):
                return file_type
        return 'unknown-binary'


class LargeFileProcessor:
    """大文件流式处理器"""
    
    DEFAULT_CHUNK_SIZE = 1024 * 1024
    MIN_CHUNK_SIZE = 256 * 1024
    
    def __init__(self, max_file_size: int = 100 * 1024 * 1024):
        """
        初始化大文件处理器
        
        Args:
            max_file_size: 最大处理文件大小 (默认 100MB)
        """
        self.max_file_size = max_file_size
    
    def should_use_streaming(self, file_path: str, content: bytes) -> bool:
        """判断是否应该使用流式处理"""
        file_size = len(content)
        
        if file_size > self.max_file_size:
            logger.warning(f"文件过大 ({file_size / 1024 / 1024:.1f}MB)，使用流式处理")
            return True
        
        return False
    
    def stream_read(self, file_path: str, encoding: str = 'utf-8') -> Iterator[str]:
        """
        流式读取文件
        
        Args:
            file_path: 文件路径
            encoding: 编码
            
        Yields:
            文本行
        """
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(self.DEFAULT_CHUNK_SIZE)
                if not chunk:
                    break
                
                try:
                    text = chunk.decode(encoding, errors='replace')
                    yield text
                except Exception as e:
                    logger.error(f"解码块失败: {e}")
                    yield chunk.decode('latin-1', errors='replace')
    
    def stream_chunk(
        self, 
        file_path: str, 
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> Iterator[Tuple[str, int, int]]:
        """
        流式分块
        
        Args:
            file_path: 文件路径
            chunk_size: 块大小
            overlap: 重叠大小
            
        Yields:
            Tuple[块内容, 起始行号, 结束行号]
        """
        lines = []
        line_number = 0
        chunk_index = 0
        
        for text_chunk in self.stream_read(file_path):
            for line in text_chunk.split('\n'):
                lines.append(line)
                
                if len('\n'.join(lines)) > chunk_size:
                    chunk_content = '\n'.join(lines)
                    
                    start_line = line_number - len(lines) + 1
                    end_line = line_number
                    
                    yield chunk_content, start_line, end_line
                    
                    if overlap > 0 and len(lines) > overlap:
                        lines = lines[-overlap:]
                        line_number -= len(lines)
                    else:
                        lines = []
                        line_number = 0
        
        if lines:
            yield '\n'.join(lines), line_number - len(lines) + 1, line_number


class BaseChunkingStrategy(ChunkingStrategy):
    """基础分块策略 (固定大小)"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
    
    def chunk(self, content: str, file_path: str) -> List[Chunk]:
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        start_line = 1
        
        for line_num, line in enumerate(lines, 1):
            line_size = len(line) + 1
            
            if len(line) > self.chunk_size:
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_content, file_path, start_line, line_num - 1, chunk_index))
                    chunk_index += 1
                    current_chunk = []
                    current_size = 0
                    start_line = line_num
                
                sub_chunks = self._split_long_line(line, self.chunk_size, self.overlap)
                for sub_chunk_content in sub_chunks:
                    chunks.append(self._create_chunk(sub_chunk_content, file_path, start_line, line_num, chunk_index))
                    chunk_index += 1
            elif current_size + line_size > self.chunk_size:
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_content, file_path, start_line, line_num - 1, chunk_index))
                    chunk_index += 1
                
                if self.overlap > 0 and len(current_chunk) > self.overlap:
                    current_chunk = current_chunk[-self.overlap:]
                    current_size = sum(len(l) + 1 for l in current_chunk)
                    start_line = line_num - len(current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
                    start_line = line_num
            
            current_chunk.append(line)
            current_size += line_size
        
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(self._create_chunk(chunk_content, file_path, start_line, len(lines), chunk_index))
        
        return chunks
    
    def _split_long_line(self, line: str, chunk_size: int, overlap: int) -> List[str]:
        """拆分过长的行"""
        if len(line) <= chunk_size:
            return [line]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(line):
            chunk_end = min(current_pos + chunk_size, len(line))
            chunk_content = line[current_pos:chunk_end]
            chunks.append(chunk_content)
            
            if chunk_end >= len(line):
                break
            
            current_pos = chunk_end - overlap
        
        return chunks
    
    def _create_chunk(
        self, 
        content: str, 
        file_path: str,
        start_line: int,
        end_line: int,
        chunk_index: int
    ) -> Chunk:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        chunk_id = f"{Path(file_path).stem}_{content_hash[:8]}_{chunk_index}"
        
        metadata = ChunkMetadata(
            file_path=file_path,
            file_type=FileType.TEXT,
            start_line=start_line,
            end_line=end_line,
            char_count=len(content),
            content_hash=content_hash
        )
        
        return Chunk(id=chunk_id, content=content, metadata=metadata)
    
    def can_handle(self, file_type: FileType) -> bool:
        return file_type in [FileType.TEXT, FileType.UNKNOWN]
    
    def is_binary_content(self, content: bytes) -> bool:
        return BinaryDetector.is_binary(content)


class RecursiveChunkingStrategy(ChunkingStrategy):
    """递归分块策略"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        min_chunk_size: int = 100,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.separators = separators or [
            "\n\n\n", "\n\n", "\n", ". ", ", ", "; ", " ", ""
        ]
    
    def chunk(self, content: str, file_path: str) -> List[Chunk]:
        chunks = self._split_recursive(content, self.separators, 0)
        chunks = self._merge_small_chunks(chunks)
        
        result = []
        for i, chunk_data in enumerate(chunks):
            content_hash = hashlib.md5(chunk_data['content'].encode()).hexdigest()
            chunk_id = f"{Path(file_path).stem}_{content_hash[:8]}_{i}"
            
            metadata = ChunkMetadata(
                file_path=file_path,
                file_type=FileType.TEXT,
                start_line=chunk_data.get('start_line', 0),
                end_line=chunk_data.get('end_line', 0),
                char_count=len(chunk_data['content']),
                content_hash=content_hash
            )
            
            result.append(Chunk(id=chunk_id, content=chunk_data['content'], metadata=metadata))
        
        return result
    
    def _split_recursive(
        self, 
        content: str, 
        separators: List[str], 
        level: int
    ) -> List[Dict[str, Any]]:
        if not separators or len(content) <= self.chunk_size:
            return [{'content': content.strip()}]
        
        first_sep = separators[0]
        remaining_seps = separators[1:]
        
        if not first_sep:
            parts = list(content)
        else:
            parts = content.split(first_sep)
        
        results = []
        current_piece = ""
        
        for part in parts:
            if len(current_piece) + len(part) + len(first_sep) <= self.chunk_size:
                current_piece += part + first_sep
            else:
                if current_piece:
                    results.append({'content': current_piece.strip()})
                
                if len(part) > self.chunk_size and remaining_seps:
                    sub_parts = self._split_recursive(part, remaining_seps, level + 1)
                    results.extend(sub_parts)
                    current_piece = ""
                else:
                    current_piece = part + first_sep
        
        if current_piece:
            results.append({'content': current_piece.strip()})
        
        return results
    
    def _merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return []
        
        merged = []
        buffer = ""
        
        for chunk in chunks:
            if len(buffer) + len(chunk['content']) <= self.chunk_size:
                buffer += f"\n\n{chunk['content']}"
            else:
                if buffer:
                    merged.append({'content': buffer.strip()})
                buffer = chunk['content']
        
        if buffer:
            merged.append({'content': buffer.strip()})
        
        return merged
    
    def can_handle(self, file_type: FileType) -> bool:
        return file_type in [FileType.JSON, FileType.YAML, FileType.SQL]
    
    def is_binary_content(self, content: bytes) -> bool:
        return BinaryDetector.is_binary(content)


class PythonCodeChunkingStrategy(ChunkingStrategy):
    """Python 代码感知分块策略 (处理复杂嵌套结构)"""
    
    CLASS_PATTERN = re.compile(r"^class\s+(\w+)", re.MULTILINE)
    DEF_PATTERN = re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)
    IMPORT_PATTERN = re.compile(r"^(?:import|from)\s+.+", re.MULTILINE)
    DECORATOR_PATTERN = re.compile(r"^@\w+", re.MULTILINE)
    
    INDENT_PATTERN = re.compile(r"^(\s*)")
    
    def __init__(
        self,
        chunk_size: int = 1200,
        min_chunk_size: int = 150,
        max_nesting_depth: int = 10
    ):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_nesting_depth = max_nesting_depth
    
    def _extract_primary_class_name(self, content: str) -> Optional[str]:
        """从内容中提取主类名（第一个类定义）"""
        lines = content.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('class ') and ':' in stripped:
                match = re.match(r'class\s+(\w+)', stripped)
                if match:
                    return match.group(1)
        return None
    
    def _extract_primary_function_name(self, content: str) -> Optional[str]:
        """从内容中提取主函数名（第一个函数定义）"""
        lines = content.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('def ', 'async def ')) and ':' in stripped:
                match = re.match(r'(?:async\s+)?def\s+(\w+)', stripped)
                if match:
                    return match.group(1)
        return None
    
    def _extract_imports(self, content: str) -> Tuple[str, List[int]]:
        """
        提取顶部的导入语句
        
        Returns:
            (导入语句字符串, 导入语句行号列表)
        """
        lines = content.split('\n')
        imports = []
        import_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if stripped.startswith('import ') or stripped.startswith('from '):
                if not stripped.startswith('#'):
                    imports.append(line)
                    import_lines.append(i)
            elif stripped and not stripped.startswith('#'):
                break
        
        return '\n'.join(imports), import_lines
    
    def _remove_imports(self, content: str, import_lines: List[int]) -> str:
        """移除内容中的导入语句"""
        lines = content.split('\n')
        result = [line for i, line in enumerate(lines) if i not in import_lines]
        return '\n'.join(line for line in result if line.strip() is not None)
    
    def chunk(self, content: str, file_path: str) -> List[Chunk]:
        chunks = self._parse_and_chunk(content)
        chunks = self._merge_small_chunks(chunks)
        
        result = []
        for i, chunk_data in enumerate(chunks):
            content_hash = hashlib.md5(chunk_data['content'].encode()).hexdigest()
            chunk_id = f"{Path(file_path).stem}_{content_hash[:8]}_{i}"
            
            class_name = chunk_data.get('class_name') or self._extract_primary_class_name(chunk_data['content'])
            function_name = chunk_data.get('function_name') or self._extract_primary_function_name(chunk_data['content'])
            
            metadata = ChunkMetadata(
                file_path=file_path,
                file_type=FileType.PYTHON,
                start_line=chunk_data.get('start_line', 0),
                end_line=chunk_data.get('end_line', 0),
                char_count=len(chunk_data['content']),
                content_hash=content_hash,
                language_structure=chunk_data.get('structure'),
                class_name=class_name,
                function_name=function_name
            )
            
            result.append(Chunk(id=chunk_id, content=chunk_data['content'], metadata=metadata))
        
        return result
    
    def _parse_and_chunk(self, content: str) -> List[Dict[str, Any]]:
        """解析 Python 代码并分块"""
        lines = content.split('\n')
        structures = self._find_structures(content)
        
        if not structures:
            return [{'content': content, 'start_line': 1, 'end_line': len(lines)}]
        
        chunks = []
        current_chunk_lines = []
        current_start = 0
        
        for i, struct in enumerate(structures):
            struct_content = content[current_start:struct['end']]
            
            if len('\n'.join(current_chunk_lines)) + len(struct_content) > self.chunk_size:
                if current_chunk_lines:
                    chunk_content = '\n'.join(current_chunk_lines)
                    chunks.append({
                        'content': chunk_content,
                        'start_line': current_start + 1,
                        'end_line': struct['start'],
                        'structure': struct.get('parent_structure'),
                        'class_name': struct.get('type') == 'class' and struct.get('name') or None,
                        'function_name': struct.get('type') == 'def' and struct.get('name') or None
                    })
                
                current_chunk_lines = list(struct_content.split('\n'))
                current_start = struct['start']
            else:
                if not current_chunk_lines:
                    current_start = struct['start']
                current_chunk_lines.extend(struct_content.split('\n'))
        
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            last_struct = structures[-1] if structures else None
            chunks.append({
                'content': chunk_content,
                'start_line': current_start + 1,
                'end_line': len(lines),
                'structure': last_struct.get('parent_structure') if last_struct else None,
                'class_name': last_struct.get('type') == 'class' and last_struct.get('name') or None if last_struct else None,
                'function_name': last_struct.get('type') == 'def' and last_struct.get('name') or None if last_struct else None
            })
        
        return chunks
    
    def _find_structures(self, content: str) -> List[Dict[str, Any]]:
        """查找代码结构 (类、函数、导入等)"""
        structures = []
        
        for match in self.CLASS_PATTERN.finditer(content):
            structures.append({
                'type': 'class',
                'name': match.group(1),
                'start': match.start(),
                'end': self._find_structure_end(content, match.start())
            })
        
        for match in self.DEF_PATTERN.finditer(content):
            structures.append({
                'type': 'def',
                'name': match.group(1),
                'start': match.start(),
                'end': self._find_structure_end(content, match.start())
            })
        
        structures.sort(key=lambda x: x['start'])
        
        for struct in structures:
            struct['parent_structure'] = self._find_parent_structure(struct, structures)
        
        return structures
    
    def _find_structure_end(self, content: str, start: int) -> int:
        """查找代码结构的结束位置（处理 docstring 和嵌套结构）"""
        lines = content.split('\n')
        line_starts = []
        pos = 0
        for i, line in enumerate(lines):
            line_starts.append(pos)
            pos += len(line) + 1
        
        line_idx = self._get_line_index(content, start)
        if line_idx >= len(lines):
            return len(content)
        
        base_indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
        
        current_indent = base_indent
        in_multiline_string = False
        string_delimiter = None
        i = line_idx
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            if in_multiline_string:
                if string_delimiter in line and self._is_closing_multiline_string(line, string_delimiter):
                    in_multiline_string = False
                    string_delimiter = None
                    i += 1
                    continue
                i += 1
                continue
            
            if not stripped:
                i += 1
                current_indent = base_indent
                continue
            
            line_indent = len(line) - len(line.lstrip())
            
            if line.startswith('"""') or line.startswith("'''"):
                delimiter = line[:3]
                if line.count(delimiter) >= 2:
                    continue
                if line.count(delimiter) == 1:
                    if not self._is_closing_multiline_string(line, delimiter):
                        in_multiline_string = True
                        string_delimiter = delimiter
                        i += 1
                        continue
                    else:
                        i += 1
                        continue
            
            if line_starts[i] > start:
                if stripped.startswith(('class ', 'def ', 'async def ')):
                    if line_indent <= base_indent:
                        return line_starts[i]
                elif line_indent <= base_indent and not stripped.startswith(('#', '"', "'")):
                    return line_starts[i]
            
            i += 1
        
        return len(content)
    
    def _is_closing_multiline_string(self, line: str, delimiter: str) -> bool:
        """检查行是否关闭多行字符串"""
        count = line.count(delimiter)
        return count >= 2 or (count == 1 and line.rstrip().endswith(delimiter))
    
    def _get_line_index(self, content: str, pos: int) -> int:
        """获取位置所在的行索引"""
        lines = content[:pos].split('\n')
        return len(lines) - 1
    
    def _find_parent_structure(
        self, 
        target: Dict, 
        all_structures: List[Dict]
    ) -> Optional[str]:
        """查找父结构"""
        target_start = target['start']
        parent = None
        
        for struct in all_structures:
            if struct['start'] < target_start and struct['end'] > target['end']:
                if parent is None or struct['end'] > parent['end']:
                    parent = struct
        
        if parent:
            return f"{parent['type']}:{parent['name']}"
        return None
    
    def _merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """合并小碎片，保留元数据"""
        if not chunks:
            return []
        
        merged = []
        buffer_content = ""
        buffer_metadata = {}
        
        for chunk in chunks:
            chunk_len = len(chunk['content'])
            buffer_len = len(buffer_content)
            
            if buffer_len + chunk_len <= self.chunk_size:
                if buffer_content:
                    buffer_content += f"\n\n{chunk['content']}"
                else:
                    buffer_content = chunk['content']
                
                if not buffer_metadata:
                    buffer_metadata = {
                        'class_name': chunk.get('class_name'),
                        'function_name': chunk.get('function_name'),
                        'structure': chunk.get('structure'),
                        'start_line': chunk.get('start_line')
                    }
            else:
                if buffer_content:
                    merged.append({
                        'content': buffer_content.strip(),
                        **buffer_metadata
                    })
                buffer_content = chunk['content']
                buffer_metadata = {
                    'class_name': chunk.get('class_name'),
                    'function_name': chunk.get('function_name'),
                    'structure': chunk.get('structure'),
                    'start_line': chunk.get('start_line')
                }
        
        if buffer_content:
            merged.append({
                'content': buffer_content.strip(),
                **buffer_metadata
            })
        
        return merged
    
    def can_handle(self, file_type: FileType) -> bool:
        return file_type == FileType.PYTHON
    
    def is_binary_content(self, content: bytes) -> bool:
        return BinaryDetector.is_binary(content)


class MarkdownChunkingStrategy(ChunkingStrategy):
    """Markdown 分块策略"""
    
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    TABLE_PATTERN = re.compile(r"\|[^\n]*\|[ \t]*(?:\n\|[^\n]*\|[ \t]*)*$", re.MULTILINE)
    
    def __init__(
        self,
        chunk_size: int = 1500,
        min_chunk_size: int = 200
    ):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
    
    def _find_tables(self, content: str) -> List[Tuple[int, int, str]]:
        """
        查找 Markdown 表格的位置范围
        
        Returns:
            [(start_pos, end_pos, table_content), ...]
        """
        tables = []
        
        for match in self.TABLE_PATTERN.finditer(content):
            table_content = match.group()
            start = match.start()
            end = match.end()
            
            lines = table_content.split('\n')
            header_count = 0
            for line in lines:
                if '|' in line and not line.strip().startswith('```'):
                    header_count += 1
                else:
                    break
            
            if header_count >= 2:
                tables.append((start, end, table_content))
        
        return tables
    
    def _protect_tables(self, content: str) -> Tuple[str, Dict[int, str]]:
        """
        保护表格不被切断
        
        Returns:
            (处理后的内容, {位置: 表格内容})
        """
        tables = self._find_tables(content)
        if not tables:
            return content, {}
        
        protected = {}
        placeholder = "___TABLE_PLACEHOLDER_{}___"
        result = content
        
        for i, (start, end, table_content) in enumerate(reversed(tables)):
            key = placeholder.format(i)
            protected[i] = table_content
            result = result[:start] + key + result[end:]
        
        return result, protected
    
    def _restore_tables(self, content: str, tables: Dict[int, str]) -> str:
        """恢复表格"""
        result = content
        for i, table_content in tables.items():
            result = result.replace(f"___TABLE_PLACEHOLDER_{i}___", table_content)
        return result
    
    def chunk(self, content: str, file_path: str) -> List[Chunk]:
        protected_content, tables = self._protect_tables(content)
        
        chunks = self._split_by_headings(protected_content)
        chunks = self._merge_small_chunks(chunks)
        
        result = []
        for i, chunk_data in enumerate(chunks):
            restored_content = self._restore_tables(chunk_data['content'], tables)
            
            content_hash = hashlib.md5(restored_content.encode()).hexdigest()
            chunk_id = f"{Path(file_path).stem}_{content_hash[:8]}_{i}"
            
            metadata = ChunkMetadata(
                file_path=file_path,
                file_type=FileType.MARKDOWN,
                start_line=0,
                end_line=restored_content.count('\n') + 1,
                char_count=len(restored_content),
                content_hash=content_hash,
                language_structure=chunk_data.get('heading')
            )
            
            result.append(Chunk(id=chunk_id, content=restored_content, metadata=metadata))
        
        return result
    
    def _split_by_headings(self, content: str) -> List[Dict[str, Any]]:
        """按标题分割"""
        sections = []
        current_section = ""
        current_heading = "unlabeled"
        
        lines = content.split('\n')
        
        for line in lines:
            heading_match = self.HEADING_PATTERN.match(line)
            if heading_match:
                if current_section:
                    sections.append({
                        'content': current_section.strip(),
                        'heading': current_heading
                    })
                current_heading = heading_match.group(2).strip()
                current_section = line + "\n"
            else:
                current_section += line + "\n"
        
        if current_section:
            sections.append({
                'content': current_section.strip(),
                'heading': current_heading
            })
        
        if not sections:
            sections = [{'content': content, 'heading': 'unlabeled'}]
        
        return sections
    
    def _merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """合并小碎片"""
        if not chunks:
            return []
        
        merged = []
        buffer = ""
        
        for chunk in chunks:
            if len(buffer) + len(chunk['content']) <= self.chunk_size:
                buffer += f"\n\n## {chunk.get('heading', '')}\n\n{chunk['content']}"
            else:
                if buffer:
                    merged.append({'content': buffer.strip()})
                buffer = f"## {chunk.get('heading', '')}\n\n{chunk['content']}"
        
        if buffer:
            merged.append({'content': buffer.strip()})
        
        return merged
    
    def can_handle(self, file_type: FileType) -> bool:
        return file_type == FileType.MARKDOWN
    
    def is_binary_content(self, content: bytes) -> bool:
        return BinaryDetector.is_binary(content)


class JavaScriptChunkingStrategy(ChunkingStrategy):
    """JavaScript/TypeScript 分块策略"""
    
    FUNCTION_PATTERN = re.compile(
        r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?function|(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)",
        re.MULTILINE
    )
    CLASS_PATTERN = re.compile(r"class\s+(\w+)", re.MULTILINE)
    IMPORT_PATTERN = re.compile(r"(?:import|export).+", re.MULTILINE)
    
    def __init__(
        self,
        chunk_size: int = 1000,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, content: str, file_path: str) -> List[Chunk]:
        chunks = self._split_by_structure(content)
        chunks = self._merge_small_chunks(chunks)
        
        result = []
        for i, chunk_data in enumerate(chunks):
            content_hash = hashlib.md5(chunk_data['content'].encode()).hexdigest()
            chunk_id = f"{Path(file_path).stem}_{content_hash[:8]}_{i}"
            
            metadata = ChunkMetadata(
                file_path=file_path,
                file_type=FileType.JAVASCRIPT if '.js' in file_path else FileType.TYPESCRIPT,
                start_line=chunk_data.get('start_line', 0),
                end_line=chunk_data.get('end_line', 0),
                char_count=len(chunk_data['content']),
                content_hash=content_hash,
                language_structure=chunk_data.get('structure')
            )
            
            result.append(Chunk(id=chunk_id, content=chunk_data['content'], metadata=metadata))
        
        return result
    
    def _split_by_structure(self, content: str) -> List[Dict[str, Any]]:
        """按代码结构分割"""
        structures = []
        
        for match in self.CLASS_PATTERN.finditer(content):
            structures.append({
                'type': 'class',
                'name': match.group(1),
                'start': match.start()
            })
        
        for match in self.FUNCTION_PATTERN.finditer(content):
            name = match.group(1) or match.group(2) or match.group(3)
            structures.append({
                'type': 'function',
                'name': name,
                'start': match.start()
            })
        
        structures.sort(key=lambda x: x['start'])
        
        if not structures:
            return [{'content': content}]
        
        chunks = []
        current_start = 0
        
        for i, struct in enumerate(structures):
            struct_content = content[current_start:struct['start']]
            
            if struct_content.strip():
                chunks.append({
                    'content': struct_content.strip(),
                    'structure': f"global_{i}"
                })
            
            next_end = structures[i + 1]['start'] if i + 1 < len(structures) else len(content)
            struct_content_full = content[struct['start']:next_end]
            chunks.append({
                'content': struct_content_full.strip(),
                'structure': f"{struct['type']}:{struct['name']}"
            })
            
            current_start = next_end
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """合并小碎片"""
        if not chunks:
            return []
        
        merged = []
        buffer = ""
        
        for chunk in chunks:
            if len(buffer) + len(chunk['content']) <= self.chunk_size:
                buffer += f"\n\n{chunk['content']}"
            else:
                if buffer:
                    merged.append({'content': buffer.strip()})
                buffer = chunk['content']
        
        if buffer:
            merged.append({'content': buffer.strip()})
        
        return merged
    
    def can_handle(self, file_type: FileType) -> bool:
        return file_type in [FileType.JAVASCRIPT, FileType.TYPESCRIPT]
    
    def is_binary_content(self, content: bytes) -> bool:
        return BinaryDetector.is_binary(content)


class SemanticBoundaryDetector:
    """语义边界检测器 - 解决语义断裂问题"""
    
    CONDITIONAL_START = re.compile(r'^\s*if\s+.+:\s*(?:#.*)?$', re.MULTILINE)
    CONDITIONAL_ELIF = re.compile(r'^\s*elif\s+.+:\s*(?:#.*)?$', re.MULTILINE)
    CONDITIONAL_ELSE = re.compile(r'^\s*else\s*:\s*(?:#.*)?$', re.MULTILINE)
    
    TRY_START = re.compile(r'^\s*try\s*:\s*(?:#.*)?$', re.MULTILINE)
    EXCEPT_CLAUSE = re.compile(r'^\s*except\s+.*:\s*(?:#.*)?$', re.MULTILINE)
    FINALLY_CLAUSE = re.compile(r'^\s*finally\s*:\s*(?:#.*)?$', re.MULTILINE)
    
    CLASS_DEF = re.compile(r'^\s*class\s+\w+.*:\s*(?:#.*)?$', re.MULTILINE)
    DEF_START = re.compile(r'^\s*(?:async\s+)?def\s+\w+\s*\([^)]*\)\s*(?:->\s*\w+\s*)?:\s*(?:#.*)?$', re.MULTILINE)
    
    LOOP_START = re.compile(r'^\s*(?:for|while)\s+.+:\s*(?:#.*)?$', re.MULTILINE)
    WITH_START = re.compile(r'^\s*with\s+.+:\s*(?:#.*)?$', re.MULTILINE)
    
    DECORATOR = re.compile(r'^\s*@\w+', re.MULTILINE)
    
    def __init__(self, preserve_semantic_boundaries: bool = True):
        self.preserve_semantic_boundaries = preserve_semantic_boundaries
    
    def find_semantic_boundaries(self, content: str) -> List[Dict[str, Any]]:
        """
        查找所有语义边界
        
        Returns:
            List[{
                'type': str,
                'start': int,
                'end': int,
                'line_start': int,
                'line_end': int,
                'name': Optional[str],
                'parent': Optional[str]
            }]
        """
        if not self.preserve_semantic_boundaries:
            return []
        
        boundaries = []
        lines = content.split('\n')
        
        self._find_conditionals(lines, boundaries)
        self._find_try_except(lines, boundaries)
        self._find_class_definitions(lines, boundaries)
        self._find_function_definitions(lines, boundaries)
        self._find_control_flow(lines, boundaries)
        
        boundaries.sort(key=lambda x: x['start'])
        self._link_conditional_groups(boundaries)
        self._link_try_except_groups(boundaries)
        
        return boundaries
    
    def _find_conditionals(self, lines: List[str], boundaries: List[Dict]):
        """查找条件语句边界 (if-elif-else)"""
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if self.CONDITIONAL_START.match(line):
                end = self._find_block_end(lines, i)
                boundaries.append({
                    'type': 'if_statement',
                    'start': self._get_pos(content='\n'.join(lines), line_idx=i),
                    'end': end,
                    'line_start': i + 1,
                    'line_end': self._get_line_number(lines, end),
                    'name': f'if_{i}',
                    'parent': None
                })
            
            elif self.CONDITIONAL_ELIF.match(line):
                boundaries.append({
                    'type': 'elif_statement',
                    'start': self._get_pos(content='\n'.join(lines), line_idx=i),
                    'end': self._find_block_end(lines, i),
                    'line_start': i + 1,
                    'line_end': 0,
                    'name': f'elif_{i}',
                    'parent': 'if'
                })
            
            elif self.CONDITIONAL_ELSE.match(line):
                boundaries.append({
                    'type': 'else_statement',
                    'start': self._get_pos(content='\n'.join(lines), line_idx=i),
                    'end': self._find_block_end(lines, i),
                    'line_start': i + 1,
                    'line_end': 0,
                    'name': f'else_{i}',
                    'parent': 'if'
                })
    
    def _find_try_except(self, lines: List[str], boundaries: List[Dict]):
        """查找异常处理边界 (try-except-finally)"""
        in_try = False
        try_start_idx = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if self.TRY_START.match(line):
                in_try = True
                try_start_idx = i
                end = self._find_block_end(lines, i)
                boundaries.append({
                    'type': 'try_block',
                    'start': self._get_pos(content='\n'.join(lines), line_idx=i),
                    'end': end,
                    'line_start': i + 1,
                    'line_end': self._get_line_number(lines, end),
                    'name': 'try',
                    'parent': None
                })
            
            elif self.EXCEPT_CLAUSE.match(line) and in_try:
                boundaries.append({
                    'type': 'except_clause',
                    'start': self._get_pos(content='\n'.join(lines), line_idx=i),
                    'end': self._find_block_end(lines, i),
                    'line_start': i + 1,
                    'line_end': 0,
                    'name': 'except',
                    'parent': 'try'
                })
            
            elif self.FINALLY_CLAUSE.match(line) and in_try:
                boundaries.append({
                    'type': 'finally_clause',
                    'start': self._get_pos(content='\n'.join(lines), line_idx=i),
                    'end': self._find_block_end(lines, i),
                    'line_start': i + 1,
                    'line_end': 0,
                    'name': 'finally',
                    'parent': 'try'
                })
                in_try = False
    
    def _find_class_definitions(self, lines: List[str], boundaries: List[Dict]):
        """查找类定义边界"""
        for i, line in enumerate(lines):
            if self.CLASS_DEF.match(line):
                end = self._find_block_end(lines, i)
                class_name = line.strip().split()[1].split('(')[0].rstrip(':')
                boundaries.append({
                    'type': 'class_definition',
                    'start': self._get_pos(content='\n'.join(lines), line_idx=i),
                    'end': end,
                    'line_start': i + 1,
                    'line_end': self._get_line_number(lines, end),
                    'name': class_name,
                    'parent': None
                })
    
    def _find_function_definitions(self, lines: List[str], boundaries: List[Dict]):
        """查找函数定义边界"""
        for i, line in enumerate(lines):
            if self.DEF_START.match(line):
                end = self._find_block_end(lines, i)
                func_name = line.strip().split('(')[0].split('def ')[1].split()[0]
                boundaries.append({
                    'type': 'function_definition',
                    'start': self._get_pos(content='\n'.join(lines), line_idx=i),
                    'end': end,
                    'line_start': i + 1,
                    'line_end': self._get_line_number(lines, end),
                    'name': func_name,
                    'parent': None
                })
    
    def _find_control_flow(self, lines: List[str], boundaries: List[Dict]):
        """查找控制流边界 (for, while, with)"""
        for i, line in enumerate(lines):
            if self.LOOP_START.match(line) or self.WITH_START.match(line):
                end = self._find_block_end(lines, i)
                boundaries.append({
                    'type': 'control_flow',
                    'start': self._get_pos(content='\n'.join(lines), line_idx=i),
                    'end': end,
                    'line_start': i + 1,
                    'line_end': self._get_line_number(lines, end),
                    'name': None,
                    'parent': None
                })
    
    def _find_block_end(self, lines: List[str], start_idx: int) -> int:
        """查找代码块的结束位置"""
        if start_idx >= len(lines):
            return len('\n'.join(lines))
        
        line = lines[start_idx]
        base_indent = len(line) - len(line.lstrip()) if line.strip() else 0
        
        for i in range(start_idx + 1, len(lines)):
            current_line = lines[i]
            stripped = current_line.strip()
            
            if not stripped or stripped.startswith('#'):
                continue
            
            current_indent = len(current_line) - len(current_line.lstrip())
            
            if current_indent <= base_indent and stripped:
                if not any(stripped.startswith(kw) for kw in ['elif', 'else', 'except', 'finally']):
                    return self._get_pos(content='\n'.join(lines), line_idx=i)
        
        return len('\n'.join(lines))
    
    def _get_pos(self, content: str, line_idx: int) -> int:
        """获取指定行索引在内容中的位置"""
        lines = content[:].split('\n')
        pos = 0
        for i in range(line_idx):
            pos += len(lines[i]) + 1
        return pos
    
    def _get_line_number(self, lines: List[str], pos: int) -> int:
        """获取位置对应的行号"""
        content = '\n'.join(lines)
        return content[:pos].count('\n') + 1
    
    def _link_conditional_groups(self, boundaries: List[Dict]):
        """链接条件语句组 (if-elif-else)"""
        if_groups = defaultdict(list)
        current_if = None
        
        for b in boundaries:
            if b['type'] == 'if_statement':
                current_if = b['name']
                if_groups[current_if].append(b)
            elif b['type'] in ('elif_statement', 'else_statement') and current_if:
                if_groups[current_if].append(b)
        
        for group in if_groups.values():
            for i, boundary in enumerate(group):
                if i > 0:
                    boundary['parent'] = group[0]['name']
                    boundary['group_members'] = [b['name'] for b in group]
    
    def _link_try_except_groups(self, boundaries: List[Dict]):
        """链接异常处理组 (try-except-finally)"""
        try_groups = defaultdict(list)
        current_try = None
        
        for b in boundaries:
            if b['type'] == 'try_block':
                current_try = b['name']
                try_groups[current_try].append(b)
            elif b['type'] in ('except_clause', 'finally_clause') and current_try:
                try_groups[current_try].append(b)
        
        for group in try_groups.values():
            for i, boundary in enumerate(group):
                if i > 0:
                    boundary['parent'] = group[0]['name']
                    boundary['group_members'] = [b['name'] for b in group]


class SemanticReferenceBuilder:
    """语义引用构建器 - 解决指代消解丢失问题"""
    
    SELF_PATTERN = re.compile(r'self\.(\w+)')
    CLASS_REF_PATTERN = re.compile(r'cls\.(\w+)')
    CONFIG_PATTERN = re.compile(r'(?:self\.)?config\.(\w+)')
    
    IMPORT_PATTERN = re.compile(r'^(?:from|import)\s+(?:\w+\s+)?(?:(\w+))', re.MULTILINE)
    TYPE_PATTERN = re.compile(r'->\s*(\w+)')
    
    def __init__(self):
        self.imports = []
        self.class_attributes = defaultdict(set)
        self.method_calls = defaultdict(list)
        self.config_refs = []
        self.type_refs = []
    
    def analyze_references(self, content: str, class_name: Optional[str] = None) -> Dict[str, Any]:
        """分析代码中的引用关系"""
        self.imports = self._extract_imports(content)
        self._extract_self_references(content, class_name)
        self._extract_config_references(content)
        self._extract_type_references(content)
        
        return {
            'imports': self.imports,
            'class_attributes': dict(self.class_attributes),
            'method_calls': dict(self.method_calls),
            'config_refs': self.config_refs,
            'type_refs': self.type_refs
        }
    
    def _extract_imports(self, content: str) -> List[Dict[str, str]]:
        """提取导入语句"""
        imports = []
        for match in self.IMPORT_PATTERN.finditer(content):
            module = match.group(1)
            line_no = content[:match.start()].count('\n') + 1
            imports.append({
                'module': module,
                'line': line_no
            })
        return imports
    
    def _extract_self_references(self, content: str, class_name: Optional[str]):
        """提取 self 引用"""
        if not class_name:
            return
        
        for match in self.SELF_PATTERN.finditer(content):
            attr = match.group(1)
            self.class_attributes[class_name].add(attr)
            
            line_no = content[:match.start()].count('\n') + 1
            self.method_calls[class_name].append({
                'attribute': attr,
                'line': line_no
            })
    
    def _extract_config_references(self, content: str):
        """提取配置引用"""
        for match in self.CONFIG_PATTERN.finditer(content):
            config_attr = match.group(1)
            self.config_refs.append(config_attr)
    
    def _extract_type_references(self, content: str):
        """提取类型引用"""
        for match in self.TYPE_PATTERN.finditer(content):
            type_name = match.group(1)
            self.type_refs.append(type_name)
    
    def get_reference_context(self) -> Dict[str, List[str]]:
        """获取引用上下文"""
        return {
            'imports': [imp['module'] for imp in self.imports],
            'config_attributes': list(set(self.config_refs)),
            'type_references': list(set(self.type_refs))
        }


class SemanticPreservingStrategy(ChunkingStrategy):
    """语义保留分块策略 - 核心策略：解决所有语义问题"""
    
    def __init__(
        self,
        chunk_size: int = 1200,
        min_chunk_size: int = 150,
        preserve_semantic_boundaries: bool = True,
        add_ghost_context: bool = True
    ):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.preserve_semantic_boundaries = preserve_semantic_boundaries
        self.add_ghost_context = add_ghost_context
        
        self.boundary_detector = SemanticBoundaryDetector(preserve_semantic_boundaries)
        self.reference_builder = SemanticReferenceBuilder()
    
    def chunk(self, content: str, file_path: str) -> List[Chunk]:
        semantic_boundaries = self.boundary_detector.find_semantic_boundaries(content)
        
        initial_chunks = self._create_initial_chunks(content, semantic_boundaries)
        merged_chunks = self._merge_with_semantic_awareness(initial_chunks, semantic_boundaries)
        
        class_name = self._extract_class_name(content)
        ref_context = self.reference_builder.analyze_references(content, class_name)
        
        result = []
        for i, chunk_data in enumerate(merged_chunks):
            chunk_id = self._generate_chunk_id(file_path, chunk_data, i)
            
            metadata = self._create_metadata(chunk_data, file_path, ref_context, semantic_boundaries)
            
            result.append(Chunk(id=chunk_id, content=chunk_data['content'], metadata=metadata))
        
        self._link_semantic_relations(result, semantic_boundaries)
        
        return result
    
    def _create_initial_chunks(
        self, 
        content: str, 
        boundaries: List[Dict]
    ) -> List[Dict[str, Any]]:
        """基于语义边界创建初始块 - 创建类和函数的独立块"""
        if not boundaries:
            return [{'content': content, 'start_line': 1, 'end_line': content.count('\n') + 1}]
        
        lines = content.split('\n')
        
        primary_boundaries = [b for b in boundaries if b['type'] in ('class_definition', 'function_definition')]
        
        if not primary_boundaries:
            return [{'content': content, 'start_line': 1, 'end_line': len(lines)}]
        
        chunks = []
        used_ranges = set()
        
        for boundary in primary_boundaries:
            start_line = boundary['line_start']
            end_line = boundary['line_end']
            
            if start_line <= 0 or start_line > len(lines):
                continue
            
            range_key = (start_line, end_line)
            if range_key in used_ranges:
                continue
            used_ranges.add(range_key)
            
            chunk_lines = lines[start_line - 1:end_line]
            if not chunk_lines:
                continue
                
            chunk_content = '\n'.join(chunk_lines)
            
            nested = [b for b in boundaries 
                     if b != boundary 
                     and b['start'] > boundary['start'] 
                     and b['end'] <= boundary['end']
                     and b['type'] in ('if_statement', 'elif_statement', 'else_statement', 'try_block', 'except_clause', 'finally_clause')]
            
            chunks.append({
                'content': chunk_content,
                'start_line': start_line,
                'end_line': end_line,
                'boundary_type': boundary['type'],
                'boundary_name': boundary['name'],
                'parent_boundary': boundary.get('parent'),
                'group_members': boundary.get('group_members', []),
                'nested_boundaries': nested
            })
        
        return chunks
    
    def _merge_with_semantic_awareness(
        self, 
        chunks: List[Dict], 
        boundaries: List[Dict]
    ) -> List[Dict[str, Any]]:
        """语义感知的合并策略 - 保持语义边界完整性"""
        if not chunks:
            return []
        
        merged = []
        buffer = ""
        buffer_metadata = {}
        
        for chunk in chunks:
            chunk_len = len(chunk['content'])
            boundary_type = chunk.get('boundary_type')
            
            should_merge = (
                buffer and 
                len(buffer) + chunk_len <= self.chunk_size and
                buffer_metadata.get('boundary_type') == boundary_type and
                buffer_metadata.get('boundary_type') in ('if_statement', 'elif_statement', 'else_statement', 'except_clause', 'finally_clause')
            )
            
            if should_merge:
                buffer += f"\n\n{chunk['content']}"
            else:
                if buffer:
                    merged.append({
                        'content': buffer.strip(),
                        **buffer_metadata
                    })
                buffer = chunk['content']
                buffer_metadata = {
                    'start_line': chunk['start_line'],
                    'boundary_type': boundary_type,
                    'boundary_name': chunk.get('boundary_name'),
                    'group_members': chunk.get('group_members', [])
                }
        
        if buffer:
            merged.append({
                'content': buffer.strip(),
                **buffer_metadata
            })
        
        return merged
    
    def _create_metadata(
        self,
        chunk_data: Dict[str, Any],
        file_path: str,
        ref_context: Dict[str, Any],
        boundaries: List[Dict]
    ) -> ChunkMetadata:
        """创建增强的元数据"""
        content_hash = hashlib.md5(chunk_data['content'].encode()).hexdigest()
        
        boundary_type = chunk_data.get('boundary_type')
        boundary_name = chunk_data.get('boundary_name')
        
        semantic_type = None
        if boundary_type == 'class_definition':
            semantic_type = 'class'
        elif boundary_type == 'function_definition':
            semantic_type = 'function'
        elif boundary_type == 'if_statement':
            semantic_type = 'conditional'
        elif boundary_type == 'try_block':
            semantic_type = 'error_handling'
        
        ghost_context = None
        if self.add_ghost_context and boundary_type in ('function_definition', 'method'):
            ghost_context = {
                'class_context': ref_context.get('class_attributes', {}),
                'imports': ref_context.get('imports', []),
                'config_refs': ref_context.get('config_refs', [])
            }
        
        return ChunkMetadata(
            file_path=file_path,
            file_type=FileType.PYTHON,
            start_line=chunk_data.get('start_line', 0),
            end_line=chunk_data.get('end_line', chunk_data.get('start_line', 0) + chunk_data.get('content', '').count('\n')),
            char_count=len(chunk_data['content']),
            content_hash=content_hash,
            language_structure=boundary_type,
            function_name=boundary_name if boundary_type == 'function_definition' else None,
            class_name=boundary_name if boundary_type == 'class_definition' else None,
            related_chunks=chunk_data.get('group_members', []),
            semantic_type=semantic_type,
            reference_context=ref_context,
            control_flow_context={
                'boundary_type': boundary_type,
                'boundary_name': boundary_name,
                'has_else': 'else' in str(chunk_data.get('group_members', [])),
                'has_except': 'except' in str(chunk_data.get('group_members', []))
            } if boundary_type in ('if_statement', 'try_block') else None
        )
    
    def _generate_chunk_id(
        self, 
        file_path: str, 
        chunk_data: Dict, 
        index: int
    ) -> str:
        """生成唯一的块 ID"""
        content_hash = hashlib.md5(chunk_data['content'].encode()).hexdigest()
        boundary_name = chunk_data.get('boundary_name', '')
        return f"{Path(file_path).stem}_{boundary_name}_{content_hash[:8]}_{index}"
    
    def _extract_class_name(self, content: str) -> Optional[str]:
        """提取类名"""
        class_match = re.search(r'^class\s+(\w+)', content, re.MULTILINE)
        return class_match.group(1) if class_match else None
    
    def _link_semantic_relations(
        self, 
        chunks: List[Chunk], 
        boundaries: List[Dict]
    ):
        """链接语义关系"""
        chunk_map = {chunk.metadata.start_line: chunk for chunk in chunks}
        
        for boundary in boundaries:
            if boundary['type'] in ('if_statement', 'try_block'):
                start_line = boundary['line_start']
                
                if start_line in chunk_map:
                    main_chunk = chunk_map[start_line]
                    
                    group_members = boundary.get('group_members', [])
                    related_ids = []
                    
                    for member_name in group_members:
                        for chunk in chunks:
                            if chunk.metadata.language_structure in ('elif_statement', 'else_statement', 'except_clause') and member_name in str(chunk.metadata.related_chunks):
                                related_ids.append(chunk.id)
                    
                    if related_ids:
                        relation = SemanticRelation(
                            relation_type=SemanticRelationType.CONDITIONAL_GROUP if boundary['type'] == 'if_statement' else SemanticRelationType.TRY_EXCEPT_GROUP,
                            related_chunk_ids=related_ids,
                            context={'group_name': boundary['name']}
                        )
                        main_chunk.metadata.semantic_relations.append(relation)
    
    def can_handle(self, file_type: FileType) -> bool:
        return file_type == FileType.PYTHON
    
    def is_binary_content(self, content: bytes) -> bool:
        return BinaryDetector.is_binary(content)


class ContextualRetrievalEnricher:
    """
    上下文检索增强器 - 实现 Anthropic 的 Contextual Retrieval 方案
    
    在向量化之前，为每个 Chunk 生成简短的上下文说明，
    解决指代消解丢失问题。
    
    原理：
    - 原始 Chunk: if self.config.timeout:...
    - 增强后: [Context: This code belongs to class DataProcessor. 
              self.config is initialized in __init__ with a configuration object.]
              if self.config.timeout:...
    """
    
    SELF_PATTERN = re.compile(r'self\.(\w+)')
    CLS_PATTERN = re.compile(r'cls\.(\w+)')
    CONFIG_PATTERN = re.compile(r'(?:self\.)?config\.(\w+)')
    RETURN_TYPE_PATTERN = re.compile(r'->\s*([A-Z]\w+)')
    
    def __init__(
        self,
        llm_service: Any = None,
        include_imports: bool = True,
        include_class_context: bool = True,
        include_type_info: bool = True
    ):
        self.llm_service = llm_service
        self.include_imports = include_imports
        self.include_class_context = include_class_context
        self.include_type_info = include_type_info
    
    def enrich_chunk(
        self,
        chunk: Chunk,
        class_context: Optional[Dict[str, Any]] = None,
        import_context: Optional[List[str]] = None
    ) -> Chunk:
        """
        为单个 Chunk 生成上下文并增强
        
        Args:
            chunk: 原始分块
            class_context: 类上下文信息
            import_context: 导入语句上下文
            
        Returns:
            增强后的分块
        """
        context_parts = []
        
        if self.include_class_context and class_context:
            context_parts.append(self._generate_class_context(chunk, class_context))
        
        if self.include_imports and import_context:
            context_parts.append(self._generate_import_context(chunk, import_context))
        
        if self.include_type_info:
            context_parts.append(self._generate_type_context(chunk))
        
        if context_parts:
            context_str = " ".join(context_parts)
            enriched_content = f"[Context: {context_str}] {chunk.content}"
            
            enriched_metadata = chunk.metadata
            enriched_metadata.ghost_context = {
                'contextual_summary': context_str,
                'is_enriched': True,
                'class_context': class_context,
                'import_context': import_context
            }
            
            return Chunk(
                id=chunk.id,
                content=enriched_content,
                metadata=enriched_metadata
            )
        
        return chunk
    
    def enrich_chunks(
        self,
        chunks: List[Chunk],
        file_content: str
    ) -> List[Chunk]:
        """
        为所有 Chunk 生成上下文并增强
        
        Args:
            chunks: 分块列表
            file_content: 完整文件内容（用于提取类信息和导入）
            
        Returns:
            增强后的分块列表
        """
        class_info = self._extract_class_info(file_content)
        import_info = self._extract_imports(file_content)
        
        enriched_chunks = []
        for chunk in chunks:
            class_context = class_info.get(chunk.metadata.class_name) if chunk.metadata.class_name else None
            enriched_chunk = self.enrich_chunk(chunk, class_context, import_info)
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks
    
    def _generate_class_context(
        self,
        chunk: Chunk,
        class_context: Dict[str, Any]
    ) -> str:
        """生成类上下文说明"""
        parts = []
        
        class_name = class_context.get('name', '')
        if class_name:
            parts.append(f"This code belongs to class {class_name}")
        
        attributes = class_context.get('attributes', [])
        if attributes:
            attr_str = ", ".join(attributes[:5])
            parts.append(f"Class attributes: {attr_str}")
        
        methods = class_context.get('methods', [])
        if methods and chunk.metadata.function_name:
            parts.append(f"Available methods: {', '.join(methods[:3])}")
        
        return "; ".join(parts) + "."
    
    def _generate_import_context(
        self,
        chunk: Chunk,
        import_context: List[str]
    ) -> str:
        """生成导入上下文说明"""
        if not import_context:
            return ""
        
        relevant_imports = []
        content_lower = chunk.content.lower()
        
        for imp in import_context:
            if any(word in content_lower for word in imp.lower().split('.')):
                relevant_imports.append(imp)
        
        if relevant_imports:
            return f"Imports: {', '.join(relevant_imports[:3])}"
        
        return ""
    
    def _generate_type_context(self, chunk: Chunk) -> str:
        """生成类型上下文说明"""
        parts = []
        
        return_types = self.RETURN_TYPE_PATTERN.findall(chunk.content)
        if return_types:
            unique_types = list(set(return_types))
            parts.append(f"Returns: {', '.join(unique_types)}")
        
        config_refs = self.CONFIG_PATTERN.findall(chunk.content)
        if config_refs:
            parts.append(f"Config accesses: {', '.join(set(config_refs))}")
        
        if parts:
            return "; ".join(parts) + "."
        
        return ""
    
    def _extract_class_info(self, content: str) -> Dict[str, Dict[str, Any]]:
        """从文件中提取类信息"""
        classes = {}
        
        class_pattern = re.compile(
            r'^class\s+(\w+)(?:\([^)]*\))?:\s*(?:#.*)?$',
            re.MULTILINE
        )
        
        attr_pattern = re.compile(r'self\.(\w+)\s*=')
        method_pattern = re.compile(r'^\s+def\s+(\w+)')
        
        for class_match in class_pattern.finditer(content):
            class_name = class_match.group(1)
            class_start = class_match.end()
            
            class_content = content[class_start:]
            lines = class_content.split('\n')
            
            attributes = set()
            methods = set()
            
            for line in lines[:50]:
                indent = len(line) - len(line.lstrip()) if line.strip() else 0
                if indent > 0 and not line.strip().startswith('#'):
                    break
                
                for attr_match in attr_pattern.finditer(line):
                    attributes.add(attr_match.group(1))
                
                for method_match in method_pattern.finditer(line):
                    methods.add(method_match.group(1))
            
            classes[class_name] = {
                'name': class_name,
                'attributes': list(attributes),
                'methods': list(methods)
            }
        
        return classes
    
    def _extract_imports(self, content: str) -> List[str]:
        """从文件中提取导入语句"""
        imports = []
        
        import_patterns = [
            re.compile(r'^(?:from|import)\s+([\w.]+)', re.MULTILINE),
            re.compile(r'^from\s+([\w.]+)\s+import', re.MULTILINE)
        ]
        
        for pattern in import_patterns:
            for match in pattern.finditer(content):
                module = match.group(1)
                if module and module not in ('__future__', 'typing'):
                    imports.append(module)
        
        return list(set(imports))


class AtomicChunkingStrategy:
    """
    AST 原子分块策略 - 解决语义断裂问题
    
    强制将 if/else、try/except、for/while 等结构作为原子单元，
    宁可对内部内容摘要，也不切断结构。
    
    原理：
    - 如果 try/except 块总长超过 Chunk Size，摘要内部内容
    - 保持条件语句完整性
    - 使用父文档检索作为后备
    """
    
    ATOMIC_STRUCTURES = {
        'if_statement',
        'elif_statement', 
        'else_statement',
        'try_block',
        'except_clause',
        'finally_clause',
        'for_statement',
        'while_statement',
        'with_statement',
        'async_for_statement',
        'async_with_statement'
    }
    
    def __init__(
        self,
        base_strategy: ChunkingStrategy,
        max_atomic_size: int = 2000,
        summary_threshold: float = 0.8
    ):
        self.base_strategy = base_strategy
        self.max_atomic_size = max_atomic_size
        self.summary_threshold = summary_threshold
    
    def chunk(self, content: str, file_path: str) -> List[Chunk]:
        """执行原子分块"""
        boundaries = self._find_atomic_boundaries(content)
        
        if not boundaries:
            return self.base_strategy.chunk(content, file_path)
        
        atomic_chunks = self._create_atomic_chunks(content, boundaries)
        atomic_chunks = self._merge_small_atomics(atomic_chunks)
        
        result = []
        for i, chunk_data in enumerate(atomic_chunks):
            chunk_id = self._generate_chunk_id(file_path, chunk_data, i)
            
            metadata = self._create_atomic_metadata(chunk_data, file_path)
            
            if len(chunk_data['content']) > self.max_atomic_size:
                summarized = self._summarize_large_chunk(chunk_data['content'])
                chunk_data['content'] = summarized
                metadata.is_truncated = True
            
            result.append(Chunk(id=chunk_id, content=chunk_data['content'], metadata=metadata))
        
        return result
    
    def _find_atomic_boundaries(self, content: str) -> List[Dict[str, Any]]:
        """查找原子结构边界"""
        boundaries = []
        lines = content.split('\n')
        
        boundary_methods = [
            self._find_conditionals,
            self._find_try_except,
            self._find_loops,
            self._find_with_statements
        ]
        
        for method in boundary_methods:
            boundaries.extend(method(lines))
        
        boundaries.sort(key=lambda x: x['start'])
        return boundaries
    
    def _find_conditionals(self, lines: List[str]) -> List[Dict[str, Any]]:
        """查找条件语句"""
        boundaries = []
        stack = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if re.match(r'^\s*if\s+.+:\s*$', stripped):
                end = self._find_block_end(lines, i)
                boundaries.append({
                    'type': 'if_statement',
                    'start': self._get_pos(lines, i),
                    'end': end,
                    'line_start': i + 1,
                    'line_end': self._get_line_number(lines, end),
                    'name': f'if_{i}'
                })
                stack.append(('if', i))
            
            elif re.match(r'^\s*elif\s+.+:\s*$', stripped) and stack:
                boundaries.append({
                    'type': 'elif_statement',
                    'start': self._get_pos(lines, i),
                    'end': self._find_block_end(lines, i),
                    'line_start': i + 1,
                    'line_end': 0,
                    'name': f'elif_{i}',
                    'parent': stack[-1][1]
                })
            
            elif re.match(r'^\s*else\s*:\s*$', stripped) and stack:
                boundaries.append({
                    'type': 'else_statement',
                    'start': self._get_pos(lines, i),
                    'end': self._find_block_end(lines, i),
                    'line_start': i + 1,
                    'line_end': 0,
                    'name': f'else_{i}',
                    'parent': stack[-1][1]
                })
                stack.pop()
        
        return boundaries
    
    def _find_try_except(self, lines: List[str]) -> List[Dict[str, Any]]:
        """查找异常处理"""
        boundaries = []
        in_try = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if re.match(r'^\s*try\s*:\s*$', stripped):
                end = self._find_block_end(lines, i)
                boundaries.append({
                    'type': 'try_block',
                    'start': self._get_pos(lines, i),
                    'end': end,
                    'line_start': i + 1,
                    'line_end': self._get_line_number(lines, end),
                    'name': 'try'
                })
                in_try = True
            
            elif re.match(r'^\s*except\b.*:\s*$', stripped) and in_try:
                boundaries.append({
                    'type': 'except_clause',
                    'start': self._get_pos(lines, i),
                    'end': self._find_block_end(lines, i),
                    'line_start': i + 1,
                    'line_end': 0,
                    'name': 'except',
                    'parent': 'try'
                })
            
            elif re.match(r'^\s*finally\s*:\s*$', stripped) and in_try:
                boundaries.append({
                    'type': 'finally_clause',
                    'start': self._get_pos(lines, i),
                    'end': self._find_block_end(lines, i),
                    'line_start': i + 1,
                    'line_end': 0,
                    'name': 'finally',
                    'parent': 'try'
                })
                in_try = False
        
        return boundaries
    
    def _find_loops(self, lines: List[str]) -> List[Dict[str, Any]]:
        """查找循环结构"""
        boundaries = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if re.match(r'^\s*(?:for|while)\s+.+:\s*$', stripped):
                end = self._find_block_end(lines, i)
                boundaries.append({
                    'type': f'{re.match(r"^\s*(for|while)", stripped).group(1)}_statement',
                    'start': self._get_pos(lines, i),
                    'end': end,
                    'line_start': i + 1,
                    'line_end': self._get_line_number(lines, end),
                    'name': f'loop_{i}'
                })
        
        return boundaries
    
    def _find_with_statements(self, lines: List[str]) -> List[Dict[str, Any]]:
        """查找 with 语句"""
        boundaries = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if re.match(r'^\s*with\s+.+:\s*$', stripped):
                end = self._find_block_end(lines, i)
                boundaries.append({
                    'type': 'with_statement',
                    'start': self._get_pos(lines, i),
                    'end': end,
                    'line_start': i + 1,
                    'line_end': self._get_line_number(lines, end),
                    'name': f'with_{i}'
                })
        
        return boundaries
    
    def _find_block_end(self, lines: List[str], start_idx: int) -> int:
        """查找代码块结束位置"""
        if start_idx >= len(lines):
            return len('\n'.join(lines))
        
        line = lines[start_idx]
        base_indent = len(line) - len(line.lstrip()) if line.strip() else 0
        
        for i in range(start_idx + 1, len(lines)):
            current_line = lines[i]
            stripped = current_line.strip()
            
            if not stripped or stripped.startswith('#'):
                continue
            
            current_indent = len(current_line) - len(current_line.lstrip())
            
            if current_indent <= base_indent and stripped:
                if not any(stripped.startswith(kw) for kw in ['elif', 'else', 'except', 'finally']):
                    return self._get_pos(lines, i)
        
        return len('\n'.join(lines))
    
    def _get_pos(self, lines: List[str], line_idx: int) -> int:
        """获取行索引在内容中的位置"""
        content = '\n'.join(lines)
        return sum(len(lines[j]) + 1 for j in range(line_idx))
    
    def _get_line_number(self, lines: List[str], pos: int) -> int:
        """获取位置对应的行号"""
        content = '\n'.join(lines)
        return content[:pos].count('\n') + 1
    
    def _create_atomic_chunks(
        self,
        content: str,
        boundaries: List[Dict]
    ) -> List[Dict[str, Any]]:
        """创建原子块"""
        if not boundaries:
            return [{'content': content, 'start_line': 1, 'end_line': len(content.split('\n'))}]
        
        lines = content.split('\n')
        chunks = []
        
        for boundary in boundaries:
            start_line = boundary['line_start']
            end_line = boundary['line_end']
            
            if start_line <= 0 or start_line > len(lines):
                continue
            
            chunk_lines = lines[start_line - 1:end_line]
            if not chunk_lines:
                continue
            
            chunks.append({
                'content': '\n'.join(chunk_lines),
                'start_line': start_line,
                'end_line': end_line,
                'boundary_type': boundary['type'],
                'boundary_name': boundary['name'],
                'is_atomic': boundary['type'] in self.ATOMIC_STRUCTURES
            })
        
        return chunks
    
    def _merge_small_atomics(self, chunks: List[Dict]) -> List[Dict]:
        """合并小型原子块"""
        if not chunks:
            return []
        
        merged = []
        buffer = ""
        buffer_start = 0
        
        for chunk in chunks:
            if chunk.get('is_atomic') and chunk['content'].count('\n') < 3:
                if buffer:
                    merged.append({
                        'content': buffer,
                        'start_line': buffer_start,
                        'end_line': buffer_start + buffer.count('\n'),
                        'boundary_type': 'atomic_group'
                    })
                merged.append(chunk)
                buffer = ""
                buffer_start = 0
            else:
                if buffer:
                    buffer += '\n\n'
                    buffer_start = merged[-1]['end_line'] if merged else buffer_start
                buffer += chunk['content']
        
        if buffer:
            merged.append({
                'content': buffer,
                'start_line': buffer_start,
                'end_line': buffer_start + buffer.count('\n'),
                'boundary_type': 'atomic_group'
            })
        
        return merged
    
    def _summarize_large_chunk(self, content: str) -> str:
        """对大型块进行摘要"""
        lines = content.split('\n')
        
        summary_lines = [
            f"// Atomic block: {content.split(chr(10))[0].strip() if content else 'unnamed'}",
            f"// Total lines: {len(lines)}",
            "// ... (content summarized for size constraints)"
        ]
        
        return '\n'.join(summary_lines[:5])
    
    def _generate_chunk_id(
        self,
        file_path: str,
        chunk_data: Dict,
        index: int
    ) -> str:
        """生成块 ID"""
        content_hash = hashlib.md5(chunk_data['content'].encode()).hexdigest()
        boundary_name = chunk_data.get('boundary_name', '')
        return f"{Path(file_path).stem}_{boundary_name}_{content_hash[:8]}_{index}"
    
    def _create_atomic_metadata(
        self,
        chunk_data: Dict[str, Any],
        file_path: str
    ) -> ChunkMetadata:
        """创建原子块元数据"""
        content_hash = hashlib.md5(chunk_data['content'].encode()).hexdigest()
        
        boundary_type = chunk_data.get('boundary_type', 'atomic')
        
        semantic_type = None
        if 'if' in boundary_type or 'elif' in boundary_type or 'else' in boundary_type:
            semantic_type = 'conditional'
        elif 'except' in boundary_type or 'finally' in boundary_type or 'try' in boundary_type:
            semantic_type = 'error_handling'
        elif 'for' in boundary_type or 'while' in boundary_type:
            semantic_type = 'loop'
        elif 'with' in boundary_type:
            semantic_type = 'context_manager'
        
        return ChunkMetadata(
            file_path=file_path,
            file_type=FileType.PYTHON,
            start_line=chunk_data.get('start_line', 0),
            end_line=chunk_data.get('end_line', 0),
            char_count=len(chunk_data['content']),
            content_hash=content_hash,
            language_structure=boundary_type,
            semantic_type=semantic_type,
            is_atomic=chunk_data.get('is_atomic', False)
        )
    
    def can_handle(self, file_type: FileType) -> bool:
        return self.base_strategy.can_handle(file_type)
    
    def is_binary_content(self, content: bytes) -> bool:
        return self.base_strategy.is_binary_content(content)


class SummaryBuilder:
    """
    摘要构建器 - 为代码块生成摘要索引
    
    用于在检索时快速理解代码逻辑，
    解决逻辑丢失问题。
    """
    
    def __init__(self, llm_service: Any = None):
        self.llm_service = llm_service
    
    def build_function_summary(self, func_code: str) -> str:
        """为函数生成摘要"""
        lines = func_code.split('\n')
        
        if len(lines) < 3:
            return func_code.strip()
        
        first_line = lines[0].strip()
        
        summary_parts = [first_line]
        
        if 'def ' in first_line:
            func_name = re.search(r'def\s+(\w+)', first_line)
            if func_name:
                summary_parts.append(f"// Function: {func_name.group(1)}")
        
        try_count = func_code.count('try:')
        if try_count > 0:
            summary_parts.append(f"// Has {try_count} try-except block(s)")
        
        if_count = func_code.count('if ')
        if if_count > 0:
            summary_parts.append(f"// Has {if_count} conditional(s)")
        
        return '\n'.join(summary_parts)
    
    def build_class_summary(self, class_code: str) -> str:
        """为类生成摘要"""
        class_match = re.search(r'class\s+(\w+)', class_code)
        class_name = class_match.group(1) if class_match else 'UnnamedClass'
        
        methods = re.findall(r'def\s+(\w+)', class_code)
        attributes = re.findall(r'self\.(\w+)\s*=', class_code)
        
        lines = class_code.split('\n')
        line_count = len(lines)
        
        summary = [
            f"class {class_name}:",
            f"// Lines: {line_count}",
            f"// Methods: {', '.join(methods[:5])}" if methods else "// Methods: None",
            f"// Attributes: {', '.join(set(attributes[:5]))}" if attributes else "// Attributes: None"
        ]
        
        return '\n'.join(summary)
    
    def build_chunk_summary(self, chunk: Chunk) -> str:
        """为分块生成摘要"""
        semantic_type = chunk.metadata.semantic_type
        
        if semantic_type == 'class':
            return self.build_class_summary(chunk.content)
        elif semantic_type == 'function':
            return self.build_function_summary(chunk.content)
        else:
            lines = chunk.content.split('\n')
            return '\n'.join(lines[:3]) + f"\n// ... {len(lines)} lines total"


class RecursiveRetriever:
    """
    递归检索器 - 实现 Agentic RAG
    
    当检索到需要外部信息的块时，
    自动生成新查询并执行二次检索。
    """
    
    REF_PATTERN = re.compile(
        r'(?:参见|refer to|见|see|如前|如上|如下|as above|as below)',
        re.IGNORECASE
    )
    
    SECTION_PATTERN = re.compile(
        r'(?:第\s*(\d+(?:\.\d+)*)\s*[节章段]|(?:section|chapter)\s*(\d+(?:\.\d+)*))',
        re.IGNORECASE
    )
    
    def __init__(
        self,
        vector_store: Any,
        llm_service: Any = None,
        max_recursion_depth: int = 3
    ):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.max_recursion_depth = max_recursion_depth
        self._retrieval_cache = {}
    
    def retrieve(
        self,
        query: str,
        initial_chunks: List[Chunk],
        depth: int = 0
    ) -> List[Chunk]:
        """
        递归检索
        
        Args:
            query: 原始查询
            initial_chunks: 初始检索结果
            depth: 当前递归深度
            
        Returns:
            合并后的检索结果
        """
        if depth >= self.max_recursion_depth:
            return initial_chunks
        
        cache_key = f"{query}:{depth}"
        if cache_key in self._retrieval_cache:
            return self._retrieval_cache[cache_key]
        
        all_chunks = list(initial_chunks)
        
        for chunk in initial_chunks:
            additional_chunks = self._handle_references(chunk, query, depth)
            all_chunks.extend(additional_chunks)
        
        self._retrieval_cache[cache_key] = all_chunks
        return all_chunks
    
    def _handle_references(
        self,
        chunk: Chunk,
        original_query: str,
        depth: int
    ) -> List[Chunk]:
        """处理 chunk 中的引用"""
        additional_chunks = []
        
        if self.REF_PATTERN.search(chunk.content):
            section_ref = self.SECTION_PATTERN.search(chunk.content)
            
            if section_ref:
                ref_query = self._generate_reference_query(chunk, section_ref, original_query)
                
                if self.llm_service:
                    refined_query = self._refine_query_with_llm(chunk, ref_query, original_query)
                else:
                    refined_query = ref_query
                
                referenced_chunks = self.vector_store.similarity_search(refined_query)
                additional_chunks.extend(referenced_chunks)
        
        return additional_chunks
    
    def _generate_reference_query(
        self,
        chunk: Chunk,
        section_ref: re.Match,
        original_query: str
    ) -> str:
        """生成引用查询"""
        section = section_ref.group(1) or section_ref.group(2)
        
        context = f"关于 {section} 节的内容"
        
        if chunk.metadata.class_name:
            context += f"，在 {chunk.metadata.class_name} 类中"
        
        if chunk.metadata.function_name:
            context += f"的 {chunk.metadata.function_name} 函数"
        
        return f"{context}; {original_query}"
    
    def _refine_query_with_llm(
        self,
        chunk: Chunk,
        ref_query: str,
        original_query: str
    ) -> str:
        """使用 LLM 精炼查询"""
        prompt = f"""
给定以下代码块和引用，需要生成一个精确的二次检索查询。

代码块上下文:
- 类: {chunk.metadata.class_name or 'N/A'}
- 函数: {chunk.metadata.function_name or 'N/A'}
- 内容: {chunk.content[:200]}...

原始查询: {original_query}
引用查询: {ref_query}

请生成一个简洁的二次查询，用于检索引用的具体内容。
"""
        
        try:
            response = self.llm_service.complete(prompt)
            return response.strip()
        except Exception:
            return ref_query
    
    def clear_cache(self):
        """清除检索缓存"""
        self._retrieval_cache.clear()


class SmartChunker:
    """智能分块器主类 - 增强版：支持语义保留分块
    
    支持的功能：
    - 基础分块：多种文件类型的智能分块
    - 上下文增强：Contextual Retrieval 上下文增强嵌入
    - 原子分组：AST 原子分块策略
    - 摘要索引：代码逻辑摘要生成
    - 递归检索：Agentic RAG 多步检索
    """
    
    STRATEGY_MAP = {
        FileType.PYTHON: SemanticPreservingStrategy,
        FileType.MARKDOWN: MarkdownChunkingStrategy,
        FileType.JAVASCRIPT: JavaScriptChunkingStrategy,
        FileType.TYPESCRIPT: JavaScriptChunkingStrategy,
        FileType.JSON: RecursiveChunkingStrategy,
        FileType.YAML: RecursiveChunkingStrategy,
        FileType.SQL: RecursiveChunkingStrategy,
    }
    
    BINARY_FILE_TYPES = {FileType.BINARY}
    
    SKIP_FILE_TYPES = {FileType.PROTO}
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        max_file_size: int = 100 * 1024 * 1024,
        llm_service: Any = None,
        enable_contextual_enrichment: bool = True,
        enable_atomic_chunking: bool = True,
        enable_rate_limiting: bool = False,
        rate_limit_rpm: int = 100,
        rate_limit_tpm: int = 10000,
        enable_batch_processing: bool = False,
        batch_max_concurrent: int = 5,
        batch_size: int = 10,
        batch_timeout: float = 30.0,
        enable_embedding_cache: bool = False,
        embedding_cache_size: int = 10000,
        embedding_cache_file: str = None,
        enable_strategy_cache: bool = True
    ):
        """
        初始化智能分块器

        Args:
            config: 配置字典
            max_file_size: 最大处理文件大小 (默认 100MB)
            llm_service: LLM 服务实例 (用于上下文增强)
            enable_contextual_enrichment: 启用上下文增强
            enable_atomic_chunking: 启用原子分块
            enable_rate_limiting: 启用令牌桶限流
            rate_limit_rpm: 每分钟请求数限制
            rate_limit_tpm: 每分钟令牌数限制
            enable_batch_processing: 启用批量处理队列
            batch_max_concurrent: 批量处理最大并发数
            batch_size: 每批处理的任务数
            batch_timeout: 任务超时时间
            enable_embedding_cache: 启用嵌入缓存
            embedding_cache_size: 嵌入缓存最大容量
            embedding_cache_file: 嵌入缓存文件路径
            enable_strategy_cache: 启用策略缓存
        """
        self.config = config or {}
        self.max_file_size = max_file_size
        self.llm_service = llm_service
        self.enable_contextual_enrichment = enable_contextual_enrichment
        self.enable_atomic_chunking = enable_atomic_chunking

        self._strategies: Dict[FileType, ChunkingStrategy] = {}
        self._init_strategies()

        self.large_file_processor = LargeFileProcessor(max_file_size)

        if enable_contextual_enrichment:
            self.contextual_enricher = ContextualRetrievalEnricher(llm_service=llm_service)
        else:
            self.contextual_enricher = None

        if enable_atomic_chunking:
            self.summary_builder = SummaryBuilder(llm_service)
        else:
            self.summary_builder = None

        self._init_performance_optimizations(
            enable_rate_limiting=enable_rate_limiting,
            rate_limit_rpm=rate_limit_rpm,
            rate_limit_tpm=rate_limit_tpm,
            enable_batch_processing=enable_batch_processing,
            batch_max_concurrent=batch_max_concurrent,
            batch_size=batch_size,
            batch_timeout=batch_timeout,
            enable_embedding_cache=enable_embedding_cache,
            embedding_cache_size=embedding_cache_size,
            embedding_cache_file=embedding_cache_file,
            enable_strategy_cache=enable_strategy_cache
        )

    def _init_performance_optimizations(
        self,
        enable_rate_limiting: bool,
        rate_limit_rpm: int,
        rate_limit_tpm: int,
        enable_batch_processing: bool,
        batch_max_concurrent: int,
        batch_size: int,
        batch_timeout: float,
        enable_embedding_cache: bool,
        embedding_cache_size: int,
        embedding_cache_file: str,
        enable_strategy_cache: bool
    ):
        """初始化性能优化组件"""
        if enable_rate_limiting:
            self.rate_limiter = RateLimiter(
                rpm=rate_limit_rpm,
                tpm=rate_limit_tpm
            )
        else:
            self.rate_limiter = None

        if enable_batch_processing:
            self.batch_processor = BatchProcessor(
                process_fn=self._process_chunk_task,
                max_concurrent=batch_max_concurrent,
                batch_size=batch_size,
                timeout=batch_timeout
            )
        else:
            self.batch_processor = None

        if enable_embedding_cache:
            self.embedding_cache = EmbeddingCache(
                max_size=embedding_cache_size,
                cache_file=embedding_cache_file
            )
        else:
            self.embedding_cache = None

        if enable_strategy_cache:
            self.strategy_cache = StrategyCache(self._strategies)
        else:
            self.strategy_cache = None

    def _process_chunk_task(self, task_data: Dict[str, Any]) -> Any:
        """处理分块任务的回调函数"""
        pass

    def set_rate_limiter(self, rpm: int = 100, tpm: int = 10000) -> None:
        """设置或更新令牌桶限流器"""
        self.rate_limiter = RateLimiter(rpm=rpm, tpm=tpm)
        self.config['rate_limit_rpm'] = rpm
        self.config['rate_limit_tpm'] = tpm

    async def acquire_rate_limit(self, tokens: int = 1) -> float:
        """获取令牌（用于异步限流控制）

        Returns:
            等待时间（秒）
        """
        if self.rate_limiter:
            return await self.rate_limiter.acquire(tokens)
        return 0.0

    async def add_batch_task(self, task_id: str, task_data: Any) -> None:
        """添加批量处理任务"""
        if self.batch_processor:
            await self.batch_processor.add_task(task_id, task_data)

    async def process_batch(self) -> Dict[str, Any]:
        """处理批量任务队列"""
        if self.batch_processor:
            return await self.batch_processor.process_batch()
        return {}

    def get_embedding(self, content: str, compute_fn: callable = None) -> Optional[List[float]]:
        """获取嵌入（使用缓存）

        Args:
            content: 文本内容
            compute_fn: 嵌入计算函数（如果未命中缓存）

        Returns:
            嵌入向量
        """
        if self.embedding_cache:
            if compute_fn:
                return self.embedding_cache.get_or_compute(content, compute_fn)
            return self.embedding_cache.get(content)
        return None

    def save_embedding_cache(self) -> None:
        """保存嵌入缓存到磁盘"""
        if self.embedding_cache:
            self.embedding_cache.save_cache()

    def clear_embedding_cache(self) -> None:
        """清空嵌入缓存"""
        if self.embedding_cache:
            self.embedding_cache.clear()

    def get_strategy(self, file_path: str, content: bytes) -> ChunkingStrategy:
        """获取分块策略（使用缓存）

        Args:
            file_path: 文件路径
            content: 文件内容

        Returns:
            分块策略
        """
        if self.strategy_cache:
            return self.strategy_cache.get_strategy(file_path, content)
        file_type = self.detect_file_type(file_path, content)
        if file_type not in self._strategies:
            file_type = FileType.TEXT
        return self._strategies[file_type]
    
    def _init_strategies(self):
        """初始化分块策略"""
        common_config = {
            'chunk_size': self.config.get('chunk_size', 1000),
            'min_chunk_size': self.config.get('min_chunk_size', 100),
        }
        
        for file_type, strategy_class in self.STRATEGY_MAP.items():
            if file_type == FileType.PYTHON:
                base_strategy = strategy_class(
                    chunk_size=common_config['chunk_size'] * 1.2,
                    min_chunk_size=common_config['min_chunk_size']
                )
                if self.enable_atomic_chunking:
                    self._strategies[file_type] = AtomicChunkingStrategy(
                        base_strategy=base_strategy,
                        max_atomic_size=int(common_config['chunk_size'] * 2)
                    )
                else:
                    self._strategies[file_type] = base_strategy
            elif file_type == FileType.MARKDOWN:
                self._strategies[file_type] = strategy_class(
                    chunk_size=common_config['chunk_size'] * 1.5,
                    min_chunk_size=common_config['min_chunk_size'] * 2
                )
            else:
                self._strategies[file_type] = strategy_class(**common_config)
        
        self._strategies[FileType.TEXT] = BaseChunkingStrategy(
            chunk_size=common_config['chunk_size'],
            min_chunk_size=common_config['min_chunk_size'],
            overlap=self.config.get('overlap', 200)
        )
        self._strategies[FileType.UNKNOWN] = BaseChunkingStrategy(
            chunk_size=common_config['chunk_size'],
            min_chunk_size=common_config['min_chunk_size'],
            overlap=self.config.get('overlap', 200)
        )
    
    def detect_file_type(
        self, 
        file_path: str, 
        content: bytes = None
    ) -> FileType:
        """
        检测文件类型
        
        Args:
            file_path: 文件路径
            content: 文件内容 (可选)
            
        Returns:
            文件类型
        """
        ext = Path(file_path).suffix.lower()
        
        type_map = {
            '.py': FileType.PYTHON,
            '.md': FileType.MARKDOWN,
            '.json': FileType.JSON,
            '.yaml': FileType.YAML,
            '.yml': FileType.YAML,
            '.sql': FileType.SQL,
            '.js': FileType.JAVASCRIPT,
            '.ts': FileType.TYPESCRIPT,
            '.jsx': FileType.JAVASCRIPT,
            '.tsx': FileType.TYPESCRIPT,
            '.proto': FileType.PROTO,
        }
        
        if ext in type_map:
            return type_map[ext]
        
        if content is not None:
            if BinaryDetector.is_binary(content):
                return FileType.BINARY
            
            if content.startswith(b'#!'):
                if b'python' in content or b'python3' in content:
                    return FileType.PYTHON
                elif b'node' in content or b'javascript' in content:
                    return FileType.JAVASCRIPT
        
        return FileType.UNKNOWN
    
    def chunk_file(
        self, 
        file_path: str,
        content: bytes = None,
        encoding: str = None
    ) -> List[Chunk]:
        """
        对文件进行分块
        
        Args:
            file_path: 文件路径
            content: 文件内容 (可选，如果未提供则从文件读取)
            encoding: 编码 (可选)
            
        Returns:
            分块列表
        """
        if content is None:
            if not Path(file_path).exists():
                logger.error(f"文件不存在: {file_path}")
                return []
            with open(file_path, 'rb') as f:
                content = f.read()
        
        try:
            file_type = self.detect_file_type(file_path, content)
            
            if file_type in self.SKIP_FILE_TYPES:
                logger.info(f"跳过不支持的文件类型: {file_path}")
                return []
            
            if file_type == FileType.BINARY:
                logger.info(f"跳过二进制文件: {file_path}")
                return []
            
            decoded_content, used_encoding = EncodingDetector.decode_content(content)
            
            if encoding and encoding != used_encoding:
                logger.debug(f"使用指定编码 {encoding} 重新解码 (检测到 {used_encoding})")
                try:
                    decoded_content = content.decode(encoding, errors='replace')
                except Exception as e:
                    logger.warning(f"使用指定编码解码失败: {e}")
            
            if self.large_file_processor.should_use_streaming(file_path, content):
                return self._chunk_large_file(file_path, decoded_content, used_encoding)
            
            return self._chunk_content(decoded_content, file_path, file_type, used_encoding)
            
        except Exception as e:
            logger.error(f"分块失败 {file_path}: {e}")
            return []
    
    def _chunk_large_file(
        self, 
        file_path: str,
        content: str,
        encoding: str
    ) -> List[Chunk]:
        """分块大文件"""
        chunks = []
        chunk_index = 0
        
        for chunk_content, start_line, end_line in self.large_file_processor.stream_chunk(
            file_path,
            chunk_size=self.config.get('chunk_size', 1000),
            overlap=self.config.get('overlap', 200)
        ):
            content_hash = hashlib.md5(chunk_content.encode()).hexdigest()
            chunk_id = f"{Path(file_path).stem}_{content_hash[:8]}_{chunk_index}"
            
            metadata = ChunkMetadata(
                file_path=file_path,
                file_type=self.detect_file_type(file_path),
                start_line=start_line,
                end_line=end_line,
                char_count=len(chunk_content),
                content_hash=content_hash,
                encoding=encoding,
                is_truncated=False
            )
            
            chunks.append(Chunk(id=chunk_id, content=chunk_content, metadata=metadata))
            chunk_index += 1
        
        return chunks
    
    def _chunk_content(
        self, 
        content: str, 
        file_path: str,
        file_type: FileType,
        encoding: str
    ) -> List[Chunk]:
        """分块内容"""
        strategy = self._strategies.get(file_type)
        
        if strategy is None:
            strategy = self._strategies[FileType.UNKNOWN]
        
        chunks = strategy.chunk(content, file_path)
        
        for chunk in chunks:
            chunk.metadata.encoding = encoding
        
        return chunks
    
    def chunk_bytes(
        self, 
        content: bytes,
        file_path: str = "memory_content"
    ) -> List[Chunk]:
        """
        对字节内容进行分块
        
        Args:
            content: 字节内容
            file_path: 文件路径 (用于检测类型)
            
        Returns:
            分块列表
        """
        return self.chunk_file(file_path, content)
    
    def chunk_text(
        self, 
        content: str,
        file_path: str = "memory_content",
        file_type: FileType = None
    ) -> List[Chunk]:
        """
        对文本内容进行分块
        
        Args:
            content: 文本内容
            file_path: 文件路径 (用于检测类型)
            file_type: 文件类型 (可选，自动检测)
            
        Returns:
            分块列表
        """
        if file_type is None:
            file_type = self.detect_file_type(file_path, content.encode('utf-8'))
        
        if file_type not in self._strategies:
            file_type = FileType.TEXT
        
        strategy = self._strategies[file_type]
        chunks = strategy.chunk(content, file_path)
        
        for chunk in chunks:
            chunk.metadata.encoding = 'utf-8'
        
        return chunks
    
    def chunk_with_context(
        self,
        content: str,
        file_path: str = "memory_content"
    ) -> List[Chunk]:
        """
        分块并应用上下文增强 (Contextual Retrieval)
        
        Args:
            content: 文本内容
            file_path: 文件路径
            
        Returns:
            增强后的分块列表
        """
        chunks = self.chunk_text(content, file_path)
        
        if self.contextual_enricher:
            chunks = self.contextual_enricher.enrich_chunks(chunks, content)
        
        return chunks
    
    def chunk_with_summaries(
        self,
        content: str,
        file_path: str = "memory_content"
    ) -> Tuple[List[Chunk], List[Dict[str, str]]]:
        """
        分块并生成摘要索引
        
        Args:
            content: 文本内容
            file_path: 文件路径
            
        Returns:
            Tuple[分块列表, 摘要列表]
        """
        chunks = self.chunk_text(content, file_path)
        summaries = []
        
        if self.summary_builder:
            for chunk in chunks:
                summary = self.summary_builder.build_chunk_summary(chunk)
                summaries.append({
                    'chunk_id': chunk.id,
                    'summary': summary,
                    'semantic_type': chunk.metadata.semantic_type or 'unknown'
                })
        
        return chunks, summaries
    
    def chunk_with_all_enhancements(
        self,
        content: str,
        file_path: str = "memory_content"
    ) -> Tuple[List[Chunk], List[Dict[str, str]]]:
        """
        分块并应用所有增强功能
        
        Args:
            content: 文本内容
            file_path: 文件路径
            
        Returns:
            Tuple[增强后的分块列表, 摘要列表]
        """
        chunks = self.chunk_text(content, file_path)
        summaries = []
        
        if self.contextual_enricher:
            chunks = self.contextual_enricher.enrich_chunks(chunks, content)
        
        if self.summary_builder:
            for chunk in chunks:
                summary = self.summary_builder.build_chunk_summary(chunk)
                summaries.append({
                    'chunk_id': chunk.id,
                    'summary': summary,
                    'semantic_type': chunk.metadata.semantic_type or 'unknown',
                    'contextual_summary': chunk.metadata.ghost_context.get('contextual_summary') if chunk.metadata.ghost_context else None
                })
        
        return chunks, summaries


class RateLimiter:
    """令牌桶限流器 - 控制并发访问速率"""

    def __init__(self, rpm: int = 100, tpm: int = 10000, max_burst: int = 10):
        self.rpm = rpm
        self.tpm = tpm
        self.max_burst = max_burst
        self.tokens = max_burst
        self.last_time = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """获取令牌

        Args:
            tokens: 需要的令牌数

        Returns:
            等待时间（秒）
        """
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            self.tokens = min(self.max_burst, self.tokens + elapsed * (self.rpm / 60))
            self.last_time = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            wait_time = (tokens - self.tokens) * (60 / self.rpm)
            self.tokens = 0
            return wait_time

    def get_tokens_remaining(self) -> float:
        """获取剩余令牌数"""
        return self.tokens


class BatchProcessor:
    """批量处理队列 - 断点续传与批量并发控制"""

    def __init__(
        self,
        process_fn: callable,
        max_concurrent: int = 5,
        batch_size: int = 10,
        timeout: float = 30.0
    ):
        self.process_fn = process_fn
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = asyncio.Queue()
        self.results = {}
        self._task_map = {}

    async def add_task(self, task_id: str, task_data: Any) -> None:
        """添加任务到队列"""
        await self.queue.put((task_id, task_data))
        self._task_map[task_id] = 'pending'

    async def process_batch(self) -> Dict[str, Any]:
        """处理一批任务"""
        tasks = []
        batch_count = 0

        while not self.queue.empty() and batch_count < self.batch_size:
            task_id, task_data = await self.queue.get()
            task = asyncio.create_task(self._run_with_semaphore(task_id, task_data))
            tasks.append(task)
            batch_count += 1

        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=self.timeout)
            for task in pending:
                task.cancel()

        return self.results

    async def _run_with_semaphore(self, task_id: str, task_data: Any) -> None:
        async with self.semaphore:
            try:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self.process_fn, task_data
                    ),
                    timeout=self.timeout
                )
                self.results[task_id] = {'status': 'success', 'result': result}
            except asyncio.TimeoutError:
                self.results[task_id] = {'status': 'timeout', 'error': '任务超时'}
            except Exception as e:
                self.results[task_id] = {'status': 'error', 'error': str(e)}
            finally:
                self._task_map[task_id] = self.results[task_id].get('status', 'unknown')


class EmbeddingCache:
    """嵌入缓存 - 基于内容哈希去重"""

    def __init__(self, max_size: int = 10000, cache_file: str = None):
        self._cache: Dict[str, List[float]] = {}
        self._content_to_hash: Dict[str, str] = {}
        self.max_size = max_size
        self.cache_file = cache_file
        self.lock = Lock()
        if cache_file:
            self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_file and Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self._cache = data.get('embeddings', {})
                    self._content_to_hash = data.get('content_map', {})
            except Exception as e:
                logger.warning(f"加载嵌入缓存失败: {e}")

    def save_cache(self) -> None:
        if self.cache_file:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump({
                        'embeddings': self._cache,
                        'content_map': self._content_to_hash
                    }, f)
            except Exception as e:
                logger.warning(f"保存嵌入缓存失败: {e}")

    def get(self, content: str) -> Optional[List[float]]:
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        with self.lock:
            return self._cache.get(content_hash)

    def get_or_compute(self, content: str, compute_fn: callable) -> List[float]:
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        with self.lock:
            if content_hash in self._cache:
                return self._cache[content_hash]

            embedding = compute_fn(content)
            if len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._content_to_hash[oldest_key]

            self._cache[content_hash] = embedding
            self._content_to_hash[content_hash] = content[:100]
            return embedding

    def clear(self) -> None:
        with self.lock:
            self._cache.clear()
            self._content_to_hash.clear()


class StrategyCache:
    """策略缓存 - 避免重复检测文件类型"""

    def __init__(self, strategies: Dict[FileType, ChunkingStrategy]):
        self._strategies = strategies
        self._cache: Dict[str, ChunkingStrategy] = {}
        self._type_cache: Dict[str, FileType] = {}

    def get_strategy(self, file_path: str, content: bytes) -> ChunkingStrategy:
        content_size = len(content)
        cache_key = f"{content_size}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        file_type = None
        ext = Path(file_path).suffix.lower()

        type_cache_key = f"{ext}:{content_size}"
        if type_cache_key in self._type_cache:
            file_type = self._type_cache[type_cache_key]
        else:
            file_type = self._detect_file_type_with_heuristics(file_path, content)
            self._type_cache[type_cache_key] = file_type

        if file_type not in self._strategies:
            file_type = FileType.TEXT

        if file_type == FileType.PYTHON and content_size > 50000:
            strategy = self._strategies.get(FileType.PYTHON)
        else:
            strategy = self._strategies.get(file_type, self._strategies[FileType.TEXT])

        self._cache[cache_key] = strategy
        return strategy

    def _detect_file_type_with_heuristics(self, file_path: str, content: bytes) -> FileType:
        ext = Path(file_path).suffix.lower()

        extension_mapping = {
            '.py': FileType.PYTHON,
            '.md': FileType.MARKDOWN,
            '.json': FileType.JSON,
            '.yaml': FileType.YAML,
            '.yml': FileType.YAML,
            '.sql': FileType.SQL,
            '.proto': FileType.PROTO,
            '.js': FileType.JAVASCRIPT,
            '.ts': FileType.TYPESCRIPT,
            '.html': FileType.HTML,
            '.css': FileType.CSS,
        }

        if ext in extension_mapping:
            return extension_mapping[ext]

        try:
            text_content = content.decode('utf-8', errors='ignore')
            if text_content.startswith('<?xml') or '<html' in text_content[:100].lower():
                return FileType.HTML
            if text_content.startswith('{') or text_content.startswith('['):
                return FileType.JSON
            if 'def ' in text_content or 'class ' in text_content:
                return FileType.PYTHON
        except Exception:
            pass

        if BinaryDetector.is_binary(content):
            return FileType.BINARY

        return FileType.TEXT


def create_smart_chunker(config: Dict[str, Any] = None) -> SmartChunker:
    """
    创建智能分块器的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        SmartChunker 实例
    """
    return SmartChunker(config=config)
