"""
父文档检索 (Parent Document Retrieval) 服务

架构原理:
- 索引时: 将代码切成小块进行向量化，确保相似度匹配精准
- 存储时: 记录小块所属的父文档信息
- 检索时: 匹配到小块，但返回给 LLM 的是完整的父文档

优势:
- 解决边界切割问题 (返回完整上下文)
- 解决格式破坏问题 (父文档保持格式完整)
- 解决重复内容问题 (无需物理复制)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import hashlib
import json


class ParentDocumentType:
    """父文档类型"""
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    SECTION = "section"
    TABLE = "table"


@dataclass
class ParentDocument:
    """
    父文档 - 包含完整上下文的大块内容
    """
    id: str
    content: str
    doc_type: str
    file_path: str
    title: Optional[str] = None
    start_line: int = 1
    end_line: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.end_line == 0:
            self.end_line = self.content.count('\n') + 1
    
    @property
    def char_count(self) -> int:
        return len(self.content)
    
    @property
    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()


@dataclass  
class ChildChunk:
    """
    子分块 - 用于精确向量匹配的小块
    """
    id: str
    content: str
    parent_doc_id: str
    file_path: str
    start_line: int
    end_line: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()


class ParentDocumentStore:
    """
    父文档存储管理器
    
    维护父子文档关系，支持 Small-to-Big Retrieval:
    - 存储父文档 (完整上下文)
    - 存储子分块 (精确匹配)
    - 提供检索和组装功能
    """
    
    def __init__(self):
        self._parent_docs: Dict[str, ParentDocument] = {}
        self._child_chunks: Dict[str, ChildChunk] = {}
        self._file_to_parent: Dict[str, Set[str]] = {}
        self._parent_to_children: Dict[str, List[str]] = {}
    
    def add_parent_document(
        self,
        doc_id: str,
        content: str,
        doc_type: str,
        file_path: str,
        title: Optional[str] = None,
        start_line: int = 1,
        end_line: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParentDocument:
        """
        添加父文档
        
        Args:
            doc_id: 文档ID
            content: 完整内容
            doc_type: 文档类型 (file/function/class/section/table)
            file_path: 文件路径
            title: 标题 (如函数名、类名)
            start_line: 起始行
            end_line: 结束行
            metadata: 额外元数据
            
        Returns:
            创建的父文档
        """
        parent = ParentDocument(
            id=doc_id,
            content=content,
            doc_type=doc_type,
            file_path=file_path,
            title=title,
            start_line=start_line,
            end_line=end_line,
            metadata=metadata or {}
        )
        
        self._parent_docs[doc_id] = parent
        self._parent_to_children[doc_id] = []
        
        if file_path not in self._file_to_parent:
            self._file_to_parent[file_path] = set()
        self._file_to_parent[file_path].add(doc_id)
        
        return parent
    
    def add_child_chunk(
        self,
        chunk_id: str,
        content: str,
        parent_doc_id: str,
        file_path: str,
        start_line: int,
        end_line: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChildChunk:
        """
        添加子分块
        
        Args:
            chunk_id: 分块ID
            content: 分块内容
            parent_doc_id: 父文档ID
            file_path: 文件路径
            start_line: 起始行
            end_line: 结束行
            metadata: 额外元数据
            
        Returns:
            创建的子分块
        """
        chunk = ChildChunk(
            id=chunk_id,
            content=content,
            parent_doc_id=parent_doc_id,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            metadata=metadata or {}
        )
        
        self._child_chunks[chunk_id] = chunk
        
        if parent_doc_id in self._parent_to_children:
            self._parent_to_children[parent_doc_id].append(chunk_id)
        
        return chunk
    
    def get_parent_document(self, doc_id: str) -> Optional[ParentDocument]:
        """获取父文档"""
        return self._parent_docs.get(doc_id)
    
    def get_parent_documents_by_file(self, file_path: str) -> List[ParentDocument]:
        """获取文件的所有父文档"""
        doc_ids = self._file_to_parent.get(file_path, set())
        return [self._parent_docs[doc_id] for doc_id in doc_ids if doc_id in self._parent_docs]
    
    def get_parent_for_chunk(self, chunk_id: str) -> Optional[ParentDocument]:
        """获取分块对应的父文档"""
        chunk = self._child_chunks.get(chunk_id)
        if chunk:
            return self._parent_docs.get(chunk.parent_doc_id)
        return None
    
    def get_parent_documents_by_ids(self, doc_ids: List[str]) -> List[ParentDocument]:
        """根据ID列表获取父文档"""
        return [self._parent_docs[doc_id] for doc_id in doc_ids if doc_id in self._parent_docs]
    
    def retrieve_with_parents(
        self,
        matched_chunk_ids: List[str],
        max_parent_docs: int = 5
    ) -> List[ParentDocument]:
        """
        检索父文档
        
        给定匹配的分块ID列表，返回对应的父文档
        
        Args:
            matched_chunk_ids: 匹配的分块ID列表
            max_parent_docs: 最大返回父文档数量
            
        Returns:
            父文档列表 (去重)
        """
        parent_ids = set()
        for chunk_id in matched_chunk_ids:
            chunk = self._child_chunks.get(chunk_id)
            if chunk:
                parent_ids.add(chunk.parent_doc_id)
        
        parent_docs = [
            self._parent_docs[pid] 
            for pid in parent_ids 
            if pid in self._parent_docs
        ]
        
        parent_docs.sort(key=lambda x: x.char_count, reverse=True)
        
        return parent_docs[:max_parent_docs]
    
    def build_context_for_llm(
        self,
        matched_chunk_ids: List[str],
        include_ghost_context: bool = True
    ) -> str:
        """
        为 LLM 构建上下文
        
        策略:
        1. 获取匹配的父文档
        2. 如果启用 ghost context，将骨架信息作为提示
        3. 拼接完整的父文档内容
        
        Args:
            matched_chunk_ids: 匹配的分块ID
            include_ghost_context: 是否包含骨架上下文
            
        Returns:
            LLM 可用的完整上下文
        """
        parent_docs = self.retrieve_with_parents(matched_chunk_ids)
        
        if not parent_docs:
            return ""
        
        context_parts = []
        
        for parent in parent_docs:
            context_parts.append(f"=== {parent.doc_type.upper()}: {parent.title or parent.id} ===")
            
            if include_ghost_context and parent.metadata:
                ghost = parent.metadata.get("ghost_context", {})
                if ghost:
                    context_parts.append(f"[骨架上下文]")
                    if ghost.get("imports"):
                        context_parts.append(f"导入: {ghost['imports']}")
                    if ghost.get("class_definition"):
                        context_parts.append(f"类定义: {ghost['class_definition']}")
                    if ghost.get("function_signature"):
                        context_parts.append(f"函数签名: {ghost['function_signature']}")
                    context_parts.append("")
            
            context_parts.append(parent.content)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def stats(self) -> Dict[str, Any]:
        """获取存储统计"""
        return {
            "parent_documents": len(self._parent_docs),
            "child_chunks": len(self._child_chunks),
            "files_indexed": len(self._file_to_parent),
            "avg_children_per_parent": (
                sum(len(c) for c in self._parent_to_children.values()) / 
                len(self._parent_to_children) if self._parent_to_children else 0
            )
        }
    
    def clear(self):
        """清空存储"""
        self._parent_docs.clear()
        self._child_chunks.clear()
        self._file_to_parent.clear()
        self._parent_to_children.clear()


class ParentDocumentRetriever:
    """
    父文档检索器
    
    封装 Small-to-Big Retrieval 逻辑:
    - 创建父子文档关系
    - 执行向量检索
    - 组装完整上下文
    """
    
    def __init__(self, store: Optional[ParentDocumentStore] = None):
        """
        初始化检索器
        
        Args:
            store: 父文档存储实例
        """
        self.store = store or ParentDocumentStore()
    
    def index_python_file(
        self,
        file_path: str,
        content: str,
        chunks: List['Chunk'],
        class_context: Optional[Dict[str, str]] = None
    ) -> List[ParentDocument]:
        """
        索引 Python 文件
        
        为每个顶级类/函数创建父文档
        
        Args:
            file_path: 文件路径
            content: 完整文件内容
            chunks: SmartChunker 生成的分块
            class_context: 类上下文映射
            
        Returns:
            创建的父文档列表
        """
        parents = []
        
        parent_id = f"{Path(file_path).stem}_file"
        parent = self.store.add_parent_document(
            doc_id=parent_id,
            content=content,
            doc_type=ParentDocumentType.FILE,
            file_path=file_path,
            title=Path(file_path).name,
            metadata={
                "ghost_context": {
                    "imports": self._extract_imports(content),
                    "classes": list(class_context.keys()) if class_context else []
                }
            }
        )
        parents.append(parent)
        
        for chunk in chunks:
            chunk_id = chunk.id
            self.store.add_child_chunk(
                chunk_id=chunk_id,
                content=chunk.content,
                parent_doc_id=parent_id,
                file_path=file_path,
                start_line=chunk.metadata.start_line,
                end_line=chunk.metadata.end_line,
                metadata={
                    "class_name": chunk.metadata.class_name,
                    "function_name": chunk.metadata.function_name
                }
            )
        
        return parents
    
    def _extract_imports(self, content: str) -> str:
        """提取导入语句"""
        lines = content.split('\n')
        imports = []
        for line in lines[:30]:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                imports.append(stripped)
            elif stripped and not stripped.startswith('#'):
                break
        return '\n'.join(imports) if imports else ""
    
    def retrieve(
        self,
        chunk_ids: List[str],
        include_ghost_context: bool = True
    ) -> str:
        """
        检索并组装上下文
        
        Args:
            chunk_ids: 匹配的分块ID
            include_ghost_context: 是否包含骨架上下文
            
        Returns:
            组装好的上下文
        """
        return self.store.build_context_for_llm(chunk_ids, include_ghost_context)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计"""
        return self.store.stats()
