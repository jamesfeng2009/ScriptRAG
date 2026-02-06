"""Enhanced Parent Document Retriever - 增强版父子文档检索

功能：
1. 连续块合并：检测连续命中的小块，合并为更大的父文档
2. 语义边界检测：基于代码语法/文档结构合并
3. 上下文窗口管理：确保不超出 LLM context limit
4. 智能去重：基于文件+内容相似度去重

解决的问题：
- 检索到小块但上下文丢失
- 同一文件的多个小块重复
- 合并后超出 context window
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from .parent_document_retriever import (
    ParentDocumentStore,
    ParentDocument,
    ChildChunk,
    ParentDocumentType
)

from ..retrieval.strategies import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class MergedContext:
    """合并后的上下文"""
    id: str
    file_path: str
    content: str
    doc_type: str
    title: Optional[str]
    start_line: int
    end_line: int
    source_chunk_ids: List[str]
    char_count: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeConfig:
    """合并配置"""
    max_context_chars: int = 4000
    min_merge_distance: int = 3
    merge_strategy: str = "contiguous"
    max_chunks_per_file: int = 5
    preserve_format: bool = True


class EnhancedParentDocumentRetriever:
    """
    增强版父子文档检索器
    
    增强功能：
    - 连续块合并
    - 语义边界合并
    - 上下文窗口管理
    - 智能去重
    """
    
    def __init__(
        self,
        store: Optional[ParentDocumentStore] = None,
        config: Optional[MergeConfig] = None
    ):
        self.store = store or ParentDocumentStore()
        self.config = config or MergeConfig()
    
    def retrieve_small_to_big(
        self,
        query: str,
        chunk_results: List[RetrievalResult],
        max_context_chars: Optional[int] = None,
        merge_strategy: str = "auto"
    ) -> List[MergedContext]:
        """
        Small-to-Big 检索
        
        策略选择：
        - "contiguous": 合并连续的代码块
        - "semantic": 基于语义边界合并
        - "file": 按文件合并
        - "auto": 自动选择最佳策略
        
        Args:
            query: 查询文本
            chunk_results: 检索到的块结果
            max_context_chars: 最大字符数
            merge_strategy: 合并策略
            
        Returns:
            合并后的上下文列表
        """
        if not chunk_results:
            return []
        
        max_chars = max_context_chars or self.config.max_context_chars
        
        # 按文件分组
        file_groups = self._group_by_file(chunk_results)
        
        merged_contexts = []
        
        for file_path, chunks in file_groups.items():
            if merge_strategy == "auto":
                strategy = self._select_strategy(chunks)
            else:
                strategy = merge_strategy
            
            if strategy == "contiguous":
                file_contexts = self._merge_contiguous_chunks(
                    chunks, file_path, max_chars
                )
            elif strategy == "semantic":
                file_contexts = self._merge_by_semantic_boundary(
                    chunks, file_path, max_chars
                )
            elif strategy == "file":
                file_contexts = self._merge_by_file(chunks, file_path, max_chars)
            else:
                file_contexts = self._merge_contiguous_chunks(
                    chunks, file_path, max_chars
                )
            
            merged_contexts.extend(file_contexts)
        
        # 去重
        deduplicated = self._deduplicate_contexts(merged_contexts)
        
        # 按置信度排序
        deduplicated.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(
            f"Small-to-Big retrieval: {len(chunk_results)} chunks → "
            f"{len(deduplicated)} contexts"
        )
        
        return deduplicated
    
    def retrieve_with_window(
        self,
        chunk_results: List[RetrievalResult],
        window_size: int = 500,
        overlap: bool = True
    ) -> List[MergedContext]:
        """
        基于窗口的上下文检索
        
        命中一个小块后，返回其周围的窗口内容
        
        Args:
            chunk_results: 检索结果
            window_size: 窗口大小（字符数）
            overlap: 是否允许重叠
            
        Returns:
            窗口上下文列表
        """
        window_contexts = []
        seen_content = set()
        
        for result in chunk_results:
            parent = self.store.get_parent_for_chunk(result.id)
            if not parent:
                continue
            
            # 计算窗口范围
            content_hash = hashlib.md5(
                parent.content.encode()
            ).hexdigest()
            
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # 截取窗口（简化处理，实际需要根据行号计算）
            window_content = self._extract_window(
                parent.content, 
                window_size
            )
            
            context = MergedContext(
                id=f"{parent.id}_window",
                file_path=parent.file_path,
                content=window_content,
                doc_type=parent.doc_type,
                title=parent.title,
                start_line=parent.start_line,
                end_line=parent.end_line,
                source_chunk_ids=[result.id],
                char_count=len(window_content),
                confidence=result.similarity,
                metadata={
                    "type": "window",
                    "window_size": window_size
                }
            )
            
            window_contexts.append(context)
        
        return window_contexts
    
    def _group_by_file(
        self,
        chunk_results: List[RetrievalResult]
    ) -> Dict[str, List[RetrievalResult]]:
        """按文件分组检索结果"""
        groups = defaultdict(list)
        for chunk in chunk_results:
            groups[chunk.file_path].append(chunk)
        return dict(groups)
    
    def _select_strategy(
        self,
        chunks: List[RetrievalResult]
    ) -> str:
        """自动选择最佳合并策略"""
        if len(chunks) < 2:
            return "contiguous"
        
        # 检查是否连续
        lines = sorted([c.metadata.get('start_line', 0) for c in chunks])
        is_contiguous = self._check_contiguity(lines)
        
        if is_contiguous:
            return "contiguous"
        elif len(chunks) >= 3:
            return "semantic"
        else:
            return "file"
    
    def _check_contiguity(self, lines: List[int]) -> bool:
        """检查行号是否连续"""
        if len(lines) < 2:
            return True
        
        lines_sorted = sorted(lines)
        
        for i in range(1, len(lines_sorted)):
            if lines_sorted[i] - lines_sorted[i-1] > self.config.min_merge_distance:
                return False
        
        return True
    
    def _merge_contiguous_chunks(
        self,
        chunks: List[RetrievalResult],
        file_path: str,
        max_chars: int
    ) -> List[MergedContext]:
        """合并连续的小块"""
        if not chunks:
            return []
        
        # 按行号排序
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.metadata.get('start_line', 0)
        )
        
        merged = []
        current_group = [sorted_chunks[0]]
        current_lines = [sorted_chunks[0].metadata.get('start_line', 0)]
        
        for chunk in sorted_chunks[1:]:
            chunk_line = chunk.metadata.get('start_line', 0)
            prev_line = current_lines[-1]
            
            if chunk_line - prev_line <= self.config.min_merge_distance:
                current_group.append(chunk)
                current_lines.append(chunk_line)
            else:
                # 完成当前组，开始新组
                context = self._build_merged_context(
                    current_group, file_path, "contiguous", max_chars
                )
                merged.append(context)
                current_group = [chunk]
                current_lines = [chunk_line]
        
        # 最后一个组
        if current_group:
            context = self._build_merged_context(
                current_group, file_path, "contiguous", max_chars
            )
            merged.append(context)
        
        return merged
    
    def _merge_by_semantic_boundary(
        self,
        chunks: List[RetrievalResult],
        file_path: str,
        max_chars: int
    ) -> List[MergedContext]:
        """基于语义边界合并"""
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.metadata.get('start_line', 0)
        )
        
        groups = self._group_by_semantic_boundary(sorted_chunks)
        
        merged = []
        for group_chunks in groups.values():
            context = self._build_merged_context(
                group_chunks, file_path, "semantic", max_chars
            )
            merged.append(context)
        
        return merged
    
    def _group_by_semantic_boundary(
        self,
        chunks: List[RetrievalResult]
    ) -> Dict[str, List[RetrievalResult]]:
        """基于语义边界分组"""
        groups = defaultdict(list)
        current_function = None
        group_id = 0
        
        for chunk in chunks:
            chunk_function = chunk.metadata.get('function_name', '')
            chunk_class = chunk.metadata.get('class_name', '')
            
            # 检测类边界
            if chunk_class and chunk_class != current_function:
                group_id += 1
                current_function = chunk_class
            # 检测函数边界
            elif chunk_function and chunk_function != current_function:
                group_id += 1
                current_function = chunk_function
            
            groups[f"group_{group_id}"].append(chunk)
        
        return groups
    
    def _merge_by_file(
        self,
        chunks: List[RetrievalResult],
        file_path: str,
        max_chars: int
    ) -> List[MergedContext]:
        """按文件合并所有块"""
        if not chunks:
            return []
        
        context = self._build_merged_context(
            chunks, file_path, "file", max_chars
        )
        
        return [context]
    
    def _build_merged_context(
        self,
        chunks: List[RetrievalResult],
        file_path: str,
        merge_type: str,
        max_chars: int
    ) -> MergedContext:
        """构建合并后的上下文"""
        # 获取父文档
        parent_docs = []
        chunk_ids = []
        
        for chunk in chunks:
            parent = self.store.get_parent_for_chunk(chunk.id)
            if parent:
                parent_docs.append(parent)
                chunk_ids.append(chunk.id)
        
        if parent_docs:
            # 合并父文档内容
            if merge_type == "file":
                # 按文件合并：取所有内容
                merged_content = self._merge_documents(parent_docs)
            else:
                # 其他合并方式：取第一个父文档（最相关的）
                merged_content = parent_docs[0].content
            
            # 截断超出 max_chars 的内容
            if len(merged_content) > max_chars:
                merged_content = self._smart_truncate(
                    merged_content, max_chars, chunks
                )
            
            first_parent = parent_docs[0]
            
            return MergedContext(
                id=f"merged_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                file_path=file_path,
                content=merged_content,
                doc_type=first_parent.doc_type,
                title=first_parent.title,
                start_line=min(p.start_line for p in parent_docs),
                end_line=max(p.end_line for p in parent_docs),
                source_chunk_ids=chunk_ids,
                char_count=len(merged_content),
                confidence=max(c.similarity for c in chunks),
                metadata={
                    "type": merge_type,
                    "parent_count": len(parent_docs),
                    "chunk_count": len(chunks)
                }
            )
        else:
            # 没有父文档，使用chunk内容
            merged_content = "\n\n".join(c.content for c in chunks)
            if len(merged_content) > max_chars:
                merged_content = merged_content[:max_chars]
            
            return MergedContext(
                id=f"merged_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                file_path=file_path,
                content=merged_content,
                doc_type="chunks",
                title=None,
                start_line=chunks[0].metadata.get('start_line', 1),
                end_line=chunks[0].metadata.get('end_line', 1),
                source_chunk_ids=chunk_ids,
                char_count=len(merged_content),
                confidence=max(c.similarity for c in chunks),
                metadata={
                    "type": merge_type,
                    "parent_count": 0,
                    "chunk_count": len(chunks)
                }
            )
    
    def _merge_documents(self, docs: List[ParentDocument]) -> str:
        """合并多个文档"""
        contents = []
        
        for doc in docs:
            header = f"=== {doc.doc_type.upper()}: {doc.title or doc.id} ==="
            contents.append(header)
            contents.append(doc.content)
            contents.append("")
        
        return "\n".join(contents)
    
    def _smart_truncate(
        self,
        content: str,
        max_chars: int,
        chunks: List[RetrievalResult]
    ) -> str:
        """智能截断（保留最相关的部分）"""
        if len(content) <= max_chars:
            return content
        
        # 找出最相关的行
        relevant_lines = set()
        for chunk in chunks:
            start = chunk.metadata.get('start_line', 0)
            end = chunk.metadata.get('end_line', start)
            
            # 扩展窗口
            window_start = max(1, start - 5)
            window_end = end + 10
            
            for line in range(window_start, window_end + 1):
                relevant_lines.add(line)
        
        lines = content.split('\n')
        
        # 构建新内容
        truncated_lines = []
        current_length = 0
        
        for i, line in enumerate(lines, 1):
            if i in relevant_lines:
                truncated_lines.append(line)
                current_length += len(line) + 1
                
                if current_length > max_chars:
                    break
        
        # 如果不够，补充一些上下文
        if current_length < max_chars * 0.8:
            # 添加开头和结尾
            truncated = "\n".join(truncated_lines)
            if len(truncated) < max_chars:
                return truncated[:max_chars]
        
        return "\n".join(truncated_lines)[:max_chars]
    
    def _extract_window(
        self,
        content: str,
        window_size: int
    ) -> str:
        """提取内容窗口"""
        if len(content) <= window_size:
            return content
        
        # 简单截取
        return content[:window_size]
    
    def _deduplicate_contexts(
        self,
        contexts: List[MergedContext]
    ) -> List[MergedContext]:
        """去重上下文"""
        seen_content = set()
        deduplicated = []
        
        for context in contexts:
            content_hash = hashlib.md5(
                context.content.encode()
            ).hexdigest()
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                deduplicated.append(context)
        
        return deduplicated
    
    def limit_per_file(
        self,
        contexts: List[MergedContext],
        max_per_file: int = None
    ) -> List[MergedContext]:
        """限制每个文件的上下文数量"""
        max_per_file = max_per_file or self.config.max_chunks_per_file
        
        file_groups = defaultdict(list)
        
        for context in contexts:
            file_groups[context.file_path].append(context)
        
        limited = []
        
        for file_path, file_contexts in file_groups.items():
            # 按置信度排序
            sorted_contexts = sorted(
                file_contexts,
                key=lambda c: c.confidence,
                reverse=True
            )
            limited.extend(sorted_contexts[:max_per_file])
        
        return limited


class SmallToBigRetrievalPipeline:
    """
    Small-to-Big 检索流水线
    
    整合查询改写、增强检索、上下文合并
    """
    
    def __init__(
        self,
        rewriter: Optional[Any] = None,
        retriever: Optional[EnhancedParentDocumentRetriever] = None
    ):
        self.rewriter = rewriter
        self.retriever = retriever or EnhancedParentDocumentRetriever()
    
    async def retrieve(
        self,
        query: str,
        chunk_results: List[RetrievalResult],
        max_context_chars: int = 4000,
        enable_rewrite: bool = False
    ) -> List[MergedContext]:
        """
        执行完整检索流程
        
        Args:
            query: 原始查询
            chunk_results: 检索到的块结果
            max_context_chars: 最大字符数
            enable_rewrite: 是否启用查询改写
            
        Returns:
            合并后的上下文
        """
        # 可选：查询改写
        if enable_rewrite and self.rewriter:
            rewritten = await self.rewriter.rewrite(query)
            logger.info(f"Query rewritten: {query[:50]}... → {rewritten.rewritten_query[:50]}...")
        
        # Small-to-Big 检索
        contexts = self.retriever.retrieve_small_to_big(
            query=query,
            chunk_results=chunk_results,
            max_context_chars=max_context_chars,
            merge_strategy="auto"
        )
        
        # 限制每个文件的数量
        contexts = self.retriever.limit_per_file(contexts)
        
        return contexts
    
    def build_prompt_context(
        self,
        contexts: List[MergedContext],
        query: str,
        max_total_chars: int = 12000
    ) -> str:
        """构建 LLM 提示词上下文"""
        context_parts = [
            f"查询: {query}\n",
            "=" * 50,
            "参考文档:\n"
        ]
        
        current_length = sum(len(c.content) for c in contexts)
        
        for context in contexts:
            if current_length + len(context.content) > max_total_chars:
                remaining = max_total_chars - current_length
                if remaining > 200:
                    context_parts.append(f"[...内容截断，仅显示前{remaining}字符...]\n")
                break
            
            header = f"\n文件: {context.file_path}"
            if context.title:
                header += f" | {context.title}"
            header += "\n"
            
            context_parts.append(header)
            context_parts.append(context.content)
            
            current_length += len(header) + len(context.content)
        
        return "".join(context_parts)
