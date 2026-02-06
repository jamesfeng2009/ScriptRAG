"""摘要服务 - 文件大小检查和基于 LLM 的摘要

本服务实现：
1. 文档 token 计数
2. 为大文件生成基于 LLM 的摘要
3. 关键元数据和代码片段保留
"""

import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel

from ..llm.service import LLMService
from ..parser.tree_sitter_parser import ParsedCode
from ...infrastructure.error_handler import (
    SummarizerError,
    handle_component_errors
)


logger = logging.getLogger(__name__)


class SummarizationConfig(BaseModel):
    """摘要配置"""
    max_tokens: int = 10000  # 最大 token 数阈值
    chunk_size: int = 2000   # 分块大小
    overlap: int = 200       # 重叠大小


class SummarizedDocument(BaseModel):
    """摘要文档"""
    original_path: str
    original_size: int  # 原始字符数
    summary: str
    key_elements: list  # 关键代码元素
    metadata: Dict[str, Any]
    was_summarized: bool = True


class SummarizationService:
    """
    摘要服务
    
    功能：
    - 检查文档 token 数
    - 为大文件生成摘要
    - 保留关键元数据和代码片段
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        config: Optional[SummarizationConfig] = None
    ):
        """
        初始化摘要服务
        
        Args:
            llm_service: LLM 服务
            config: 摘要配置
        """
        self.llm_service = llm_service
        self.config = config or SummarizationConfig()
    
    def check_size(self, content: str) -> bool:
        """
        检查文档大小是否超过阈值
        
        Args:
            content: 文档内容
            
        Returns:
            是否需要摘要
        """
        # 简单的 token 估算：1 token ≈ 4 字符
        estimated_tokens = len(content) // 4
        needs_summary = estimated_tokens > self.config.max_tokens
        
        if needs_summary:
            logger.info(f"Document size {estimated_tokens} tokens exceeds threshold {self.config.max_tokens}")
        
        return needs_summary
    
    async def summarize_document(
        self,
        content: str,
        file_path: str,
        parsed_code: Optional[ParsedCode] = None
    ) -> SummarizedDocument:
        """
        为大文件生成摘要（带错误处理）
        
        实现需求 9.1, 9.2, 9.3, 9.5, 18.3:
        - 在添加到上下文前检查文档 token 数
        - 为大文件实现基于 LLM 的摘要
        - 保留关键元数据和关键代码片段
        - 使用截断处理摘要器失败
        
        Args:
            content: 文档内容
            file_path: 文件路径
            parsed_code: 解析后的代码（可选）
            
        Returns:
            摘要文档
        """
        @handle_component_errors(
            component_name="summarizer",
            fallback_value=None,
            log_level="warning"
        )
        async def _summarize_with_error_handling():
            # 提取关键元素
            key_elements = []
            if parsed_code:
                # 提取函数和类
                for element in parsed_code.elements:
                    if element.type in ['function', 'class']:
                        key_elements.append({
                            'type': element.type,
                            'name': element.name,
                            'line': element.line_start
                        })
            
            # 构建摘要提示
            prompt = self._build_summary_prompt(
                content=content,
                file_path=file_path,
                key_elements=key_elements,
                parsed_code=parsed_code
            )
            
            # 调用 LLM 生成摘要
            logger.info(f"Generating summary for {file_path}...")
            messages = [
                {"role": "system", "content": "You are a code summarization assistant. Generate concise summaries that preserve key information."},
                {"role": "user", "content": prompt}
            ]
            
            try:
                summary = await self.llm_service.chat_completion(
                    messages=messages,
                    task_type="lightweight",  # 使用轻量级模型
                    temperature=0.3,
                    max_tokens=1000
                )
            except Exception as e:
                # 如果 LLM 调用失败，抛出 SummarizerError 以触发错误处理
                logger.error(f"LLM summarization failed: {str(e)}")
                raise SummarizerError(f"LLM summarization failed: {str(e)}")
            
            # 构建元数据
            metadata = {
                'original_size': len(content),
                'estimated_tokens': len(content) // 4,
                'num_key_elements': len(key_elements)
            }
            
            if parsed_code:
                metadata.update({
                    'has_deprecated': parsed_code.has_deprecated,
                    'has_fixme': parsed_code.has_fixme,
                    'has_todo': parsed_code.has_todo,
                    'has_security': parsed_code.has_security,
                    'language': parsed_code.language
                })
            
            logger.info(f"Summary generated for {file_path}: {len(summary)} characters")
            
            return SummarizedDocument(
                original_path=file_path,
                original_size=len(content),
                summary=summary,
                key_elements=key_elements,
                metadata=metadata,
                was_summarized=True
            )
        
        # 执行带错误处理的摘要
        result = await _summarize_with_error_handling()
        
        # 如果错误处理返回了 None（回退值），使用截断策略
        if result is None:
            logger.warning(f"Summarization failed for {file_path}, using truncation fallback")
            return self._fallback_truncate(content, file_path, parsed_code)
        
        return result
    
    def _build_summary_prompt(
        self,
        content: str,
        file_path: str,
        key_elements: list,
        parsed_code: Optional[ParsedCode] = None
    ) -> str:
        """构建摘要提示"""
        prompt_parts = [
            f"Please summarize the following code file: {file_path}",
            "",
            "Focus on:",
            "1. Main purpose and functionality",
            "2. Key functions and classes",
            "3. Important comments and warnings (especially @deprecated, FIXME, TODO, Security)",
            "4. Dependencies and imports",
            "",
        ]
        
        # 添加关键元素信息
        if key_elements:
            prompt_parts.append("Key elements found:")
            for elem in key_elements[:10]:  # 限制数量
                prompt_parts.append(f"- {elem['type']}: {elem['name']} (line {elem['line']})")
            prompt_parts.append("")
        
        # 添加标记信息
        if parsed_code:
            markers = []
            if parsed_code.has_deprecated:
                markers.append("@deprecated")
            if parsed_code.has_fixme:
                markers.append("FIXME")
            if parsed_code.has_todo:
                markers.append("TODO")
            if parsed_code.has_security:
                markers.append("Security")
            
            if markers:
                prompt_parts.append(f"Important markers found: {', '.join(markers)}")
                prompt_parts.append("")
        
        # 添加内容（截断）
        max_content_length = 8000  # 限制内容长度
        if len(content) > max_content_length:
            truncated_content = content[:max_content_length] + "\n... (truncated)"
        else:
            truncated_content = content
        
        prompt_parts.append("Code content:")
        prompt_parts.append("```")
        prompt_parts.append(truncated_content)
        prompt_parts.append("```")
        
        return '\n'.join(prompt_parts)
    
    def _fallback_truncate(
        self,
        content: str,
        file_path: str,
        parsed_code: Optional[ParsedCode] = None
    ) -> SummarizedDocument:
        """
        回退策略：截断内容
        
        当 LLM 摘要失败时使用
        """
        logger.warning(f"Using fallback truncation for {file_path}")
        
        # 截断到阈值
        max_chars = self.config.max_tokens * 4
        truncated_content = content[:max_chars]
        
        # 提取关键元素
        key_elements = []
        if parsed_code:
            for element in parsed_code.elements:
                if element.type in ['function', 'class']:
                    key_elements.append({
                        'type': element.type,
                        'name': element.name,
                        'line': element.line_start
                    })
        
        # 构建简单摘要
        summary = f"[Truncated content from {file_path}]\n\n"
        summary += f"File size: {len(content)} characters\n"
        summary += f"Truncated to: {len(truncated_content)} characters\n\n"
        
        if key_elements:
            summary += f"Key elements ({len(key_elements)}):\n"
            for elem in key_elements[:10]:
                summary += f"- {elem['type']}: {elem['name']}\n"
        
        summary += f"\n{truncated_content}"
        
        metadata = {
            'original_size': len(content),
            'truncated_size': len(truncated_content),
            'fallback_used': True
        }
        
        if parsed_code:
            metadata.update({
                'has_deprecated': parsed_code.has_deprecated,
                'has_fixme': parsed_code.has_fixme,
                'has_todo': parsed_code.has_todo,
                'has_security': parsed_code.has_security
            })
        
        return SummarizedDocument(
            original_path=file_path,
            original_size=len(content),
            summary=summary,
            key_elements=key_elements,
            metadata=metadata,
            was_summarized=True
        )
