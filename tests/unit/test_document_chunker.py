"""
智能分块器单元测试

测试覆盖:
- 编码检测 (UTF-8, GBK, binary)
- 二进制文件检测
- Python 代码分块 (嵌套结构)
- Markdown 分块
- 大文件流式处理
- 特殊格式处理
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.services.documents.document_chunker import (
    SmartChunker,
    EncodingDetector,
    BinaryDetector,
    LargeFileProcessor,
    FileType,
    Chunk,
)


class TestEncodingDetector:
    """编码检测测试"""
    
    def test_utf8_plain(self):
        content = b"Hello, World!"
        encoding, success = EncodingDetector.detect(content)
        assert success
        assert encoding in ['utf-8', 'utf-8-sig']
    
    def test_utf8_with_bom(self):
        content = b'\xef\xbb\xbfHello'
        encoding, success = EncodingDetector.detect(content)
        assert success
        assert encoding == 'utf-8-sig'
    
    def test_gbk_content(self):
        content = "你好世界".encode('gbk')
        encoding, success = EncodingDetector.detect(content)
        assert success
        assert encoding == 'gbk'
    
    def test_empty_content(self):
        encoding, success = EncodingDetector.detect(b"")
        assert success
        assert encoding == 'utf-8'
    
    def test_decode_content(self):
        content = "测试文本".encode('utf-8')
        text, enc = EncodingDetector.decode_content(content)
        assert text == "测试文本"
        assert enc in ['utf-8', 'utf-8-sig']


class TestBinaryDetector:
    """二进制文件检测测试"""
    
    def test_text_content(self):
        content = b"Hello, World!\nThis is text."
        assert BinaryDetector.is_binary(content) == False
    
    def test_png_file(self):
        content = b'\x89PNG\r\n\x1a\n...'
        assert BinaryDetector.is_binary(content) == True
    
    def test_pdf_file(self):
        content = b'%PDF-1.4...'
        assert BinaryDetector.is_binary(content) == True
    
    def test_null_bytes(self):
        content = b'\x00\x00\x00' + b'a' * 50
        assert BinaryDetector.is_binary(content) == True
    
    def test_high_null_ratio(self):
        content = b'\x00' * 100 + b'text'
        assert BinaryDetector.is_binary(content) == True
    
    def test_get_file_type(self):
        content = b'\x89PNG\r\n\x1a\n'
        file_type = BinaryDetector.get_file_type(content)
        assert file_type == 'png'


class TestSmartChunker:
    """智能分块器测试"""
    
    @pytest.fixture
    def chunker(self):
        return SmartChunker(config={
            'chunk_size': 500,
            'min_chunk_size': 50,
            'overlap': 100
        })
    
    def test_detect_python_file(self, chunker):
        file_type = chunker.detect_file_type("test.py")
        assert file_type == FileType.PYTHON
    
    def test_detect_markdown_file(self, chunker):
        file_type = chunker.detect_file_type("README.md")
        assert file_type == FileType.MARKDOWN
    
    def test_detect_json_file(self, chunker):
        file_type = chunker.detect_file_type("data.json")
        assert file_type == FileType.JSON
    
    def test_detect_binary_file(self, chunker):
        content = b'\x89PNG\r\n\x1a\n'
        file_type = chunker.detect_file_type("image.png", content)
        assert file_type == FileType.BINARY
    
    def test_chunk_python_code(self, chunker):
        code = '''
class UserService:
    def __init__(self, user_id: int):
        self.user_id = user_id
    
    async def get_user(self) -> User:
        """获取用户信息"""
        return await self.db.query(User, id=self.user_id)
    
    class InnerService:
        def handle(self):
            pass
'''
        chunks = chunker.chunk_text(code, "test.py", FileType.PYTHON)
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        
        for chunk in chunks:
            assert len(chunk.content) > 0
            assert chunk.metadata.file_type == FileType.PYTHON
            assert chunk.metadata.char_count == len(chunk.content)
    
    def test_chunk_python_nested_class(self, chunker):
        code = '''
class Outer:
    class Inner:
        class DeepInner:
            def method(self):
                pass
'''
        chunks = chunker.chunk_text(code, "nested.py", FileType.PYTHON)
        
        assert len(chunks) > 0
        content = '\n'.join(c.content for c in chunks)
        assert 'Outer' in content
        assert 'Inner' in content
        assert 'DeepInner' in content
    
    def test_chunk_markdown(self, chunker):
        md = '''
# Title

## Section 1

Content of section 1

## Section 2

Content of section 2
'''
        chunks = chunker.chunk_text(md, "test.md", FileType.MARKDOWN)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata.file_type == FileType.MARKDOWN
    
    def test_chunk_json(self, chunker):
        json_content = '{"name": "test", "value": 123}'
        chunks = chunker.chunk_text(json_content, "test.json", FileType.JSON)
        
        assert len(chunks) > 0
    
    def test_chunk_text_long(self, chunker):
        text = "word " * 250
        chunks = chunker.chunk_text(text, "test.txt", FileType.TEXT)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata.char_count == len(chunk.content)
    
    def test_chunk_binary_file(self, chunker):
        content = b'\x89PNG\r\n\x1a\n'
        chunks = chunker.chunk_bytes(content, "test.png")
        
        assert len(chunks) == 0
    
    def test_chunk_empty_file(self, chunker):
        chunks = chunker.chunk_text("", "empty.txt")
        
        assert len(chunks) == 0 or len(chunks[0].content.strip()) == 0
    
    def test_chunk_file_not_found(self, chunker):
        chunks = chunker.chunk_file("/nonexistent/file.py")
        
        assert len(chunks) == 0
    
    def test_chunk_encoding_detection(self, chunker):
        content = "测试文本".encode('utf-8')
        chunks = chunker.chunk_bytes(content, "test.txt")
        
        assert len(chunks) > 0
        assert chunks[0].metadata.encoding in ['utf-8', 'utf-8-sig']
    
    def test_chunk_content_hash_unique(self, chunker):
        code = '''
def func1():
    return 1

def func2():
    return 2
'''
        chunks = chunker.chunk_text(code, "test.py", FileType.PYTHON)
        
        hashes = [c.metadata.content_hash for c in chunks]
        assert len(hashes) == len(set(hashes))
    
    def test_chunk_id_format(self, chunker):
        code = "x = 1"
        chunks = chunker.chunk_text(code, "test.py", FileType.PYTHON)
        
        assert len(chunks) == 1
        chunk_id = chunks[0].id
        assert chunk_id.startswith("test_")
        assert "_" in chunk_id


class TestLargeFileProcessor:
    """大文件流式处理测试"""
    
    @pytest.fixture
    def processor(self):
        return LargeFileProcessor(max_file_size=1024)
    
    def test_should_use_streaming_small(self, processor):
        content = b"small content"
        assert processor.should_use_streaming("test.txt", content) == False
    
    def test_should_use_streaming_large(self, processor):
        content = b"x" * 2048
        assert processor.should_use_streaming("test.txt", content) == True
    
    def test_stream_chunk_file(self, processor):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("line1\nline2\nline3\n")
            temp_path = f.name
        
        try:
            chunks = list(processor.stream_chunk(temp_path, chunk_size=100))
            assert len(chunks) >= 1
            
            content, start, end = chunks[0]
            assert "line1" in content
        finally:
            os.unlink(temp_path)
    
    def test_stream_with_overlap(self, processor):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("line1\nline2\nline3\nline4\nline5\n")
            temp_path = f.name
        
        try:
            chunks = list(processor.stream_chunk(temp_path, chunk_size=50, overlap=1))
            assert len(chunks) >= 1
        finally:
            os.unlink(temp_path)


class TestEdgeCases:
    """边界情况测试"""
    
    @pytest.fixture
    def chunker(self):
        return SmartChunker()
    
    def test_minified_js(self, chunker):
        js = 'function test(){console.log("test");return 1;}'
        chunks = chunker.chunk_text(js, "test.js", FileType.JAVASCRIPT)
        assert len(chunks) > 0
    
    def test_very_long_line(self, chunker):
        line = "x" * 2000
        text = f"{line}\nnormal line\n"
        chunks = chunker.chunk_text(text, "test.txt")
        
        for chunk in chunks:
            assert len(chunk.content) > 0
    
    def test_special_characters(self, chunker):
        text = 'Special: "quotes" & <brackets>'
        chunks = chunker.chunk_text(text, "test.txt")
        assert len(chunks) > 0
        assert chunks[0].content == text
    
    def test_unicode_emoji(self, chunker):
        text = "Hello 🌍 World 🚀"
        chunks = chunker.chunk_text(text, "test.txt")
        assert len(chunks) > 0
        assert "🌍" in chunks[0].content
    
    def test_only_whitespace(self, chunker):
        text = "   \n\n   \n\n   "
        chunks = chunker.chunk_text(text, "test.txt")
        assert len(chunks) == 0 or len(chunks[0].content.strip()) == 0
    
    def test_python_with_decorators(self, chunker):
        code = '''
@decorator1
@decorator2
def decorated_func():
    pass

class DecoratedClass:
    @property
    def value(self):
        return 1
'''
        chunks = chunker.chunk_text(code, "test.py", FileType.PYTHON)
        assert len(chunks) > 0
        content = '\n'.join(c.content for c in chunks)
        assert 'decorated_func' in content
        assert 'DecoratedClass' in content
    
    def test_python_async_functions(self, chunker):
        code = '''
async def async_func():
    await asyncio.sleep(1)

async def another_async():
    return await inner()
'''
        chunks = chunker.chunk_text(code, "test.py", FileType.PYTHON)
        assert len(chunks) > 0
        content = '\n'.join(c.content for c in chunks)
        assert 'async_func' in content
        assert 'another_async' in content
    
    def test_python_multiline_strings(self, chunker):
        code = '''
def func():
    s = """
    Multi-line
    string
    """
    return s
'''
        chunks = chunker.chunk_text(code, "test.py", FileType.PYTHON)
        assert len(chunks) > 0
        content = '\n'.join(c.content for c in chunks)
        assert 'Multi-line' in content
    
    def test_chunk_metadata_completeness(self, chunker):
        code = "def test(): pass"
        chunks = chunker.chunk_text(code, "test.py", FileType.PYTHON)
        
        if chunks:
            chunk = chunks[0]
            assert hasattr(chunk, 'id')
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'metadata')
            assert hasattr(chunk.metadata, 'file_path')
            assert hasattr(chunk.metadata, 'file_type')
            assert hasattr(chunk.metadata, 'char_count')
            assert hasattr(chunk.metadata, 'content_hash')
    
    def test_to_dict_method(self, chunker):
        code = "x = 1"
        chunks = chunker.chunk_text(code, "test.py", FileType.PYTHON)
        
        if chunks:
            chunk_dict = chunks[0].to_dict()
            assert 'id' in chunk_dict
            assert 'content' in chunk_dict
            assert 'metadata' in chunk_dict
            assert chunk_dict['metadata']['file_path'] == 'test.py'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
