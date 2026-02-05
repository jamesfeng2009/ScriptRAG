"""Pytest configuration and shared fixtures"""

import pytest
import asyncio
import os
from pathlib import Path
from typing import AsyncGenerator


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip integration tests when database is not available"""
    try:
        import asyncpg
        has_asyncpg = True
    except ImportError:
        has_asyncpg = False

    # Check if database is available
    db_available = False
    if has_asyncpg:
        try:
            import asyncio
            async def check_db():
                try:
                    conn = await asyncpg.connect(
                        host="localhost",
                        port=5433,
                        database="Screenplay",
                        user="postgres",
                        password="123456",
                        timeout=2
                    )
                    await conn.close()
                    return True
                except Exception:
                    return False
            loop = asyncio.get_event_loop()
            db_available = loop.run_until_complete(check_db())
        except Exception:
            db_available = False

    # Skip integration tests if database is not available
    if not db_available:
        for item in items:
            marker = item.get_closest_marker("integration")
            if marker or "integration" in item.nodeid:
                item.add_marker(pytest.mark.skip(
                    reason="Integration test skipped - PostgreSQL database not available"
                ))


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory"""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_code_file(test_data_dir: Path, tmp_path: Path) -> Path:
    """Create a sample code file for testing"""
    code_content = '''
def example_function(x: int, y: int) -> int:
    """Example function for testing"""
    # TODO: Add more functionality
    return x + y

class ExampleClass:
    """Example class for testing"""
    
    def __init__(self, value: int):
        self.value = value
    
    @deprecated
    def old_method(self):
        """This method is deprecated"""
        pass
'''
    
    test_file = tmp_path / "example.py"
    test_file.write_text(code_content)
    return test_file


@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
