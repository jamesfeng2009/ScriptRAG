"""Pytest configuration and shared fixtures"""

import pytest
from pathlib import Path


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
