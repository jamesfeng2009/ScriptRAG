"""Setup script for RAG Screenplay Multi-Agent System"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag-screenplay-multi-agent",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="RAG-based screenplay generation multi-agent system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag-screenplay-multi-agent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "langgraph>=0.2.0",
        "pydantic>=2.5.0",
        "hypothesis>=6.92.0",
        "openai>=1.10.0",
        "dashscope>=1.14.0",
        "zhipuai>=2.0.0",
        "asyncpg>=0.29.0",
        "psycopg2-binary>=2.9.9",
        "pgvector>=0.2.4",
        "redis>=5.0.1",
        "tree-sitter>=0.21.0",
        "tree-sitter-python>=0.21.0",
        "tree-sitter-javascript>=0.21.0",
        "tree-sitter-typescript>=0.21.0",
        "python-dotenv>=1.0.0",
        "aiohttp>=3.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "ruff>=0.1.9",
            "mypy>=1.8.0",
        ],
    },
)
