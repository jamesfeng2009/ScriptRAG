#!/usr/bin/env python3
"""Verify project setup and dependencies"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    required_packages = [
        "langgraph",
        "pydantic",
        "hypothesis",
        "openai",
        "asyncpg",
        "redis",
        "tree_sitter",
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (not installed)")
            all_installed = False
    
    return all_installed


def check_project_structure():
    """Check if project structure is correct"""
    print("\nChecking project structure...")
    required_dirs = [
        "src/presentation",
        "src/application",
        "src/domain/agents",
        "src/services/llm",
        "src/services/database",
        "src/services/parser",
        "src/infrastructure",
        "tests/unit",
        "tests/property",
        "tests/integration",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
            all_exist = False
    
    return all_exist


def check_config_files():
    """Check if configuration files exist"""
    print("\nChecking configuration files...")
    required_files = [
        "pyproject.toml",
        "requirements.txt",
        "config.yaml",
        ".env.example",
        "README.md",
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists() and path.is_file():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_exist = False
    
    # Check if .env exists (optional but recommended)
    env_path = Path(".env")
    if env_path.exists():
        print("✓ .env (configured)")
    else:
        print("⚠ .env (not configured - copy from .env.example)")
    
    return all_exist


def main():
    """Run all checks"""
    print("=" * 60)
    print("RAG Screenplay Multi-Agent System - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_dependencies()),
        ("Project Structure", check_project_structure()),
        ("Configuration Files", check_config_files()),
    ]
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All checks passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Configure .env file with your API keys")
        print("2. Set up PostgreSQL database")
        print("3. Run tests: make test")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the output above.")
        print("\nRefer to SETUP.md for detailed setup instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
