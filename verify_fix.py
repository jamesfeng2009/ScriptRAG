#!/usr/bin/env python3
"""
验证依赖修复是否成功
"""

import sys
import subprocess
import time

def test_imports():
    """测试关键模块导入"""
    print("🧪 测试模块导入...")
    
    tests = [
        ("langgraph.graph", "StateGraph, END"),
        ("langchain_core.runnables", "Runnable"),
        ("pydantic", "BaseModel"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "run"),
    ]
    
    for module, items in tests:
        try:
            exec(f"from {module} import {items}")
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module}: {e}")
            return False
    
    return True

def test_api_import():
    """测试 API 模块导入"""
    print("\n🧪 测试 API 模块导入...")
    
    try:
        from src.presentation.api import app
        print("✓ API 模块导入成功")
        return True
    except Exception as e:
        print(f"✗ API 模块导入失败: {e}")
        return False

def test_api_startup():
    """测试 API 启动"""
    print("\n🧪 测试 API 启动...")
    
    try:
        # 启动 API 服务
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.presentation.api:app", 
            "--host", "127.0.0.1", 
            "--port", "8001"  # 使用不同端口避免冲突
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待启动
        time.sleep(3)
        
        # 检查进程是否还在运行
        if process.poll() is None:
            print("✓ API 启动成功")
            process.terminate()
            process.wait()
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"✗ API 启动失败")
            print(f"stdout: {stdout.decode()}")
            print(f"stderr: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"✗ API 启动测试失败: {e}")
        return False

def main():
    """主函数"""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                                                                ║")
    print("║                    验证依赖修复结果                            ║")
    print("║                                                                ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    success = True
    
    # 测试模块导入
    if not test_imports():
        success = False
    
    # 测试 API 模块导入
    if not test_api_import():
        success = False
    
    # 测试 API 启动
    if not test_api_startup():
        success = False
    
    print("\n" + "="*60)
    
    if success:
        print("🎉 所有测试通过！依赖修复成功！")
        print("\n现在你可以启动 API:")
        print("  uvicorn src.presentation.api:app --reload")
        print("\n然后访问:")
        print("  http://localhost:8000/docs")
        return 0
    else:
        print("❌ 某些测试失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())