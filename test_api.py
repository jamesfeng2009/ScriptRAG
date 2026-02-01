#!/usr/bin/env python3
"""
API 测试脚本

用途：快速测试 API 功能
使用：python test_api.py
"""

import requests
import json
import time
import sys
from typing import Optional, Dict, Any


# 配置
API_BASE_URL = "http://localhost:8000"
API_KEY = "your-secret-api-key"
TIMEOUT = 300  # 5 分钟超时


class Colors:
    """颜色定义"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'


def print_header(text: str):
    """打印标题"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{Colors.END}\n")


def print_success(text: str):
    """打印成功信息"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """打印错误信息"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """打印警告信息"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str):
    """打印信息"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")


def test_root() -> bool:
    """测试根路径"""
    print_header("1. 测试根路径 (GET /)")
    
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"API 可用")
            print_info(f"版本: {data.get('version')}")
            print_info(f"文档: {data.get('docs')}")
            return True
        else:
            print_error(f"HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("无法连接到 API，请确保服务已启动")
        print_info("启动命令: python -m src.presentation.api")
        return False
    except Exception as e:
        print_error(f"错误: {e}")
        return False


def test_health() -> bool:
    """测试健康检查"""
    print_header("2. 测试健康检查 (GET /health)")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            status = data.get('status')
            
            if status == 'healthy':
                print_success("所有服务健康")
            else:
                print_warning(f"服务状态: {status}")
            
            # 打印服务状态
            services = data.get('services', {})
            for service, health in services.items():
                if health == 'healthy':
                    print_success(f"{service}: {health}")
                else:
                    print_warning(f"{service}: {health}")
            
            return True
        else:
            print_error(f"HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"错误: {e}")
        return False


def test_generate(topic: str = "What is Python?") -> Optional[str]:
    """测试生成请求"""
    print_header("3. 测试生成请求 (POST /generate)")
    
    try:
        payload = {
            "topic": topic,
            "workspace_id": "test-workspace",
            "context": "Testing the API",
            "skill": "standard_tutorial",
            "tone": "professional",
            "max_retries": 3
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": API_KEY
        }
        
        print_info(f"提交请求: {topic}")
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            task_id = data.get('task_id')
            status = data.get('status')
            
            print_success(f"请求已提交")
            print_info(f"Task ID: {task_id}")
            print_info(f"状态: {status}")
            
            return task_id
        else:
            print_error(f"HTTP {response.status_code}")
            print_error(f"响应: {response.text}")
            return None
            
    except Exception as e:
        print_error(f"错误: {e}")
        return None


def test_status(task_id: str, max_wait: int = 60) -> bool:
    """测试获取任务状态"""
    print_header(f"4. 轮询任务状态 (GET /status/{task_id})")
    
    try:
        headers = {"X-API-Key": API_KEY}
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            response = requests.get(
                f"{API_BASE_URL}/status/{task_id}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                progress = data.get('progress', {})
                
                # 打印进度
                if progress:
                    current = progress.get('current_step', 0)
                    total = progress.get('total_steps', 0)
                    agent = progress.get('current_agent', 'unknown')
                    print_info(f"[{elapsed:.0f}s] 状态: {status} | 步骤: {current}/{total} | Agent: {agent}")
                else:
                    print_info(f"[{elapsed:.0f}s] 状态: {status}")
                
                # 检查是否完成
                if status == 'completed':
                    print_success("任务完成！")
                    
                    # 打印结果摘要
                    result = data.get('result', {})
                    if result:
                        print_info(f"生成的片段数: {result.get('fragments_count', 'N/A')}")
                        print_info(f"检索的文档数: {result.get('documents_retrieved', 'N/A')}")
                        print_info(f"执行时间: {result.get('execution_time_seconds', 'N/A')}s")
                    
                    return True
                
                elif status == 'failed':
                    error = data.get('error')
                    print_error(f"任务失败: {error}")
                    return False
                
                # 检查超时
                if elapsed > max_wait:
                    print_warning(f"等待超时 ({max_wait}s)，任务仍在运行")
                    print_info("你可以稍后使用 task_id 查询状态")
                    return True
                
                # 等待后重试
                time.sleep(2)
                
            else:
                print_error(f"HTTP {response.status_code}")
                return False
                
    except Exception as e:
        print_error(f"错误: {e}")
        return False


def test_list_tasks() -> bool:
    """测试列出所有任务"""
    print_header("5. 测试列出任务 (GET /tasks)")
    
    try:
        headers = {"X-API-Key": API_KEY}
        
        response = requests.get(
            f"{API_BASE_URL}/tasks",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            tasks = response.json()
            print_success(f"获取 {len(tasks)} 个任务")
            
            for task in tasks[:5]:  # 只显示前 5 个
                task_id = task.get('task_id')
                status = task.get('status')
                print_info(f"  {task_id[:8]}... - {status}")
            
            if len(tasks) > 5:
                print_info(f"  ... 还有 {len(tasks) - 5} 个任务")
            
            return True
        else:
            print_error(f"HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"错误: {e}")
        return False


def test_metrics() -> bool:
    """测试获取指标"""
    print_header("6. 测试获取指标 (GET /metrics)")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/metrics",
            timeout=10
        )
        
        if response.status_code == 200:
            print_success("获取指标成功")
            # 只显示前几行
            lines = response.text.split('\n')[:5]
            for line in lines:
                if line and not line.startswith('#'):
                    print_info(f"  {line[:60]}...")
            return True
        else:
            print_error(f"HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"错误: {e}")
        return False


def print_summary(results: Dict[str, bool]):
    """打印测试总结"""
    print_header("测试总结")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    print(f"总测试数: {total}")
    print(f"通过: {Colors.GREEN}{passed}{Colors.END}")
    print(f"失败: {Colors.RED}{failed}{Colors.END}")
    
    print("\n详细结果:")
    for test_name, result in results.items():
        if result:
            print_success(test_name)
        else:
            print_error(test_name)
    
    if failed == 0:
        print_success("\n所有测试通过！")
        return True
    else:
        print_error(f"\n有 {failed} 个测试失败")
        return False


def print_usage():
    """打印使用说明"""
    print_header("API 测试脚本")
    
    print("用途: 测试 RAG Screenplay API 的各个端点")
    print("\n使用方法:")
    print("  python test_api.py              # 运行所有测试")
    print("  python test_api.py --quick      # 快速测试（不等待生成完成）")
    print("  python test_api.py --topic \"Your Topic\"  # 自定义主题")
    print("\n前置条件:")
    print("  1. API 服务已启动: python -m src.presentation.api")
    print("  2. .env 文件已配置")
    print("  3. GLM API 密钥已设置")
    print("\n配置:")
    print(f"  API URL: {API_BASE_URL}")
    print(f"  API Key: {API_KEY}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="API 测试脚本")
    parser.add_argument("--quick", action="store_true", help="快速测试（不等待生成完成）")
    parser.add_argument("--topic", type=str, default="What is Python?", help="自定义主题")
    parser.add_argument("--wait", type=int, default=60, help="最大等待时间（秒）")
    
    args = parser.parse_args()
    
    print_usage()
    
    results = {}
    
    # 运行测试
    results["根路径"] = test_root()
    
    if not results["根路径"]:
        print_error("\n无法连接到 API，请先启动服务")
        return 1
    
    results["健康检查"] = test_health()
    
    # 生成请求
    task_id = test_generate(args.topic)
    results["生成请求"] = task_id is not None
    
    # 如果生成成功，查询状态
    if task_id and not args.quick:
        results["任务状态"] = test_status(task_id, max_wait=args.wait)
    
    # 列出任务
    results["列出任务"] = test_list_tasks()
    
    # 获取指标
    results["获取指标"] = test_metrics()
    
    # 打印总结
    success = print_summary(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
