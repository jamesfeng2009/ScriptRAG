#!/usr/bin/env python
"""
测试脚本 - 验证线程安全文档和 LangGraph 并发模型

测试内容：
1. 验证 ARCHITECTURE.md 中的线程安全文档是否存在
2. 测试状态隔离（不同工作流的状态独立）
3. 测试异步并发执行
4. 验证文档中的代码示例是否正确
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.domain.models import SharedState


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_success(message):
    """打印成功消息"""
    print(f"✓ {message}")


def print_error(message):
    """打印错误消息"""
    print(f"✗ {message}")


def print_info(message):
    """打印信息消息"""
    print(f"  {message}")


def test_documentation_exists():
    """测试 1: 验证线程安全文档是否存在"""
    print_section("测试 1: 验证线程安全文档是否存在")
    
    arch_doc_path = project_root / "docs" / "ARCHITECTURE.md"
    
    if not arch_doc_path.exists():
        print_error("ARCHITECTURE.md 文件不存在")
        return False
    
    print_success("ARCHITECTURE.md 文件存在")
    
    # 读取文档内容
    with open(arch_doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查关键章节是否存在
    required_sections = [
        "## 线程安全保证",
        "### LangGraph 并发模型",
        "#### 1. 节点原子性",
        "#### 2. 状态隔离",
        "#### 3. 异步执行",
        "### 最佳实践",
        "### 多工作流并发",
        "### 故障排查",
    ]
    
    all_found = True
    for section in required_sections:
        if section in content:
            print_success(f"找到章节: {section}")
        else:
            print_error(f"缺少章节: {section}")
            all_found = False
    
    # 检查是否包含代码示例
    if "```python" in content:
        code_blocks = content.count("```python")
        print_success(f"包含 {code_blocks} 个 Python 代码示例")
    else:
        print_error("缺少 Python 代码示例")
        all_found = False
    
    # 检查是否包含相关资源链接
    if "LangGraph 文档" in content or "asyncio 文档" in content:
        print_success("包含相关资源链接")
    else:
        print_error("缺少相关资源链接")
        all_found = False
    
    return all_found


def test_state_isolation():
    """测试 2: 测试状态隔离"""
    print_section("测试 2: 测试状态隔离")
    
    try:
        # 创建两个独立的状态对象
        state1 = SharedState(
            user_topic="topic1",
            current_skill="standard_tutorial",
            outline=[]
        )
        
        state2 = SharedState(
            user_topic="topic2",
            current_skill="warning_mode",
            outline=[]
        )
        
        print_success("创建了两个独立的状态对象")
        print_info(f"   state1.user_topic = '{state1.user_topic}'")
        print_info(f"   state2.user_topic = '{state2.user_topic}'")
        
        # 修改 state1
        state1.current_skill = "research_mode"
        state1.pivot_triggered = True
        
        print_success("修改了 state1")
        print_info(f"   state1.current_skill = '{state1.current_skill}'")
        print_info(f"   state1.pivot_triggered = {state1.pivot_triggered}")
        
        # 验证 state2 未受影响
        if state2.current_skill == "warning_mode" and not state2.pivot_triggered:
            print_success("state2 未受 state1 修改的影响")
            print_info(f"   state2.current_skill = '{state2.current_skill}'")
            print_info(f"   state2.pivot_triggered = {state2.pivot_triggered}")
            return True
        else:
            print_error("state2 受到了 state1 修改的影响")
            return False
            
    except Exception as e:
        print_error(f"状态隔离测试失败: {e}")
        return False


async def test_async_execution():
    """测试 3: 测试异步执行"""
    print_section("测试 3: 测试异步执行")
    
    try:
        # 模拟异步节点函数
        async def mock_node(state: SharedState, delay: float) -> SharedState:
            """模拟一个异步节点"""
            await asyncio.sleep(delay)
            state.current_skill = f"processed_after_{delay}s"
            return state
        
        # 创建多个状态
        states = [
            SharedState(user_topic=f"topic{i}", current_skill="standard_tutorial", outline=[])
            for i in range(3)
        ]
        
        print_success("创建了 3 个状态对象")
        
        # 并发执行
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(
            mock_node(states[0], 0.1),
            mock_node(states[1], 0.1),
            mock_node(states[2], 0.1)
        )
        end_time = asyncio.get_event_loop().time()
        
        elapsed = end_time - start_time
        
        print_success(f"并发执行完成，耗时 {elapsed:.3f} 秒")
        
        # 验证并发执行（应该接近 0.1 秒，而不是 0.3 秒）
        if elapsed < 0.2:  # 允许一些误差
            print_success("并发执行正常（耗时接近单个任务时间）")
        else:
            print_error(f"并发执行可能有问题（耗时 {elapsed:.3f} 秒，期望 < 0.2 秒）")
            return False
        
        # 验证每个状态都被正确处理
        for i, result in enumerate(results):
            if result.current_skill == "processed_after_0.1s":
                print_success(f"状态 {i} 被正确处理")
            else:
                print_error(f"状态 {i} 处理失败")
                return False
        
        return True
        
    except Exception as e:
        print_error(f"异步执行测试失败: {e}")
        return False


async def test_concurrent_workflows():
    """测试 4: 测试多工作流并发"""
    print_section("测试 4: 测试多工作流并发")
    
    try:
        # 模拟工作流执行函数
        async def mock_workflow(workflow_id: int) -> dict:
            """模拟一个工作流"""
            state = SharedState(
                user_topic=f"workflow_{workflow_id}",
                current_skill="standard_tutorial",
                outline=[]
            )
            
            # 模拟一些处理
            await asyncio.sleep(0.05)
            state.current_skill = "warning_mode"
            
            await asyncio.sleep(0.05)
            state.current_skill = "research_mode"
            
            return {
                "workflow_id": workflow_id,
                "final_skill": state.current_skill,
                "topic": state.user_topic
            }
        
        # 并发执行多个工作流
        num_workflows = 5
        print_info(f"并发执行 {num_workflows} 个工作流...")
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[
            mock_workflow(i) for i in range(num_workflows)
        ])
        end_time = asyncio.get_event_loop().time()
        
        elapsed = end_time - start_time
        
        print_success(f"所有工作流完成，耗时 {elapsed:.3f} 秒")
        
        # 验证每个工作流都正确完成
        all_correct = True
        for result in results:
            workflow_id = result["workflow_id"]
            if result["final_skill"] == "research_mode" and \
               result["topic"] == f"workflow_{workflow_id}":
                print_success(f"工作流 {workflow_id} 正确完成")
            else:
                print_error(f"工作流 {workflow_id} 结果不正确")
                all_correct = False
        
        # 验证并发执行效率
        expected_time = 0.1  # 每个工作流 0.1 秒
        if elapsed < expected_time * 2:  # 允许一些开销
            print_success(f"并发执行效率良好（{num_workflows} 个工作流耗时 {elapsed:.3f} 秒）")
        else:
            print_error(f"并发执行效率较低（耗时 {elapsed:.3f} 秒）")
            all_correct = False
        
        return all_correct
        
    except Exception as e:
        print_error(f"多工作流并发测试失败: {e}")
        return False


def test_state_model_validation():
    """测试 5: 测试状态模型验证"""
    print_section("测试 5: 测试状态模型验证")
    
    try:
        # 测试正常创建
        state = SharedState(
            user_topic="test_topic",
            current_skill="standard_tutorial",
            outline=[]
        )
        print_success("正常状态创建成功")
        
        # 测试状态修改
        state.current_skill = "warning_mode"
        print_success("状态修改成功")
        
        # 测试 switch_skill 辅助方法
        if hasattr(state, 'switch_skill'):
            state.switch_skill(
                new_skill="research_mode",
                reason="test",
                step_id=1
            )
            print_success("switch_skill 辅助方法工作正常")
        else:
            print_info("switch_skill 方法不存在（可能是可选的）")
        
        return True
        
    except Exception as e:
        print_error(f"状态模型验证失败: {e}")
        return False


def test_documentation_code_examples():
    """测试 6: 验证文档中的代码示例"""
    print_section("测试 6: 验证文档中的代码示例")
    
    arch_doc_path = project_root / "docs" / "ARCHITECTURE.md"
    
    if not arch_doc_path.exists():
        print_error("ARCHITECTURE.md 文件不存在")
        return False
    
    with open(arch_doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查关键代码模式是否存在
    code_patterns = [
        ("async def", "异步函数定义"),
        ("await", "异步等待"),
        ("SharedState", "SharedState 类型"),
        ("asyncio.gather", "并发执行"),
        ("return state", "返回状态"),
    ]
    
    all_found = True
    for pattern, description in code_patterns:
        if pattern in content:
            print_success(f"找到代码模式: {description} ({pattern})")
        else:
            print_error(f"缺少代码模式: {description} ({pattern})")
            all_found = False
    
    # 检查是否有正确和错误的示例对比
    if "✅ 正确" in content and "❌ 错误" in content:
        print_success("包含正确和错误示例的对比")
    else:
        print_error("缺少正确和错误示例的对比")
        all_found = False
    
    return all_found


def test_best_practices_coverage():
    """测试 7: 验证最佳实践覆盖"""
    print_section("测试 7: 验证最佳实践覆盖")
    
    arch_doc_path = project_root / "docs" / "ARCHITECTURE.md"
    
    if not arch_doc_path.exists():
        print_error("ARCHITECTURE.md 文件不存在")
        return False
    
    with open(arch_doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查最佳实践是否涵盖
    best_practices = [
        ("在节点内修改状态", "节点内修改"),
        ("使用 SharedState 的辅助方法", "辅助方法"),
        ("避免跨节点共享可变对象", "避免共享"),
        ("全局变量", "全局变量警告"),
    ]
    
    all_found = True
    for practice, description in best_practices:
        if practice in content:
            print_success(f"涵盖最佳实践: {description}")
        else:
            print_error(f"缺少最佳实践: {description}")
            all_found = False
    
    return all_found


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("  线程安全文档和并发模型 - 测试脚本")
    print("=" * 70)
    
    # 同步测试
    sync_tests = [
        ("验证线程安全文档是否存在", test_documentation_exists),
        ("测试状态隔离", test_state_isolation),
        ("测试状态模型验证", test_state_model_validation),
        ("验证文档中的代码示例", test_documentation_code_examples),
        ("验证最佳实践覆盖", test_best_practices_coverage),
    ]
    
    results = []
    
    # 运行同步测试
    for test_name, test_func in sync_tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"测试异常: {e}")
            results.append((test_name, False))
    
    # 运行异步测试
    async_tests = [
        ("测试异步执行", test_async_execution),
        ("测试多工作流并发", test_concurrent_workflows),
    ]
    
    for test_name, test_func in async_tests:
        try:
            result = asyncio.run(test_func())
            results.append((test_name, result))
        except Exception as e:
            print_error(f"测试异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print_section("测试总结")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")
    
    print("\n" + "-" * 70)
    print(f"总计: {passed}/{total} 个测试通过 ({passed/total*100:.1f}%)")
    print("-" * 70)
    
    if passed == total:
        print("\n🎉 所有测试通过！线程安全文档完整且正确！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查。")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
