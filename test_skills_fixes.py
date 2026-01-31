#!/usr/bin/env python
"""
测试脚本 - 验证 Skills 兼容性修复

测试内容：
1. 验证所有 Skills 的兼容性规则已更新
2. 测试 BFS 路径查找功能
3. 测试 find_closest_compatible_skill 函数
4. 验证所有 Skills 可相互到达
5. 测试 SkillManager 类的功能
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.domain.skills import (
    SKILLS,
    check_skill_compatibility,
    find_skill_path,
    find_closest_compatible_skill,
    SkillManager
)


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


def test_skills_compatibility_rules():
    """测试 1: 验证所有 Skills 的兼容性规则已更新"""
    print_section("测试 1: 验证 Skills 兼容性规则")
    
    expected_compatibility = {
        "standard_tutorial": ["visualization_analogy", "warning_mode", "research_mode", "fallback_summary"],
        "warning_mode": ["standard_tutorial", "research_mode", "fallback_summary"],
        "visualization_analogy": ["standard_tutorial", "meme_style", "research_mode"],
        "research_mode": ["standard_tutorial", "warning_mode", "visualization_analogy", "fallback_summary"],
        "meme_style": ["visualization_analogy", "fallback_summary", "standard_tutorial"],
        "fallback_summary": ["standard_tutorial", "research_mode", "warning_mode", "meme_style"]
    }
    
    all_passed = True
    
    for skill_name, expected_compat in expected_compatibility.items():
        actual_compat = SKILLS[skill_name].compatible_with
        
        if set(actual_compat) == set(expected_compat):
            print_success(f"{skill_name}: {len(actual_compat)} 个兼容 Skills")
            print_info(f"   → {', '.join(actual_compat)}")
        else:
            print_error(f"{skill_name}: 兼容性规则不匹配")
            print_info(f"   期望: {expected_compat}")
            print_info(f"   实际: {actual_compat}")
            all_passed = False
    
    return all_passed


def test_direct_compatibility():
    """测试 2: 测试直接兼容性"""
    print_section("测试 2: 测试直接兼容性")
    
    test_cases = [
        ("standard_tutorial", "research_mode", True, "新增连接"),
        ("standard_tutorial", "fallback_summary", True, "新增连接"),
        ("warning_mode", "fallback_summary", True, "新增连接"),
        ("visualization_analogy", "research_mode", True, "新增连接"),
        ("meme_style", "standard_tutorial", True, "新增连接"),
        ("fallback_summary", "warning_mode", True, "新增连接"),
        ("fallback_summary", "meme_style", True, "新增连接"),
        ("meme_style", "warning_mode", False, "不直接兼容"),
    ]
    
    all_passed = True
    
    for current, target, expected, note in test_cases:
        try:
            result = check_skill_compatibility(current, target)
            if result == expected:
                status = "✓" if expected else "○"
                print(f"{status} {current} → {target}: {result} ({note})")
            else:
                print_error(f"{current} → {target}: 期望 {expected}, 实际 {result}")
                all_passed = False
        except Exception as e:
            print_error(f"{current} → {target}: 异常 - {e}")
            all_passed = False
    
    return all_passed


def test_bfs_path_finding():
    """测试 3: 测试 BFS 路径查找功能"""
    print_section("测试 3: 测试 BFS 路径查找")
    
    test_cases = [
        ("meme_style", "warning_mode", 3),
        ("meme_style", "research_mode", 3),
        ("fallback_summary", "visualization_analogy", 3),
        ("standard_tutorial", "meme_style", 3),
        ("warning_mode", "meme_style", 3),
    ]
    
    all_passed = True
    
    for source, target, max_hops in test_cases:
        try:
            path = find_skill_path(source, target, max_hops=max_hops)
            if path:
                print_success(f"{source} → {target}")
                print_info(f"   路径 ({len(path)-1} 步): {' → '.join(path)}")
            else:
                print_error(f"{source} → {target}: 找不到路径")
                all_passed = False
        except Exception as e:
            print_error(f"{source} → {target}: 异常 - {e}")
            all_passed = False
    
    return all_passed


def test_all_skills_reachable():
    """测试 4: 验证所有 Skills 可相互到达"""
    print_section("测试 4: 验证所有 Skills 可相互到达")
    
    skills_list = list(SKILLS.keys())
    unreachable = []
    total_pairs = 0
    
    for source in skills_list:
        for target in skills_list:
            if source != target:
                total_pairs += 1
                path = find_skill_path(source, target, max_hops=3)
                if path is None:
                    unreachable.append((source, target))
    
    if unreachable:
        print_error(f"找到 {len(unreachable)} 个无法到达的对：")
        for source, target in unreachable:
            print_info(f"   {source} → {target}")
        return False
    else:
        print_success(f"所有 {len(skills_list)} 个 Skills 都可以相互到达")
        print_info(f"   总共测试了 {total_pairs} 个 Skill 对")
        print_info(f"   所有路径都在 3 步以内")
        return True


def test_find_closest_compatible_skill():
    """测试 5: 测试 find_closest_compatible_skill 函数"""
    print_section("测试 5: 测试 find_closest_compatible_skill")
    
    test_cases = [
        ("meme_style", "warning_mode", True, "应该找到路径"),
        ("meme_style", "warning_mode", False, "禁用多步跳转"),
        ("standard_tutorial", "warning_mode", True, "直接兼容"),
    ]
    
    all_passed = True
    
    for current, desired, allow_multi_hop, note in test_cases:
        try:
            result = find_closest_compatible_skill(
                current, 
                desired, 
                allow_multi_hop=allow_multi_hop
            )
            print_success(f"{current} → {desired} (multi_hop={allow_multi_hop})")
            print_info(f"   下一步: {result} ({note})")
            
            # 验证返回的 Skill 是否与当前 Skill 兼容
            if result != desired:
                is_compatible = check_skill_compatibility(current, result)
                if not is_compatible:
                    print_error(f"   返回的 Skill {result} 与 {current} 不兼容")
                    all_passed = False
        except Exception as e:
            print_error(f"{current} → {desired}: 异常 - {e}")
            all_passed = False
    
    return all_passed


def test_skill_manager():
    """测试 6: 测试 SkillManager 类"""
    print_section("测试 6: 测试 SkillManager 类")
    
    try:
        manager = SkillManager()
        
        # 测试 list_skills
        skills = manager.list_skills()
        print_success(f"SkillManager.list_skills(): {len(skills)} 个 Skills")
        
        # 测试 check_compatibility
        is_compat = manager.check_compatibility("standard_tutorial", "research_mode")
        if is_compat:
            print_success("SkillManager.check_compatibility(): 正常工作")
        else:
            print_error("SkillManager.check_compatibility(): 应该返回 True")
            return False
        
        # 测试 find_compatible_skill
        result = manager.find_compatible_skill(
            "meme_style",
            "warning_mode",
            allow_multi_hop=True
        )
        print_success(f"SkillManager.find_compatible_skill(): {result}")
        
        # 测试 get_compatible_skills
        compatible = manager.get_compatible_skills("meme_style")
        print_success(f"SkillManager.get_compatible_skills(): {len(compatible)} 个兼容 Skills")
        print_info(f"   → {', '.join(compatible)}")
        
        return True
    except Exception as e:
        print_error(f"SkillManager 测试失败: {e}")
        return False


def test_path_lengths():
    """测试 7: 统计路径长度"""
    print_section("测试 7: 统计路径长度")
    
    skills_list = list(SKILLS.keys())
    path_lengths = {1: 0, 2: 0, 3: 0}
    
    for source in skills_list:
        for target in skills_list:
            if source != target:
                path = find_skill_path(source, target, max_hops=3)
                if path:
                    length = len(path) - 1
                    if length in path_lengths:
                        path_lengths[length] += 1
    
    total = sum(path_lengths.values())
    print_info(f"路径长度统计（总共 {total} 个路径）：")
    for length, count in sorted(path_lengths.items()):
        percentage = (count / total * 100) if total > 0 else 0
        print_info(f"   {length} 步: {count} 个路径 ({percentage:.1f}%)")
    
    # 验证大部分路径都在 1-2 步
    short_paths = path_lengths[1] + path_lengths[2]
    if short_paths / total > 0.7:
        print_success(f"大部分路径（{short_paths/total*100:.1f}%）都在 1-2 步以内")
        return True
    else:
        print_error(f"只有 {short_paths/total*100:.1f}% 的路径在 1-2 步以内")
        return False


def test_specific_improvements():
    """测试 8: 测试特定的改进"""
    print_section("测试 8: 测试特定的改进")
    
    print_info("修复前的问题：meme_style → warning_mode 需要 4 步")
    
    # 测试修复后的路径
    path = find_skill_path("meme_style", "warning_mode", max_hops=3)
    
    if path:
        steps = len(path) - 1
        print_success(f"修复后：meme_style → warning_mode 只需 {steps} 步")
        print_info(f"   路径: {' → '.join(path)}")
        
        if steps <= 3:
            print_success("改进成功：路径长度从 4 步减少到 3 步以内")
            return True
        else:
            print_error(f"路径仍然太长：{steps} 步")
            return False
    else:
        print_error("找不到路径")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("  Skills 兼容性修复 - 测试脚本")
    print("=" * 70)
    
    tests = [
        ("验证 Skills 兼容性规则", test_skills_compatibility_rules),
        ("测试直接兼容性", test_direct_compatibility),
        ("测试 BFS 路径查找", test_bfs_path_finding),
        ("验证所有 Skills 可相互到达", test_all_skills_reachable),
        ("测试 find_closest_compatible_skill", test_find_closest_compatible_skill),
        ("测试 SkillManager 类", test_skill_manager),
        ("统计路径长度", test_path_lengths),
        ("测试特定的改进", test_specific_improvements),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
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
        print("\n🎉 所有测试通过！修复成功！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查修复。")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
