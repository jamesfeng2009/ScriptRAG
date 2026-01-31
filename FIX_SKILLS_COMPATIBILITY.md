# Skills 兼容性矩阵修复指南

## 问题分析

### 当前问题

**问题**: `meme_style` 不能直接切换到 `warning_mode`

**原因**: 兼容性矩阵不够完整

**当前兼容性**:
```
standard_tutorial      → [visualization_analogy, warning_mode]
warning_mode           → [standard_tutorial, research_mode]
visualization_analogy  → [standard_tutorial, meme_style]
research_mode          → [standard_tutorial, warning_mode]
meme_style             → [visualization_analogy, fallback_summary]
fallback_summary       → [standard_tutorial, research_mode]
```

**问题场景**:
```
当前 Skill: meme_style
目标 Skill: warning_mode

直接切换: ❌ 不兼容
需要中间步骤: meme_style → visualization_analogy → standard_tutorial → warning_mode
```

---

## 修复方案

### 方案 1: 增加直接兼容性（推荐）

**文件**: `src/domain/skills.py`

**修改内容**:

```python
# 修改前
SKILLS: Dict[str, SkillConfig] = {
    "standard_tutorial": SkillConfig(
        description="清晰、结构化的教程格式",
        tone="professional",
        compatible_with=["visualization_analogy", "warning_mode"]
    ),
    "warning_mode": SkillConfig(
        description="突出显示废弃/风险内容",
        tone="cautionary",
        compatible_with=["standard_tutorial", "research_mode"]
    ),
    "visualization_analogy": SkillConfig(
        description="使用类比和可视化解释复杂概念",
        tone="engaging",
        compatible_with=["standard_tutorial", "meme_style"]
    ),
    "research_mode": SkillConfig(
        description="承认信息缺口并建议研究方向",
        tone="exploratory",
        compatible_with=["standard_tutorial", "warning_mode"]
    ),
    "meme_style": SkillConfig(
        description="轻松幽默的呈现方式",
        tone="casual",
        compatible_with=["visualization_analogy", "fallback_summary"]
    ),
    "fallback_summary": SkillConfig(
        description="详情不可用时的高层概述",
        tone="neutral",
        compatible_with=["standard_tutorial", "research_mode"]
    )
}

# 修改后
SKILLS: Dict[str, SkillConfig] = {
    "standard_tutorial": SkillConfig(
        description="清晰、结构化的教程格式",
        tone="professional",
        compatible_with=[
            "visualization_analogy",
            "warning_mode",
            "research_mode",      # 新增
            "fallback_summary"    # 新增
        ]
    ),
    "warning_mode": SkillConfig(
        description="突出显示废弃/风险内容",
        tone="cautionary",
        compatible_with=[
            "standard_tutorial",
            "research_mode",
            "fallback_summary"    # 新增
        ]
    ),
    "visualization_analogy": SkillConfig(
        description="使用类比和可视化解释复杂概念",
        tone="engaging",
        compatible_with=[
            "standard_tutorial",
            "meme_style",
            "research_mode"       # 新增
        ]
    ),
    "research_mode": SkillConfig(
        description="承认信息缺口并建议研究方向",
        tone="exploratory",
        compatible_with=[
            "standard_tutorial",
            "warning_mode",
            "visualization_analogy",  # 新增
            "fallback_summary"        # 新增
        ]
    ),
    "meme_style": SkillConfig(
        description="轻松幽默的呈现方式",
        tone="casual",
        compatible_with=[
            "visualization_analogy",
            "fallback_summary",
            "standard_tutorial"   # 新增
        ]
    ),
    "fallback_summary": SkillConfig(
        description="详情不可用时的高层概述",
        tone="neutral",
        compatible_with=[
            "standard_tutorial",
            "research_mode",
            "warning_mode",       # 新增
            "meme_style"          # 新增
        ]
    )
}
```

**优点**:
- ✅ 简单直接
- ✅ 减少中间步骤
- ✅ 提高用户体验
- ✅ 易于维护

**缺点**:
- ❌ 兼容性规则变得复杂
- ❌ 需要更多的测试

---

### 方案 2: 改进查找算法（备选）

如果不想增加太多直接兼容性，可以改进 `find_closest_compatible_skill()` 算法，使用 BFS 查找最短路径。

**文件**: `src/domain/skills.py`

**添加函数**:

```python
from collections import deque
from typing import List, Tuple, Optional


def find_skill_path(
    current_skill: str,
    desired_skill: str,
    max_hops: int = 2
) -> Optional[List[str]]:
    """
    使用 BFS 查找从当前 Skill 到目标 Skill 的最短路径
    
    Args:
        current_skill: 当前 Skill
        desired_skill: 目标 Skill
        max_hops: 最大跳转次数
        
    Returns:
        Skill 路径列表，如果找不到返回 None
        
    示例:
        find_skill_path("meme_style", "warning_mode")
        → ["meme_style", "visualization_analogy", "standard_tutorial", "warning_mode"]
    """
    if current_skill == desired_skill:
        return [current_skill]
    
    if current_skill not in SKILLS or desired_skill not in SKILLS:
        return None
    
    # BFS 查找
    queue = deque([(current_skill, [current_skill])])
    visited = {current_skill}
    
    while queue:
        skill, path = queue.popleft()
        
        # 检查是否超过最大跳转次数
        if len(path) - 1 > max_hops:
            continue
        
        # 获取兼容 Skills
        compatible = SKILLS[skill].compatible_with
        
        for next_skill in compatible:
            if next_skill == desired_skill:
                # 找到目标
                return path + [next_skill]
            
            if next_skill not in visited:
                visited.add(next_skill)
                queue.append((next_skill, path + [next_skill]))
    
    # 找不到路径
    return None


def find_closest_compatible_skill_v2(
    current_skill: str,
    desired_skill: str,
    global_tone: Optional[str] = None,
    max_hops: int = 2
) -> str:
    """
    改进的查找算法，支持多步跳转
    
    Args:
        current_skill: 当前 Skill
        desired_skill: 目标 Skill
        global_tone: 全局语调偏好
        max_hops: 最大跳转次数
        
    Returns:
        最接近的兼容 Skill
    """
    if current_skill not in SKILLS:
        raise ValueError(f"Invalid current skill: {current_skill}")
    if desired_skill not in SKILLS:
        raise ValueError(f"Invalid desired skill: {desired_skill}")
    
    # 如果直接兼容，返回目标 Skill
    if check_skill_compatibility(current_skill, desired_skill):
        return desired_skill
    
    # 尝试找到路径
    path = find_skill_path(current_skill, desired_skill, max_hops)
    if path and len(path) > 1:
        # 返回路径中的下一个 Skill
        return path[1]
    
    # 如果找不到路径，使用原有逻辑
    compatible = get_compatible_skills(current_skill)
    
    if not compatible:
        return current_skill
    
    # 优先选择与全局语调匹配的 Skill
    if global_tone:
        tone_matches = [
            skill for skill in compatible
            if SKILLS[skill].tone == global_tone
        ]
        if tone_matches:
            return tone_matches[0]
    
    # 返回第一个兼容 Skill
    return compatible[0]
```

**优点**:
- ✅ 保持兼容性矩阵简洁
- ✅ 自动找到最短路径
- ✅ 易于扩展

**缺点**:
- ❌ 算法复杂度更高
- ❌ 可能需要多步切换

---

## 推荐方案：混合方案

结合两种方案的优点：

1. **增加关键的直接兼容性**（方案 1 的部分）
2. **实现 BFS 查找算法**（方案 2）

**修改步骤**:

### 步骤 1: 更新兼容性矩阵

```python
# 增加关键的直接兼容性
SKILLS: Dict[str, SkillConfig] = {
    "standard_tutorial": SkillConfig(
        description="清晰、结构化的教程格式",
        tone="professional",
        compatible_with=[
            "visualization_analogy",
            "warning_mode",
            "research_mode",      # 新增：连接到研究模式
            "fallback_summary"    # 新增：连接到降级模式
        ]
    ),
    "warning_mode": SkillConfig(
        description="突出显示废弃/风险内容",
        tone="cautionary",
        compatible_with=[
            "standard_tutorial",
            "research_mode",
            "fallback_summary"    # 新增：连接到降级模式
        ]
    ),
    "visualization_analogy": SkillConfig(
        description="使用类比和可视化解释复杂概念",
        tone="engaging",
        compatible_with=[
            "standard_tutorial",
            "meme_style",
            "research_mode"       # 新增：连接到研究模式
        ]
    ),
    "research_mode": SkillConfig(
        description="承认信息缺口并建议研究方向",
        tone="exploratory",
        compatible_with=[
            "standard_tutorial",
            "warning_mode",
            "visualization_analogy",  # 新增：连接到可视化
            "fallback_summary"        # 新增：连接到降级模式
        ]
    ),
    "meme_style": SkillConfig(
        description="轻松幽默的呈现方式",
        tone="casual",
        compatible_with=[
            "visualization_analogy",
            "fallback_summary",
            "standard_tutorial"   # 新增：连接到标准教程
        ]
    ),
    "fallback_summary": SkillConfig(
        description="详情不可用时的高层概述",
        tone="neutral",
        compatible_with=[
            "standard_tutorial",
            "research_mode",
            "warning_mode",       # 新增：连接到警告模式
            "meme_style"          # 新增：连接到梗风格
        ]
    )
}
```

### 步骤 2: 实现 BFS 查找

```python
# 在 src/domain/skills.py 中添加
from collections import deque

def find_skill_path(
    current_skill: str,
    desired_skill: str,
    max_hops: int = 2
) -> Optional[List[str]]:
    """使用 BFS 查找最短路径"""
    # ... 实现如上所示
```

### 步骤 3: 更新 SkillManager

```python
class SkillManager:
    # ... 现有代码
    
    def find_compatible_skill(
        self,
        current_skill: str,
        desired_skill: str,
        global_tone: Optional[str] = None,
        allow_multi_hop: bool = True
    ) -> str:
        """
        查找兼容 Skill
        
        Args:
            current_skill: 当前 Skill
            desired_skill: 目标 Skill
            global_tone: 全局语调
            allow_multi_hop: 是否允许多步跳转
            
        Returns:
            兼容 Skill 名称
        """
        if current_skill not in self._skills:
            raise ValueError(f"Invalid current skill: {current_skill}")
        if desired_skill not in self._skills:
            raise ValueError(f"Invalid desired skill: {desired_skill}")
        
        # 如果直接兼容，返回目标 Skill
        if self.check_compatibility(current_skill, desired_skill):
            return desired_skill
        
        # 如果允许多步跳转，尝试找到路径
        if allow_multi_hop:
            path = find_skill_path(current_skill, desired_skill, max_hops=2)
            if path and len(path) > 1:
                return path[1]  # 返回下一步
        
        # 回退到原有逻辑
        compatible = self.get_compatible_skills(current_skill)
        
        if not compatible:
            return current_skill
        
        if global_tone:
            tone_matches = [
                skill for skill in compatible
                if self._skills[skill].tone == global_tone
            ]
            if tone_matches:
                return tone_matches[0]
        
        return compatible[0]
```

---

## 测试

### 单元测试

```python
# tests/unit/test_skills_compatibility.py

import pytest
from src.domain.skills import (
    SKILLS,
    check_skill_compatibility,
    find_skill_path,
    find_closest_compatible_skill_v2,
    SkillManager
)


class TestSkillsCompatibility:
    """Skills 兼容性测试"""
    
    def test_direct_compatibility(self):
        """测试直接兼容性"""
        # standard_tutorial 应该兼容 warning_mode
        assert check_skill_compatibility("standard_tutorial", "warning_mode")
        assert check_skill_compatibility("warning_mode", "standard_tutorial")
    
    def test_indirect_compatibility(self):
        """测试间接兼容性"""
        # meme_style 应该能通过路径到达 warning_mode
        path = find_skill_path("meme_style", "warning_mode", max_hops=3)
        assert path is not None
        assert path[0] == "meme_style"
        assert path[-1] == "warning_mode"
    
    def test_find_closest_skill(self):
        """测试查找最接近的 Skill"""
        # 从 meme_style 到 warning_mode
        next_skill = find_closest_compatible_skill_v2(
            "meme_style",
            "warning_mode"
        )
        
        # 应该返回一个兼容的 Skill
        assert check_skill_compatibility("meme_style", next_skill)
    
    def test_skill_manager_compatibility(self):
        """测试 SkillManager 兼容性"""
        manager = SkillManager()
        
        # 测试直接兼容
        assert manager.check_compatibility("standard_tutorial", "warning_mode")
        
        # 测试查找兼容 Skill
        next_skill = manager.find_compatible_skill(
            "meme_style",
            "warning_mode"
        )
        assert next_skill is not None
    
    def test_all_skills_reachable(self):
        """测试所有 Skill 都可以相互到达"""
        skills = list(SKILLS.keys())
        
        for source in skills:
            for target in skills:
                if source != target:
                    path = find_skill_path(source, target, max_hops=3)
                    assert path is not None, f"No path from {source} to {target}"
    
    def test_tone_preference(self):
        """测试语调偏好"""
        manager = SkillManager()
        
        # 从 meme_style 到 warning_mode，偏好 professional 语调
        next_skill = manager.find_compatible_skill(
            "meme_style",
            "warning_mode",
            global_tone="professional"
        )
        
        # 应该返回 standard_tutorial（professional 语调）
        assert SKILLS[next_skill].tone == "professional"
```

### 属性测试

```python
# tests/property/test_skill_compatibility_property.py

from hypothesis import given, strategies as st
import pytest
from src.domain.skills import (
    SKILLS,
    check_skill_compatibility,
    find_skill_path
)


class TestSkillCompatibilityProperty:
    """Skills 兼容性属性测试"""
    
    @given(
        source=st.sampled_from(list(SKILLS.keys())),
        target=st.sampled_from(list(SKILLS.keys()))
    )
    def test_skill_path_exists(self, source, target):
        """属性：任意两个 Skill 之间都存在路径"""
        if source == target:
            return
        
        path = find_skill_path(source, target, max_hops=3)
        
        # 路径应该存在
        assert path is not None, f"No path from {source} to {target}"
        
        # 路径应该以源 Skill 开始
        assert path[0] == source
        
        # 路径应该以目标 Skill 结束
        assert path[-1] == target
        
        # 路径中相邻的 Skill 应该兼容
        for i in range(len(path) - 1):
            assert check_skill_compatibility(path[i], path[i + 1])
    
    @given(
        skill=st.sampled_from(list(SKILLS.keys()))
    )
    def test_skill_self_compatible(self, skill):
        """属性：Skill 与自己兼容"""
        assert check_skill_compatibility(skill, skill)
    
    @given(
        skill1=st.sampled_from(list(SKILLS.keys())),
        skill2=st.sampled_from(list(SKILLS.keys()))
    )
    def test_compatibility_symmetry(self, skill1, skill2):
        """属性：兼容性应该是对称的"""
        compat_1_2 = check_skill_compatibility(skill1, skill2)
        compat_2_1 = check_skill_compatibility(skill2, skill1)
        
        # 如果 skill1 兼容 skill2，skill2 也应该兼容 skill1
        assert compat_1_2 == compat_2_1
```

---

## 验证清单

修复后验证：

- [ ] 所有 Skills 之间都有路径（最多 3 步）
- [ ] 直接兼容性测试通过
- [ ] 间接兼容性测试通过
- [ ] 属性测试通过
- [ ] 没有循环依赖
- [ ] 兼容性矩阵文档更新

**运行测试**:

```bash
# 运行单元测试
pytest tests/unit/test_skills_compatibility.py -v

# 运行属性测试
pytest tests/property/test_skill_compatibility_property.py -v

# 运行所有 Skills 相关测试
pytest tests/ -k "skill" -v
```

---

## 总结

### 修复方案对比

| 方案 | 工作量 | 复杂度 | 效果 | 推荐 |
|------|--------|--------|------|------|
| 方案 1: 增加直接兼容性 | 30 分钟 | 低 | 好 | ✅ |
| 方案 2: BFS 算法 | 1 小时 | 中 | 很好 | ⚠️ |
| 混合方案 | 1.5 小时 | 中 | 最好 | ✅✅ |

### 推荐：混合方案

- 增加关键的直接兼容性（减少常见切换的步骤）
- 实现 BFS 查找算法（处理复杂的切换场景）
- 总工作量: 1.5 小时
- 优先级: P1

