# 执行清单 - 两个关键问题修复

## 修复 1: 线程安全文档

### 步骤 1: 打开文件
- [ ] 打开 `docs/ARCHITECTURE.md`
- [ ] 定位到"设计决策"章节（约第 400 行）

### 步骤 2: 添加新章节
- [ ] 在"设计决策"章节后添加"线程安全保证"章节
- [ ] 参考 `FIX_THREAD_SAFETY.md` 中的内容

### 步骤 3: 添加内容
- [ ] 添加"LangGraph 并发模型"小节
- [ ] 添加"节点原子性"说明
- [ ] 添加"状态隔离"说明
- [ ] 添加"异步执行"说明
- [ ] 添加"最佳实践"小节
- [ ] 添加"多工作流并发"示例
- [ ] 添加"故障排查"小节

### 步骤 4: 验证
- [ ] 检查格式是否正确
- [ ] 检查代码示例是否有效
- [ ] 检查链接是否正确

### 步骤 5: 提交
- [ ] 保存文件
- [ ] 运行 `git diff docs/ARCHITECTURE.md` 检查变更
- [ ] 提交更改

**预计时间**: 30 分钟

---

## 修复 2: Skills 兼容性矩阵

### 步骤 1: 更新兼容性规则

#### 1.1 打开文件
- [ ] 打开 `src/domain/skills.py`
- [ ] 定位到 SKILLS 字典（约第 60 行）

#### 1.2 修改 standard_tutorial
```python
# 查找这一行
"standard_tutorial": SkillConfig(
    description="清晰、结构化的教程格式",
    tone="professional",
    compatible_with=["visualization_analogy", "warning_mode"]
),

# 修改为
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
```

- [ ] 修改 standard_tutorial 的 compatible_with

#### 1.3 修改 warning_mode
```python
# 修改为
"warning_mode": SkillConfig(
    description="突出显示废弃/风险内容",
    tone="cautionary",
    compatible_with=[
        "standard_tutorial",
        "research_mode",
        "fallback_summary"    # 新增
    ]
),
```

- [ ] 修改 warning_mode 的 compatible_with

#### 1.4 修改 visualization_analogy
```python
# 修改为
"visualization_analogy": SkillConfig(
    description="使用类比和可视化解释复杂概念",
    tone="engaging",
    compatible_with=[
        "standard_tutorial",
        "meme_style",
        "research_mode"       # 新增
    ]
),
```

- [ ] 修改 visualization_analogy 的 compatible_with

#### 1.5 修改 research_mode
```python
# 修改为
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
```

- [ ] 修改 research_mode 的 compatible_with

#### 1.6 修改 meme_style
```python
# 修改为
"meme_style": SkillConfig(
    description="轻松幽默的呈现方式",
    tone="casual",
    compatible_with=[
        "visualization_analogy",
        "fallback_summary",
        "standard_tutorial"   # 新增
    ]
),
```

- [ ] 修改 meme_style 的 compatible_with

#### 1.7 修改 fallback_summary
```python
# 修改为
"fallback_summary": SkillConfig(
    description="详情不可用时的高层概述",
    tone="neutral",
    compatible_with=[
        "standard_tutorial",
        "research_mode",
        "warning_mode",       # 新增
        "meme_style"          # 新增
    ]
),
```

- [ ] 修改 fallback_summary 的 compatible_with

### 步骤 2: 添加 BFS 查找函数

#### 2.1 在 RETRIEVAL_CONFIG 后添加
```python
# 在 RETRIEVAL_CONFIG = RetrievalConfig() 后添加

from collections import deque
from typing import Optional, List

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
```

- [ ] 添加 find_skill_path 函数

#### 2.2 更新 find_closest_compatible_skill 函数
```python
# 查找现有的 find_closest_compatible_skill 函数
# 替换为新版本

def find_closest_compatible_skill(
    current_skill: str,
    desired_skill: str,
    global_tone: Optional[str] = None,
    allow_multi_hop: bool = True
) -> str:
    """Find the closest compatible skill when direct switch is not possible
    
    Args:
        current_skill: Current active skill
        desired_skill: Desired target skill
        global_tone: Optional global tone preference
        allow_multi_hop: Whether to allow multi-hop paths
        
    Returns:
        Name of the closest compatible skill
        
    Raises:
        ValueError: If skill names are invalid
    """
    if current_skill not in SKILLS:
        raise ValueError(f"Invalid current skill: {current_skill}")
    if desired_skill not in SKILLS:
        raise ValueError(f"Invalid desired skill: {desired_skill}")
    
    # If directly compatible, return desired skill
    if check_skill_compatibility(current_skill, desired_skill):
        return desired_skill
    
    # If multi-hop allowed, try to find a path
    if allow_multi_hop:
        path = find_skill_path(current_skill, desired_skill, max_hops=2)
        if path and len(path) > 1:
            return path[1]  # Return next step
    
    # Get compatible skills
    compatible = get_compatible_skills(current_skill)
    
    if not compatible:
        # No compatible skills, stay with current
        return current_skill
    
    # If global tone is specified, prefer skills with matching tone
    if global_tone:
        tone_matches = [
            skill for skill in compatible
            if SKILLS[skill].tone == global_tone
        ]
        if tone_matches:
            return tone_matches[0]
    
    # Check if any compatible skill is compatible with desired skill
    for skill in compatible:
        if check_skill_compatibility(skill, desired_skill):
            return skill
    
    # Return first compatible skill as fallback
    return compatible[0]
```

- [ ] 更新 find_closest_compatible_skill 函数

### 步骤 3: 更新 SkillManager 类

#### 3.1 更新 find_compatible_skill 方法
```python
# 在 SkillManager 类中找到 find_compatible_skill 方法
# 替换为新版本

def find_compatible_skill(
    self,
    current_skill: str,
    desired_skill: str,
    global_tone: Optional[str] = None,
    allow_multi_hop: bool = True
) -> str:
    """Find closest compatible skill for switching
    
    Args:
        current_skill: Current active skill
        desired_skill: Desired target skill
        global_tone: Optional global tone preference
        allow_multi_hop: Whether to allow multi-hop paths
        
    Returns:
        Name of the closest compatible skill
    """
    if current_skill not in self._skills:
        raise ValueError(f"Invalid current skill: {current_skill}")
    if desired_skill not in self._skills:
        raise ValueError(f"Invalid desired skill: {desired_skill}")
    
    # If directly compatible, return desired skill
    if self.check_compatibility(current_skill, desired_skill):
        return desired_skill
    
    # If multi-hop allowed, try to find a path
    if allow_multi_hop:
        path = find_skill_path(current_skill, desired_skill, max_hops=2)
        if path and len(path) > 1:
            return path[1]  # Return next step
    
    # Get compatible skills
    compatible = self.get_compatible_skills(current_skill)
    
    if not compatible:
        return current_skill
    
    # Prefer skills with matching tone
    if global_tone:
        tone_matches = [
            skill for skill in compatible
            if self._skills[skill].tone == global_tone
        ]
        if tone_matches:
            return tone_matches[0]
    
    # Check if any compatible skill can reach desired skill
    for skill in compatible:
        if self.check_compatibility(skill, desired_skill):
            return skill
    
    # Return first compatible skill
    return compatible[0]
```

- [ ] 更新 SkillManager.find_compatible_skill 方法

### 步骤 4: 验证修改

#### 4.1 检查语法
```bash
python -m py_compile src/domain/skills.py
```

- [ ] 运行语法检查，确保没有错误

#### 4.2 运行单元测试
```bash
pytest tests/unit/test_skills.py -v
pytest tests/unit/test_skills_compatibility.py -v
```

- [ ] 运行单元测试，确保所有测试通过

#### 4.3 运行属性测试
```bash
pytest tests/property/test_skill_compatibility.py -v
```

- [ ] 运行属性测试，验证兼容性规则

#### 4.4 验证所有 Skills 可相互到达
```bash
pytest tests/ -k "skill" -v
```

- [ ] 运行所有 Skills 相关测试

### 步骤 5: 提交

- [ ] 保存文件
- [ ] 运行 `git diff src/domain/skills.py` 检查变更
- [ ] 提交更改

**预计时间**: 1.5 小时

---

## 验证总清单

### 线程安全修复验证

- [ ] `docs/ARCHITECTURE.md` 已更新
- [ ] 包含"线程安全保证"章节
- [ ] 包含 LangGraph 并发模型说明
- [ ] 包含最佳实践示例
- [ ] 包含故障排查指南
- [ ] 文档格式正确
- [ ] 所有链接有效

### Skills 兼容性修复验证

- [ ] `src/domain/skills.py` 已更新
- [ ] 所有 6 个 Skills 的兼容性规则已更新
- [ ] `find_skill_path()` 函数已添加
- [ ] `find_closest_compatible_skill()` 函数已更新
- [ ] `SkillManager.find_compatible_skill()` 方法已更新
- [ ] 语法检查通过
- [ ] 单元测试通过
- [ ] 属性测试通过
- [ ] 所有 Skills 可相互到达（最多 3 步）

### 最终验证

- [ ] 所有测试通过
- [ ] 没有新的警告或错误
- [ ] 代码风格一致
- [ ] 文档完整清晰
- [ ] 变更已提交

---

## 时间估计

| 任务 | 预计时间 | 实际时间 |
|------|---------|---------|
| 修复线程安全文档 | 30 分钟 | _____ |
| 修复 Skills 兼容性 | 1.5 小时 | _____ |
| 验证和测试 | 30 分钟 | _____ |
| **总计** | **2.5 小时** | _____ |

---

## 完成标志

修复完成时，应该看到：

1. **线程安全**
   - ✅ 文档清晰说明了 LangGraph 的线程安全保证
   - ✅ 开发者知道如何正确使用 API
   - ✅ 包含最佳实践和故障排查指南

2. **Skills 兼容性**
   - ✅ 所有 Skills 之间都有路径（最多 3 步）
   - ✅ 常见切换只需 1-2 步
   - ✅ 所有测试通过
   - ✅ 用户体验改善

---

## 遇到问题？

### 问题 1: 测试失败

**解决**:
1. 检查修改是否正确
2. 查看错误信息
3. 参考 `FIX_SKILLS_COMPATIBILITY.md` 中的测试部分
4. 运行 `pytest -vv` 获取详细输出

### 问题 2: 导入错误

**解决**:
1. 检查 `from collections import deque` 是否已添加
2. 检查 `from typing import Optional, List` 是否已添加
3. 确保函数定义在正确的位置

### 问题 3: 兼容性规则冲突

**解决**:
1. 检查是否有重复的兼容性规则
2. 确保所有 Skills 都已更新
3. 运行属性测试验证一致性

---

## 下一步

修复完成后：

1. **文档更新**
   - [ ] 更新 README.md 中的 Skills 说明
   - [ ] 更新 CONFIGURATION.md 中的 Skills 配置

2. **性能测试**
   - [ ] 测试 BFS 查找的性能
   - [ ] 验证没有性能回退

3. **用户通知**
   - [ ] 更新 CHANGELOG.md
   - [ ] 通知用户新的兼容性规则

---

**开始时间**: ___________  
**完成时间**: ___________  
**总耗时**: ___________

