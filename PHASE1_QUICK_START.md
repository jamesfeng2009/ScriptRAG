# 🚀 Phase 1 快速开始指南

## ✨ 新功能概览

Skill系统现在支持配置文件管理和热重载！

## 📦 安装依赖

```bash
pip install watchdog
```

## 🎯 快速使用

### 1. 基础使用

```python
from src.domain.skills import SkillManager

# 从配置文件加载skills
manager = SkillManager(config_path="config/skills.yaml")

# 列出所有可用skills
print(manager.list_skills())
# ['standard_tutorial', 'warning_mode', 'visualization_analogy', ...]

# 获取skill配置
skill = manager.get_skill("standard_tutorial")
print(skill.description)  # "清晰、结构化的教程格式"
```

### 2. 启用热重载

```python
# 方法1：初始化时启用
manager = SkillManager(
    config_path="config/skills.yaml",
    enable_hot_reload=True  # 🔥 启用热重载
)

# 方法2：手动启用
manager = SkillManager(config_path="config/skills.yaml")
manager.enable_hot_reload()

# 现在修改 config/skills.yaml 会自动重新加载！
```

### 3. 添加自定义Skill

编辑 `config/skills.yaml`：

```yaml
my_custom_skill:
  description: "我的自定义skill"
  tone: "friendly"
  compatible_with:
    - standard_tutorial
  prompt_config:
    system_prompt: |
      你是一个友好的助手，擅长用简单的语言解释复杂概念。
    user_template: |
      任务: {step_description}
      参考内容: {retrieved_content}
      请生成友好易懂的内容。
    temperature: 0.8
    max_tokens: 2000
  enabled: true
  metadata:
    category: "custom"
    author: "your_name"
```

保存后：
- ✅ 如果启用了热重载，自动生效
- ✅ 否则调用 `manager.reload_from_config("config/skills.yaml")`

### 4. 禁用Skill

在配置文件中设置 `enabled: false`：

```yaml
meme_style:
  description: "轻松幽默的呈现方式"
  # ... 其他配置 ...
  enabled: false  # ❌ 禁用此skill
```

## 🧪 运行测试

```bash
# 运行所有skill loader测试
python -m pytest tests/unit/test_skill_loader.py -v

# 运行所有skill相关测试
python -m pytest tests/unit/test_skills.py tests/unit/test_skill_loader.py -v
```

## 📚 配置文件位置

```
project/
├── config/
│   └── skills.yaml          # 主配置文件
├── src/domain/
│   ├── skills.py            # SkillManager
│   └── skill_loader.py      # 配置加载器
└── docs/
    └── SKILL_CONFIGURATION_GUIDE.md  # 详细文档
```

## 🔧 常用操作

### 导出当前配置

```python
manager = SkillManager()
manager.export_to_config("config/my_skills.yaml")
```

### 创建默认配置

```python
from src.domain.skill_loader import create_default_config
from pathlib import Path

create_default_config(Path("config/default_skills.yaml"))
```

### 验证配置

```python
from src.domain.skill_loader import SkillConfigLoader
import yaml

loader = SkillConfigLoader("config/skills.yaml")

with open("config/skills.yaml", 'r') as f:
    config = yaml.safe_load(f)

if loader.validate_config(config):
    print("✅ 配置有效")
else:
    print("❌ 配置有错误")
```

### 重新加载配置

```python
# 手动重新加载
manager.reload_from_config("config/skills.yaml")

# 或者使用热重载（自动）
manager.enable_hot_reload()
```

## 💡 最佳实践

### 1. Prompt设计

```yaml
prompt_config:
  system_prompt: |
    # 明确角色
    你是一个专业的技术写作专家。
    
    # 明确要求
    要求：
    - 使用清晰、简洁的语言
    - 提供具体的代码示例
    - 基于检索内容，不要编造
    
  user_template: |
    # 必须包含这两个占位符
    步骤描述: {step_description}
    检索内容: {retrieved_content}
    
    请生成内容。
```

### 2. Temperature设置

```yaml
# 技术文档 - 需要精确
temperature: 0.3-0.5

# 教程解释 - 平衡
temperature: 0.6-0.7

# 创意内容 - 更自由
temperature: 0.8-0.9
```

### 3. 兼容性设计

```yaml
# 创建兼容性链
skill_a:
  compatible_with: [skill_b, skill_c]

skill_b:
  compatible_with: [skill_a, skill_d]

# 可以实现: skill_a -> skill_b -> skill_d
```

## 🐛 故障排除

### 问题：配置文件找不到

```python
# 使用绝对路径
from pathlib import Path
config_path = Path(__file__).parent / "config" / "skills.yaml"
manager = SkillManager(config_path=str(config_path))
```

### 问题：YAML格式错误

```bash
# 在线验证YAML
# https://www.yamllint.com/

# 或使用Python验证
python -c "import yaml; yaml.safe_load(open('config/skills.yaml'))"
```

### 问题：热重载不工作

```python
# 检查状态
print(f"Hot reload enabled: {manager.is_hot_reload_enabled()}")
print(f"Config path: {manager.get_config_path()}")

# 确保路径正确
manager = SkillManager(
    config_path="config/skills.yaml",  # 确保路径正确
    enable_hot_reload=True
)
```

## 📖 更多资源

- 📘 [完整配置指南](docs/SKILL_CONFIGURATION_GUIDE.md)
- 📗 [Skills系统概述](src/domain/README_SKILLS.md)
- 📙 [实施总结](docs/SKILL_SYSTEM_ENHANCEMENT_SUMMARY.md)
- 📕 [API文档](docs/API.md)

## 🎉 下一步

Phase 1 完成！准备进入：

- **Phase 2**: 优化RAG检索系统
  - 改进向量搜索算法
  - 添加重排序机制
  - 提升检索质量

- **Phase 3**: 增强监控和可观测性
  - 添加性能指标
  - 改进日志分析
  - 实时告警

---

**需要帮助？** 查看 [SKILL_CONFIGURATION_GUIDE.md](docs/SKILL_CONFIGURATION_GUIDE.md) 获取详细说明。
