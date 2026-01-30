# Skills System Documentation

## Overview

The Skills system provides a flexible framework for managing generation style modes in the RAG screenplay generation system. Skills define how screenplay fragments are generated, with different tones and styles for different contexts.

## Available Skills

### 1. standard_tutorial
- **Description**: 清晰、结构化的教程格式
- **Tone**: professional
- **Compatible with**: visualization_analogy, warning_mode
- **Use case**: Default mode for standard technical explanations

### 2. warning_mode
- **Description**: 突出显示废弃/风险内容
- **Tone**: cautionary
- **Compatible with**: standard_tutorial, research_mode
- **Use case**: When dealing with deprecated features or security concerns

### 3. visualization_analogy
- **Description**: 使用类比和可视化解释复杂概念
- **Tone**: engaging
- **Compatible with**: standard_tutorial, meme_style
- **Use case**: Complex concepts that benefit from analogies

### 4. research_mode
- **Description**: 承认信息缺口并建议研究方向
- **Tone**: exploratory
- **Compatible with**: standard_tutorial, warning_mode
- **Use case**: When information is insufficient or missing

### 5. meme_style
- **Description**: 轻松幽默的呈现方式
- **Tone**: casual
- **Compatible with**: visualization_analogy, fallback_summary
- **Use case**: Casual, humorous content presentation

### 6. fallback_summary
- **Description**: 详情不可用时的高层概述
- **Tone**: neutral
- **Compatible with**: standard_tutorial, research_mode
- **Use case**: Degraded mode when details are unavailable

## Usage Examples

### Basic Usage

```python
from src.domain.skills import SKILLS, check_skill_compatibility

# Check if two skills are compatible
is_compatible = check_skill_compatibility("standard_tutorial", "warning_mode")
print(f"Compatible: {is_compatible}")  # True

# Get compatible skills for a given skill
from src.domain.skills import get_compatible_skills
compatible = get_compatible_skills("standard_tutorial")
print(f"Compatible skills: {compatible}")  # ['visualization_analogy', 'warning_mode']
```

### Using SkillManager

```python
from src.domain.skills import SkillManager, SkillConfig

# Create a skill manager
manager = SkillManager()

# List all available skills
skills = manager.list_skills()
print(f"Available skills: {skills}")

# Check compatibility
if manager.check_compatibility("standard_tutorial", "warning_mode"):
    print("Skills are compatible!")

# Find closest compatible skill
closest = manager.find_compatible_skill(
    current_skill="standard_tutorial",
    desired_skill="research_mode",
    global_tone="professional"
)
print(f"Closest compatible skill: {closest}")
```

### Registering Custom Skills

```python
from src.domain.skills import SkillManager, SkillConfig

# Create a custom skill
custom_skill = SkillConfig(
    description="Technical API documentation style",
    tone="technical",
    compatible_with=["standard_tutorial", "warning_mode"]
)

# Register the custom skill
manager = SkillManager()
manager.register_skill("api_documentation", custom_skill)

# Use the custom skill
if manager.check_compatibility("standard_tutorial", "api_documentation"):
    print("Can switch to custom skill!")
```

### Integration with SharedState

```python
from src.domain.models import SharedState, OutlineStep
from src.domain.skills import SkillManager

# Create state with initial skill
state = SharedState(
    user_topic="Deprecated API usage",
    project_context="Legacy codebase",
    outline=[
        OutlineStep(step_id=1, description="Explain deprecated API", status="pending")
    ],
    current_skill="standard_tutorial"
)

# Detect need to switch skills
manager = SkillManager()
desired_skill = "warning_mode"

if manager.check_compatibility(state.current_skill, desired_skill):
    state.current_skill = desired_skill
else:
    # Find closest compatible skill
    state.current_skill = manager.find_compatible_skill(
        state.current_skill,
        desired_skill,
        state.global_tone
    )

print(f"Current skill: {state.current_skill}")
```

## Retrieval Configuration

The system also provides a comprehensive retrieval configuration:

```python
from src.domain.skills import RETRIEVAL_CONFIG

# Vector search configuration
vector_config = RETRIEVAL_CONFIG.vector_search
print(f"Top K: {vector_config.top_k}")  # 5
print(f"Similarity threshold: {vector_config.similarity_threshold}")  # 0.7
print(f"Embedding model: {vector_config.embedding_model}")  # text-embedding-3-large

# Keyword search configuration
keyword_config = RETRIEVAL_CONFIG.keyword_search
print(f"Markers: {keyword_config.markers}")  # ['@deprecated', 'FIXME', 'TODO', ...]
print(f"Boost factor: {keyword_config.boost_factor}")  # 1.5

# Hybrid merge configuration
hybrid_config = RETRIEVAL_CONFIG.hybrid_merge
print(f"Vector weight: {hybrid_config.vector_weight}")  # 0.6
print(f"Keyword weight: {hybrid_config.keyword_weight}")  # 0.4

# Summarization configuration
summary_config = RETRIEVAL_CONFIG.summarization
print(f"Max tokens: {summary_config.max_tokens}")  # 10000
print(f"Chunk size: {summary_config.chunk_size}")  # 2000
```

## Skill Compatibility Graph

The skills form a compatibility graph that ensures smooth transitions:

```
standard_tutorial <-> visualization_analogy
standard_tutorial <-> warning_mode
warning_mode <-> research_mode
visualization_analogy <-> meme_style
meme_style <-> fallback_summary
fallback_summary <-> standard_tutorial
fallback_summary <-> research_mode
```

## Best Practices

1. **Always check compatibility** before switching skills to ensure smooth transitions
2. **Use SkillManager** for complex workflows with multiple skill switches
3. **Consider global_tone** when finding compatible skills to maintain consistency
4. **Register custom skills** for domain-specific requirements
5. **Validate skill graph** after registering custom skills to ensure integrity

## Requirements Mapping

This implementation satisfies the following requirements:

- **需求 4.1**: standard_tutorial Skill support
- **需求 4.2**: warning_mode Skill support
- **需求 4.3**: visualization_analogy Skill support
- **需求 4.4**: research_mode Skill support
- **需求 4.5**: meme_style Skill support
- **需求 4.6**: fallback_summary Skill support
- **需求 11.3**: Skill compatibility management and dynamic loading
