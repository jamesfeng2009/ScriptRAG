"""
Planner Agent - Generates initial screenplay outline

本模块实现规划器智能体，负责分析用户主题和项目上下文，
生成包含 5-10 步的结构化剧本大纲。
"""

import logging
from typing import List
from ..models import SharedState, OutlineStep
from ...services.llm.service import LLMService

logger = logging.getLogger(__name__)


async def plan_outline(state: SharedState, llm_service: LLMService) -> SharedState:
    """
    规划器智能体主函数 - 生成初始剧本大纲
    
    功能：
    1. 分析用户主题和项目上下文
    2. 生成包含 5-10 步的结构化大纲
    3. 为每个步骤创建 OutlineStep 对象
    4. 为每个步骤估算预期的检索关键词
    
    Args:
        state: 共享状态对象
        llm_service: LLM 服务实例
        
    Returns:
        更新后的共享状态对象
    """
    logger.info("Planner agent started: Generating outline")
    
    # 记录开始日志
    state.add_log_entry(
        agent_name="planner",
        action="start_planning",
        details={
            "user_topic": state.user_topic,
            "project_context": state.project_context
        }
    )
    
    try:
        # 构建提示词
        prompt = _build_planning_prompt(state.user_topic, state.project_context)
        
        # 调用 LLM 生成大纲（使用高性能模型）
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的剧本规划师，擅长将复杂主题分解为清晰的步骤。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = await llm_service.chat_completion(
            messages=messages,
            task_type="high_performance",
            temperature=0.7,
            max_tokens=2000
        )
        
        # 解析 LLM 响应生成大纲步骤
        outline_steps = _parse_outline_response(response)
        
        # 验证大纲步骤数量（5-10 步）
        if len(outline_steps) < 5:
            logger.warning(f"Generated outline has only {len(outline_steps)} steps, padding to 5")
            outline_steps = _pad_outline(outline_steps, 5)
        elif len(outline_steps) > 10:
            logger.warning(f"Generated outline has {len(outline_steps)} steps, truncating to 10")
            outline_steps = outline_steps[:10]
        
        # 更新状态
        state.outline = outline_steps
        state.current_step_index = 0
        
        # 记录成功日志
        state.add_log_entry(
            agent_name="planner",
            action="outline_generated",
            details={
                "num_steps": len(outline_steps),
                "steps": [
                    {"step_id": step.step_id, "description": step.description}
                    for step in outline_steps
                ]
            }
        )
        
        logger.info(f"Planner agent completed: Generated {len(outline_steps)} steps")
        
        return state
        
    except Exception as e:
        logger.error(f"Planner agent failed: {str(e)}")
        
        # 记录错误日志
        state.add_log_entry(
            agent_name="planner",
            action="planning_failed",
            details={"error": str(e)}
        )
        
        # 创建回退大纲
        state.outline = _create_fallback_outline(state.user_topic)
        state.current_step_index = 0
        
        logger.warning("Using fallback outline due to planning failure")
        
        return state


def _build_planning_prompt(user_topic: str, project_context: str) -> str:
    """
    构建规划提示词
    
    Args:
        user_topic: 用户主题
        project_context: 项目上下文
        
    Returns:
        格式化的提示词
    """
    prompt = f"""请为以下主题生成一个结构化的剧本大纲：

主题：{user_topic}

项目上下文：{project_context if project_context else "无特定上下文"}

要求：
1. 生成 5-10 个步骤，每个步骤应该是一个独立的章节或主题
2. 每个步骤应该有清晰的描述，说明该步骤要讲解的内容
3. 步骤之间应该有逻辑连贯性，形成完整的叙述流程
4. 为每个步骤估算可能需要检索的关键词（用于后续 RAG 检索）

请按以下格式输出（每个步骤一行）：
步骤1: [步骤描述] | 关键词: [关键词1, 关键词2, ...]
步骤2: [步骤描述] | 关键词: [关键词1, 关键词2, ...]
...

示例：
步骤1: 介绍 FastAPI 框架的基本概念和优势 | 关键词: FastAPI, ASGI, 异步框架
步骤2: 创建第一个 FastAPI 应用程序 | 关键词: FastAPI, 路由, 装饰器
"""
    
    return prompt


def _parse_outline_response(response: str) -> List[OutlineStep]:
    """
    解析 LLM 响应生成大纲步骤
    
    Args:
        response: LLM 响应文本
        
    Returns:
        OutlineStep 对象列表
    """
    outline_steps = []
    lines = response.strip().split('\n')
    
    step_id = 0
    for line in lines:
        line = line.strip()
        
        # 跳过空行和非步骤行
        if not line or not line.startswith('步骤'):
            continue
        
        try:
            # 解析步骤描述
            # 格式: 步骤1: [描述] | 关键词: [关键词列表]
            if ':' in line:
                # 移除 "步骤X:" 前缀
                content = line.split(':', 1)[1].strip()
                
                # 提取描述（关键词之前的部分）
                if '|' in content:
                    description = content.split('|')[0].strip()
                else:
                    description = content
                
                # 创建 OutlineStep 对象
                outline_step = OutlineStep(
                    step_id=step_id,
                    description=description,
                    status="pending",
                    retry_count=0
                )
                
                outline_steps.append(outline_step)
                step_id += 1
                
        except Exception as e:
            logger.warning(f"Failed to parse outline line: {line}, error: {str(e)}")
            continue
    
    return outline_steps


def _pad_outline(outline_steps: List[OutlineStep], min_steps: int) -> List[OutlineStep]:
    """
    填充大纲步骤到最小数量
    
    Args:
        outline_steps: 现有大纲步骤
        min_steps: 最小步骤数
        
    Returns:
        填充后的大纲步骤列表
    """
    while len(outline_steps) < min_steps:
        step_id = len(outline_steps)
        outline_steps.append(
            OutlineStep(
                step_id=step_id,
                description=f"补充步骤 {step_id + 1}：进一步探讨相关主题",
                status="pending",
                retry_count=0
            )
        )
    
    return outline_steps


def _create_fallback_outline(user_topic: str) -> List[OutlineStep]:
    """
    创建回退大纲（当 LLM 调用失败时使用）
    
    Args:
        user_topic: 用户主题
        
    Returns:
        回退大纲步骤列表
    """
    logger.info("Creating fallback outline")
    
    fallback_steps = [
        OutlineStep(
            step_id=0,
            description=f"介绍主题：{user_topic}",
            status="pending",
            retry_count=0
        ),
        OutlineStep(
            step_id=1,
            description="核心概念和基础知识",
            status="pending",
            retry_count=0
        ),
        OutlineStep(
            step_id=2,
            description="实践示例和应用场景",
            status="pending",
            retry_count=0
        ),
        OutlineStep(
            step_id=3,
            description="常见问题和解决方案",
            status="pending",
            retry_count=0
        ),
        OutlineStep(
            step_id=4,
            description="总结和最佳实践",
            status="pending",
            retry_count=0
        )
    ]
    
    return fallback_steps
