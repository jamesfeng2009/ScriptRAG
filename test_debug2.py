import asyncio
from demo_saint_seiya import init_workflow_services

async def test():
    llm_service, theme_loader, skill_service, task_service, orchestrator = init_workflow_services()
    
    initial_state = {
        'user_topic': '圣斗士星矢攻打十二宫',
        'chat_history': [],
        'messages': [],
        'enable_dynamic_adjustment': True,
        'current_skill': 'heated_battle'
    }
    
    result = await orchestrator.execute(initial_state=initial_state, recursion_limit=100)
    
    print('=' * 70)
    print('Success:', result.get('success'))
    
    if 'state' in result:
        state = result['state']
        
        # 检查关键字段
        print('current_step_index:', state.get('current_step_index'))
        print('outline:', state.get('outline'))
        print('error_flag:', state.get('error_flag'))
        print('workflow_complete:', state.get('workflow_complete'))
        
        fragments = state.get('fragments', [])
        print('\nFragments count:', len(fragments))
        if fragments:
            print('First fragment:', fragments[0])
        
        # 检查是否有 director_feedback
        director_feedback = state.get('director_feedback')
        print('\ndirector_feedback:', director_feedback)
        
        # 检查 execution_log
        logs = state.get('execution_log', [])
        print('\nExecution logs:')
        for log in logs[-10:]:  # 只显示最后 10 条
            agent = log.get('agent', '')
            action = log.get('action', '')
            if agent == 'writer':
                print(f"  Writer: {action}")
        
    print('=' * 70)

if __name__ == '__main__':
    asyncio.run(test())
