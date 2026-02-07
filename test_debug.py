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
    print('Result keys:', list(result.keys()))
    print('Success:', result.get('success'))
    if 'state' in result:
        state = result['state']
        print('State keys:', list(state.keys()))
        print('Fragments count:', len(state.get('fragments', [])))
        fragments = state.get('fragments', [])
        if fragments:
            print('First fragment:', fragments[0])
        print('Final screenplay length:', len(state.get('final_screenplay', '') or state.get('screenplay', '')))
    else:
        print('No state key')
    print('=' * 70)

if __name__ == '__main__':
    asyncio.run(test())
