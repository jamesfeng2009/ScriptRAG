import asyncio
from demo_saint_seiya import init_workflow_services, SAINT_SEIYA_DOCUMENTS
from src.services.mocks import MockRetrievalService
from src.services.parser.tree_sitter_parser import TreeSitterParser
from src.services.core.summarization_service import SummarizationService
from src.domain.agents.node_factory import NodeFactory
from src.domain.agents.navigator import retrieve_content, _parallel_process_results, RetrievedDocument

async def test():
    llm_service, theme_loader, skill_service, task_service, orchestrator = init_workflow_services()
    
    # 创建服务
    retrieval_service = MockRetrievalService(documents=SAINT_SEIYA_DOCUMENTS)
    parser_service = TreeSitterParser()
    summarization_service = SummarizationService(llm_service)
    
    # 创建节点工厂
    node_factory = NodeFactory(
        llm_service=llm_service,
        retrieval_service=retrieval_service,
        parser_service=parser_service,
        summarization_service=summarization_service
    )
    
    # 测试 Navigator 检索
    print("=" * 70)
    print("测试 Navigator 检索")
    print("=" * 70)
    
    state = {
        'user_topic': '圣斗士星矢攻打十二宫',
        'current_step_index': 0,
        'outline': [
            {'step_id': 0, 'title': '第 1 部分', 'description': '关于圣斗士星矢的第一部分内容'}
        ],
        'chat_history': [],
        'messages': [],
        'enable_dynamic_adjustment': True,
        'current_skill': 'heated_battle',
        'execution_log': []
    }
    
    from src.domain.intent_parser import IntentParser
    intent_parser = IntentParser(llm_service)
    intent_result = await intent_parser.analyze(state['user_topic'])
    print(f"Intent: {intent_result}")
    
    # 检索内容
    updated_state, quality_eval = await retrieve_content(
        state=state,
        retrieval_service=retrieval_service,
        parser_service=parser_service,
        summarization_service=summarization_service,
        enable_parallel=True,
        enable_quality_eval=False,
        llm_service=llm_service,
        intent=intent_result
    )
    
    retrieved_docs = updated_state.get('retrieved_docs', [])
    print(f"\n检索到的文档数量: {len(retrieved_docs)}")
    for i, doc in enumerate(retrieved_docs):
        print(f"  文档 {i+1}: source={getattr(doc, 'source', 'unknown')}")
        print(f"         content preview: {getattr(doc, 'content', '')[:100]}...")
    
    # 测试 Writer
    print("\n" + "=" * 70)
    print("测试 Writer node")
    print("=" * 70)
    
    writer_state = {
        'user_topic': '圣斗士星矢攻打十二宫',
        'current_step_index': 0,
        'outline': [
            {'step_id': 0, 'title': '第 1 部分', 'description': '关于圣斗士星矢的第一部分内容'}
        ],
        'retrieved_docs': retrieved_docs,
        'current_skill': 'heated_battle',
        'execution_log': []
    }
    
    writer_result = await node_factory.writer_node(writer_state)
    
    print(f"\nWriter 结果:")
    print(f"  keys: {list(writer_result.keys())}")
    print(f"  error_flag: {writer_result.get('error_flag')}")
    
    if 'fragments' in writer_result:
        fragments = writer_result['fragments']
        print(f"  fragments count: {len(fragments)}")
        if fragments:
            print(f"  First fragment content preview: {fragments[0].get('content', '')[:200]}...")
    else:
        print("  No fragments in result")
    
    if 'execution_log' in writer_result:
        logs = writer_result['execution_log']
        for log in logs:
            if isinstance(log, dict):
                print(f"  Log: {log.get('action')}")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    asyncio.run(test())
