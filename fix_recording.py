#!/usr/bin/env python
"""批量修改 orchestrator.py，将 _record_agent_execution 从结束后改为开始前"""

import re

file_path = "/Users/fengyu/Downloads/myproject/workspace/agent-skills-demo/src/application/orchestrator.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

nodes_to_fix = [
    "_intent_parser_node",
    "_quality_eval_node", 
    "_rag_analyzer_node",
    "_dynamic_director_node",
    "_skill_recommender_node",
    "_director_node",
    "_pivot_manager_node",
    "_retry_protection_node",
    "_writer_node",
    "_fact_checker_node",
    "_step_advancer_node",
    "_compiler_node",
    "_collaboration_manager_node",
    "_execution_tracer_node",
    "_agent_reflection_node",
    "_editor_node"
]

for node_name in nodes_to_fix:
    pattern = rf'(async def {node_name}.*?{{\n)(.*?await self\._record_agent_execution\(\n.*?node_name="{node_name.lstrip("_")}",\n.*?state=state,\n.*?updates=updates\n\)\n)'
    
    match = re.search(pattern, content, re.DOTALL)
    if match:
        print(f"Found {node_name}")
        before = match.group(1)
        after = match.group(2)
        
        new_after = re.sub(
            r'await self\._record_agent_execution\(\n.*?node_name=".*?",\n.*?state=state,\n.*?updates=updates\n\)\n',
            '',
            after,
            flags=re.DOTALL
        )
        
        content = content.replace(before + after, before + new_after)
        print(f"  - Removed _record_agent_execution from end")
    else:
        print(f"Not found or already fixed: {node_name}")

print("\nDone!")
