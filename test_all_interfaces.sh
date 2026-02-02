# RAG剧本生成Agent测试脚本
# 测试时间: 2026-02-02
# 测试所有接口和数据持久化

BASE_URL="http://localhost:8000"

echo "========================================="
echo "   RAG剧本生成Agent接口测试"
echo "========================================="
echo ""

# 1. 健康检查
echo "【1】健康检查"
echo "-----------------------------------"
curl -s "${BASE_URL}/health" | jq .
echo ""

# 2. 技能管理接口
echo "【2】技能管理接口测试"
echo "-----------------------------------"

# 2.1 获取所有技能
echo "2.1 获取所有技能列表"
SKILLS=$(curl -s "${BASE_URL}/skills")
echo "$SKILLS" | jq '{total_count: .skills | length, default_skill: .default_skill}'
echo ""

# 2.2 获取特定技能详情
echo "2.2 获取standard_tutorial技能详情"
curl -s "${BASE_URL}/skills/standard_tutorial" | jq '{skill_name: .skill_name, description: .description, tone: .tone}'
echo ""

# 2.3 获取技能配置
echo "2.3 获取standard_tutorial技能配置"
curl -s "${BASE_URL}/skills/standard_tutorial/config" | jq '.compatible_with'
echo ""

# 2.4 更新技能启用状态
echo "2.4 禁用meme_style技能"
curl -s -X PATCH "${BASE_URL}/skills/meme_style" \
  -H "Content-Type: application/json" \
  -d '{"is_enabled": false}' | jq '{skill_name: .skill_name, is_enabled: .is_enabled}'
echo ""

# 2.5 重新启用技能
echo "2.5 重新启用meme_style技能"
curl -s -X PATCH "${BASE_URL}/skills/meme_style" \
  -H "Content-Type: application/json" \
  -d '{"is_enabled": true}' | jq '{skill_name: .skill_name, is_enabled: .is_enabled}'
echo ""

# 2.6 创建新技能
echo "2.6 创建新技能custom_style"
curl -s -X POST "${BASE_URL}/skills" \
  -H "Content-Type: application/json" \
  -d '{
    "skill_name": "custom_style",
    "description": "自定义风格测试",
    "tone": "professional",
    "compatible_with": ["standard_tutorial", "research_mode"],
    "is_enabled": true,
    "is_default": false
  }' | jq '{skill_name: .skill_name, description: .description, tone: .tone}'
echo ""

# 3. 文档管理接口
echo ""
echo "【3】文档管理接口测试"
echo "-----------------------------------"

# 3.1 创建测试文档1 - 剧本创作指南
echo "3.1 上传文档：剧本创作指南"
DOC1=$(curl -s -X POST "${BASE_URL}/documents" \
  -F "title=剧本创作指南2026版" \
  -F "file_name=screenplay_guide_2026.txt" \
  -F "content=剧本创作的核心要素包括：1) 明确的主题和立意；2) 鲜活的人物塑造；3) 紧凑的情节结构；4) 精彩的对话设计。剧本应该有清晰的三幕结构：开端、发展、高潮。人物对话要符合人物性格，推动情节发展。" \
  -F "category=guide")
echo "$DOC1" | jq '{doc_id: .doc_id, title: .title, category: .category}'
DOC1_ID=$(echo "$DOC1" | jq -r '.doc_id')
echo ""

# 3.2 创建测试文档2 - 技术文档规范
echo "3.2 上传文档：技术文档规范"
DOC2=$(curl -s -X POST "${BASE_URL}/documents" \
  -F "title=技术文档写作规范" \
  -F "file_name=tech_writing_guide.txt" \
  -F "content=技术文档应该遵循以下原则：1) 清晰简洁；2) 结构化呈现；3) 代码示例完整；4) 术语统一；5) 循序渐进。使用Markdown格式，包含标题层级、代码块、表格等元素。技术文档的目标读者通常是开发人员，需要提供准确的技术细节和可操作的指导。" \
  -F "category=standard")
echo "$DOC2" | jq '{doc_id: .doc_id, title: .title, category: .category}'
DOC2_ID=$(echo "$DOC2" | jq -r '.doc_id')
echo ""

# 3.3 创建测试文档3 - 产品需求文档
echo "3.3 上传文档：产品需求文档模板"
DOC3=$(curl -s -X POST "${BASE_URL}/documents" \
  -F "title=产品需求文档PRD模板" \
  -F "file_name=prd_template.txt" \
  -F "content=产品需求文档(PRD)应该包含以下章节：1) 文档概述和版本历史；2) 产品愿景和目标用户；3) 功能需求清单；4) 用户故事和验收标准；5) 交互设计和原型链接；6) 非功能需求如性能、安全性；7) 依赖关系和风险说明。PRD是产品、开发、设计、测试沟通的重要桥梁。" \
  -F "category=template")
echo "$DOC3" | jq '{doc_id: .doc_id, title: .title, category: .category}'
DOC3_ID=$(echo "$DOC3" | jq -r '.doc_id')
echo ""

# 3.4 获取文档列表
echo "3.4 获取文档列表"
curl -s "${BASE_URL}/documents" | jq '{total: .total, documents: [.documents[] | {id: .doc_id, title: .title, category: .category}]}'
echo ""

# 3.5 搜索文档
echo "3.5 搜索包含'技术'的文档"
curl -s "${BASE_URL}/documents/search?query=技术" | jq '{results: [.documents[] | {id: .doc_id, title: .title, score: .score}]}'
echo ""

# 4. 剧本生成接口
echo ""
echo "【4】剧本生成接口测试"
echo "-----------------------------------"

# 4.1 生成剧本 - 使用standard_tutorial技能
echo "4.1 生成剧本：主题='如何编写技术文档'，技能=standard_tutorial"
TASK1=$(curl -s -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "如何编写技术文档",
    "context": "目标读者是初级开发人员",
    "skill_name": "standard_tutorial",
    "max_retries": 2
  }')
echo "$TASK1" | jq '{task_id: .task_id, status: .status, skill_name: .skill_name}'
TASK1_ID=$(echo "$TASK1" | jq -r '.task_id')
echo ""

# 4.2 生成剧本 - 使用research_mode技能
echo "4.2 生成剧本：主题='API设计最佳实践'，技能=research_mode"
TASK2=$(curl -s -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "API设计最佳实践",
    "context": "需要涵盖RESTful API设计原则",
    "skill_name": "research_mode",
    "max_retries": 2
  }')
echo "$TASK2" | jq '{task_id: .task_id, status: .status, skill_name: .skill_name}'
TASK2_ID=$(echo "$TASK2" | jq -r '.task_id')
echo ""

# 4.3 生成剧本 - 使用meme_style技能
echo "4.3 生成剧本：主题='代码审查的重要性'，技能=meme_style"
TASK3=$(curl -s -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "代码审查的重要性",
    "context": "用轻松幽默的方式讲解",
    "skill_name": "meme_style",
    "max_retries": 2
  }')
echo "$TASK3" | jq '{task_id: .task_id, status: .status, skill_name: .skill_name}'
TASK3_ID=$(echo "$TASK3" | jq -r '.task_id')
echo ""

# 4.4 查询任务结果
echo "4.4 查询任务结果 (等待5秒后查询)"
sleep 5
echo ""
echo "4.4.1 任务1结果:"
curl -s "${BASE_URL}/result/${TASK1_ID}" | jq '{task_id: .task_id, status: .status, skill_name: .skill_name, has_outline: (.outline != null), outline_length: (.outline | length)}'
echo ""

echo "4.4.2 任务2结果:"
curl -s "${BASE_URL}/result/${TASK2_ID}" | jq '{task_id: .task_id, status: .status, skill_name: .skill_name, has_screenplay: (.screenplay != null)}'
echo ""

echo "4.4.3 任务3结果:"
curl -s "${BASE_URL}/result/${TASK3_ID}" | jq '{task_id: .task_id, status: .status, skill_name: .skill_name}'
echo ""

# 5. 任务调整接口
echo ""
echo "【5】任务调整接口测试"
echo "-----------------------------------"

# 5.1 使用不同技能调整任务1
echo "5.1 调整任务1：切换到research_mode技能"
ADJUST1=$(curl -s -X POST "${BASE_URL}/adjust/${TASK1_ID}" \
  -H "Content-Type: application/json" \
  -d '{"skill_name": "research_mode"}')
echo "$ADJUST1" | jq '{task_id: .task_id, status: .status, skill_name: .skill_name}'
echo ""

# 5.2 再次查询调整后的结果
sleep 3
echo "5.2 查询调整后的任务1结果:"
curl -s "${BASE_URL}/result/${TASK1_ID}" | jq '{task_id: .task_id, status: .status, skill_name: .skill_name, direction_changes: (.direction_changes | length)}'
echo ""

# 6. 删除测试
echo ""
echo "【6】删除接口测试"
echo "-----------------------------------"

# 6.1 删除测试文档
echo "6.1 删除测试文档3"
curl -s -X DELETE "${BASE_URL}/documents/${DOC3_ID}" | jq '{success: .success, doc_id: .doc_id}'
echo ""

# 6.2 验证文档已删除
echo "6.2 验证文档列表（应该只有2个文档）"
curl -s "${BASE_URL}/documents" | jq '{total: .total}'
echo ""

# 7. 数据库持久化验证
echo ""
echo "【7】数据库持久化验证"
echo "-----------------------------------"

echo "7.1 tasks表记录数:"
PGPASSWORD=123456 psql -h localhost -p 5433 -U postgres -d Screenplay -t -c "SELECT COUNT(*) FROM screenplay.tasks;" 2>/dev/null | xargs echo

echo "7.2 documents表记录数:"
PGPASSWORD=123456 psql -h localhost -p 5433 -U postgres -d Screenplay -t -c "SELECT COUNT(*) FROM screenplay.documents;" 2>/dev/null | xargs echo

echo "7.3 workspace_skills表记录数:"
PGPASSWORD=123456 psql -h localhost -p 5433 -U postgres -d Screenplay -t -c "SELECT COUNT(*) FROM screenplay.workspace_skills;" 2>/dev/null | xargs echo

echo ""
echo "7.4 查看所有任务详情:"
PGPASSWORD=123456 psql -h localhost -p 5433 -U postgres -d Screenplay -c "SELECT task_id, status, skill_name, topic, created_at FROM screenplay.tasks ORDER BY created_at;" 2>/dev/null

echo ""
echo "7.5 查看所有文档详情:"
PGPASSWORD=123456 psql -h localhost -p 5433 -U postgres -d Screenplay -c "SELECT doc_id, title, category, indexed_at FROM screenplay.documents ORDER BY indexed_at;" 2>/dev/null

echo ""
echo "========================================="
echo "   测试完成!"
echo "========================================="
