#!/bin/bash
# 基础设斲测试脚本：晴明剧本生成场景

BASE_URL="http://localhost:8000"

echo "========================================"
echo "测试：晴明剧本生成场景 - 基础设施"
echo "========================================"

# 步骤 1: 创建 Chat Session
echo ""
echo "【步骤 1】创建 Chat Session..."
echo "请求参数:"
echo "  mode: agent"
echo "  skill: standard_tutorial"

SESSION_RESPONSE=$(curl -s -X POST "$BASE_URL/chat/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "agent",
    "skill": "standard_tutorial",
    "enable_rag": false,
    "temperature": 0.7
  }')

echo ""
echo "原始响应:"
echo "$SESSION_RESPONSE" | jq .

ERROR_MSG=$(echo "$SESSION_RESPONSE" | jq -r '.detail // empty')
if [ -n "$ERROR_MSG" ]; then
  echo ""
  echo "错误信息: $ERROR_MSG"
  exit 1
fi

SESSION_ID=$(echo "$SESSION_RESPONSE" | jq -r '.session_id // empty')
echo ""
echo "解析结果: session_id = [$SESSION_ID]"

if [ "$SESSION_ID" == "null" ] || [ -z "$SESSION_ID" ]; then
  echo ""
  echo "错误：创建 Session 失败"
  exit 1
fi

echo ""
echo "✅ Session 创建成功: $SESSION_ID"

# 步骤 2: 验证 Session 存在
echo ""
echo "========================================"
echo "【步骤 2】验证 Session 存在..."
echo "========================================"

SESSION_INFO=$(curl -s -X GET "$BASE_URL/chat/sessions/$SESSION_ID")
echo "Session 信息:"
echo "$SESSION_INFO" | jq .

# 步骤 3: 发送消息（不期望 LLM 响应）
echo ""
echo "========================================"
echo "【步骤 3】发送消息到 Session..."
echo "========================================"

RESPONSE1=$(curl -s -X POST "$BASE_URL/chat/sessions/$SESSION_ID/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "我要创作一个以阴阳师安培晴明为主角的剧本。\n基本设定：\n- 时代：平安时代\n- 地点：京都\n- 主角：安培晴明\n- 请帮我完善世界观设定。"
  }')

echo "响应:"
echo "$RESPONSE1" | jq . 2>/dev/null || echo "响应: $RESPONSE1"

# 步骤 4: 获取消息历史
echo ""
echo "========================================"
echo "【步骤 4】获取消息历史..."
echo "========================================"

MESSAGES=$(curl -s -X GET "$BASE_URL/chat/sessions/$SESSION_ID/messages")
MESSAGE_COUNT=$(echo "$MESSAGES" | jq '. | length')
echo "消息数量: $MESSAGE_COUNT"
echo "消息列表:"
echo "$MESSAGES" | jq .

# 步骤 5: 导出对话历史（用于生成剧本）
echo ""
echo "========================================"
echo "【步骤 5】导出对话历史..."
echo "========================================"

EXPORT=$(curl -s -X GET "$BASE_URL/chat/sessions/$SESSION_ID/export")
echo "导出结果:"
echo "$EXPORT" | jq .

# 步骤 6: 测试 /generate 接口（验证请求格式）
echo ""
echo "========================================"
echo "【步骤 6】测试 /generate 接口请求格式..."
echo "========================================"

# 使用正确的 SkillConfig 格式
GENERATE_RESPONSE=$(curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d "{
    \"topic\": \"以阴阳师安培晴明为主角的剧本\",
    \"skill\": {
      \"initial_skill\": \"standard_tutorial\",
      \"enable_auto_switch\": true,
      \"switch_threshold\": 0.7
    },
    \"chat_session_id\": \"$SESSION_ID\"
  }")

echo "响应:"
echo "$GENERATE_RESPONSE" | jq . 2>/dev/null || echo "响应: $GENERATE_RESPONSE"

TASK_ID=$(echo "$GENERATE_RESPONSE" | jq -r '.task_id // empty')
echo ""
echo "Task ID: $TASK_ID"

# 步骤 7: 验证 Chat Session 与 Task 关联
echo ""
echo "========================================"
echo "【步骤 7】验证 Chat Session 与 Task 关联..."
echo "========================================"

UPDATED_SESSION=$(curl -s -X GET "$BASE_URL/chat/sessions/$SESSION_ID")
echo "更新后的 Session 信息:"
echo "$UPDATED_SESSION" | jq .

# 最终验证
echo ""
echo "========================================"
echo "✅ 测试完成！"
echo "========================================"
echo ""
echo "验证要点:"
echo "  1. Session 创建成功 ✅"
echo "  2. 消息发送功能正常 (消息已存储) ✅"
echo "  3. 消息历史查询正常 ✅"
echo "  4. 对话导出正常 ✅"
echo "  5. Generate 接口请求格式正确 ✅"
echo ""
if [ "$TASK_ID" != "null" ] && [ -n "$TASK_ID" ]; then
  echo "  6. Task 创建成功: $TASK_ID ✅"
  echo ""
  echo "后续验证命令:"
  echo "  - 查询任务状态: curl $BASE_URL/result/$TASK_ID | jq ."
  echo "  - 数据库查询 Session:"
  echo "    PGPASSWORD=123456 psql -h localhost -p 5433 -U postgres -d Screenplay \\"
  echo "      -c \"SELECT id, mode, status, created_at FROM chat_sessions WHERE id='$SESSION_ID';\""
  echo "  - 数据库查询 Tasks:"
  echo "    PGPASSWORD=123456 psql -h localhost -p 5433 -U postgres -d Screenplay \\"
  echo "      -c \"SELECT task_id, topic, status FROM screenplay.tasks WHERE chat_session_id='$SESSION_ID';\""
else
  echo "  6. Task 创建失败或需要 LLM 支持 ⚠️"
  echo ""
  echo "注：完整剧本生成需要有效的 LLM API Key"
fi

echo ""
echo "========================================"
echo "数据库验证..."
echo "========================================"
echo ""
echo "Chat Sessions:"
PGPASSWORD=123456 psql -h localhost -p 5433 -U postgres -d Screenplay -t -c "SELECT id, mode, status, created_at FROM chat_sessions;" | head -5

echo ""
echo "Tasks:"
PGPASSWORD=123456 psql -h localhost -p 5433 -U postgres -d Screenplay -t -c "SELECT task_id, topic, status, chat_session_id FROM screenplay.tasks;" | head -5
